from __future__ import annotations

import gc
import copy
import math
import os
import subprocess
import threading
import time
import types
from pathlib import Path
from typing import Callable

from .config import Settings
from .encoding import encode_video_crf
from .schemas import STILL_IMAGE_MODES, GenerateRequest


PIXEL_UPSCALER_FILENAME = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"

# fp8 layerwise-casting skip list, verified by module enumeration against
# Lightricks/LTX-2.5-Diffusers subfolder=transformer at the pinned revision
# (faithful copy of scratch_fp8_probe/fp8_common.py::FP8_SKIP_MODULES_PATTERN;
# protects embeddings / patchify / AdaLN / scale-shift / gate projections and
# casts only the big attention/FF Linear layers inside transformer_blocks).
FP8_SKIP_MODULES_PATTERN = (
    "norm",
    "^proj_in$",
    "^proj_out$",
    "^audio_proj_in$",
    "^audio_proj_out$",
    "time_embed",
    "audio_time_embed",
    "av_cross_attn_video_scale_shift",
    "av_cross_attn_audio_scale_shift",
    "av_cross_attn_video_a2v_gate",
    "av_cross_attn_audio_v2a_gate",
    "prompt_adaln",
    "audio_prompt_adaln",
    "to_gate_logits",  # small per-block gate projections, keep precise
)


def _patch_flex_for_real_kernels() -> None:
    """diffusers' flex processor calls torch flex_attention eagerly, which falls back to the
    math kernel (materializes the full NxN score matrix -> OOM at production grids), and
    create_block_mask without _compile=True materializes a full bool mask. Compile both so
    the actual block-sparse flex kernel runs."""
    import torch
    import torch.nn.attention.flex_attention as fa_mod

    if getattr(fa_mod, "_ltx25_flex_patched", False):
        return
    import torch._dynamo as dynamo

    # Many distinct grids/head-counts across decoder stages: the default recompile limit (8)
    # silently drops back to the eager math fallback (full NxN scores -> OOM).
    dynamo.config.recompile_limit = 256
    dynamo.config.cache_size_limit = 256
    orig_flex = fa_mod.flex_attention
    orig_cbm = fa_mod.create_block_mask
    compiled_flex = torch.compile(orig_flex)

    def cbm(*args, **kwargs):
        # `_compile=True` internally calls `torch.compile(create_block_mask)(...)`, which
        # resolves the module-global again -> restore the original around the call to
        # avoid infinite recursion through this wrapper.
        kwargs.setdefault("_compile", True)
        fa_mod.create_block_mask = orig_cbm
        try:
            return orig_cbm(*args, **kwargs)
        finally:
            fa_mod.create_block_mask = cbm

    fa_mod.flex_attention = compiled_flex
    fa_mod.create_block_mask = cbm
    fa_mod._ltx25_flex_patched = True


class _ProgressRamp:
    """Smoothly advances progress from `lo` toward `hi` while a long opaque step
    (diffusion decode has no step callback) is running."""

    def __init__(self, progress: Callable[[float], None], lo: float, hi: float, tau_s: float):
        self._progress = progress
        self._lo, self._hi, self._tau = lo, hi, tau_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        start = time.monotonic()
        while not self._stop.wait(2.0):
            elapsed = time.monotonic() - start
            self._progress(self._lo + (self._hi - self._lo) * (1 - math.exp(-elapsed / self._tau)))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._progress(self._hi)


class GenerationInterrupted(Exception):
    """Raised from `progress_callback()` when a `POST /api/interrupt` request has been
    accepted for the currently-running job.

    Only checked at diffusers `callback_on_step_end` boundaries (i.e. once per
    denoise step), so reaction time is bounded by one step's wall-clock cost --
    this mirrors the H3 backend's `core/runner.py::GenerationInterrupted` (same
    endpoint name/shape, same HTTP 499 on the caller side) so `mv_studio_V2` does not
    need to special-case either backend. This does not abort any in-flight CUDA op;
    it only stops the pipeline from starting its *next* step. The existing
    `JobManager._run()`'s `except Exception` / `finally: self._save(job)` path already
    handles marking the job failed and releasing the worker for the next job, so no
    additional cleanup is required here.
    """


class _InterruptController:
    """Process-wide single-flight interrupt flag.

    Only one job runs at a time (`JobManager._run()`'s single worker thread pulls
    from a `queue.Queue` one at a time), so a single flag is enough. Guarded by a
    lock since it is written from the FastAPI request-handling thread and read from
    the worker thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = False
        self._job_id: str | None = None

    def begin(self, job_id: str | None) -> None:
        """Call exactly once, right before a job's `generate()` call starts, so a
        stale interrupt request from a previous job cannot instantly kill this one."""
        with self._lock:
            self._requested = False
            self._job_id = job_id

    def request(self, job_id: str | None) -> bool:
        """Request an interrupt. If `job_id` is given and does not match the
        currently-running job, does nothing and returns False (so a request racing
        with the start of an unrelated job cannot kill it). Returns True if the flag
        was actually set."""
        with self._lock:
            if self._job_id is None:
                return False
            if job_id is not None and job_id != self._job_id:
                return False
            self._requested = True
            return True

    def check(self) -> None:
        """Call from a step-boundary callback. Raises `GenerationInterrupted` if an
        interrupt has been requested for the currently-running job."""
        with self._lock:
            requested = self._requested
        if requested:
            raise GenerationInterrupted("generation was stopped by an interrupt request")

    def current_job_id(self) -> str | None:
        with self._lock:
            return self._job_id

    def end(self) -> None:
        """Call once the job has finished (success, failure, or interruption)."""
        with self._lock:
            self._job_id = None
            self._requested = False


interrupt_controller = _InterruptController()



def _subsample_distilled_sigmas(sigmas, steps):
    """蒸留σ列の間引き(steps<len のときのみ。先頭・末尾を含む等間隔選択)。"""
    try:
        steps = int(steps or 0)
    except (TypeError, ValueError):
        return sigmas
    n = len(sigmas)
    if steps <= 0 or steps >= n:
        return sigmas
    if steps == 1:
        picked = [sigmas[0]]
    else:
        idx = [round(i * (n - 1) / (steps - 1)) for i in range(steps)]
        picked = [sigmas[i] for i in idx]
    print(f"[ltx25] distilled sigma subsample: {steps}/{n} steps -> {picked}", flush=True)
    return picked


class LTXGenerator:
    """Lazily loads the gated model so health checks remain cheap."""

    def __init__(self, config: Settings):
        self.config = config
        self._pipe = None
        self._upsample_pipe = None
        self._temporal_upsample_pipe = None
        self._diffusion_decode_pipe = None
        self._graph_runner = None
        self._load_lock = threading.Lock()

    def load(self):
        if self._pipe is not None:
            return self._pipe
        with self._load_lock:
            if self._pipe is not None:
                return self._pipe
            import torch
            from diffusers import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline, LTX2VideoTransformer3DModel
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
            from transformers import Gemma4UnifiedForConditionalGeneration

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required for LTX-2.5 inference")
            model_dir = self.config.quantized_model_dir.resolve()
            text_encoder_dir = model_dir / "text_encoder_bnb_4bit"
            transformer_dir = model_dir / "transformer_bnb_4bit"
            if not text_encoder_dir.is_dir() or not transformer_dir.is_dir():
                raise RuntimeError(
                    f"Quantized LTX-2.5 components are missing under {model_dir}. "
                    "Run scripts/download_quantize_ltx25.py first."
                )
            upsampler_dir = model_dir / "latent_upsampler"
            if self.config.ltx25_load_upsamplers and not upsampler_dir.is_dir():
                raise RuntimeError(
                    f"LTX-2.5 latent upsampler is missing under {model_dir}. "
                    "Run scripts/download_quantize_ltx25.py --component quality first."
                )
            temporal_upsampler_dir = model_dir / "temporal_latent_upsampler"
            # Keep non-quantized norms/embeddings and activations in bf16 too.
            # The NF4 config only controls Linear4bit compute dtype on reload.
            text_encoder = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                text_encoder_dir, dtype=torch.bfloat16
            )
            if self.config.ltx25_transformer_precision == "bf16":
                # Release bf16 weights (~38GB, 96GB-class GPUs). text_encoder stays NF4.
                transformer = LTX2VideoTransformer3DModel.from_pretrained(
                    self.config.model_id,
                    subfolder="transformer",
                    revision=self.config.model_revision,
                    torch_dtype=torch.bfloat16,
                )
            elif self.config.ltx25_transformer_precision == "fp8":
                # Verified recipe from scratch_fp8_probe/ (F1-F5): load the bf16 Hub
                # shards on CPU, then apply diffusers layerwise casting
                # (storage=fp8_e4m3fn / compute=bf16) *while still on CPU*. Casting
                # on GPU instead needs a transient ~43GB bf16-on-GPU peak that OOMs
                # a real 48GB card (probe F4b established the CPU-cast path as the
                # 48GB-class recipe). Afterwards the module joins the normal
                # offload_mode handling below (model_cpu_offload verified: F2/F4).
                from diffusers.hooks import apply_layerwise_casting

                fp8_t0 = time.time()
                transformer = LTX2VideoTransformer3DModel.from_pretrained(
                    self.config.model_id,
                    subfolder="transformer",
                    revision=self.config.model_revision,
                    torch_dtype=torch.bfloat16,
                )
                apply_layerwise_casting(
                    transformer,
                    storage_dtype=torch.float8_e4m3fn,
                    compute_dtype=torch.bfloat16,
                    skip_modules_pattern=FP8_SKIP_MODULES_PATTERN,
                    non_blocking=False,
                )
                resident_gb = sum(
                    p.numel() * p.element_size() for p in transformer.parameters()
                ) / 1024**3
                print(
                    f"[ltx25] transformer precision=fp8 (layerwise cast applied on CPU "
                    f"in {time.time() - fp8_t0:.1f}s, resident weights {resident_gb:.1f}GB)",
                    flush=True,
                )
            elif self.config.ltx25_transformer_precision == "nvfp4":
                # Official Blackwell-native FP4 distilled transformer (single-file
                # ComfyUI format). Loaded straight onto the GPU by app/nvfp4.py:
                # quantized Linears become NVFP4Linear (torch._scaled_mm FP4 GEMM,
                # ~3.4x raw / ~1.8x per-layer vs bf16 incl. activation-quant cost).
                # The bnb transformer_dir is only used for its config.json (same
                # architecture); its weights are not read.
                import json as _json

                from .nvfp4 import load_nvfp4_transformer

                nvfp4_ckpt = self.config.ltx25_nvfp4_ckpt
                if not nvfp4_ckpt:
                    from huggingface_hub import hf_hub_download

                    nvfp4_ckpt = hf_hub_download(
                        "Lightricks/LTX-2.5",
                        "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
                    )
                with open(transformer_dir / "config.json") as fh:
                    nvfp4_cfg = _json.load(fh)
                nvfp4_t0 = time.time()
                transformer = load_nvfp4_transformer(
                    str(nvfp4_ckpt), nvfp4_cfg, torch.device("cuda")
                )
                print(
                    f"[ltx25] transformer precision=nvfp4 loaded in "
                    f"{time.time() - nvfp4_t0:.1f}s (FP4 GEMM, resident ~19GB)",
                    flush=True,
                )
            else:
                transformer = LTX2VideoTransformer3DModel.from_pretrained(
                    transformer_dir, dtype=torch.bfloat16
                )
            pipe = LTX2ConditionPipeline.from_pretrained(
                model_dir,
                text_encoder=text_encoder,
                transformer=transformer,
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            pipe.vae.enable_tiling()
            upsample_pipe = None
            temporal_upsample_pipe = None
            if self.config.ltx25_load_upsamplers:
                latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                    model_dir,
                    subfolder="latent_upsampler",
                    dtype=torch.bfloat16,
                    local_files_only=True,
                )
                upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
                if temporal_upsampler_dir.is_dir():
                    temporal_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                        model_dir,
                        subfolder="temporal_latent_upsampler",
                        dtype=torch.bfloat16,
                        local_files_only=True,
                    )
                    temporal_upsample_pipe = LTX2LatentUpsamplePipeline(
                        vae=pipe.vae, latent_upsampler=temporal_upsampler
                    )
            else:
                # リアルタイム用途向け(LTX25_LOAD_UPSAMPLERS=0): 常に upscale:false の
                # ため latent(950MB)/temporal(250MB)upsampler を常駐させない。
                print("[ltx25] upsamplers not loaded (LTX25_LOAD_UPSAMPLERS=0, -1.2GB)", flush=True)
            if self.config.offload_mode == "none":
                pipe.to("cuda")
                if upsample_pipe is not None:
                    upsample_pipe.to("cuda")
                if temporal_upsample_pipe is not None:
                    temporal_upsample_pipe.to("cuda")
            elif self.config.offload_mode == "sequential":
                pipe.enable_sequential_cpu_offload()
                if upsample_pipe is not None:
                    upsample_pipe.enable_sequential_cpu_offload()
                if temporal_upsample_pipe is not None:
                    temporal_upsample_pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
                if upsample_pipe is not None:
                    upsample_pipe.enable_model_cpu_offload()
                if temporal_upsample_pipe is not None:
                    temporal_upsample_pipe.enable_model_cpu_offload()
            if self.config.ltx25_te_diet:
                if self.config.offload_mode != "none":
                    print(
                        "[ltx25] LTX25_TE_DIET=1 ignored: requires OFFLOAD_MODE=none "
                        f"(got {self.config.offload_mode!r})",
                        flush=True,
                    )
                else:
                    from .tediet import apply_te_diet

                    freed = apply_te_diet(pipe.text_encoder)
                    torch.cuda.empty_cache()
                    print(
                        f"[ltx25] TE diet applied: embed_tokens -> CPU ({freed:.2f}GiB freed), "
                        "lm_head skipped (~0.5GB transient logits removed)",
                        flush=True,
                    )
            if self.config.ltx25_te_stream:
                if self.config.offload_mode != "none":
                    print(
                        "[ltx25] LTX25_TE_STREAM=1 ignored: requires OFFLOAD_MODE=none "
                        f"(got {self.config.offload_mode!r})",
                        flush=True,
                    )
                else:
                    # tediet の後に適用する(streamed_forward が diet_forward を内側に
                    # 取り込む)。tediet なしでも動作する(元 forward を包むだけ)。
                    from .testream import apply_te_stream

                    pinned = apply_te_stream(
                        pipe.text_encoder, window=self.config.ltx25_te_stream_window
                    )
                    torch.cuda.empty_cache()
                    print(
                        f"[ltx25] TE stream applied: {pinned:.2f}GiB layers -> pinned host "
                        f"(window={self.config.ltx25_te_stream_window}, encode +~0.16s, bit-exact)",
                        flush=True,
                    )
            if self.config.ltx25_compile_blocks != "off":
                # per-block torch.compile(app/compileblocks.py)。CUDA Graph の
                # install より前に適用する(graph は compile 済みブロックの呼び出しを
                # capture する必要がある)。compiled eager 単体は素の eager より遅い
                # ため、graph 無効時・nvfp4 以外・offload 有効時は適用しない。
                _cb = self.config.ltx25_compile_blocks
                if (
                    self.config.ltx25_transformer_precision == "nvfp4"
                    and self.config.ltx25_cuda_graph
                    and self.config.offload_mode == "none"
                ):
                    from .compileblocks import apply_block_compile

                    apply_block_compile(pipe.transformer, _cb)
                else:
                    print(
                        f"[ltx25] LTX25_COMPILE_BLOCKS={_cb} ignored: requires "
                        "precision=nvfp4 + LTX25_CUDA_GRAPH=1 + OFFLOAD_MODE=none "
                        f"(got precision={self.config.ltx25_transformer_precision!r}, "
                        f"cuda_graph={self.config.ltx25_cuda_graph}, "
                        f"offload={self.config.offload_mode!r})",
                        flush=True,
                    )
            if self.config.ltx25_cuda_graph:
                # transformer.forward 全体の CUDA Graph 化(app/cudagraph.py 参照)。
                # OFFLOAD_MODE=none 限定: model/sequential offload は重みのデバイスが
                # リクエスト間で動き、capture 済み graph が焼き込んだアドレスと
                # 食い違って黙って壊れるため適用しない。
                if self.config.offload_mode == "none":
                    from .cudagraph import ForwardGraphRunner

                    self._graph_runner = ForwardGraphRunner(
                        pipe.transformer,
                        max_captures=self.config.ltx25_cuda_graph_max_captures,
                    )
                    self._graph_runner.install()
                    print(
                        "[ltx25] CUDA graph capture enabled for transformer.forward "
                        f"(max_captures={self.config.ltx25_cuda_graph_max_captures})",
                        flush=True,
                    )
                else:
                    print(
                        "[ltx25] LTX25_CUDA_GRAPH=1 ignored: requires OFFLOAD_MODE=none "
                        f"(got {self.config.offload_mode!r})",
                        flush=True,
                    )
            self._pipe = pipe
            self._upsample_pipe = upsample_pipe
            self._temporal_upsample_pipe = temporal_upsample_pipe
        return self._pipe

    def unload(self) -> dict:
        """Release every pipeline/model reference so VRAM returns to (near) zero while
        the process stays alive (Phase 5a resident switching). The next generate()
        simply goes through load() again -- load() only checks `self._pipe is None`,
        so dropping the references restores the exact lazy-load entry state.
        Callers must ensure no job is running (app.main checks before calling)."""
        freed = []
        with self._load_lock:
            if self._graph_runner is not None:
                # capture 済み graph は transformer の重み・mempool を参照し続けるため、
                # パイプライン参照を落とす前に破棄しないと VRAM が返らない。
                self._graph_runner.reset()
                self._graph_runner.uninstall()
                self._graph_runner = None
                freed.append("cuda_graphs")
            for attr in ("_pipe", "_upsample_pipe", "_temporal_upsample_pipe",
                         "_diffusion_decode_pipe"):
                if getattr(self, attr) is not None:
                    setattr(self, attr, None)
                    freed.append(attr.lstrip("_"))
            gc.collect()
            allocated_gb = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    allocated_gb = round(torch.cuda.memory_allocated() / 1024**3, 3)
            except Exception:  # torch never imported / no CUDA: nothing to free
                pass
        return {"freed": freed, "allocated_gb": allocated_gb}

    def _configure_decode_tiling(self, num_frames: int, height: int, width: int) -> None:
        """decode 直前にタイル構成を決める(ジョブごと、decoder は常駐共有のため毎回設定)。

        既定タイル(768^2x80f / stride 704^2x56f)は 24GB 級を想定した保守値で、
        大出力ではタイル数と重複(オーバーラップ)計算が膨らむ(1536x896x121f で
        12 タイル・重複 約1.4x)。ここでは **空き VRAM の範囲で最大のタイル**
        (最小の分割数)を選ぶ: 分割は幅→高さ→フレームの順に増やし、
        タイル体積 <= 予算(空きVRAM / 0.34GB/Mpx / 1.25 マージン、
        probes/probe_decode_tiling.py の実測係数)を満たす最初の構成を採る。
        1024x576x121f では単一タイルになり decode 9.6s -> 7.9s(継ぎ目も消える)。
        LTX25_DECODE_SINGLE_TILE: auto(既定)/ on(常に単一タイル、VRAM検査なし)/
        off(常に既定タイル)。"""
        import torch

        decoder = self._diffusion_decode_pipe.diffusion_decoder
        mode = self.config.ltx25_decode_single_tile
        if mode == "off":
            decoder.enable_tiling()  # 既定値へ戻す
            return

        OVERLAP_PX = 64   # 既定タイルと同じ空間オーバーラップ(768-704)
        OVERLAP_F = 8     # 時間方向(既定24は保守的すぎるため最小の8n)
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        budget_mpx = free_gb / (0.34 * 1.25) * 1e6

        def tile_dims(nw: int, nh: int, nf: int) -> tuple[int, int, int]:
            tw = -(-width // nw) + (OVERLAP_PX if nw > 1 else 0)
            th = -(-height // nh) + (OVERLAP_PX if nh > 1 else 0)
            tf = -(-num_frames // nf) + (OVERLAP_F if nf > 1 else 0)
            return tw, th, tf

        # 幅→高さ→フレームの順で分割を増やし、予算に収まる最初の構成を採用
        candidates = [(1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 1), (2, 2, 2),
                      (3, 2, 2), (3, 3, 2), (4, 3, 2)]
        chosen = None
        for nw, nh, nf in candidates:
            tw, th, tf = tile_dims(nw, nh, nf)
            if mode == "on" or tw * th * tf <= budget_mpx:
                chosen = (nw, nh, nf, tw, th, tf)
                break
        if chosen is None:
            decoder.enable_tiling()
            print(
                f"[ltx25] decode tiling: default tiles (budget {budget_mpx/1e6:.0f}Mpx "
                f"too small for {width}x{height}x{num_frames}f)",
                flush=True,
            )
            return
        nw, nh, nf, tw, th, tf = chosen
        # stride は「各次元の刻み = ceil(dim/n)」。オーバーラップは tile_min 側にだけ
        # 足してあるため、分割数1の次元では引かない(引くと2タイル化してしまう)。
        decoder.enable_tiling(
            tile_sample_min_height=th,
            tile_sample_min_width=tw,
            tile_sample_min_num_frames=tf,
            tile_sample_stride_height=th - (OVERLAP_PX if nh > 1 else 0),
            tile_sample_stride_width=tw - (OVERLAP_PX if nw > 1 else 0),
            tile_sample_stride_num_frames=tf - (OVERLAP_F if nf > 1 else 0),
        )
        print(
            f"[ltx25] decode tiling: {nw}x{nh}x{nf} tiles of {tw}x{th}x{tf}f "
            f"(free {free_gb:.1f}GB, budget {budget_mpx/1e6:.0f}Mpx)",
            flush=True,
        )

    def load_diffusion_decoder(self):
        """Lazily load the LTX-2.5 diffusion decoder pipeline (~0.83GB, kept resident)."""
        if self._diffusion_decode_pipe is not None:
            return self._diffusion_decode_pipe
        with self._load_lock:
            if self._diffusion_decode_pipe is not None:
                return self._diffusion_decode_pipe
            import torch
            from diffusers import FlowMatchEulerDiscreteScheduler
            from diffusers.models.autoencoders import LTX2VideoDiffusionDecoderModel
            from diffusers.pipelines.ltx2 import LTX2VideoDiffusionDecodePipeline

            decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
                self.config.model_id,
                subfolder="diffusion_decoder",
                revision=self.config.model_revision,
                torch_dtype=torch.bfloat16,
            )
            decoder.to("cuda")
            # Bound per-tile attention grids (121f x 1024^2 untiled OOMs even at 96GB).
            decoder.enable_tiling()
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(self.config.quantized_model_dir.resolve()), subfolder="scheduler"
            )
            # Prefer NATTEN's fused na3d kernel (fetched from the Hub via `kernels`,
            # needs torch>=2.11 prebuilt variants); fall back to the compiled
            # flex-attention patch when unavailable.
            try:
                from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
                    LTX2VideoVaeNeighborhoodNattenProcessor,
                )

                decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
                print("[ltx25] diffusion decoder attention: NATTEN na3d (shi-labs/natten)", flush=True)
            except Exception as exc:
                # Compile flex-attention so the block-sparse kernel actually runs (the eager
                # fallback materializes full NxN scores and OOMs at production grids).
                print(
                    f"[ltx25] NATTEN unavailable ({type(exc).__name__}: {exc}); "
                    "diffusion decoder attention: compiled flex-attention fallback",
                    flush=True,
                )
                _patch_flex_for_real_kernels()
            self._diffusion_decode_pipe = LTX2VideoDiffusionDecodePipeline(
                diffusion_decoder=decoder, scheduler=scheduler, vae=None
            )
        return self._diffusion_decode_pipe

    def _decode_audio(self, pipe, audio_latents):
        """audio latents -> mel (audio_vae) -> waveform (vocoder).

        Needed when the main pipeline runs with output_type="latent" (diffusion decoder
        path). audio_vae.decode() bypasses the cpu-offload forward hook, so briefly move
        the (small) audio_vae to CUDA if it is offloaded.
        """
        import torch

        audio_vae = pipe.audio_vae
        original_device = next(audio_vae.parameters()).device
        with torch.no_grad():
            if original_device.type != "cuda" and torch.cuda.is_available():
                audio_vae.to("cuda")
            try:
                latents = audio_latents.to(next(audio_vae.parameters()).device, audio_vae.dtype)
                mel = audio_vae.decode(latents, return_dict=False)[0]
            finally:
                if original_device.type != "cuda":
                    audio_vae.to(original_device)
            # vocoder forward goes through the offload hook (inputs are moved too).
            audio = pipe.vocoder(mel)
        return audio

    def _finish_retake(self, source: Path, generated: Path, target: Path, request: GenerateRequest) -> None:
        """Keep source media outside the selected interval while using generated media inside it."""
        start, end = request.retake_start, request.retake_end
        video_input = "1:v:0" if request.regenerate_video else "0:v:0"
        command = ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-i", str(generated)]
        if request.regenerate_audio:
            audio_filter = (
                f"[0:a]atrim=0:{start},asetpts=PTS-STARTPTS[a0];"
                f"[1:a]atrim={start}:{end},asetpts=PTS-STARTPTS[a1];"
                f"[0:a]atrim=start={end},asetpts=PTS-STARTPTS[a2];"
                "[a0][a1][a2]concat=n=3:v=0:a=1[a]"
            )
            command += ["-filter_complex", audio_filter, "-map", video_input, "-map", "[a]"]
        else:
            command += ["-map", video_input, "-map", "0:a:0?"]
        command += ["-c:v", "copy", "-c:a", "aac", "-shortest", str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Retake media merge failed: {result.stderr.strip()[-500:]}")

    def _finish_extend(
        self, source: Path, generated: Path, target: Path, direction: str,
        context_seconds: float, extension_seconds: float,
    ) -> None:
        """Extract the newly generated region and append/prepend it to the source."""
        if direction == "end":
            generated_range = f"start={context_seconds}:end={context_seconds + extension_seconds}"
            order = "[sv][sa][gv][ga]concat=n=2:v=1:a=1[v][a]"
        else:
            generated_range = f"start=0:end={extension_seconds}"
            order = "[gv][ga][sv][sa]concat=n=2:v=1:a=1[v][a]"
        filters = (
            "[0:v]setpts=PTS-STARTPTS[sv];[0:a]asetpts=PTS-STARTPTS[sa];"
            f"[1:v]trim={generated_range},setpts=PTS-STARTPTS[gv];"
            f"[1:a]atrim={generated_range},asetpts=PTS-STARTPTS[ga];" + order
        )
        command = [
            "ffmpeg", "-y", "-v", "error", "-i", str(source), "-i", str(generated),
            "-filter_complex", filters, "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
            "-crf", str(self.config.ltx25_video_crf), "-pix_fmt", "yuv420p", "-c:a", "aac", str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Extend media merge failed: {result.stderr.strip()[-500:]}")

    def _get_mel_transform_16k(self, device):
        """a2v 条件付け用 MelSpectrogram(16kHz 固定パラメータ)のデバイス別キャッシュ。

        毎リクエストの構築(フィルタバンク計算 + .to(device))を省く(高速化②-2)。
        パラメータは従来のインライン構築と完全同一。
        """
        cache = getattr(self, "_mel_transform_cache", None)
        if cache is None:
            cache = {}
            self._mel_transform_cache = cache
        key = str(device)
        if key not in cache:
            import torchaudio
            cache[key] = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160,
                f_min=0.0, f_max=8000.0, n_mels=64, center=True, pad_mode="reflect",
                power=1.0, mel_scale="slaney", norm="slaney",
            ).to(device)
        return cache[key]

    @staticmethod
    def _decode_audio_file(source: Path, sample_rate: int, start: float, duration: float):
        """Decode a selected region to stereo float32 with ffmpeg."""
        import numpy as np
        command = [
            "ffmpeg", "-v", "error", "-ss", str(start), "-t", str(duration), "-i", str(source),
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sample_rate), "pipe:1",
        ]
        result = subprocess.run(command, capture_output=True)
        if result.returncode != 0:
            raise ValueError("Input audio could not be decoded")
        samples = np.frombuffer(result.stdout, dtype=np.float32)
        if samples.size < 2:
            raise ValueError("Selected audio region is empty")
        samples = samples[: samples.size // 2 * 2].reshape(-1, 2).T.copy()
        wanted = round(duration * sample_rate)
        if samples.shape[1] < wanted:
            samples = np.pad(samples, ((0, 0), (0, wanted - samples.shape[1])))
        return samples[:, :wanted]

    def _cast_lora_layers_to_bf16(self, pipe) -> None:
        """fp8 LoRA compatibility workaround (verified probe F5): lora_A/lora_B
        Linear layers created by load_lora_weights() *after* layerwise casting are
        materialized at the base layer's current storage dtype (fp8_e4m3fn), and the
        forward pass then fails with NotImplementedError '"addmm_cuda" not
        implemented for Float8_e4m3fn'. Cast the (tiny) LoRA modules back to bf16.
        No-op for nf4/bf16 precisions."""
        if self.config.ltx25_transformer_precision != "fp8":
            return
        import torch

        n_cast = 0
        for name, module in pipe.transformer.named_modules():
            if "lora_A" in name or "lora_B" in name:
                module.to(torch.bfloat16)
                n_cast += 1
        if n_cast:
            print(f"[ltx25] fp8 LoRA compat: cast {n_cast} lora_A/lora_B modules to bf16", flush=True)

    def generate(self, request: GenerateRequest, target: Path, progress: Callable[[float], None]) -> dict[str, float]:
        pipe = self.load()
        adapter_names = []
        lora_root = self.config.lora_dir.resolve()
        # LoRA を載せるジョブ(job loras / pixel upscale の IC-LoRA)は CUDA graph 不可:
        # capture は重みテンソルのアドレスを焼き込むため、adapter の付け外しをまたぐ
        # replay は stale な重みを黙って使う。eager に落とし、ジョブ後に capture を捨てる。
        _graph_lora_guard = self._graph_runner is not None and (
            bool(request.loras) or request.upscale_method == "pixel"
        )
        if _graph_lora_guard:
            self._graph_runner.enabled = False
        try:
            for index, item in enumerate(request.loras):
                path = (lora_root / item.id).resolve()
                if path.parent != lora_root or path.suffix.lower() != ".safetensors" or not path.is_file():
                    raise ValueError(f"LoRA file not found: {item.id}")
                adapter_name = f"job_lora_{index}"
                try:
                    pipe.load_lora_weights(path, adapter_name=adapter_name)
                except Exception as exc:
                    raise RuntimeError(f"LoRA could not be loaded ({item.id}): {exc}") from exc
                adapter_names.append(adapter_name)
            if adapter_names:
                pipe.set_adapters(adapter_names, adapter_weights=[item.strength for item in request.loras])
                self._cast_lora_layers_to_bf16(pipe)
            if request.mode in STILL_IMAGE_MODES:
                return self._generate_still_impl(request, target, progress)
            return self._generate_impl(request, target, progress)
        finally:
            if request.loras or request.upscale_method == "pixel":
                try:
                    pipe.unload_lora_weights()
                except Exception as exc:
                    print(f"[ltx25] LoRA cleanup failed: {exc}", flush=True)
            if _graph_lora_guard:
                # unload_lora_weights 後の構造復元を信用せず、防御的に capture を捨てる
                # (次の非 LoRA ジョブが ~1s で再 capture する)。
                self._graph_runner.reset()
                self._graph_runner.enabled = True

    def _generate_still_impl(
        self, request: GenerateRequest, target: Path, progress: Callable[[float], None]
    ) -> dict[str, float]:
        """Still-image modes (t2i / refine_image / ref2i), kept independent from the
        video path. Recipes are faithful ports of the verified probes under
        scratch_t2i_probe/ (P3 for t2i, P4_nf17 for t2i+diffusion decoder, P6/P6b for
        refine_image, P7a/P7b for ref2i)."""
        import numpy as np
        import torch
        from PIL import Image
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.utils import load_image

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pipe = self.load()
        generator = torch.Generator(device="cpu").manual_seed(request.seed)

        image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        input_dir = self.config.input_dir.resolve()
        conditions = []
        for condition in request.conditions:
            matches = list(input_dir.glob(f"{condition.asset_id}.*"))
            if len(matches) != 1:
                raise ValueError(f"Input asset not found: {condition.asset_id}")
            source = matches[0]
            if condition.kind != "image" or source.suffix.lower() not in image_suffixes:
                raise ValueError(f"Still-image modes accept image assets only: {condition.asset_id}")
            strength = request.strength if request.mode == "refine_image" else condition.strength
            conditions.append(
                LTX2VideoCondition(frames=load_image(str(source)), index=condition.index, strength=strength)
            )

        # t2i (no image conditioning) follows the configured default decoder — the
        # diffusion decoder is dramatically sharper for pure text-to-still (probe P4).
        # Image-conditioned stills stay on VAE (diffusion decoder blurs them, probe P8).
        if request.mode == "t2i":
            decoder_kind = request.decoder or self.config.ltx25_decoder
        else:
            decoder_kind = request.decoder or "vae"
        if request.mode in {"refine_image", "ref2i"} and decoder_kind == "diffusion":
            # Probe result: image-conditioned latents through the diffusion decoder blur.
            print(
                f"[ltx25] {request.mode}: diffusion decoder is not supported for "
                "image-conditioned still output (probe: blurred results); falling back to the VAE decoder",
                flush=True,
            )
            decoder_kind = "vae"

        # Sharpness A/B (Q5): for pure text-to-still, a clarity suffix measurably
        # improves fine detail at zero cost. Image-conditioned stills keep the
        # user's prompt untouched (untested there; fidelity to the input matters more).
        still_prompt = request.prompt
        still_negative = request.negative_prompt
        if request.mode == "t2i":
            still_prompt = f"{still_prompt.rstrip()} sharp focus, crisp fine detail, high clarity, minimal haze."
            extra_neg = "soft focus, hazy, bloom"
            still_negative = f"{still_negative}, {extra_neg}" if still_negative else extra_neg

        def progress_callback(offset: float, span: float, step_count: int):
            def on_step(_pipe, step: int, _timestep, callback_kwargs):
                # Step-boundary interrupt check (see GenerationInterrupted's docstring):
                # runs once per denoise step, right where diffusers already calls back
                # into us, so no extra hook into the pipeline internals is needed.
                interrupt_controller.check()
                progress(min(offset + ((step + 1) / step_count) * span, 0.96))
                return callback_kwargs

            return on_step

        if request.mode == "ref2i":
            # Probe P7a/P7b: single-stage base 30-step schedule, tiled VAE decode.
            with torch.no_grad():
                video_np, _audio = pipe(
                    prompt=still_prompt,
                    negative_prompt=still_negative,
                    conditions=conditions,
                    width=request.width,
                    height=request.height,
                    num_frames=request.num_frames,
                    frame_rate=request.fps,
                    num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    stg_scale=1.0,
                    modality_scale=3.0,
                    audio_guidance_scale=7.0,
                    audio_stg_scale=1.0,
                    audio_modality_scale=3.0,
                    enable_prompt_enhancement=request.enhance_prompt,
                    generator=generator,
                    output_type="np",
                    return_dict=False,
                    callback_on_step_end=progress_callback(0.0, 0.9, request.steps),
                )
            frames = video_np[0]
            frame_index = frames.shape[0] - 1 if request.frame_position == "last" else frames.shape[0] // 2
        else:
            # t2i / refine_image: distilled two-stage
            # (8 sigmas -> 2x latent upsample -> 3-sigma refine), probes P3 / P6.
            num_frames = request.num_frames or 9
            use_diffusion_decoder = request.mode == "t2i" and decoder_kind == "diffusion"
            if self._upsample_pipe is None:
                # t2i の品質パイプライン(2x latent upsample -> refine)は upsampler 必須。
                raise RuntimeError(
                    "latent upsampler が未ロードです(LTX25_LOAD_UPSAMPLERS=0 のリアルタイム構成)。"
                    "t2i を使うには LTX25_LOAD_UPSAMPLERS=1 のプリセットで起動し直してください。"
                )
            if use_diffusion_decoder and num_frames < 17:
                # NATTEN na3d needs >= its (11,11,11) kernel per tile: nf=9 fails, nf=17 works
                # (probes P4 vs P4_nf17). Promote internally.
                print(
                    f"[ltx25] t2i: diffusion decoder requires num_frames>=17 "
                    f"(NATTEN kernel size); promoting num_frames {num_frames} -> 17",
                    flush=True,
                )
                num_frames = 17
            with torch.no_grad():
                video, audio = pipe(
                    prompt=still_prompt,
                    negative_prompt=still_negative,
                    conditions=conditions or None,
                    width=request.width,
                    height=request.height,
                    num_frames=num_frames,
                    frame_rate=request.fps,
                    sigmas=DISTILLED_SIGMA_VALUES,
                    guidance_scale=1.0,
                    audio_guidance_scale=1.0,
                    stg_scale=0.0,
                    audio_stg_scale=0.0,
                    modality_scale=1.0,
                    audio_modality_scale=1.0,
                    enable_prompt_enhancement=request.enhance_prompt,
                    generator=generator,
                    output_type="latent",
                    return_dict=False,
                    callback_on_step_end=progress_callback(0.0, 0.45, len(DISTILLED_SIGMA_VALUES)),
                )
                progress(0.5)
                video = self._upsample_pipe(
                    latents=video,
                    height=request.height,
                    width=request.width,
                    num_frames=num_frames,
                    output_type="latent",
                    return_dict=False,
                )[0]
                video, audio = pipe(
                    prompt=still_prompt,
                    negative_prompt=still_negative,
                    latents=video,
                    audio_latents=audio,
                    width=request.width * 2,
                    height=request.height * 2,
                    num_frames=num_frames,
                    frame_rate=request.fps,
                    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                    guidance_scale=1.0,
                    audio_guidance_scale=1.0,
                    stg_scale=0.0,
                    audio_stg_scale=0.0,
                    modality_scale=1.0,
                    audio_modality_scale=1.0,
                    generator=generator,
                    output_type="latent" if use_diffusion_decoder else "np",
                    return_dict=False,
                    callback_on_step_end=progress_callback(
                        0.52, 0.3, len(STAGE_2_DISTILLED_SIGMA_VALUES)
                    ),
                )
                if use_diffusion_decoder:
                    progress(0.85)
                    decode_pipe = self.load_diffusion_decoder()
                    self._configure_decode_tiling(
                        num_frames, request.height * 2, request.width * 2
                    )
                    decode_generator = torch.Generator(device="cpu").manual_seed(request.seed)
                    video = decode_pipe(
                        latents=video.to("cuda"),
                        generator=decode_generator,
                        output_type="np",
                        return_dict=False,
                        denormalize=False,  # latent-path outputs are already denormalized
                    )[0]
            frames = video[0]
            frame_index = frames.shape[0] // 2

        frame = (np.clip(frames[frame_index], 0, 1) * 255).round().astype("uint8")
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(target)
        print(
            f"[ltx25] {request.mode}: saved frame {frame_index}/{frames.shape[0]} "
            f"({frame.shape[1]}x{frame.shape[0]}) -> {target.name}",
            flush=True,
        )
        progress(1.0)
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        gc.collect()
        torch.cuda.empty_cache()
        return {"peak_vram_gb": peak_vram_gb}

    def _generate_impl(self, request: GenerateRequest, target: Path, progress: Callable[[float], None]) -> dict[str, float]:
        import torch
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.utils import load_image, load_video

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pipe = self.load()
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        effective_width, effective_height, effective_fps = request.width, request.height, request.fps
        input_audio_wave = None
        input_audio_latents = None
        restore_audio_prepare = None
        previous_audio_scheduler = None

        conditions = []
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        video_suffixes = {".mp4", ".mov", ".webm", ".mkv", ".gif"}
        input_dir = self.config.input_dir.resolve()
        source_edit_path = None
        source_frames = None
        for condition in request.conditions:
            matches = list(input_dir.glob(f"{condition.asset_id}.*"))
            if len(matches) != 1:
                raise ValueError(f"Input asset not found: {condition.asset_id}")
            source = matches[0]
            if request.mode in {"retake", "extend"} and condition.kind == "video":
                source_edit_path = source
                source_frames = load_video(str(source))
                continue
            # extend の画像条件はキーフレームアンカー(schemas.py の extend 分岐参照)。
            # 通常の LTX2VideoCondition として下の共通経路へ流し、prepare_extend 側で
            # context ロックとマージする(index は生成窓の latent インデックス)。
            if condition.kind == "image" and source.suffix.lower() in image_suffixes:
                frames = load_image(str(source))
                if request.mode == "iclora":
                    # IC-LoRA expects a reference video. Turn a reference sheet into
                    # a static full-length clip so its latent tokens span the output.
                    frames = [frames.copy() for _ in range(request.num_frames)]
            elif condition.kind == "video" and source.suffix.lower() in video_suffixes:
                frames = load_video(str(source))
            else:
                raise ValueError(f"Input asset type mismatch: {condition.asset_id}")
            conditions.append(
                LTX2VideoCondition(frames=frames, index=condition.index, strength=condition.strength)
            )

        def progress_callback(offset: float, span: float, step_count: int):
            def on_step(_pipe, step: int, _timestep, callback_kwargs):
                # Step-boundary interrupt check (see GenerationInterrupted's docstring):
                # runs once per denoise step, right where diffusers already calls back
                # into us, so no extra hook into the pipeline internals is needed.
                interrupt_controller.check()
                progress(min(offset + ((step + 1) / step_count) * span, 0.96))
                return callback_kwargs

            return on_step

        effective_num_frames = request.num_frames
        restore_prepare_latents = None
        extend_context_duration = extend_duration = None
        if request.mode == "retake":
            if not source_frames:
                raise ValueError("Retake source video has no decodable frames")
            effective_num_frames = min(len(source_frames), 481)
            effective_num_frames = ((effective_num_frames - 1) // 8) * 8 + 1
            if effective_num_frames < 9:
                raise ValueError("Retake source must contain at least 9 frames")
            source_frames = source_frames[:effective_num_frames]
            import av
            with av.open(str(source_edit_path)) as container:
                stream = container.streams.video[0]
                effective_width, effective_height = stream.width, stream.height
                effective_fps = float(stream.average_rate or request.fps)
            if effective_width % 32 or effective_height % 32:
                raise ValueError("Retake source resolution must be divisible by 32")
            duration = (effective_num_frames - 1) / effective_fps
            if request.retake_end > duration:
                raise ValueError(f"Retake end exceeds source duration ({duration:.2f}s)")

            pixels = pipe.video_processor.preprocess_video(
                source_frames, height=effective_height, width=effective_width
            ).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
            from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents
            with torch.no_grad():
                source_latents = retrieve_latents(pipe.vae.encode(pixels), sample_mode="argmax")

            original_prepare_latents = pipe.prepare_latents

            def prepare_retake(this, *args, **kwargs):
                latents, _mask, _clean, coords = original_prepare_latents(*args, **kwargs)
                clean_5d = this._normalize_latents(
                    source_latents, this.vae.latents_mean, this.vae.latents_std, this.vae.config.scaling_factor
                ).to(device=latents.device, dtype=latents.dtype)
                clean = this._pack_latents(
                    clean_5d, this.transformer_spatial_patch_size, this.transformer_temporal_patch_size
                )
                latent_frames = clean_5d.shape[2]
                keep = clean_5d.new_ones((clean_5d.shape[0], 1, latent_frames, clean_5d.shape[3], clean_5d.shape[4]))
                if request.regenerate_video:
                    start_latent = max(0, math.floor(request.retake_start * effective_fps / 8))
                    end_latent = min(latent_frames, math.ceil(request.retake_end * effective_fps / 8) + 1)
                    keep[:, :, start_latent:end_latent] = 0
                mask = this._pack_latents(
                    keep, this.transformer_spatial_patch_size, this.transformer_temporal_patch_size
                )
                latents = latents * (1 - mask) + clean * mask
                return latents, mask, clean, coords

            restore_prepare_latents = original_prepare_latents
            pipe.prepare_latents = types.MethodType(prepare_retake, pipe)

        elif request.mode == "extend":
            if not source_frames:
                raise ValueError("Extend source video has no decodable frames")
            import av
            with av.open(str(source_edit_path)) as container:
                stream = container.streams.video[0]
                effective_width, effective_height = stream.width, stream.height
                effective_fps = float(stream.average_rate or request.fps)
            if effective_width % 32 or effective_height % 32:
                raise ValueError("Extend source resolution must be divisible by 32")

            extension_intervals = max(8, round(request.extend_seconds * effective_fps / 8) * 8)
            available_intervals = max(8, ((len(source_frames) - 1) // 8) * 8)
            context_intervals = max(8, round(request.extend_context_seconds * effective_fps / 8) * 8)
            context_intervals = min(context_intervals, available_intervals, 480 - extension_intervals)
            if context_intervals < 8 or extension_intervals + context_intervals > 480:
                raise ValueError("Extend context plus extension exceeds the 481-frame generation limit")
            context_frames_count = context_intervals + 1
            context_frames = (
                source_frames[-context_frames_count:]
                if request.extend_direction == "end" else source_frames[:context_frames_count]
            )
            effective_num_frames = context_intervals + extension_intervals + 1
            extend_context_duration = context_intervals / effective_fps
            extend_duration = extension_intervals / effective_fps

            # 画像キーフレームアンカーの位置検証(2026-09-05): index は生成窓の
            # latent インデックス。context 区間(latent 0..context_end)への指定は
            # ロック済み領域と衝突するため拒否し、窓外も黙殺(pipeline 側は warning
            # でスキップする)ではなく明示エラーにする。
            last_latent = (effective_num_frames - 1) // 8
            context_end_latent = context_intervals // 8 if request.extend_direction == "end" else -1
            for _c in request.conditions:
                if _c.kind != "image":
                    continue
                resolved_idx = _c.index if _c.index >= 0 else last_latent
                if resolved_idx > last_latent:
                    raise ValueError(
                        f"extend image keyframe latent index {_c.index} exceeds the "
                        f"generation window (last latent {last_latent})"
                    )
                if request.extend_direction == "end" and resolved_idx <= context_end_latent:
                    raise ValueError(
                        f"extend image keyframe latent index {_c.index} falls inside the "
                        f"locked context region (latents 0..{context_end_latent}); use a "
                        f"larger index or -1"
                    )

            pixels = pipe.video_processor.preprocess_video(
                context_frames, height=effective_height, width=effective_width
            ).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
            from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents
            with torch.no_grad():
                context_latents = retrieve_latents(pipe.vae.encode(pixels), sample_mode="argmax")
            original_prepare_latents = pipe.prepare_latents

            def prepare_extend(this, *args, **kwargs):
                # 元の prepare_latents は画像キーフレーム条件を処理済みの
                # (latents, mask, clean, coords) を返す(キーフレームは base 系列の
                # 後ろに追加トークンとして連結される)。旧実装は mask/clean を
                # 丸ごと自前のものに差し替えていたため条件が無効化されていた。
                # 2026-09-05: base 区間(先頭 base_len トークン)だけ context ロックを
                # 適用し、キーフレーム由来の mask/clean/coords は温存するマージ方式へ
                # 変更(キーフレーム無しなら従来と同値)。
                latents, orig_mask, orig_clean, coords = original_prepare_latents(*args, **kwargs)
                normalized_context = this._normalize_latents(
                    context_latents, this.vae.latents_mean, this.vae.latents_std, this.vae.config.scaling_factor
                ).to(device=latents.device, dtype=latents.dtype)
                total_latent_frames = (effective_num_frames - 1) // 8 + 1
                clean_5d = normalized_context.new_zeros(
                    (1, normalized_context.shape[1], total_latent_frames,
                     normalized_context.shape[3], normalized_context.shape[4])
                )
                keep = normalized_context.new_zeros(
                    (1, 1, total_latent_frames, normalized_context.shape[3], normalized_context.shape[4])
                )
                count = normalized_context.shape[2]
                region = slice(0, count) if request.extend_direction == "end" else slice(-count, None)
                clean_5d[:, :, region] = normalized_context
                keep[:, :, region] = 1
                clean = this._pack_latents(
                    clean_5d, this.transformer_spatial_patch_size, this.transformer_temporal_patch_size
                )
                mask = this._pack_latents(
                    keep, this.transformer_spatial_patch_size, this.transformer_temporal_patch_size
                )
                base_len = mask.shape[1]
                latents[:, :base_len] = latents[:, :base_len] * (1 - mask) + clean * mask
                merged_mask = orig_mask.clone()
                merged_mask[:, :base_len] = torch.maximum(orig_mask[:, :base_len], mask)
                merged_clean = orig_clean.clone()
                merged_clean[:, :base_len] = orig_clean[:, :base_len] * (1 - mask) + clean * mask
                return latents, merged_mask, merged_clean, coords

            restore_prepare_latents = original_prepare_latents
            pipe.prepare_latents = types.MethodType(prepare_extend, pipe)

        # LTX25_STAGE_DEBUG=1: リアルタイム経路の未計測区間を分解する一時計測
        # (2026-09-03 調査、既定OFFで挙動不変)。a2v 前処理は ffprobe + ffmpeg×2 の
        # subprocess 3回 + MelSpectrogram 毎回構築 + audio_vae.encode を含む。
        _stage_debug = os.getenv("LTX25_STAGE_DEBUG", "0").strip() == "1"
        _a2v_prep_t0 = time.time()
        if request.mode == "a2v":
            matches = list(input_dir.glob(f"{request.audio_asset_id}.*"))
            if len(matches) != 1:
                raise ValueError("Audio input asset not found")
            audio_source = matches[0]
            # 高速化②-1: 尺の取得は torchaudio.info(メタデータのみ・subprocess なし)を
            # 優先し、読めないコンテナだけ従来の ffprobe(プロセス起動 ~40ms)へ落とす。
            # デコード本体は従来どおり ffmpeg のまま(リサンプラを替えると a2v の
            # 条件付け mel が数値的に変わるため、意図的に触らない)。
            # 注意: torchaudio.info はこの venv の torchaudio には存在しない
            # (AttributeError、2026-09-03 実機確認)。PCM wav なら標準 wave モジュールで
            # ヘッダから正確な尺が取れる(依存ゼロ・subprocess ゼロ)。
            total_duration = None
            if audio_source.suffix.lower() == ".wav":
                try:
                    import wave as _wave
                    with _wave.open(str(audio_source), "rb") as _w:
                        _n, _sr = _w.getnframes(), _w.getframerate()
                    if _n > 0 and _sr > 0:
                        total_duration = _n / float(_sr)
                except Exception:
                    total_duration = None
            if total_duration is None:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_source)],
                    capture_output=True, text=True,
                )
                if probe.returncode != 0:
                    raise ValueError("Input audio duration could not be read")
                total_duration = float(probe.stdout.strip())
            remaining = total_duration - request.audio_start
            requested_duration = request.audio_duration or min(remaining, 20.0)
            if remaining < 1 or requested_duration > remaining + 0.05:
                raise ValueError("Selected audio range exceeds the input audio")
            intervals = min(480, max(8, round(requested_duration * effective_fps / 8) * 8))
            effective_num_frames = intervals + 1
            actual_duration = intervals / effective_fps

            audio_16k = self._decode_audio_file(audio_source, 16000, request.audio_start, actual_duration)
            input_audio_wave = torch.from_numpy(
                self._decode_audio_file(audio_source, pipe.vocoder.config.output_sampling_rate,
                                        request.audio_start, actual_duration)
            )
            waveform = torch.from_numpy(audio_16k).unsqueeze(0).to(pipe._execution_device)
            # 高速化②-2: MelSpectrogram はパラメータ固定なのでプロセス内で1回だけ
            # 構築してデバイス別にキャッシュする(毎リクエストの module 構築+
            # フィルタバンク計算+.to(device) を省く。数値は同一 module の再利用なので不変)。
            mel_transform = self._get_mel_transform_16k(waveform.device)
            mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5)).permute(0, 1, 3, 2)
            with torch.no_grad():
                posterior = pipe.audio_vae.encode(mel.to(pipe.audio_vae.dtype), return_dict=False)[0]
                input_audio_latents = posterior.mode()

            original_audio_prepare = pipe.prepare_audio_latents

            def prepare_frozen_audio(this, *args, **kwargs):
                source_latents = kwargs.get("latents")
                if source_latents is None:
                    source_latents = input_audio_latents
                if source_latents.ndim == 4:
                    source_latents = this._pack_audio_latents(source_latents)
                source_latents = this._normalize_audio_latents(
                    source_latents, this.audio_vae.latents_mean, this.audio_vae.latents_std
                )
                return source_latents.to(device=kwargs.get("device"), dtype=kwargs.get("dtype"))

            frozen_scheduler = copy.deepcopy(pipe.scheduler)
            original_set_timesteps = frozen_scheduler.set_timesteps

            def set_frozen_timesteps(
                _self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None
            ):
                result = original_set_timesteps(
                    num_inference_steps=num_inference_steps, device=device, sigmas=sigmas, mu=mu, timesteps=timesteps
                )
                _self.timesteps = torch.zeros_like(_self.timesteps)
                return result

            def frozen_step(_self, _model_output, _timestep, sample, return_dict=True, **_kwargs):
                return (sample,) if not return_dict else type("FrozenOutput", (), {"prev_sample": sample})()

            frozen_scheduler.set_timesteps = types.MethodType(set_frozen_timesteps, frozen_scheduler)
            frozen_scheduler.step = types.MethodType(frozen_step, frozen_scheduler)
            restore_audio_prepare = original_audio_prepare
            previous_audio_scheduler = pipe.audio_scheduler
            pipe.prepare_audio_latents = types.MethodType(prepare_frozen_audio, pipe)
            pipe.audio_scheduler = frozen_scheduler
            if _stage_debug:
                print(f"[ltx25] STAGE_DEBUG a2v_prep {time.time() - _a2v_prep_t0:.3f}s", flush=True)

        decoder_kind = request.decoder or self.config.ltx25_decoder
        use_refine = request.upscale or request.temporal_upscale
        if use_refine and self._upsample_pipe is None:
            raise RuntimeError(
                "latent upsampler が未ロードです(LTX25_LOAD_UPSAMPLERS=0 のリアルタイム構成)。"
                "upscale / temporal_upscale を使うには LTX25_LOAD_UPSAMPLERS=1 のプリセットで"
                "バックエンドを起動し直してください。"
            )
        use_diffusion_decoder = use_refine and decoder_kind == "diffusion"
        final_fps = effective_fps * (2 if request.temporal_upscale else 1)

        args = {
            "prompt": request.prompt,
            "conditions": conditions or None,
            "negative_prompt": request.negative_prompt,
            "width": effective_width,
            "height": effective_height,
            "num_frames": effective_num_frames,
            "min_seconds": request.min_seconds,
            "max_seconds": request.max_seconds,
            "frame_rate": effective_fps,
            # 蒸留8σスケジュール。steps<8 を明示されたときだけ間引く(2026-09-04、
            # リアルタイム高速化)。従来 request.steps はこの経路で黙殺されており
            # 「4steps 指定」は効いていなかった。間引きは先頭・末尾を必ず含む
            # 等間隔インデックス(例 4steps: idx 0,2,5,7)。蒸留の学習分布外に
            # なるため品質は A/B 前提。steps>=8・未指定は従来どおり8σ固定。
            "sigmas": _subsample_distilled_sigmas(DISTILLED_SIGMA_VALUES, request.steps),
            "guidance_scale": 1.0,
            # リップシンク調整ノブ(schemas.GenerateRequest の同名フィールド参照)。
            # None(既定)なら従来どおり 1.0 固定 = 完全互換。
            "audio_guidance_scale": (
                request.audio_guidance_scale if request.audio_guidance_scale is not None else 1.0
            ),
            "stg_scale": 0.0,
            "audio_stg_scale": 0.0,
            # 映像側 modality_scale がリップシンクの実効ノブ(schemas の訂正コメント参照)
            "modality_scale": (
                request.modality_scale if request.modality_scale is not None else 1.0
            ),
            "audio_modality_scale": (
                request.audio_modality_scale if request.audio_modality_scale is not None else 1.0
            ),
            "enable_prompt_enhancement": request.enhance_prompt,
            "generator": generator,
            # 非refine(リアルタイム)経路は "pt" で受ける(2026-09-03 高速化①):
            # postprocess 済み (B,F,C,H,W)・GPU 上の float [0,1] が返るので、
            # uint8 化を GPU で行ってから CPU へ下ろす(従来の "np" は float32
            # ~190MB を CPU へ転送して numpy で clip/mul/round していた。実測 0.20s)。
            "output_type": "latent" if use_refine else "pt",
            "return_dict": False,
            "callback_on_step_end": progress_callback(
                0.0, 0.55 if use_refine else 0.96, len(DISTILLED_SIGMA_VALUES)
            ),
        }
        if request.mode == "a2v":
            args["audio_latents"] = input_audio_latents
        # LTX25_STAGE_DEBUG=1: pipe() 内部のコンポーネント別時間を計測する一時フック
        # (text_encoder / video VAE decode / audio VAE decode / vocoder)。
        # instance 属性で forward/decode を差し替え、finally で必ず原状復帰する。
        _dbg_acc: dict = {}
        _dbg_restore: list = []
        if _stage_debug:
            def _dbg_wrap(obj, attr, name):
                orig = getattr(obj, attr)
                def timed(*a, **k):
                    t0 = time.time()
                    try:
                        return orig(*a, **k)
                    finally:
                        _dbg_acc[name] = _dbg_acc.get(name, 0.0) + (time.time() - t0)
                setattr(obj, attr, timed)
                _dbg_restore.append((obj, attr, orig))
            _dbg_wrap(pipe.text_encoder, "forward", "text_encode")
            _dbg_wrap(pipe.vae, "decode", "video_vae_decode")
            _dbg_wrap(pipe.audio_vae, "decode", "audio_vae_decode")
            _dbg_wrap(pipe.vocoder, "forward", "vocoder")
        stage_t0 = time.time()
        try:
            video, audio = pipe(**args)
        except Exception:
            if restore_audio_prepare is not None:
                pipe.prepare_audio_latents = restore_audio_prepare
                pipe.audio_scheduler = previous_audio_scheduler
            raise
        finally:
            if restore_prepare_latents is not None:
                pipe.prepare_latents = restore_prepare_latents
            for _obj, _attr, _orig in _dbg_restore:
                setattr(_obj, _attr, _orig)
        if _stage_debug:
            _parts = " ".join(f"{k}={v:.3f}s" for k, v in _dbg_acc.items())
            print(
                f"[ltx25] STAGE_DEBUG pipe_total {time.time() - stage_t0:.3f}s ({_parts})",
                flush=True,
            )
        if not use_refine:
            # 高速化①: uint8 変換を GPU で実行(output_type="pt" とセット)。
            # (B,F,C,H,W) float [0,1] → (B,F,H,W,3) uint8 numpy。値は従来の
            # np.clip*255→round(半数偶数丸め)→astype と bit 一致する
            # (torch.round も half-to-even、float32 演算は IEEE で同一)。
            # 以降の video の使われ方(video[0] を encode へ)は従来の "np" と
            # 同じレイアウトなので下流は無変更。encode_video_crf 側の uint8
            # 変換は dtype==uint8 のため素通りになる。
            _cvt_t0 = time.time()
            with torch.no_grad():
                # .float() が必須: "pt" は VAE 出力の dtype(bf16)のまま返るが、
                # 従来の "np" 経路は numpy 変換時に float32 へキャストしてから
                # *255/round していた。bf16 のまま演算すると丸めが変わり
                # framemd5 が一致しない(実測で確認)。float32 に揃えると
                # 旧経路と同一の IEEE 演算列になる。
                video = (
                    video.permute(0, 1, 3, 4, 2)
                    .float()
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
            if _stage_debug:
                print(f"[ltx25] STAGE_DEBUG gpu_uint8_convert {time.time() - _cvt_t0:.3f}s", flush=True)
        generated_num_frames = effective_num_frames
        if generated_num_frames is None:
            # Auto-duration returns unpacked video latents [B, C, latent_F, H, W].
            generated_num_frames = (video.shape[2] - 1) * pipe.vae_temporal_compression_ratio + 1
        if use_refine:
            print(f"[ltx25] stage timing: base denoise {time.time() - stage_t0:.1f}s", flush=True)
            stage_t0 = time.time()
            pixel_reference_latents = video.detach().clone() if request.upscale_method == "pixel" else None
            progress(0.58)
            if request.upscale:
                video = self._upsample_pipe(
                    latents=video,
                    height=request.height,
                    width=request.width,
                    num_frames=generated_num_frames,
                    output_type="latent",
                    return_dict=False,
                )[0]
            if request.temporal_upscale:
                if self._temporal_upsample_pipe is None:
                    raise RuntimeError(
                        "Temporal latent upsampler is missing. Run "
                        "scripts/download_quantize_ltx25.py --component temporal first."
                    )
                video = self._temporal_upsample_pipe(
                    latents=video,
                    height=request.height * (2 if request.upscale else 1),
                    width=request.width * (2 if request.upscale else 1),
                    num_frames=generated_num_frames,
                    output_type="latent",
                    return_dict=False,
                )[0]
                generated_num_frames = (generated_num_frames - 1) * 2 + 1
            progress(0.64)
            print(f"[ltx25] stage timing: latent upsample {time.time() - stage_t0:.1f}s", flush=True)
            stage_t0 = time.time()
            stage2_span = 0.14 if use_diffusion_decoder else 0.32
            restore_stage2_prepare = None
            if request.upscale_method == "pixel":
                pixel_lora = (
                    self.config.quantized_model_dir.resolve()
                    / "pixel_spatial_upscaler"
                    / PIXEL_UPSCALER_FILENAME
                )
                if not pixel_lora.is_file():
                    raise RuntimeError(
                        "Pixel Spatial Upscaler IC-LoRA is missing. Run "
                        "scripts/download_quantize_ltx25.py --component pixel_upscaler first."
                    )
                try:
                    pipe.load_lora_weights(pixel_lora, adapter_name="pixel_spatial_upscaler")
                    names = [f"job_lora_{index}" for index in range(len(request.loras))]
                    pipe.set_adapters(
                        [*names, "pixel_spatial_upscaler"],
                        adapter_weights=[*[item.strength for item in request.loras], 1.0],
                    )
                    self._cast_lora_layers_to_bf16(pipe)
                except Exception as exc:
                    raise RuntimeError(f"Pixel Spatial Upscaler IC-LoRA could not be loaded: {exc}") from exc

                # Diffusers does not yet expose VideoConditionByReferenceLatent. Append the
                # clean half-resolution Stage-1 latent exactly as the official LTX DFR
                # pipeline does, and scale its spatial RoPE coordinates into the target grid.
                original_prepare_latents = pipe.prepare_latents

                def prepare_pixel_reference(this, *args, **kwargs):
                    latents, mask, clean, coords = original_prepare_latents(*args, **kwargs)
                    reference = this._normalize_latents(
                        pixel_reference_latents,
                        this.vae.latents_mean,
                        this.vae.latents_std,
                        this.vae.config.scaling_factor,
                    ).to(device=latents.device, dtype=latents.dtype)
                    reference_tokens = this._pack_latents(
                        reference,
                        this.transformer_spatial_patch_size,
                        this.transformer_temporal_patch_size,
                    ).expand(latents.shape[0], -1, -1)
                    reference_mask = torch.ones(
                        (*reference_tokens.shape[:2], 1), device=latents.device, dtype=mask.dtype
                    )
                    reference_coords = this.transformer.rope.prepare_video_coords(
                        latents.shape[0],
                        reference.shape[2],
                        reference.shape[3],
                        reference.shape[4],
                        latents.device,
                        fps=final_fps,
                    )
                    reference_coords[:, 1:, :, :] *= 2
                    combined_coords = (
                        reference_coords if coords is None else torch.cat([coords, reference_coords], dim=2)
                    )
                    return (
                        torch.cat([latents, torch.zeros_like(reference_tokens)], dim=1),
                        torch.cat([mask, reference_mask], dim=1),
                        torch.cat([clean, reference_tokens], dim=1),
                        combined_coords,
                    )

                restore_stage2_prepare = original_prepare_latents
                pipe.prepare_latents = types.MethodType(prepare_pixel_reference, pipe)
            try:
                video, audio = pipe(
                    prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                latents=video,
                audio_latents=audio,
                width=effective_width * (2 if request.upscale else 1),
                height=effective_height * (2 if request.upscale else 1),
                num_frames=generated_num_frames,
                min_seconds=request.min_seconds,
                max_seconds=request.max_seconds,
                frame_rate=final_fps,
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                guidance_scale=1.0,
                audio_guidance_scale=1.0,
                stg_scale=0.0,
                audio_stg_scale=0.0,
                modality_scale=1.0,
                audio_modality_scale=1.0,
                generator=generator,
                output_type="latent" if use_diffusion_decoder else "np",
                return_dict=False,
                    callback_on_step_end=progress_callback(
                        0.64, stage2_span, len(STAGE_2_DISTILLED_SIGMA_VALUES)
                    ),
                )
            except Exception:
                if restore_audio_prepare is not None:
                    pipe.prepare_audio_latents = restore_audio_prepare
                    pipe.audio_scheduler = previous_audio_scheduler
                raise
            finally:
                if restore_stage2_prepare is not None:
                    pipe.prepare_latents = restore_stage2_prepare
            print(f"[ltx25] stage timing: stage2 refine {time.time() - stage_t0:.1f}s", flush=True)
        if restore_audio_prepare is not None:
            pipe.prepare_audio_latents = restore_audio_prepare
            pipe.audio_scheduler = previous_audio_scheduler
        sample_rate = pipe.vocoder.config.output_sampling_rate
        if use_diffusion_decoder:
            # `output_type="latent"` returned de-normalized video latents and audio latents.
            audio_wave = self._decode_audio(pipe, audio)[0].float().cpu()
            progress(0.80)
            decode_pipe = self.load_diffusion_decoder()
            self._configure_decode_tiling(
                generated_num_frames,
                effective_height * (2 if request.upscale else 1),
                effective_width * (2 if request.upscale else 1),
            )
            decode_generator = torch.Generator(device="cpu").manual_seed(request.seed)
            decode_start = time.time()
            with _ProgressRamp(progress, 0.80, 0.955, tau_s=90.0), torch.no_grad():
                video = decode_pipe(
                    latents=video.to("cuda"),
                    generator=decode_generator,
                    output_type="np",
                    return_dict=False,
                    denormalize=False,  # latent-path outputs are already denormalized
                )[0]
            print(f"[ltx25] diffusion decode {time.time() - decode_start:.1f}s", flush=True)
        else:
            audio_wave = audio[0].float().cpu()
        if request.mode == "a2v":
            audio_wave = input_audio_wave
        target.parent.mkdir(parents=True, exist_ok=True)
        encode_target = target.with_suffix(".edit-generated.mp4") if request.mode in {"retake", "extend"} else target

        def _encode_and_finalize():
            encode_t0 = time.time()
            encode_video_crf(
                video[0],
                fps=final_fps,
                audio=audio_wave,
                audio_sample_rate=sample_rate,
                output_path=encode_target,
                crf=self.config.ltx25_video_crf,
                encoder=self.config.ltx25_video_encoder,
                nvenc_preset=self.config.ltx25_nvenc_preset,
            )
            print(f"[ltx25] stage timing: mp4 encode {time.time() - encode_t0:.1f}s", flush=True)
            if request.mode == "retake":
                self._finish_retake(source_edit_path, encode_target, target, request)
                encode_target.unlink(missing_ok=True)
            elif request.mode == "extend":
                self._finish_extend(
                    source_edit_path, encode_target, target, request.extend_direction,
                    extend_context_duration, extend_duration,
                )
                encode_target.unlink(missing_ok=True)

        # 高速化④: 基本モードの mp4 encode(実測 0.6s、CPU+NVENC のみで VRAM 不使用)は
        # ジョブワーカーへ closure として返し、次ジョブの denoise と重ねられるようにする
        # (jobs.py 側の encode 専用スレッドが実行し、完了時に completed へ遷移させる)。
        # retake/extend は encode 後にファイル差し替えの後処理があるため従来どおり同期。
        # LTX25_ASYNC_ENCODE=0 で従来の同期動作へ戻る。
        defer_encode = (
            os.getenv("LTX25_ASYNC_ENCODE", "1").strip() == "1"
            and request.mode in {"t2v", "a2v", "i2v", "flf2v"}
        )
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        if defer_encode:
            gc.collect()
            torch.cuda.empty_cache()
            # progress(1.0) は encode 完了時に jobs 側が立てる(ここではまだ完了ではない)。
            return {"peak_vram_gb": peak_vram_gb, "deferred_encode": _encode_and_finalize}
        # 同期エンコード(retake/extend 等)も NVENC を開く**前**に torch の予約
        # キャッシュを返す(2026-09-06)。従来はエンコード後にしか empty_cache して
        # おらず、直前の大きいジョブでキャッシュが育っていると NVENC の
        # avcodec_open2 が VRAM を確保できず失敗した(実機: 1024×576×361f の
        # i2v 連発後の extend で "avcodec_open2(h264_nvenc)" エラー)。
        gc.collect()
        torch.cuda.empty_cache()
        _encode_and_finalize()
        progress(1.0)
        gc.collect()
        torch.cuda.empty_cache()
        return {"peak_vram_gb": peak_vram_gb}
