from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_id: str = "Lightricks/LTX-2.5-Diffusers"
    # Pinned default: opt in to newer Hub weights by changing MODEL_REVISION.
    model_revision: str | None = "69009ff070135c693ad1ad1ef2cc149c227963da"
    quantized_model_dir: Path = Path("LTX-2.5-Diffusers-bnb-4bit")
    hf_token: str | None = None
    offload_mode: str = "model"
    output_dir: Path = Path("outputs")
    input_dir: Path = Path("inputs")
    lora_dir: Path = Path("loras")
    max_upload_size_mb: int = 500
    max_queue_size: int = 4
    history_db: Path = Path("outputs/history.sqlite3")
    # OpenAI-compatible chat-completions endpoint used only for prompt rewriting.
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 60.0
    # Decode path after the 2x latent upscale/refine stage: "diffusion"
    # (better fine detail, +~18s with NATTEN) or "vae" (fastest).
    # Non-upscaled jobs always use VAE. Env: LTX25_DECODER
    ltx25_decoder: str = "diffusion"
    # libx264 CRF for all output videos (lower = higher quality). Env: LTX25_VIDEO_CRF
    ltx25_video_crf: int = 18
    # mp4 encoder: "nvenc" (h264_nvenc p7/tune hq, GPU; default -- measured
    # PSNR 43.8dB vs x264 crf18 on identical frames, 1024x576x121f encode
    # 3.3s -> 1.9s / 1536x896 6.9s -> 4.4s, falls back to x264 automatically
    # when NVENC is unavailable) or "x264" (libx264 preset=slower, CPU).
    # Env: LTX25_VIDEO_ENCODER
    ltx25_video_encoder: str = "nvenc"
    # NVENC のエンコードプリセット(p1=最速 .. p7=最遅・最高品質)。既定 p7 は
    # MV 等の最終出力品質を優先した値。リアルタイム用途(低画素・短尺)では
    # LTX25_NVENC_PRESET=p4 で mp4 encode を ~0.1-0.15s 短縮できる(2026-09-04)。
    ltx25_nvenc_preset: str = "p7"
    # Diffusion-decoder tiling: "auto" (single tile when free VRAM allows --
    # ~1.23x faster and seam-free; falls back to default tiles otherwise),
    # "on" (always single tile), "off" (always default 768^2x80f tiles).
    # Probe (probes/probe_decode_tiling.py, 1024x576x121f): default 9.67s /
    # single 7.85s, decode activations ~0.34GB per Mpixel of output volume.
    # Env: LTX25_DECODE_SINGLE_TILE
    ltx25_decode_single_tile: str = "auto"
    # Transformer weights: "nf4" (bnb 4bit, default), "fp8" (bf16-equivalent
    # quality via layerwise casting storage=fp8_e4m3fn / compute=bf16, resident
    # ~18GB / peak ~29GB, for 48GB-class GPUs; requires the ~38GB bf16
    # transformer shards in the HF cache) or "bf16" (release weights, ~38GB,
    # for 96GB-class GPUs) or "nvfp4" (official Blackwell-native FP4 distilled
    # transformer, resident ~19GB, FP4 tensor-core matmul via torch._scaled_mm;
    # requires sm_120+. See app/nvfp4.py). Env: LTX25_TRANSFORMER_PRECISION
    ltx25_transformer_precision: str = "nf4"
    # Optional local path to the ComfyUI-format nvfp4 checkpoint. When unset,
    # hf_hub_download("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-
    # distilled-transformer-nvfp4.safetensors") resolves it (18.7GB, cached).
    # Env: LTX25_NVFP4_CKPT
    ltx25_nvfp4_ckpt: str | None = None
    # transformer.forward 全体を CUDA Graph capture/replay して denoise の
    # カーネル起動律速を潰す(app/cudagraph.py。実測 512x288x121f t2v 8steps:
    # denoise 1.82s→0.48s(3.8x)、映像・音声とも eager と bit 一致)。
    # OFFLOAD_MODE=none 前提(それ以外では警告して無効)。LoRA ジョブは自動で
    # eager に落ちる。Env: LTX25_CUDA_GRAPH
    ltx25_cuda_graph: bool = False
    # capture を保持する shape 数の上限。超過分の新 shape は eager フォールバック。
    # 1 shape あたり静的入出力バッファ+graph 中間メモリ(共有 mempool)を保持する。
    # Env: LTX25_CUDA_GRAPH_MAX_CAPTURES
    ltx25_cuda_graph_max_captures: int = 8
    # latent/temporal upsampler をロードするか。0 で GPU 常駐 -1.2GB
    # (latent 950MB + temporal 250MB、OFFLOAD_MODE=none 時)。リアルタイム
    # 用途は常に upscale:false のため不要。0 のとき upscale/temporal_upscale
    # 要求は明確なエラーになる。Env: LTX25_LOAD_UPSAMPLERS
    ltx25_load_upsamplers: bool = True
    # Gemma TE のダイエット(app/tediet.py): embed_tokens(bf16 1.88GB)を
    # CPU へ移して gather をブリッジし、lm_head(tied)の全トークン logits
    # (1024x262144 bf16 ≈ 0.5GB の一時確保)をスキップする。合計で常駐
    # -1.9GB + エンコード時一時 -0.5GB。OFFLOAD_MODE=none 専用(それ以外は
    # 警告して無効)。32GB 級(RTX 5090)で全常駐を成立させるための削減。
    # Env: LTX25_TE_DIET
    ltx25_te_diet: bool = False
    # Gemma TE の窓付き層ストリーミング(app/testream.py): NF4言語モデル層
    # (~5.0GB)を pinned host に常駐させ、エンコード中だけ層単位でGPUへ流す
    # (先読み window=2、転送は計算に完全に隠れる)。実測: TE常駐 5.75->0.77GiB、
    # encode +0.156s、出力は全常駐と bit 完全一致。32GB級で常駐 ~23.8GB になり
    # CUDA Graph 再有効化の余地を作る。OFFLOAD_MODE=none 専用。
    # Env: LTX25_TE_STREAM
    ltx25_te_stream: bool = False
    # ストリーミングの先読み窓(層数)。2で転送が計算に隠れる(W=4/8でも同速)。
    # Env: LTX25_TE_STREAM_WINDOW
    ltx25_te_stream_window: int = 2
    # 【実験的・非推奨】transformer_blocks の per-block torch.compile
    # (app/compileblocks.py のモジュール docstring の結論を必ず読むこと)。
    # "off"(既定)/ "islands" / "fusion"。probe では graph 単体に勝つが、
    # サーバ E2E では同 shape で誤差範囲・リアルタイム小 shape では退行
    # (2.39s vs 1.83s)のため本番では使わない。品質面も eager と bit 一致しない
    # (軌道差、ユーザー判定では同等)。nvfp4 + LTX25_CUDA_GRAPH=1 +
    # OFFLOAD_MODE=none が前提(それ以外は警告して無効)。
    # Env: LTX25_COMPILE_BLOCKS
    ltx25_compile_blocks: str = "off"


settings = Settings()
