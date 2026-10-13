"""transformer.forward 全体の CUDA Graph capture/replay(LTX25_CUDA_GRAPH=1)。

Lightricks 公式 LTX-2 パッケージの cudagraph_capture.py(v1.2.0/v1.3.0、ブロック
ループのみを capture)の方式 -- side stream での warmup → clearCublasWorkspaces →
共有 mempool での capture → 毎回 replay -- を踏襲しつつ、こちらは diffusers の
`LTX2VideoTransformer3DModel.forward` **全体**を capture する。パイプライン
(LTX2ConditionPipeline)が video_coords / audio_coords を事前計算して渡し、
蒸留経路(stg_scale=0 / cfg=1)では forward 内に CPU 依存の分岐が残らないため、
公式より広い境界が成立する(実測は probes/probe_cudagraph.py: 512x288x121f t2v
8steps で映像・音声とも eager と bit 完全一致、denoise 1.82s→0.48s(3.8x)、
peak VRAM 増加なし、capture 固定費 ~1s/shape)。

成立条件(破ると黙って壊れるので generator 側でガードする):
- OFFLOAD_MODE=none(transformer 常駐)。model offload は重みのデバイス/アドレスが
  リクエスト間で動くため capture 済み graph が無効になる。
- LoRA を載せるジョブ(request.loras / pixel upscale の IC-LoRA)は eager に落とす。
  capture は重みテンソルのアドレスを焼き込むため、adapter の付け外しをまたいだ
  replay は stale な重みを黙って使う。ジョブ後は reset() で捨てる。
- 生成は単一スレッド(jobs.py の ltx-worker)。runner はスレッドセーフではない。

その他の性質:
- capture キーは (テンソル引数の shape/dtype) + (非テンソル引数の repr)。
  同一ジョブ内の modality guidance 用 2nd forward(isolate_modalities=True)も
  別キーとして独立に capture される。
- 新しいキーの初回は warmup 3 回 + capture(~1s)。キー数が max_captures を超えたら
  以後の新キーは eager フォールバック(既存キーの replay は継続)。
- capture 中の例外はそのキーを恒久 eager 化して伝播させない(結果は eager で返す)。
"""
from __future__ import annotations

import logging
import time
from typing import Any

import torch

logger = logging.getLogger("ltx25.cudagraph")

_EAGER = object()  # blacklist sentinel


class ForwardGraphRunner:
    def __init__(self, module: torch.nn.Module, warmup_iters: int = 3, max_captures: int = 8,
                 max_tokens: int = 0):
        self._module = module
        # capture する shape の上限(hidden_states のトークン数、0 = 無制限)。
        # 2026-10-13: MV の大画角(1536×896×121f ≈ 21.5k トークン)を capture すると、
        # graph の private pool が中間テンソルごと常駐し、ジョブ後の empty_cache() でも
        # 返らない(実測: 5秒後 39.4GB / 10秒後 46.7GB をプロセスが保持し続け、同居 H3 の
        # load が gateway の空き VRAM ガードで 409 になった)。大画角は GPU 計算律速で
        # graph の効果(起動オーバーヘッド削減)が小さいので eager に落とす。
        self._max_tokens = int(max_tokens or 0)
        self._warned_large = False
        self._orig_forward = module.forward
        self._warmup_iters = warmup_iters
        self._max_captures = max_captures
        self._pool = None
        self._captures: dict[tuple, Any] = {}
        self.enabled = True
        self.replays = 0
        self._warned_args = False
        self._warned_keynone = False

    # -- public ------------------------------------------------------------
    def install(self) -> None:
        self._module.forward = self  # nn.Module.__call__ resolves instance attr first

    def uninstall(self) -> None:
        if self._module.forward is self:
            del self._module.forward  # type: ignore[attr-defined]

    def reset(self) -> None:
        """capture 済み graph を全て破棄する(LoRA ジョブ後の防御的リセット)。"""
        n = sum(1 for v in self._captures.values() if v is not _EAGER)
        self._captures.clear()
        if n:
            torch.cuda.synchronize()
            print(f"[ltx25] cudagraph: reset ({n} captures dropped)", flush=True)

    # -- internal ----------------------------------------------------------
    def _key(self, kwargs: dict) -> tuple | None:
        parts = []
        for k in sorted(kwargs):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                parts.append((k, "T", tuple(v.shape), str(v.dtype), str(v.device)))
            elif isinstance(v, (int, float, str, bool, type(None))):
                parts.append((k, "S", v))
            elif isinstance(v, dict):
                if any(isinstance(x, torch.Tensor) for x in v.values()):
                    return None  # dict 内のテンソルは静的バッファ化できない -> eager
                parts.append((k, "D", tuple(sorted((dk, repr(dv)) for dk, dv in v.items()))))
            elif isinstance(v, (list, tuple)):
                if any(isinstance(x, torch.Tensor) for x in v):
                    return None
                parts.append((k, "L", repr(v)))
            else:
                return None  # 未知の型 -> eager
        return tuple(parts)

    def __call__(self, *args, **kwargs):
        if not self.enabled or args:
            # 位置引数呼び出しは想定外(パイプラインは kwargs のみ)。安全側で eager。
            if args and not self._warned_args:
                self._warned_args = True
                print("[ltx25] cudagraph: positional-arg call -> eager", flush=True)
            return self._orig_forward(*args, **kwargs)
        key = self._key(kwargs)
        if key is None:
            if not self._warned_keynone:
                self._warned_keynone = True
                bad = [k for k, v in kwargs.items()
                       if not isinstance(v, (torch.Tensor, int, float, str, bool, type(None), dict, list, tuple))]
                print(f"[ltx25] cudagraph: unkeyable kwargs -> eager (offenders: {bad})", flush=True)
            return self._orig_forward(**kwargs)
        cap = self._captures.get(key)
        if cap is _EAGER:
            return self._orig_forward(**kwargs)
        if cap is None:
            hs = kwargs.get("hidden_states")
            if (
                self._max_tokens > 0
                and isinstance(hs, torch.Tensor)
                and hs.dim() >= 2
                and hs.shape[1] > self._max_tokens
            ):
                # 大きい shape は capture しない(上の __init__ のコメント参照)。
                # max_captures の枠も消費しない(小さい shape の capture を妨げない)。
                if not self._warned_large:
                    self._warned_large = True
                    print(f"[ltx25] cudagraph: {hs.shape[1]} tokens > max_tokens="
                          f"{self._max_tokens} -> eager (not captured)", flush=True)
                self._captures[key] = _EAGER
                return self._orig_forward(**kwargs)
            n_live = sum(1 for v in self._captures.values() if v is not _EAGER)
            if n_live >= self._max_captures:
                if _EAGER not in self._captures.values():
                    print(f"[ltx25] cudagraph: max_captures={self._max_captures} reached; "
                          "new shapes fall back to eager", flush=True)
                self._captures[key] = _EAGER
                return self._orig_forward(**kwargs)
            try:
                cap = self._capture(kwargs)
            except Exception as exc:
                print(f"[ltx25] cudagraph: capture failed; this shape stays eager: {exc!r}", flush=True)
                torch.cuda.synchronize()
                self._captures[key] = _EAGER
                return self._orig_forward(**kwargs)
            self._captures[key] = cap
        else:
            for k, static in cap["inputs"].items():
                static.copy_(kwargs[k])
        cap["graph"].replay()
        self.replays += 1
        return cap["outputs"]

    def _capture(self, kwargs: dict) -> dict:
        statics: dict[str, torch.Tensor] = {}
        static_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                s = v.clone()
                statics[k] = s
                static_kwargs[k] = s
            else:
                static_kwargs[k] = v
        t0 = time.time()
        with torch.no_grad():
            # warmup を side stream で回し、Triton JIT / cuBLAS ワークスペース確保 /
            # カーネル選択を capture の外へ出す(公式実装と同じ手順)。
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self._warmup_iters):
                    self._orig_forward(**static_kwargs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            # 公式に倣い、warmup 後に入力を入れ直してから capture する
            for k, s in statics.items():
                s.copy_(kwargs[k])
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="global"):
                outputs = self._orig_forward(**static_kwargs)
        print(f"[ltx25] cudagraph: captured shape #"
              f"{sum(1 for v in self._captures.values() if v is not _EAGER) + 1} "
              f"in {time.time() - t0:.2f}s ({len(statics)} input tensors)", flush=True)
        return {"graph": graph, "inputs": statics, "outputs": outputs}
