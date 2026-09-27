"""Gemma TE の窓付き層ストリーミング(LTX25_TE_STREAM=1)。

言語モデル本体(NF4 量子化層、~5.0GB)を pinned host memory に常駐させ、
エンコード中だけ層単位で GPU へ流す。狙いは 32GB 級での常駐削減
(28.8GB -> ~23.8GB)と、それによる CUDA Graph 再有効化の余地。

方式(probe_te_stream_a4/a5 で確立、いずれも出力 bit 完全一致を実測):
- セットアップ時: 各層の param.data / buffer / bnb quant_state テンソル
  (absmax 等)を pinned CPU へ移す。
- エンコード開始時: 先頭 window 層分のコピーをコピー専用 stream へ投入。
- 各層の forward_pre_hook: 自層のコピー完了 event を計算 stream が wait し、
  GPU テンソルに record_stream(計算 stream)を打つ(アロケータの早期再利用防止)。
- forward_hook(層完了後): その層を CPU ポインタへ復元して GPU コピーを手放し、
  window 先の層のコピーを投入する。
- エンコード終了時: 全層を CPU ポインタへ復元(コピーは発生しない)。

実測(probe A5、1024トークン、RTX PRO 5000):
  window=2 で encode 0.213s -> 0.369s(+0.156s)、TE 常駐 5.75 -> 0.77GiB、
  エンコード中の一時ピーク +1.5GB、出力は全常駐と bit 完全一致(2連続実行で
  復元経路も検証済み)。層あたり転送(~2.4ms)< 層あたり計算(~4.4ms)のため
  window=2 で転送は完全に計算へ隠れる(W=4/8 にしても速度は変わらない)。

diffusers の apply_group_offloading(block_level + use_stream)を使わない理由
(probe A/A2/A3 実測): stream 使用時は num_blocks_per_group=1 が強制され、
48 グループ×フック固定費 ~22ms で +1.03s/encode。stream なしは pinned に
ならず +2.0s。いずれも実時間チャンク予算に対して過大だった。

制約: OFFLOAD_MODE=none 専用。tediet(LTX25_TE_DIET)と併用可(embed は
tediet が CPU ブリッジ済み、本モジュールは transformer 層のみを扱う)。
サーバはジョブを直列実行するため、begin/end の非再入は考慮不要。
"""

from __future__ import annotations

import torch


def _collect_movable(module: torch.nn.Module):
    """層内のデバイス移動対象テンソルを (owner, attr, tensor) で列挙する。

    nn.Parameter の .data、named_buffers、および bnb Params4bit の
    quant_state 内テンソル(absmax / code / offset、double-quant 時は state2 も)。
    """
    items = []
    for _, p in module.named_parameters(recurse=True):
        items.append((p, "data", p.data))
        qs = getattr(p, "quant_state", None)
        if qs is not None:
            for attr in ("absmax", "code", "offset"):
                t = getattr(qs, attr, None)
                if isinstance(t, torch.Tensor):
                    items.append((qs, attr, t))
            nested = getattr(qs, "state2", None)
            if nested is not None:
                for attr in ("absmax", "code"):
                    t = getattr(nested, attr, None)
                    if isinstance(t, torch.Tensor):
                        items.append((nested, attr, t))
    for name, b in module.named_buffers(recurse=True):
        owner = module
        parts = name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        items.append((owner, parts[-1], b))
    return items


class WindowedLayerStreamer:
    def __init__(self, layers, window: int = 2):
        self.layers = list(layers)
        self.window = max(1, int(window))
        self.copy_stream = torch.cuda.Stream()
        self.events = [torch.cuda.Event() for _ in self.layers]
        self.cpu_snap = []
        self.gpu_hold = [None] * len(self.layers)
        pinned_gib = 0.0
        for layer in self.layers:
            snap = []
            for owner, attr, t in _collect_movable(layer):
                cpu = t.detach().to("cpu")
                if not cpu.is_pinned():
                    cpu = cpu.pin_memory()
                pinned_gib += cpu.numel() * cpu.element_size() / 1024**3
                snap.append((owner, attr, cpu))
                self._set(owner, attr, cpu)
            self.cpu_snap.append(snap)
        self.pinned_gib = pinned_gib
        for i, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._pre(i))
            layer.register_forward_hook(self._post(i))

    @staticmethod
    def _set(owner, attr, tensor):
        if isinstance(owner, torch.nn.Parameter):
            owner.data = tensor
        else:
            setattr(owner, attr, tensor)

    def _onload(self, i: int) -> None:
        gpu_list = []
        with torch.cuda.stream(self.copy_stream):
            for owner, attr, cpu in self.cpu_snap[i]:
                g = cpu.to("cuda", non_blocking=True)
                self._set(owner, attr, g)
                gpu_list.append(g)
            self.events[i].record(self.copy_stream)
        self.gpu_hold[i] = gpu_list

    def _pre(self, i: int):
        def hook(_m, _args):
            cur = torch.cuda.current_stream()
            cur.wait_event(self.events[i])
            for g in self.gpu_hold[i] or []:
                g.record_stream(cur)
        return hook

    def _post(self, i: int):
        def hook(_m, _args, _out):
            for owner, attr, cpu in self.cpu_snap[i]:
                self._set(owner, attr, cpu)
            self.gpu_hold[i] = None
            nxt = i + self.window
            if nxt < len(self.layers):
                self._onload(nxt)
        return hook

    def begin(self) -> None:
        for i in range(min(self.window, len(self.layers))):
            self._onload(i)

    def end(self) -> None:
        # 計算完了を待ってから全層を pinned CPU ポインタへ戻す(重みは不変なので
        # D2H コピーは発生しない。GPU コピーは参照が切れて解放される)。
        torch.cuda.current_stream().synchronize()
        for i, snap in enumerate(self.cpu_snap):
            for owner, attr, cpu in snap:
                self._set(owner, attr, cpu)
            self.gpu_hold[i] = None


def apply_te_stream(text_encoder, window: int = 2) -> float:
    """TE の言語モデル層を窓付きストリーミング化し、forward を begin/end で包む。

    tediet 適用後(text_encoder.forward が diet_forward)でも適用前でも動く:
    現在の text_encoder.forward をそのまま内側に取り込む。
    Returns: pinned host へ移した重みサイズ(GiB)。冪等。
    """
    if getattr(text_encoder, "_te_stream_applied", False):
        return 0.0

    layers = text_encoder.model.language_model.layers
    streamer = WindowedLayerStreamer(layers, window=window)
    inner_forward = text_encoder.forward

    def streamed_forward(*args, **kwargs):
        streamer.begin()
        try:
            return inner_forward(*args, **kwargs)
        finally:
            streamer.end()

    text_encoder.forward = streamed_forward
    text_encoder._te_stream_applied = True
    text_encoder._te_streamer = streamer
    return streamer.pinned_gib
