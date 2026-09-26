"""Gemma4Unified text encoder の VRAM ダイエット(LTX25_TE_DIET=1)。

狙い(合計で常駐 -1.9GB + エンコード時一時 -0.5GB)。32GB 級 GPU で
全常駐(OFFLOAD_MODE=none)を成立させるための削減で、48GB 以上では不要。

1. embed_tokens([262144, 3840] bf16 = 1.88GB。bnb 4bit 量子化の対象は
   Linear のみで、埋め込みは bf16 のまま GPU 常駐していた)をモジュール
   ごと CPU へ移し、forward を「input_ids を CPU へ → CPU で gather+scale
   → 呼び出し元デバイスへ戻す」ブリッジ版に差し替える。1024 トークンの
   往復は ~7.5MB で無視できる。モジュール単位の差し替えなので、モデル
   内部の全呼び出し箇所(通常の埋め込み lookup と、get_placeholder_mask()
   が特殊トークンの埋め込みベクトルを比較する分岐)が無変更で動く。
   input_ids=None + inputs_embeds 渡しに変える案は不採用: その経路は
   get_placeholder_mask() が inputs_embeds と特殊トークン埋め込みの比較に
   落ち、埋め込みモジュールが CPU だと device mismatch になる上、マスク
   導出の分岐自体が本番経路と変わってしまう。

2. text_encoder.forward を「self.model(Gemma4UnifiedModel)を同一引数で
   呼んで出力をそのまま返す」薄いラッパーに差し替え、lm_head を経由しない。
   diffusers の LTX2 パイプライン(_get_gemma_prompt_embeds)は出力の
   .hidden_states しか読まず、Gemma4UnifiedForConditionalGeneration.forward
   は logits_to_keep=0(=全トークン)で lm_head を必ず計算して捨てていた。
   hidden_states の収集は TextModel の capture_outputs 機構が担い、
   Gemma4UnifiedModel の出力へそのまま passthrough される(実装確認済み)
   ため、self.model を直接呼んでも返る値は完全に同一の経路で作られる。
   lm_head.weight は embed_tokens.weight と tied のため embed の CPU 移動で
   lm_head も CPU に行くが、この forward は lm_head を呼ばないので無害
   (= TE_DIET 有効時、この text_encoder で logits は取れない)。

制約: OFFLOAD_MODE=none 専用。enable_model_cpu_offload() の accelerate
hook はコンポーネント全体を .to() で往復させるため、「embed だけ常時 CPU」
という配置と干渉する(generator 側で none 以外では適用せず警告する)。
"""

from __future__ import annotations

import torch


def apply_te_diet(text_encoder) -> float:
    """embed_tokens を CPU へ移し、lm_head をスキップする forward に差し替える。

    Returns: GPU から解放された embed_tokens の重みサイズ(GiB)。
    冪等(2回目以降は 0.0 を返して何もしない)。
    """
    embed = text_encoder.model.language_model.embed_tokens
    if getattr(embed, "_te_diet_applied", False):
        return 0.0

    freed_gib = embed.weight.numel() * embed.weight.element_size() / 1024**3

    # 1) 埋め込みの CPU ブリッジ化。bound method を先に取ってから移動する
    #    (クラス forward = gather + embed_scale 乗算をそのまま CPU で実行)。
    orig_embed_forward = embed.forward

    def bridged_embed_forward(input_ids: torch.Tensor) -> torch.Tensor:
        return orig_embed_forward(input_ids.to("cpu")).to(input_ids.device)

    embed.to("cpu")
    embed.forward = bridged_embed_forward
    embed._te_diet_applied = True

    # 2) lm_head スキップ。inner(Gemma4UnifiedModel)の出力は .hidden_states を
    #    持ち(TextModel からの passthrough)、パイプラインはそれしか読まない。
    inner = text_encoder.model

    def diet_forward(input_ids=None, attention_mask=None, **kwargs):
        kwargs.pop("labels", None)  # loss 計算は不要(lm_head を呼ばない)
        return inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )

    text_encoder.forward = diet_forward
    return freed_gib
