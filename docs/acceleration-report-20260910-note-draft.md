# 22Bの動画・音声生成モデル「LTX-2.5」を、diffusers直組みでリアルタイム化した話

※この記事は技術検証の下書きです。数値は2026年9月10日時点の実測結果です（「32GBのGPU一枚でも〜」の節のみ2026年9月27日の追検証）。

LTX-2.5は、映像と音声を同時に生成できる22BパラメータのDiTモデルです。

今回、このモデルをComfyUI経由ではなくdiffusersのパイプラインへ直接組み込み、単一GPUでどこまで高速化できるかを検証しました。

結果から書くと、約5秒の動画クリップを1.83秒で生成できました。リアルタイム比では0.37倍、つまり再生時間の半分以下です。

実測環境は次のとおりです。

- GPU：RTX PRO 6000 Blackwell Workstation 96GB（sm_120）
- PyTorch：2.11.0+cu130
- diffusers：Git版
- GPU構成：単GPU

コードと検証スクリプトは、GitHubの[animede/diffusers-ltx2_5](https://github.com/animede/diffusers-ltx2_5)で公開しています。検証用コードは主に`probes/`以下にあります。

## 先に結論：4つの高速化を重ねた

今回の高速化は、ひとつの大技で達成したものではありません。異なるボトルネックを狙った、次の4層を重ねています。

1. 蒸留ステップの間引き

   公式の8段のσ列から4段を等間隔で選び、denoise回数そのものを削減しました。8stepから4stepへの変更で、実測31％短縮です。

2. NVFP4

   公式FP4蒸留チェックポイントを読み込み、Blackwellネイティブの`torch._scaled_mm`でGEMMを実行します。GEMM単体では3.2〜3.8倍、層全体では約1.8倍になりました。

3. CUDA Graph

   transformerのforward全体をcaptureし、CPUからのカーネル起動コストをほぼ消しました。小解像度・4stepでは、出力をbit完全一致に保ったまま約20％短縮できました。

4. NVENCと非同期エンコード

   `h264_nvenc`、MP4の非同期エンコード、GPU上でのuint8変換を組み合わせ、クリップあたり合計0.5秒前後を削減しました。

なお、2倍高解像度化パスで使うdiffusion decoderにはNATTENのプリビルトカーネルも適用しています。こちらは非リアルタイム向けですが、decode時間は293秒から18.3秒まで短縮しました。

## なぜdiffusersへ直接組み込んだのか

ComfyUIは実験やワークフローの組み替えにはとても便利です。一方、今回狙ったのは「固定shapeのクリップを常駐サーバで何本も生成する」というserving寄りの用途でした。

この条件では、モデルをGPUへ常駐させ、forwardの境界や入力のshapeを厳密に管理できるdiffusers直組みが有利です。特にCUDA Graphは、モデルのメモリアドレスや実行経路が安定しているほど扱いやすくなります。

サーバはFastAPIで構成し、LTX-2.5のdiffusersパイプラインを直接呼び出しています。

## NVFP4：BlackwellのFP4を直接使う

まず大きく効いたのがNVFP4です。

Lightricksが公式配布しているNVFP4量子化済みの蒸留transformerは、ComfyUIの単一ファイル形式で約18.7GBあります。これをdiffusers側で直接ロードし、cuBLASのblock-scaled FP4を使う`torch._scaled_mm`で実行しました。実装は`app/nvfp4.py`です。

PyTorchにはbf16からFP4へ直接castする仕組みがないため、活性化の動的FP4量子化には自作のTritonカーネルを使っています。

### 実装で踏んだ3つの罠

NVFP4対応は、単にチェックポイントを読んで`_scaled_mm`へ渡せば終わり、とはいきませんでした。

1つ目は、チェックポイントのnibble順です。

チェックポイントではhigh nibbleが先頭要素ですが、これはcuBLASのパック規約と逆です。そのためロード時にbyte内のnibbleを入れ替える必要があります。

2つ目は、`weight_scale`の配置です。

この値は、すでにcuBLASのblocked layoutへswizzleされた状態で保存されています。ここへ通常のswizzleをもう一度かけると、二重変換になって値が壊れます。

厄介なのは、壊れても分布が一見正常に見えることです。ブロック単位でスケールの位置が入れ替わるだけなので、自作dequantと自作GEMMを同じ解釈で比較するとcosine値も良好に見えてしまいます。発見には、公式bf16重みとの直接比較が必要でした。

3つ目は、`_scaled_mm`の挙動です。

ブロックスケールを通常の行順のまま渡してもエラーにはなりません。しかし、結果は黙って壊れます。この種の不具合は「実行できた」ことを正しさの根拠にできないので、公式bf16版との数値照合が必須です。

## CUDA Graph：小さいshapeほど効いた

NVFP4で演算を速くすると、今度はCPUからGPUへカーネルを発行するコストが無視できなくなります。

そこで、Lightricks公式LTX-2パッケージの`cudagraph_capture.py`（v1.2.0以降）を参考に、`LTX2VideoTransformer3DModel.forward`全体へCUDA Graphを適用しました。実装は`app/cudagraph.py`です。

流れは次のとおりです。

- side streamでwarmupする
- cuBLAS workspaceをクリアする
- 共有mempoolを使ってcaptureする
- 2回目以降は静的入力バッファへ値をコピーしてreplayする

captureはshapeごとに初回だけ必要で、1.5〜2秒ほどの追加コストがかかります。その後、同じshapeを繰り返す場合に効果を回収できます。

出力は、eager実行とbit単位で完全一致しました。E2Eでは映像のframemd5と音声md5を使って確認しています。

LoRAを使うジョブは自動でeagerへフォールバックさせています。CUDA Graphは重みのGPUアドレスを記録するため、LoRAの着脱と相性が悪いためです。

### 実測結果

4stepでの結果は次のとおりです。

- 512×288、121フレーム
  - eager：2.48秒
  - CUDA Graph：2.38秒
  - 短縮率：約4％

- 384×288、81フレーム
  - eager：2.30〜2.37秒
  - CUDA Graph：1.83〜1.88秒
  - 短縮率：約20％

小さいshapeほど効果が大きいのがポイントです。

CUDA Eventで切り分けると、eagerではホスト側のカーネル発行に1.95秒、GPU側の実行に約2.0秒かかっていました。CPUがカーネルを発行しきれず、GPUの実行列に隙間が生じていた状態です。

CUDA Graph化するとCPU側は約0.001秒になり、GPU実行そのものの下限が見えるようになりました。一方、大きなshapeではGPU演算自体が支配的になるため、差は小さくなります。

### 非同期計測の落とし穴

ここでは一度、計測方法を誤りました。

非同期実行中のdenoiseループでコールバック間隔を測り、「1.82秒から0.48秒、約3.8倍」と判断してしまったのです。しかし、この値はGPUの実行時間ではなく、ホストが処理を発行した時間でした。

実際のGPU時間は、1 forwardあたり250msから225msで約1.10倍。E2Eでの効果は先ほどの約20％です。

GPU処理時間は、必ずCUDA Eventで測る必要があります。

## なぜComfyUIではCUDA Graphの事例を見かけないのか

ComfyUIでCUDA Graphが不可能というわけではありません。ただし、仕組み上いくつかの条件が噛み合いにくいと考えています。

### 重みを動かす仕組みと相性が悪い

CUDA Graphは重みのGPUアドレスを記録します。一方、ComfyUIのlowvram部分ロードやモデル退避は、重みを動かしてメモリを管理する仕組みです。

### 動的なパッチでcaptureが無効になる

ModelPatcher、LoRA、ControlNet、attentionパッチなどは、実行時にforwardを変更します。ワークフローを少し変えただけでも再captureが必要になり、管理を誤ると古いgraphから誤った出力が返る危険があります。

### shapeが固定されにくい

CUDA Graphは、同じshapeを繰り返し使って初めてcaptureコストを回収できます。固定shapeの動画を大量に流す常駐サーバは、一般的なComfyUIの使い方からは少し外れます。

本家LTX-2パッケージにはCUDA Graph実装がありますが、ComfyUI統合では使われていません。また、diffusersへ直接組み込んで運用する例もまだ少ない。この3つが重なり、事例が表に出にくかったのだと思います。

## torch.compileも試したが、本番では速くならなかった

CUDA GraphはCPU側の起動コストを減らしますが、GPU側にはまだ小さなカーネル群が残ります。そこで「Inductorで融合すれば、さらに速くできるのではないか」と考え、`torch.compile`も検証しました。

検証コードは`probes/probe_compile_*.py`にあります。

結果は、単体テストでは高速化したものの、本番E2Eでは効果なしでした。

経過は次のとおりです。

1. 素のper-block compileはfullgraphにできませんでした。自作FP4カーネルの`float4_e2m1fn_x2` dtypeで、InductorのTriton codegenがクラッシュします。

2. NVFP4 linearを`torch.library.custom_op`でopaque化すると、graph breakとFP4 dtypeの問題を同時に回避でき、`fullgraph=True`が通りました。custom op単体はbit一致しています。

3. 単体プローブでは、compiled＋CUDA Graphで1 forwardが250msから168msになり、1.48倍高速化しました。

4. ところが本番E2Eでは同じshapeで誤差範囲でした。リアルタイム向けの小shapeでは、1.83秒から2.39秒へ逆に遅くなりました。極小テンソルでは、Inductorが生成したカーネルの固定費がeagerのATenカーネルを上回ったと見ています。

5. `dynamic=False`を付けない場合、2つ目のshapeで動的な汎用カーネルがcaptureされ、2.77秒まで悪化しました。

数値面でも注意が必要でした。compile版はeager版とbit一致せず、ブロック単体ではcosine約0.9999でも、48ブロックを通した1 forwardでは約0.965まで差が累積します。`aot_eager`はbit一致し、`emulate_precision_casts`でも結果が変わらなかったため、単一箇所のバグではなく、融合カーネル群の微小差が積み重なったものと考えています。

実映像では構図のわずかなドリフトとして現れました。目視品質は同等でしたが、速度上の利得もないため本番では採用していません。

実装自体は実験フラグ`LTX25_COMPILE_BLOCKS`として残し、既定値はoffにしています。

## 約5秒クリップのリアルタイム運用ライン

4stepで、NVFP4、CUDA Graph、NVENC、非同期処理をすべて有効にした場合の実測です。

20fps、97フレームの場合：

- 安全圏（リアルタイム比0.90倍以下）：768×512
- ほぼ上限（同0.98倍前後）：768×576

24fps、121フレームの場合：

- 安全圏（リアルタイム比0.90倍以下）：704×480
- ほぼ上限（同0.98倍前後）：768×480

8stepの高品質設定では、おおむねこの半分が境界になります。

音声条件とアンカー画像を使うa2v、つまりリップシンク経路は、t2avと比べて平均0.15秒しか増えませんでした。

長尺単発では、704×416、8stepの条件でAPI上限の30秒まで速度とVRAMの両方が成立しました。実測は29.4秒、VRAM使用量は40.9GBです。

## 意外な発見：16fpsでは4秒周期の揺らぎが出る

計算量を減らすため16fpsも試しましたが、品質上の問題が見つかりました。

具体的には、ちょうど4.0秒、つまり64フレーム周期で、モーションが「淀んでからジャンプする」アーティファクトが出ます。64フレームで折り畳んだ位相プロファイルでも確認できました。

同じseed、同じ内容で比較すると、20fpsと24fpsではこの現象が消えます。

原因は、LTX-2.5の学習データが24fps・25fps中心で、秒単位のRoPE時間座標が16fpsでは学習分布から外れるため、と考えると説明できます。

そのため、リアルタイム用途では20fpsを推奨します。

## 長尺生成では、停止からのジャンプもQCする

25秒以上の単発生成では、seedによって「モーションが一度止まり、その後ジャンプする」現象が散発しました。

同じseedなら同じ位置に決定論的に再現しますが、seedを変えると消えるか、発生位置が移動します。

そこで`probes/detect_motion_stall.py`を用意しました。フレーム間差分から停滞区間と、その直後のジャンプを検出します。処理時間は1本あたり約0.5秒で、検出時はexit code 1を返します。

生成後のQCへ組み込み、検出された場合だけ別seedで再生成する運用が現実的です。

## 実運用で注意すること

### capture数の上限

`LTX25_CUDA_GRAPH_MAX_CAPTURES`の既定値は8です。上限を超えたshapeは、警告ログを出したうえでeagerへフォールバックします。

サービス自体は止まらないため、気づかずベンチマークすると結果が混ざります。多shape運用時や性能測定時は上限を引き上げ、ログも確認してください。今回の測定でも一度、この挙動で汚染されたデータを取得しました。

### t2avとa2vは別capture

a2vは音声トークン数がt2avと異なるため、CUDA Graphのcaptureキーも別です。モードごとに初回captureコストが発生します。

### モデルオフロードとは併用できない

CUDA Graphは`OFFLOAD_MODE=none`、つまり全モデルをGPUへ常駐させる構成が前提です。model offloadやsequential offloadとは併用できません。

## 追記：32GBのGPU一枚でもリアルタイムは成立する（2026-09-27検証）

ここまでの全常駐構成は、重みのロードだけで約32.5GBの空きVRAMを必要とします。空きVRAMを31GB（ヘッドレスのRTX 5090 32GB相当）に制限したテストでは、ロード段階でOOMになり、そもそも起動できませんでした。CUDA Graphを切っても結果は同じで、足りないのは純粋に重みの常駐分です。

代替として試したmodel offload構成は、ピーク22.2GBで動くものの、チャンクごとに重みがPCIeを往復するため約5秒のクリップに約15秒かかり、リアルタイムには使えません。

そこで、リアルタイム経路では一度も使われないものを削りました。

1. latent/temporal upsamplerを読み込まない（`LTX25_LOAD_UPSAMPLERS=0`、−1.2GB）

   リアルタイム用途は常に`upscale: false`です。それにもかかわらず、latent upsampler（950MB）とtemporal upsampler（250MB）は無条件にGPUへ常駐していました。

2. テキストエンコーダのダイエット（`LTX25_TE_DIET=1`、常駐−1.9GB＋一時−0.5GB）

   Gemmaテキストエンコーダはbnb 4bit量子化ですが、量子化されるのはLinear層だけで、語彙262,144×隠れ3,840の埋め込みテーブル（bf16で1.88GB）はそのままGPUに常駐していました。埋め込みはただのテーブル参照なので、モジュールのforwardを「インデックスをCPUへ送り、CPUでgatherして結果だけGPUへ返す」ブリッジに差し替えれば、リクエストあたりの転送は約7.5MBで済みます。さらに、`ForConditionalGeneration`のforwardは誰も使わない全トークンのlogits（1024×262144、bf16で約0.5GB分の一時確保）を毎回計算していたため、lm_headを経由しない薄いラッパーでこれも除去しました。ちなみにlm_headの重みは埋め込みテーブルとtiedなので、重み側の追加節約はゼロです。

   実装上の注意をひとつ。`inputs_embeds`を自前で計算して渡す方式は一見きれいですが、この経路ではモデル内部が特殊トークンの判定を「埋め込みベクトルとの比較」で行うため、埋め込みテーブルをCPUへ置いた瞬間にデバイス不一致で壊れます。モジュールのforward差し替え方式なら、内部の全呼び出し箇所が無変更で動きます。

結果、空き31GB制限下で常駐・ピークとも28.77GBに収まり、512×384・20fps・97フレーム・4stepのリップシンクチャンクが定常3.8秒（リアルタイム比0.79倍）で生成できました。48GB構成の3.5秒とほぼ同速です。品質面では、同一プロセス内なら同じseedでframemd5まで完全に一致する決定論を維持しています。

つまり、**リアルタイムの制約は演算性能ではなくVRAM容量であり、それも「本当に使うものだけを載せる」ことで32GB級に収まります**。検証はRTX PRO 5000 Blackwell 48GBの空きVRAMを31GBに制限して行ったもので、5090の実カードでは演算性能に余裕がある分、むしろ有利なはずです。

実アプリでの通し検証（会話アプリ＝TTSは2枚目GPU、LLMは別ホスト、同じ空き31GB制限）では、さらに2つの罠がありました。合成テスト（単一shape）では28.8GBで収まったのに、実アプリは待機動画・発話アンカー・ターン先頭の低解像度チャンク・ターン内連結チャンクとshapeが多く、まずアロケータ断片化で「予約済み未使用」領域が1.3〜1.6GB発生して境界OOMになりました。これは`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`で回収でき、7チャンク連続朗読が完走するようになります。しかし会話モードではshape種がさらに増え、CUDA Graphのcaptureごとのプール蓄積で新shapeのワークスペース確保（364MB）が失敗しました。既にcapture済みのshapeのreplayだけ成功し、新shapeのチャンクだけ交互に落ちる、という症状です。ここで一度は「32GB構成ではCUDA Graphを無効にする」と結論しましたが、最終的にはもう一段の削減で覆せました。テキストエンコーダのNF4言語モデル層（5.7GB）をpinned host memoryへ常駐させ、エンコード中だけ層単位の窓付き先読みでGPUへ流すストリーミング（`app/testream.py`、window=2）です。層あたりの転送（約2.4ms）は層あたりの計算（約4.4ms）より短いため、先読み2層で転送は計算に完全に隠れ、エンコードは0.213秒→0.369秒（+0.16秒）、出力は全常駐とbit完全一致でした。ちなみにdiffusersの`apply_group_offloading`はstream使用時に1層グループが強制され、48グループ分のフック固定費で+1.0秒/エンコードになるため使えず、自前実装です。この5GBの削減で常駐は約23.6GBになり、graph poolの余裕が戻って**CUDA Graphを32GB構成で再有効化**できました。会話7チャンク連続の通しで全チャンク成功・ピーク27.1GB・約4秒/チャンクを実測しています。境界VRAMとの戦いは「削る→速くする→また削る」の反復だった、というのが実運用の結論です。

この構成の前提条件は3つです。第一に、upsamplerを載せないためupscaleとt2i品質経路は使えません（要求すると明確なエラーを返します）。第二に、ヘッドルームは約2GBしかないため、**GPUは動画エンジンの専有が前提です。TTSやLLMを同居させる場合は、それらをCPU実行（または別ホスト）にしてください**。第三に、ヘッドレス運用（画面出力は別GPUかiGPU）が前提です。

### おまけ：プロセスをまたぐと同一seedでも出力が変わる

この検証の過程で、もうひとつ計測上の罠を見つけました。nvfp4構成は同一プロセス内では完全に決定論的（同じseedでframemd5完全一致）ですが、**プロセスを再起動すると同じseed・同じ設定でも出力が変わります**（PSNR 27dB程度の軌道分岐。cuBLAS等のアルゴリズム選択がプロセスごとに変わるためと推測）。bit再現が必要な比較実験は、必ず同一プロセス内で行ってください。当初この差分を新機能のせいと誤認しかけ、対照実験（同一構成でプロセスだけ再起動）でようやく切り分けられました。

## 再現したい人向けの実装メモ

ここからは、実装を追いたい人向けの詳細です。

### CUDA Graphを成立させる条件

次の条件がひとつでも欠けると、今回の方式は成立しません。

- transformerのforward内に、CPU依存の分岐、`.item()`、CPUテンソル生成がないこと
- RoPE座標の`video_coords`と`audio_coords`をforwardの外で事前計算して渡すこと
- 蒸留経路、つまり`cfg=1`かつ`stg=0`であること
- 全モデルがGPUへ常駐していること
- 生成が単一スレッドであること
- LoRAを使用していないこと

LTX2パイプラインではRoPE座標を外から渡せるため、forward全体をcaptureできます。座標を渡さない場合はforward内でCPUテンソルが生成され、capture中のHost-to-Device copyで失敗します。

STG経路にも`torch.zeros((B,))`によるCPUテンソル生成があるため、今回の対象外です。

### captureの手順

実装全体は約170行で、`app/cudagraph.py`にあります。要点は次のコードです。

```python
# 1. テンソル引数をcloneし、静的入力バッファを作る
statics = {
    k: v.clone()
    for k, v in kwargs.items()
    if isinstance(v, torch.Tensor)
}
static_kwargs = {**kwargs, **statics}

# 2. side streamで3回warmupする
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3):
        orig_forward(**static_kwargs)
torch.cuda.current_stream().wait_stream(side)
torch.cuda.synchronize()

# 3. cuBLAS workspaceをクリアする
torch._C._cuda_clearCublasWorkspaces()

# 4. warmup後、capture前に入力を入れ直す
for k, sbuf in statics.items():
    sbuf.copy_(kwargs[k])

# 5. 共有mempoolを使ってcaptureする
pool = torch.cuda.graph_pool_handle()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(
    graph,
    pool=pool,
    capture_error_mode="global",
):
    outputs = orig_forward(**static_kwargs)
```

warmupではTriton JIT、cuBLAS workspaceの確保、カーネル選択をcaptureの外へ出します。その後にworkspaceをクリアするのは、graph mempoolへメモリが二重計上されるのを防ぐためです。

replayは次の形です。

```python
for k, sbuf in statics.items():
    sbuf.copy_(new_kwargs[k])

graph.replay()
return outputs
```

返り値はcapture時に確保した静的出力バッファです。そのため、呼び出し側は次のreplayより前に出力を消費する必要があります。

LTX2パイプラインではtransformer出力を直後に`.float()`でコピーするため安全です。別のパイプラインへ組み込む場合は、この契約を確認してください。

forwardの差し替えは、インスタンスへ`transformer.forward = runner`を代入するだけです。`nn.Module.__call__`はインスタンス属性のforwardを優先します。

captureキーには、すべてのテンソル引数の名前、shape、dtype、deviceと、非テンソル引数の値を含めています。キーごとにgraphを1本作ります。

未知の引数型、capture数の上限超過、capture中の例外ではeagerへフォールバックします。例外になったキーは、その後も恒久的にeagerで実行し、サービス全体は止めません。

モデルオフロードやLoRAの着脱では重みのアドレスが変わるため、graphを無効化する必要があります。LoRAジョブはeagerで処理し、ジョブ後に全captureを破棄しています。

### NVFP4チェックポイントの構造

量子化対象の各層には、次のテンソルがあります。

- `<module>.weight`
  - dtypeとshape：U8、`[out, in/2]`
  - e2m1を1byteあたり2値パック
  - high nibbleが先頭要素。cuBLAS規約とは逆なので、ロード時にbyte内swapが必要

- `<module>.weight_scale`
  - dtypeとshape：F8_E4M3、`[out, in/16]`
  - 16要素ごとのブロックスケール
  - すでにcuBLAS blocked layoutへswizzle済み。再度swizzleしてはいけない

- `<module>.weight_scale_2`
  - F32スカラー
  - グローバルスケール

- `<module>.input_scale`
  - F32スカラー
  - 活性化の静的スケール

norm、bias、adalnなど、量子化されていない層はbf16です。ヘッダの`_quantization_metadata`に量子化対象層の一覧があります。

forwardでは、次の順序で処理します。

```text
x（bf16、[..., in]）
→ 2次元化
→ Mを16の倍数へpadding
→ Tritonで動的量子化
   - packed fp4：[M, in/2]
   - scales F8：[M, in/16]
→ 活性化側scaleをto_blocked()でswizzle
→ torch._scaled_mm(..., out_dtype=bf16)
→ input_scale × weight_scale_2を乗算
→ biasを加算
→ 元のshapeへ戻す
```

`to_blocked`は、cuBLAS向けの128行×4列タイル並べ替えです。通常の行順のまま渡してもエラーにならないため、検証には必ず公式bf16重みとの出力比較を使います。

### 4stepで使うσ列

公式蒸留モデルの8段のσ列は次のとおりです。

```text
[1.0, 0.99609375, 0.9765625, 0.9375,
 0.8515625, 0.578125, 0.28125, 0.109375]
```

8未満の`steps=n`を指定した場合は、先頭と末尾を必ず含め、等間隔でn個を選びます。インデックスとしては`round(linspace(0, 7, n))`です。

4stepでは実測31％短縮し、静止フレーム品質に大きな破綻はありませんでした。

### エンコード側の実装

動画エンコードには`h264_nvenc`を使い、presetは用途に応じてp4〜p7を選べます。NVENCがない環境ではlibx264へ自動でフォールバックします。

MP4エンコードはワーカースレッドへ渡し、次のジョブのdenoiseと重ねます。ただし、クライアントが前のジョブの完了前に次のジョブを投入する運用でなければ、この重畳効果は得られません。

フレームのuint8化は`output_type="pt"`で受け、GPU上で255倍します。ここでbf16のまま255倍するとbit不一致になるため、先に`.float()`を挟みます。

## 実装ファイルと検証コード

最後に、各機能の場所をまとめます。

- CUDA Graph
  - 実装：`app/cudagraph.py`
  - 導入とガード：`app/generator.py`
  - 検証：`probes/probe_cudagraph.py`
  - GPU時間の検証：`probes/probe_cudagraph_gputime.py`

- NVFP4
  - 実装：`app/nvfp4.py`
  - 検証：`probes/probe_nvfp4_*.py`

- torch.compileの実験
  - 実装：`app/compileblocks.py`
  - 検証：`probes/probe_compile_*.py`

- 蒸留stepの間引き
  - 実装：`app/generator.py`の`_subsample_distilled_sigmas()`

- モーション停止の検出
  - 検証・QC：`probes/detect_motion_stall.py`

## まとめ

今回、蒸留、NVFP4、CUDA Graph、NVENCという4層の高速化を組み合わせ、22Bの音声同時生成モデルでリアルタイムを超える0.37倍を単一GPUで確認できました。

特に効果が大きいのは、「小解像度、少step、固定shapeを繰り返す」というserving用途です。この領域では、モデルと実行経路を固定しやすいdiffusers直組みが強みを発揮します。

一方で、`torch.compile`は単体プローブほど本番では効かず、16fpsでは4秒周期の品質問題も見つかりました。高速化では成功した結果だけでなく、効かなかった条件や品質上の境界を残すことも重要だと感じています。

再現方法と最新の実装は、GitHubの[animede/diffusers-ltx2_5](https://github.com/animede/diffusers-ltx2_5)にまとめています。READMEと`probes/`もあわせて参照してください。
