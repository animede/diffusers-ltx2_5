# 生成AI推論を速くするための設計ノート

この文書は、LTX-2.5 の「リアルタイム動画生成」という到達点ではなく、そこへ至るまでに使った高速化手法を、別のモデルや推論サーバにも応用できる形で整理したものです。

対象実装は 22B の動画・音声生成 DiT ですが、中心となる考え方は一般的です。高速化を「GPUを速くすること」と一括りにせず、処理回数、行列演算、カーネル起動、デコード、データ変換、エンコードという別々のボトルネックに分解し、それぞれに違う手段を当てます。

数値は 2026-09-10 時点、RTX PRO 6000 Blackwell Workstation 96GB（sm_120）、PyTorch 2.11.0+cu130、diffusers Git版、単一GPUでの実測です。絶対時間は環境に依存しますが、どこに効き、どこに効かなかったかが本稿の主題です。

## 1. まず「何を減らすか」を分ける

推論時間は大まかに次のように分解できます。

```text
総レイテンシ
  = 前処理
  + denoise回数 × transformer 1回の時間
  + latentのデコード
  + 画素形式への変換・GPU→CPU転送
  + 動画エンコード・音声mux
  + 同期、キュー待ち、その他の固定費
```

今回採用した手法は、同じ場所を重複して最適化しているのではありません。

| 手法 | 直接減らすもの | 実測上の効果 | 主な制約 |
|---|---|---:|---|
| 蒸留スケジュールの間引き | transformerの実行回数 | 8→4 stepで約31%短縮 | 品質評価が必要 |
| NVFP4 | 各層のGEMM時間と重み帯域 | GEMM 3.2〜3.8倍、層全体約1.8倍 | Blackwell sm_120専用 |
| CUDA Graph | CPUからのカーネル起動コスト | 小shapeのE2Eで約20%短縮 | 固定shape・重み常駐が前提 |
| NATTEN | diffusion decoderの近傍attention | decode 293秒→18.3秒 | 対応カーネルと入力条件が必要 |
| GPU上のuint8化 | CPU変換と転送量 | 後処理の固定費を削減 | 丸め順序に注意 |
| NVENC・非同期エンコード | MP4生成とクリティカルパス | 後処理全体で0.5秒級 | 並列ジョブ投入時に特に有効 |

表の倍率を単純に掛け合わせてはいけません。手法ごとに対象区間が違い、ある区間を速くすると次の区間が律速になります。評価すべき値は常に、単体カーネル、1 forward、ステージ別時間、E2E時間の4段階です。

## 2. アルゴリズム側：実行回数そのものを減らす

低レベル最適化より先に、同じ巨大モデルを何回呼ぶかを見直すのが最も素直です。本実装では公式の蒸留済み8段σ列から、先頭と末尾を必ず残して等間隔に4点を選びました。

```python
idx = [round(i * (len(sigmas) - 1) / (steps - 1)) for i in range(steps)]
picked = [sigmas[i] for i in idx]
```

実装は `app/generator.py` の `_subsample_distilled_sigmas()` にあります。8 stepから4 stepへの変更で約31%短縮しました。

ここで重要なのは、計算回数を半分にしてもE2E時間が半分になるとは限らないことです。テキストエンコード、VAEデコード、エンコードなどの固定費は残ります。また、任意のschedulerを機械的に間引くのではなく、蒸留モデルが想定する軌道から選ぶ必要があります。

### 検証の観点

- 同一seedで構図、動き、音声同期をA/B比較する
- 静止画品質だけでなく、フレーム間の速度変化や停止も調べる
- 速度はdenoise区間とE2Eの両方を記録する
- 用途別に「高速設定」と「品質設定」を分ける

これは近似を導入する高速化なので、bit一致ではなく用途上の受容基準で判断します。

## 3. 演算側：NVFP4でGEMMを短くする

transformerの主要コストはLinear層の行列積です。本実装は公式NVFP4量子化チェックポイントを読み込み、Blackwellネイティブのblock-scaled FP4 GEMMである `torch._scaled_mm` を直接呼びます。重みは約18.7GBで、活性化はforwardごとに自作Tritonカーネルで動的にFP4量子化します。

処理の流れは次のとおりです。

```text
bf16 activation
  → 16要素ごとの動的量子化
  → FP4 packed値 + FP8 block scale
  → scaleをcuBLAS用blocked layoutへ変換
  → torch._scaled_mm
  → global scaleとbiasを適用
  → bf16 output
```

GEMM単体ではbf16比3.2〜3.8倍、量子化やscale処理を含む層全体では約1.8倍でした。低精度化の評価では、理論FLOPSやGEMMだけでなく、量子化・padding・レイアウト変換を含む層全体を測る必要があります。

### フォーマットはdtypeだけでは決まらない

今回もっとも危険だったのは、テンソルのshapeとdtypeが正しくても意味が一致しないケースです。

- チェックポイントのpacked weightは、high nibbleが先頭要素で、cuBLASの規約と逆だった
- `weight_scale` はすでにcuBLASのblocked layoutで保存されていた
- そこへ通常のswizzleをもう一度かけると二重変換になった
- block scaleを行優先のまま `_scaled_mm` に渡しても、例外ではなく誤った数値が返った

この不具合は、自作dequantと自作GEMMを同じ誤解釈で比較すると見逃します。両方が同じように間違うためです。最終的には公式bf16重みとの直接照合で特定しました。

量子化フォーマットを移植するときは、次を独立に検証すべきです。

1. 値の符号化とnibble順
2. block scaleの意味とglobal scaleの適用順
3. scaleテンソルの物理レイアウト
4. 量子化Linear単体と公式高精度Linearの比較
5. 1層だけでなく、全ブロック通過後の誤差

実装は `app/nvfp4.py`、検証は `probes/probe_nvfp4_*.py` にあります。

## 4. 実行系：CUDA Graphでカーネル起動をまとめる

GEMMを速くすると、相対的にCPUからGPUへ多数の小さなカーネルを発行する時間が目立ちます。GPUが速くなった結果、GPU演算ではなくlaunchが律速になるわけです。

CUDA Graphは一連のGPU処理を一度captureし、2回目以降をgraph replayとしてまとめて発行します。本実装では個々のblockではなく、diffusersのtransformer forward全体をcaptureしました。

```text
初回:
  静的入力を確保 → side streamでwarmup → workspaceを整理 → capture

2回目以降:
  静的入力へcopy_ → graph.replay() → 静的出力を直ちに消費
```

4 stepでの実測は次のとおりです。

| shape | eager | CUDA Graph | 差 |
|---|---:|---:|---:|
| 512×288、121 frames | 2.48秒 | 2.38秒 | -4% |
| 384×288、81 frames | 2.30〜2.37秒 | 1.83〜1.88秒 | -20% |

小さいshapeほど効果が大きくなりました。大きいshapeではGPU演算が長いためlaunchコストが隠れますが、小さいshapeではCPUが発行しきれずGPUの実行列に隙間ができます。これは「CUDA Graphは常に一定割合速くするもの」ではなく、「launch-boundな領域だけに効く」ことを示しています。

### 成立条件

- forward内に `.item()`、CPU依存分岐、CPUテンソル生成がない
- shape、dtype、非テンソル引数をcaptureキーで区別できる
- モデルと重みのGPUアドレスが変わらない
- モデルオフロードを使わない
- LoRAなど重みやforwardを動的に変更する処理をまたいでreplayしない
- 静的出力を次のreplay前に消費する
- 同じshapeを繰り返し、初回1.5〜2秒のcapture費を償却できる

本実装は未知の引数、capture失敗、capture数上限超過ではeagerへフォールバックします。サービス継続には有効ですが、性能測定では「知らないうちにeagerへ落ちていた」という混在を生むため、ログとreplay数の確認が必要です。

実装は `app/cudagraph.py`、検証は `probes/probe_cudagraph.py` と `probes/probe_cudagraph_gputime.py` にあります。

## 5. 専用カーネル：一般実装より演算構造に合うものを選ぶ

高解像度化パスのdiffusion decoderでは、近傍attentionの一般的なflex-attention実装が支配的でした。ここをNATTENのプリビルト `na3d` カーネルへ置き換えると、デコードは293秒から18.3秒へ短縮し、ピークVRAMも35.8GBから17.3GBへ低下しました。

これは本稿の中で最大の単一区間改善ですが、理由は「コンパイルしたから」ではありません。処理が本質的に局所近傍attentionであるのに、より汎用な実装を使っていたためです。演算の意味に合った専用カーネルを選ぶことは、dtype変更やgraph化より先に検討する価値があります。

品質はLaplacian分散、平滑領域のpatch分散、raw frame差で比較し、従来経路と実質同等であることを確認しました。高速なカーネルへ差し替える際も、速度、VRAM、数値・画質の3軸で評価します。

一方、専用カーネルには対応GPU、PyTorch/CUDA ABI、kernel size、最小入力shapeなどの制約があります。本実装は取得・ロードできない環境では従来のcompiled flex-attentionへ戻し、選ばれた経路を起動ログへ出します。

## 6. データ経路：変換、転送、エンコードをクリティカルパスから外す

denoiseが数秒まで短くなると、0.1〜0.6秒の後処理も無視できません。本実装では次の3点を行いました。

### GPU上でuint8へ変換する

VAE出力をfloatのままCPUへ転送してNumPyで変換する代わりに、GPU上でclamp、255倍、round、uint8化してから転送します。これによりCPU側の変換と転送量を減らします。

ただし、VAE出力がbf16のまま `* 255` すると旧経路と丸め結果が一致しませんでした。先にfloat32へ変換することで、従来のNumPy経路とframemd5が一致します。

```python
video = (
    video.permute(0, 1, 3, 4, 2)
    .float()
    .clamp(0, 1)
    .mul(255)
    .round()
    .to(torch.uint8)
    .cpu()
    .numpy()
)
```

### NVENCへ仕事を移す

H.264をlibx264ではなく `h264_nvenc` で処理します。品質基準を合わせるため、CRF相当の値をVBRのconstant-qualityへ対応付けています。NVENCが使えない場合や、VRAM逼迫によりcodec openが失敗した場合は、途中生成物を除いてlibx264で再試行します。

### エンコードを次のdenoiseと重ねる

MP4エンコードを専用の単一ワーカースレッドへ渡し、メインの生成ワーカーは次ジョブのdenoiseへ進みます。これは1件の完了通知を早める最適化ではなく、連続投入時のスループット最適化です。ジョブはエンコードが終わるまで `running` のままにし、ファイル完成前に `completed` を返さないようにしています。

実装は `app/encoding.py`、GPU変換と遅延エンコードの組み立ては `app/generator.py`、完了管理は `app/jobs.py` にあります。

## 7. `torch.compile`が不採用になった理由

CUDA GraphでCPU launchを減らした後、Inductorでblock内の小カーネルを融合する案も検証しました。FP4 dtypeをInductorから隠すため、NVFP4 Linearを `torch.library.custom_op` としてopaque化し、`fullgraph=True` まで通しています。

transformer単体のプローブでは、compiled + graphが1 forward 250msから168msへ改善しました。しかし本番E2Eでは同等か、384×288×81 framesの小shapeで1.83秒から2.39秒へ退行しました。極小テンソルでは生成されたTritonカーネルの固定費がeagerのATenカーネルを上回ったためと考えられます。

また、`dynamic=False` がないと2つ目のshapeで動的汎用カーネルがcaptureされ、2.77秒まで悪化しました。compile後はblock単体でcosine約0.9999の差でも、48 block後には1 forwardで約0.965まで軌道差が累積しました。

ここから得られる教訓は明確です。

- compile成功は高速化成功ではない
- 単体プローブの勝利はE2Eの勝利ではない
- 動的shapeの再コンパイルと生成カーネルを観察する
- 微小な数値差は深い反復モデルで累積する
- 不採用の実装も、再検証条件とともに実験フラグとして隔離する

実装は `app/compileblocks.py`、再現用プローブは `probes/probe_compile_*.py` にあります。本番既定値はoffです。

## 8. 正しく測る

CUDAは非同期なので、Pythonの関数が戻った時間はGPU処理の完了時間とは限りません。実際、denoiseコールバックの間隔から「1.82秒→0.48秒、3.8倍」と誤読しましたが、それはホスト側の発行時間でした。CUDA Eventで測ったGPU時間は1 forward 250ms→225ms、E2E利得は小shapeで約20%でした。

測定は次の階層で行います。

| 階層 | 測るもの | 方法 |
|---|---|---|
| kernel / op | GEMM、量子化、attention | CUDA Event、十分なwarmup |
| module | Linear層、transformer 1 forward、decoder | CUDA Event + synchronize |
| stage | text encode、denoise、decode、encode | 区間ログ。GPU区間は同期位置を明示 |
| E2E | API受付から成果物完成まで | wall clock。初回と定常状態を分離 |

さらに、次を測定条件へ含めます。

- cold startかwarmか
- 初回compile/captureを含むか
- shape、frame数、fps、step数、生成モード
- precision、offload、LoRAの有無
- エンコード完了まで含むか
- 同期点をどこへ置いたか
- 同一seedで比較したか

速度だけでなく、framemd5・音声md5、cosine、画質指標、目視A/Bを手法に応じて使い分けます。bit一致するはずの最適化で「見た目は同じ」は弱すぎ、近似を含む量子化やstep削減で「bit不一致」は当然です。

## 9. 適用判断の早見表

| 状況 | 最初に検討する手法 | 避ける判断 |
|---|---|---|
| denoise回数が支配的 | 蒸留、step削減 | 品質確認なしの任意間引き |
| Linear/GEMMが支配的、対応GPUあり | ネイティブ低精度GEMM | checkpoint形式をshapeだけで判断 |
| 小shapeでGPU utilizationに隙間 | CUDA Graph | 可変shape、一発生成、offloadとの無理な併用 |
| 特定attentionが極端に遅い | 演算専用カーネル | 汎用compileだけで押し切る |
| decode後が相対的に重い | GPU上変換、NVENC | denoiseだけ測って終了する |
| ジョブが連続投入される | 非同期エンコード | ファイル完成前の完了通知 |
| compileプローブだけ速い | E2Eと品質を再測定 | そのまま本番有効化 |

このプロジェクトの最速構成は、大容量VRAMへモデルを常駐させ、少数の固定shapeを繰り返すservingに特化しています。低VRAM化、可変shape、LoRAの頻繁な着脱、単発の高品質生成では、最適解が変わります。

## 番外: 速度を保ったままVRAMを削り、成立するGPUを広げる

高速化と同じ分解の考え方は、VRAM削減にも使えます。全常駐＋固定shapeという最速構成は大容量VRAMを要求しますが、「本当に使うものだけを載せる」だけで、必要容量が一段下のGPUに収まることがあります。この実装では次の3つで常駐を約3.6GB削り、32GB級GPU一枚でのリアルタイム動作を成立させました（空きVRAMを31GBに制限した実測で常駐28.8GB、速度は48GB構成とほぼ同等）。

1. **使わないコンポーネントは読み込まない。** 対象用途（リアルタイム）では一度も呼ばれないアップスケーラ2種が、無条件に1.2GB常駐していました。ロードを設定フラグで飛ばし、要求されたら明確なエラーを返します。
2. **巨大な埋め込みテーブルはCPUに置ける。** LLM系テキストエンコーダの語彙埋め込み（bf16で1.9GB）は、演算ではなくテーブル参照です。モジュールのforwardを「インデックスをCPUへ・gather結果だけGPUへ」というブリッジに差し替えれば、リクエストあたり数MBの転送で済み、速度影響は実測でほぼゼロでした。4bit量子化モデルでも埋め込みは量子化されずbf16で残っている、という点も見落としがちです。
3. **出力ヘッドの不要な計算を疑う。** テキストエンコーダとして隠れ状態しか使っていないのに、`ForConditionalGeneration`のforwardは全トークン×全語彙のlogits（一時0.5GB）を毎回計算していました。lm_headを経由しない薄いラッパーで除去できます。

注意点は2つ。埋め込みをCPUへ置く際は、モデル内部に「埋め込みベクトルとの比較」のような隠れた参照経路がないかをソースで確認すること（この実装では`inputs_embeds`渡し方式がこれで壊れ、モジュール差し替え方式に切り替えました）。そして、削った後のヘッドルームが小さい場合、そのGPUに他のワークロード（TTS・LLM等）を同居させないことです。

## 10. この実装から得た原則

1. 高速化はボトルネックごとに別の層で行う。
2. 速くした結果、次に何が律速になったかを毎回測り直す。
3. ハードウェアネイティブ形式は、dtypeだけでなく物理レイアウトまで検証する。
4. 固定shape・固定アドレスを運用要件にできるなら、CUDA Graphは強い。
5. 専用カーネルがある演算を、汎用コンパイラだけで最適化しない。
6. 非同期GPUではhost時間とdevice時間を混同しない。
7. 単体ベンチマーク、E2E速度、再現性、品質の4つが揃って初めて採用する。
8. 失敗した最適化も、適用域を知るための成果として残す。

リアルタイム動作の到達値、運用可能な解像度、fps固有の品質問題など、LTX-2.5の結果そのものについては [高速化の全記録](acceleration-report-20260910.md) を参照してください。
