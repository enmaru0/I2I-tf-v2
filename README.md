# I2I-tf

グレースケール3D/2D医用画像のimage-to-image translation（デノイズ・ぼかし修正など）を
TensorFlow/Keras 3で学習・比較するプロジェクト。
セグメンテーションプロジェクト（Med3D-DL）のデータパイプラインを踏襲している。

## データ形式

`data.mode` でデータ供給方法を選べる（source/targetは位置合わせ済み＝同一サイズ・同一スペーシングが前提）。
ディレクトリ構成はいずれも `data_dir/{train,val}/<データセット名>/`。

- **paired**（デフォルト）: 同一フォルダに source `xxx.hdr` と target `xxx.target.hdr`
  （サフィックスは `data.target_suffix` で変更可）
- **paired_dir**: source は `data_dir`、target は `data.target_data_dir` の別フォルダに置く。
  通常は同名ファイルを対応付けるが、`data.pair_matching.method=stem_token`でfilenameの
  一部をpair keyにできる
- **denoise**: クリーン画像1枚から source=合成劣化 / target=クリーンを作る自己教師デノイジング。
  `data.self_noise.simulation` でGaussian、Rician、k-space、low-fieldを選択できる
  （`self_noise`は後方互換の別名）
- **self_sr**: PSFぼかし、低解像度スライス標本化、再補間からthrough-plane SRペアを合成
  （撮像条件は `data.self_sr.protocols` または範囲指定、スライス位相もランダム化可能）

動作確認用の合成ペアデータは `python utils/make_synthetic_dataset.py` で生成できる。

```bash
# paired_dir の例（source=劣化画像フォルダ、target=クリーン画像フォルダ、同名がペア）
python main.py --overrides exp_dir=results/pd \
    data.mode=paired_dir data_dir=noisy_root data.target_data_dir=clean_root
```

## CAC non-gated → gated motion correction

CAC専用設定は [`conf/config_cac.yaml`](conf/config_cac.yaml) に分離している。
通常の`conf/config.yaml`は変更せず、`extends: config.yaml`で共通設定を継承する。
3.0 mm（AX）×0.5 mm×0.5 mmのnative spacing、mixed U-Net、-1000～2000 HU、
mixed precision、gradient accumulation、EMAを初期値とする。

位置合わせ済み実ペアでは`conf/config_cac_real.yaml`を使う。例えば
`P001_non_gated.hdr`と`P001_gated.hdr`は、stemを`_`で分割した先頭の`P001`が
一致するためpairになる。1つの患者IDに複数のgated targetがある場合はerrorにする。
最初にgated targetからCAC crop用sidecarを作る。

```bash
python utils/prepare_cac_boxes.py \
    --config conf/config_cac_real.yaml \
    --source-dir /data/cac/non_gated \
    --target-dir /data/cac/gated

python main.py --config conf/config_cac_real.yaml --overrides \
    exp_dir=results/cac_real \
    data_dir=/data/cac/non_gated \
    data.target_data_dir=/data/cac/gated
```

単一時相gated CTから合成ペアを事前生成する場合、入力は既存datasetと同じ
`{train,val}/<dataset>/*.hdr`構造にする。

```bash
python utils/generate_cac_motion_dataset.py \
    --config conf/config_cac.yaml \
    --input-dir /data/cac/gated_clean \
    --output-dir /data/cac/simulated \
    --variants-per-case 3

# 出力: simulated/source と simulated/target
python main.py --config conf/config_cac.yaml --overrides \
    exp_dir=results/cac_synthetic_pretrain \
    data_dir=/data/cac/simulated/source \
    data.target_data_dir=/data/cac/simulated/target
```

`data.cac_motion.simulator`は2種類ある。

- `image_blend`: 複数の局所warp時相を平均する高速な画像領域近似。
- `parallel_fbp`: viewごとに異なる時相をAX parallel-beam投影し、Poisson noiseと
  FBP再構成を行う。angle依存streakを生成できるが、実scannerのcone/helical geometryを
  完全には再現しない。学習中ではなく事前生成に使う。

heart maskは画像と同じdirectoryのsidecar（`case.hdr`に対する
`case.mask.hdr`）を自動で読み込む。別rootに置く場合は同じ相対directory構造で
`case.mask.hdr`（または`case.heart.mask.hdr`）を配置し、`--heart-mask-dir`を渡す。
旧仕様の同名`case.hdr`も外部mask rootに限り読み込める。maskが見つからず
`--heart-mask-dir`も未指定の場合はconfigのsoft ellipsoidへfallbackするため、
FOVや心臓位置が一定でないdatasetではsidecar maskの利用を推奨する。

実ペアと合成ペアのCAC統計（130 HU mask Dice、体積比、peak HU比、重心距離、
心臓ROI MAE）を比較してsimulatorを校正できる。

```bash
python utils/summarize_cac_pairs.py \
    --config conf/config_cac_real.yaml \
    --source-dir /data/cac/non_gated \
    --target-dir /data/cac/gated \
    --output results/cac_real_pair_stats.json

python utils/summarize_cac_pairs.py \
    --source-dir /data/cac/simulated/source \
    --target-dir /data/cac/simulated/target \
    --output results/cac_simulated_pair_stats.json
```

推奨手順は、合成ペアで`cac_regression`をpretrainし、実ペアだけでfinetuneする方法。
`cac_regression`は通常L1に加え、130 HU以上のtarget領域と0.5 mm面内edgeを重くする。
これらの統計・lossは研究用であり、診断用Agatston score実装ではない。

## アルゴリズム

`--overrides algorithm.name=<名前>` で切り替える。全アルゴリズムが同一のU-Netバックボーン・
データパイプライン・評価指標（PSNR/SSIM、`val_psnr`でベストモデル選択）を共有する。

| algorithm.name | 手法 | 推論コスト | 備考 |
|---|---|---|---|
| `regression` | U-Net回帰 (L1/L2/SSIM損失) | 1 forward | ベースライン。residual/direct出力 |
| `cac_regression` | CAC重み付き残差回帰 | 1 forward | 130 HU以上と面内edgeを追加監督。CAC専用configで使用 |
| `pix2pix` | 条件付きGAN + L1 | 1 forward | 3D PatchGAN。optimizerはadamw推奨 |
| `edm_karras` | EDM (Karras 2022) | 2N-1 forwards | Heun/Euler、`edm`は後方互換名 |
| `conditional_restoration_ode` | 条件付きrestoration ODE | 2N-1 forwards | PyTorch版の従来EDM目的関数を移植 |
| `rectified_flow` | Rectified Flow / Flow Matching | N forwards | オイラー積分 |
| `i2i_rfr` | source起点Rectified Flow | N forwards | source→targetを直接輸送する独自variant |
| `i2i_rfr_x0` | I2I-RFR x0予測 | N forwards | ノイズ化targetからx0を予測 |
| `resshift` | Residual Shifting拡散 | N forwards | source近傍から少ステップ復元 |
| `split_mean_flow` | 区間平均速度場 | N forwards | 1～数ステップ生成 |

生成系はU-Net入力を3～4チャンネル
（作業画像・source・ノイズレベル/時刻の定数チャンネル）にして条件付けする。

## 使い方

```bash
# 学習（regression）
python main.py --overrides exp_dir=results/reg

# 学習（他アルゴリズムの例）
python main.py --overrides exp_dir=results/p2p algorithm.name=pix2pix \
    optimizer.name=adamw optimizer.adamw.max_lr=2e-4
python main.py --overrides exp_dir=results/edm algorithm.name=edm_karras \
    optimizer.name=adamw optimizer.adamw.max_lr=2e-4

# EMA（重みの指数移動平均）を有効にして学習（拡散/フロー系で推奨）
python main.py --overrides exp_dir=results/edm algorithm.name=edm_karras \
    ema.enabled=True ema.decay=0.999 \
    optimizer.name=adamw optimizer.adamw.max_lr=2e-4

# データローダーの確認（サンプル画像をrawで保存）
python debug_dataloader.py --overrides exp_dir=results/debug_dl

# 評価（検証パッチでPSNR/SSIM/MAEを集計。commonlib不要）
python eval.py results/reg/checkpoints/model_best.keras

# 生成系はサンプリングステップ数をスイープして最適値を探せる
python eval.py results/edm/checkpoints/model_best.keras --num_steps 1,2,4,8,16

# 推論（元画像空間に戻してrawで保存。標準ではsliding-window推論）
python predict.py results/reg/checkpoints/model_best.keras

# Through-plane SR: 1 mm model残差だけをnative XYへ戻し、z=1 mmで保存
python predict.py results/sr/checkpoints/model_best.keras \
    --native-xy-residual --target-z-spacing-mm 1.0

# 同一seed・optimizer・step予算で主要手法を比較（時間予算はbudget_mode=minutes）
python compare_algorithms.py --exp_root results/compare \
    --algorithms regression i2i_rfr_x0 resshift \
    --budget_mode steps --budget 100000 --seeds 0 1 2 \
    --overrides data.mode=self_sr
```

比較時は `reproducibility.seed` からcrop・劣化・生成初期ノイズを固定する。評価は症例単位の
PSNR/MAEとz/y/x方向SSIMを使い、`evaluation.val_patches_per_volume` 個の固定パッチを取る。
`mixed_precision_policy=mixed_float16` でloss scaling付きmixed precisionを有効化できる。
`gradient_accumulation_steps=N` では`batch_size`をmicro batch sizeとして、N回の平均勾配で
generator（pix2pixではdiscriminatorも）を1回更新する。`num_train_steps`と`eval_every`は
従来どおりoptimizer更新回数であり、LR scheduleの意味は変わらない。

MRIでは`data.contrast_augmentation.enabled=True`により、gamma/scale/shift/inversion/
smooth bias fieldを劣化simulation前のcleanへ適用できる。pairedデータでは同じ変換を
source/targetへ共有する。

`predict.py --native-xy-residual`は`output_mode=residual`のmodel専用opt-in推論。
入力全体を学習時`norm_spacing_zyx`へ変換してmodel残差を求める一方、別経路では
native XYを保持してzだけを`target-z-spacing-mm`へ補間する。残差だけをnative gridへ
戻して加算し、出力spacingを`[target_z, native_y, native_x]`として保存する。
学習時の`image.share_normalization=True`が必要で、標準推論の挙動は変更しない。

## EMA（指数移動平均）

`ema.enabled=True` で generator の重みのEMAを保持する。拡散/フロー系（edm /
rectified_flow / i2i_rfr）で生成品質が安定・向上しやすい。有効時は検証メトリクス・
チェックポイント・ログ画像がすべてEMA重みで計算される。ただし model_latest.keras も
EMA重みで保存されるため、EMA有効時の `restore` による厳密な学習再開はできない。

## 構成

- `data/` — ペア読み込みデータローダー（同一アフィン変換をsource/targetに適用、
  有効領域マスク生成、CT窓/MRパーセンタイル正規化）
- `trainers/` — アルゴリズム別の学習ループ。`BaseI2ITrainer` がgeneratorを内包し、
  `MODEL_REGISTRY` と `build_trainer` に登録して追加する
- `models/` — U-Net（マスク付きBatchRenorm）と3D PatchGAN discriminator
- `losses/` — マスク付きL1/L2/PSNR/SSIM（+旧DICE/CE）
- `callbacks/` — TensorBoardへのsource/target/予測/誤差マップの記録

## 注意点

- 強度系データ拡張（ノイズ付加・ぼかし等）はsourceのみに適用され、デフォルト無効。
  デノイズ/ぼかし修正では劣化過程そのものを変えてしまうため、有効化は慎重に
- pix2pixの`restore`ではdiscriminator optimizerの状態は復元されない（重みは復元される）
- DGX Spark (GB10)ではコンテナ内ptxasが古いとXLAが失敗する。
  ホストの`/usr/local/cuda/bin/ptxas`をコンテナへマウント/コピーすること
