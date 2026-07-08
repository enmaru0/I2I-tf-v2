# I2I-tf

グレースケール3D/2D医用画像のimage-to-image translation（デノイズ・ぼかし修正など）を
TensorFlow/Keras 3で学習・比較するプロジェクト。
セグメンテーションプロジェクト（Med3D-DL）のデータパイプラインを踏襲している。

## データ形式

`data.mode` で3種類のデータ供給方法を選べる（source/targetは位置合わせ済み＝同一サイズ・同一スペーシングが前提）。
ディレクトリ構成はいずれも `data_dir/{train,val}/<データセット名>/`。

- **paired**（デフォルト）: 同一フォルダに source `xxx.hdr` と target `xxx.target.hdr`
  （サフィックスは `data.target_suffix` で変更可）
- **paired_dir**: source は `data_dir`、target は `data.target_data_dir` の別フォルダに置き、
  **同名ファイル**（`{split}/{データセット名}/{同じ名前}.hdr`）をペアとする
- **self_noise**: クリーン画像1枚から source=クリーン+合成劣化（ぼかし→ノイズ） / target=クリーン
  を作る自己教師デノイジング（targetファイル不要。`data.self_noise.*` で劣化を設定）

動作確認用の合成ペアデータは `python utils/make_synthetic_dataset.py` で生成できる。

```bash
# paired_dir の例（source=劣化画像フォルダ、target=クリーン画像フォルダ、同名がペア）
python main.py --overrides exp_dir=results/pd \
    data.mode=paired_dir data_dir=noisy_root data.target_data_dir=clean_root
```

## アルゴリズム

`--overrides algorithm.name=<名前>` で切り替える。全アルゴリズムが同一のU-Netバックボーン・
データパイプライン・評価指標（PSNR/SSIM、`val_psnr`でベストモデル選択）を共有する。

| algorithm.name | 手法 | 推論コスト | 備考 |
|---|---|---|---|
| `regression` | U-Net回帰 (L1/L2/SSIM損失) | 1 forward | ベースライン。residual/direct出力 |
| `pix2pix` | 条件付きGAN + L1 | 1 forward | 3D PatchGAN。optimizerはadamw推奨 |
| `edm` | 拡散モデル (Karras 2022) | 2N-1 forwards | Heunサンプラー。EMA未実装 |
| `rectified_flow` | Rectified Flow / Flow Matching | N forwards | オイラー積分 |
| `i2i_rfr` | source起点のRectified Flow再定式化 | N forwards (N=1～4) | source→targetを直接輸送。少ステップ向き |

生成系（edm / rectified_flow / i2i_rfr）はU-Net入力を3チャンネル
（作業画像・source・ノイズレベル/時刻の定数チャンネル）にして条件付けする。

## 使い方

```bash
# 学習（regression）
python main.py --overrides exp_dir=results/reg

# 学習（他アルゴリズムの例）
python main.py --overrides exp_dir=results/p2p algorithm.name=pix2pix \
    optimizer.name=adamw optimizer.adamw.max_lr=2e-4
python main.py --overrides exp_dir=results/edm algorithm.name=edm \
    optimizer.name=adamw optimizer.adamw.max_lr=2e-4

# EMA（重みの指数移動平均）を有効にして学習（拡散/フロー系で推奨）
python main.py --overrides exp_dir=results/edm algorithm.name=edm \
    ema.enabled=True ema.decay=0.999 \
    optimizer.name=adamw optimizer.adamw.max_lr=2e-4

# データローダーの確認（サンプル画像をrawで保存）
python debug_dataloader.py --overrides exp_dir=results/debug_dl

# 評価（検証パッチでPSNR/SSIM/MAEを集計。commonlib不要）
python eval.py results/reg/checkpoints/model_best.keras

# 生成系はサンプリングステップ数をスイープして最適値を探せる
python eval.py results/edm/checkpoints/model_best.keras --num_steps 1,2,4,8,16

# 推論（元画像空間に戻してrawで保存。要pycommonlib）
python predict.py results/reg/checkpoints/model_best.keras
```

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
