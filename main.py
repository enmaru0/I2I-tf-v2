import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import keras
import numpy as np
from absl import logging
from keras.api.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.api.optimizers import SGD, AdamW
from keras.api.optimizers.schedules import CosineDecay
from omegaconf import ListConfig, OmegaConf

from callbacks import (
    EMACallback,
    ImageLogger,
    TestImageLogger,
    UnifiedTensorBoardLogger,
)
from data.dataloader import create_dataloader, create_test_batch, resolve_target_hdr_path
from trainers import MODEL_REGISTRY, BaseI2ITrainer, attach_aux_optimizers, build_trainer


def read_cfg_and_parse_arg():
    # コマンドライン引数と設定ファイルを読み込む関数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="設定を上書きするフォーマット (例: 'batch_size=12 aug.crop_size_zyx=[64,64,64]')",
    )
    args = parser.parse_args()
    cmd_overrides = args.overrides

    config_path = "conf/config.yaml"
    cfg = OmegaConf.load(config_path)

    # コマンドライン引数で設定を上書きする
    override_config = OmegaConf.from_dotlist(cmd_overrides)
    for key in override_config:
        if key not in cfg:
            raise KeyError(f"設定ファイルに存在しないキー: {key}")
    cfg = OmegaConf.merge(cfg, override_config)

    # ディレクトリを Path 型に変換
    cfg.exp_dir = Path(cfg.exp_dir)
    cfg.data_dir = Path(cfg.data_dir)
    cfg.restore = Path(cfg.restore) if cfg.restore else None
    cfg.finetune = Path(cfg.finetune) if cfg.finetune else None

    if cfg.restore and cfg.finetune:
        raise ValueError("restoreとfinetuneの両方を指定することはできません")

    # リスケール済みのディレクトリを探す
    target_scale_zyx = np.array(cfg.aug.affine.norm_spacing_zyx)
    target_scale_zyx = target_scale_zyx.astype(np.float32)
    rescaled_dir = cfg.data_dir.parent / (
        cfg.data_dir.name + "_" + "_".join(map(str, target_scale_zyx))
    )
    if rescaled_dir.exists():
        cfg.data_dir = rescaled_dir
    elif target_scale_zyx[0] > 2:
        raise ValueError(
            "Thickスライスで学習する場合は./utils/rescale_dataset.pyで予めリスケールすることを推奨します"
        )

    # その他cfgのチェック
    assert cfg.image.modality in ["CT", "MR"], cfg.image.modality
    # U-Netのmodeに応じてクロップzサイズの割り切れ制約をチェックする
    unet_mode = cfg.model.unet.get("mode", "3d")
    assert unet_mode in ["3d", "mixed", "2d"], unet_mode
    crop_z = int(cfg.aug.crop_size_zyx[0])
    z_div = {"3d": 2**cfg.model.unet.depth, "mixed": 2, "2d": 1}[unet_mode]
    assert crop_z % z_div == 0, (
        f"model.unet.mode={unet_mode}ではcrop_size_zyxのzは{z_div}の倍数にしてください"
        f"（現在: {crop_z}）"
    )
    assert cfg.data.mode in ["paired", "paired_dir", "self_noise", "self_sr"], (
        cfg.data.mode
    )
    if cfg.data.mode == "self_sr":
        for axis_name in list(cfg.data.self_sr.axes) + [cfg.data.self_sr.val_axis]:
            assert str(axis_name).upper() in ("AX", "COR", "SAG"), (
                f"self_sr.axes/val_axisはAX/COR/SAGで指定してください: {axis_name}"
            )
        sr_profile = str(cfg.data.self_sr.slice_profile).lower()
        assert sr_profile in ("gaussian", "box", "continuous", "none"), (
            "data.self_sr.slice_profileはgaussian/box/continuous/noneで"
            f"指定してください（現在: {cfg.data.self_sr.slice_profile}）"
        )
        if sr_profile == "continuous":
            assert float(cfg.data.self_sr.continuous_sigma_k) >= 0.0, (
                "data.self_sr.continuous_sigma_kは0以上で指定してください"
                f"（現在: {cfg.data.self_sr.continuous_sigma_k}）"
            )
        sr_interp = str(cfg.data.self_sr.slice_interpolation).lower().replace("_", "-")
        assert sr_interp in (
            "linear",
            "spline",
            "b-spline",
            "bspline",
            "b-spine",
            "bspine",
        ), (
            "data.self_sr.slice_interpolationはlinear/spline/b-splineで"
            f"指定してください（現在: {cfg.data.self_sr.slice_interpolation}）"
        )
        if sr_interp in ("b-spline", "bspline", "b-spine", "bspine"):
            b_spline_order = int(cfg.data.self_sr.b_spline_order)
            assert 0 <= b_spline_order <= 5, (
                "data.self_sr.b_spline_orderは0〜5で指定してください"
                f"（現在: {cfg.data.self_sr.b_spline_order}）"
            )
    if cfg.data.mode == "paired_dir":
        assert cfg.data.target_data_dir, (
            "paired_dirモードではdata.target_data_dirを指定してください"
        )
        assert Path(cfg.data.target_data_dir).exists(), (
            f"target_data_dirが存在しません: {cfg.data.target_data_dir}"
        )
    if cfg.data.real_dc.enabled:
        assert cfg.data.mode == "self_sr", (
            "data.real_dcは現状self_srタスク向けです（劣化演算子がthrough-plane SR前提）"
        )
        assert cfg.data.real_dc.input_dir, (
            "real_dc.enabled時はreal_dc.input_dirを指定してください"
        )
        assert Path(cfg.data.real_dc.input_dir).exists(), (
            f"real_dc.input_dirが存在しません: {cfg.data.real_dc.input_dir}"
        )
        assert str(cfg.data.real_dc.axis).upper() in ("AX", "COR", "SAG"), (
            cfg.data.real_dc.axis
        )
        dc_supported = {
            "regression",
            "pix2pix",
            "rectified_flow",
            "i2i_rfr",
            "i2i_rfr_x0",
            "resshift",
            "split_mean_flow",
        }
        assert cfg.algorithm.name in dc_supported, (
            f"real_dcは{sorted(dc_supported)}のみ対応です（現在: {cfg.algorithm.name}）"
        )
    if cfg.test.enabled:
        assert cfg.test.input_dir, "test.enabled時はtest.input_dirを指定してください"
        assert Path(cfg.test.input_dir).exists(), (
            f"test.input_dirが存在しません: {cfg.test.input_dir}"
        )
    assert cfg.algorithm.name in MODEL_REGISTRY, (
        f"algorithm.name={cfg.algorithm.name} は未実装です。"
        f"利用可能: {list(MODEL_REGISTRY.keys())}"
    )
    assert cfg.model.num_channel == 1, (
        "グレースケール画像変換なのでmodel.num_channelは1を指定してください"
    )
    return cfg


def gpu_setting(gpu_str: str, gpu_allow_growth: bool) -> None:
    if gpu_str == "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(os.getenv("SGE_GPU", 0))
    else:
        if isinstance(gpu_str, ListConfig):
            gpu_str = ",".join(map(str, gpu_str))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_str)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = str(gpu_allow_growth).lower()


def prepare_data_dict(cfg):
    """
    sourceのhdrパスのリストをデータセットごとに列挙する。
    - paired: source xxx.hdr, target xxx{target_suffix}.hdr（同一フォルダ、命名規則）
    - paired_dir: source(data_dir)とtarget(target_data_dir)を別フォルダの同名ファイルで対応
    - self_noise / self_sr: クリーン画像のみ（targetファイルは不要。劣化はローダ内で合成）
    targetやマスク類のrawファイルは除外する。
    """
    data_dir = Path(cfg.data_dir)
    target_suffix = cfg.data.target_suffix
    no_target = cfg.data.mode in ("self_noise", "self_sr")

    def _collect(split_name):
        data_dict = defaultdict(dict)
        split_dir = data_dir / split_name
        if not split_dir.exists():
            return data_dict
        for dataset_dir in sorted(split_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            hdr_list = []
            for raw_path in sorted(dataset_dir.glob("*.raw")):
                if raw_path.stem.endswith(target_suffix) or ".mask" in raw_path.name:
                    continue
                hdr_path = raw_path.with_suffix(".hdr")
                if not no_target:
                    tgt_hdr_path = resolve_target_hdr_path(hdr_path, cfg)
                    assert tgt_hdr_path.exists(), (
                        f"targetが見つかりません: {tgt_hdr_path}"
                    )
                hdr_list.append(hdr_path)
            assert len(hdr_list) > 0, f"データが見つかりません: {dataset_dir}"
            data_dict[dataset_dir.name]["img_hdr_list"] = hdr_list
        return data_dict

    # 学習用：サンプリング頻度はデータ数に比例させる（全画像で一様サンプリング）
    train_dict = _collect("train")
    num_total = sum(len(v["img_hdr_list"]) for v in train_dict.values())
    for value in train_dict.values():
        value["freq"] = len(value["img_hdr_list"]) / num_total

    # 検証用データリストを作成
    val_dict = _collect("val")
    for value in val_dict.values():
        value["freq"] = -1
    return train_dict, val_dict


def select_optimizer(cfg):
    cfg_opt = cfg.optimizer[cfg.optimizer.name]
    # スケジューラーを設定
    warmup_steps = int(cfg_opt.warmup_ratio * cfg.num_train_steps)
    lr_schedule = CosineDecay(
        cfg_opt.warmup_lr,
        cfg.num_train_steps - warmup_steps,
        alpha=0.0,
        name="CosineDecay",
        warmup_target=cfg_opt.max_lr,
        warmup_steps=warmup_steps,
    )

    # オプティマイザを設定
    if cfg.optimizer.name == "sgd":
        optimizer = SGD(
            learning_rate=lr_schedule,
            momentum=cfg_opt.momentum,
            nesterov=cfg_opt.use_nesterov,
            weight_decay=cfg_opt.wd,
            clipvalue=cfg_opt.clip_value,  # 勾配クリッピング
        )
    elif cfg.optimizer.name == "adamw":
        optimizer = AdamW(
            learning_rate=lr_schedule,
            weight_decay=cfg_opt.wd,
            clipvalue=cfg_opt.clip_value,  # 勾配クリッピング
        )
    else:
        raise NotImplementedError(cfg.optimizer.name)
    return optimizer


if __name__ == "__main__":
    # ログのレベルを設定する：INFO以上を表示
    logging.set_verbosity(logging.INFO)

    cfg = read_cfg_and_parse_arg()

    # 実験フォルダを作成
    cfg.exp_dir.mkdir(exist_ok=True, parents=True)

    # すでにチェックポイントがあり、restoreが指定されていない場合は終了する
    _checkpoint_path = cfg.exp_dir / "checkpoints" / "model_latest.keras"
    if _checkpoint_path.exists() and not cfg.restore:
        logging.error(f"すでにチェックポイントが存在します。{_checkpoint_path}")
        logging.error("checkpointを削除するか、restoreを指定してください")
        exit(1)

    # 設定ファイルを保存
    OmegaConf.save(cfg, cfg.exp_dir / "output.yaml")

    gpu_setting(cfg.gpu, cfg.gpu_allow_growth)
    tensorboard_dir = cfg.exp_dir / "tensorboard_logs"

    # 学習データリストを準備
    """ 下記のような辞書を作成する
    {
       "DataSetA":
            {
                "img_hdr_list": [path1.hdr, path2.hdr, ...]
                "freq": 0.8, # 80%の確率でDataSetAからサンプリング
            },
        "DataSetB": 
            {
                "img_hdr_list": [path3.hdr, path4.hdr, ...]
                "freq": 0.2, # 20%の確率でDataSetAからサンプリング
            },
    }
    """
    train_dict, val_dict = prepare_data_dict(cfg)

    # トレーニングおよび検証用のDataLoaderを作成
    train_loader = create_dataloader(train_dict, is_training=True, cfg=cfg)
    val_loader = create_dataloader(val_dict, is_training=False, cfg=cfg)

    # モデルを作成（アルゴリズムに応じてgeneratorと学習ループを構築する）
    input_shape = tuple(cfg.aug.crop_size_zyx) + (1,)
    model: BaseI2ITrainer = build_trainer(cfg, input_shape)

    # オプティマイザを選択する
    optimizer = select_optimizer(cfg)

    # モデルをコンパイル
    # lossとmetricsはいろいろとカスタマイズしたい場所なので、
    # trainers/以下の各アルゴリズムのクラスで手動設定する。
    model.compile(
        optimizer=optimizer,
        loss=None,
        metrics=None,
        weighted_metrics=None,
        jit_compile=True,  # 実行を JIT コンパイルで高速化
    )

    # TensorBoard コールバックを設定
    # trainとvalを同じログに記録するように変更している。
    # デフォルト仕様がいい場合はkeras.callback.TensorBoardに書き換える。
    profile_batch = (32, 64) if cfg.enable_profiling else 0  # プロファイリングする範囲
    tensorboard_callback = UnifiedTensorBoardLogger(
        log_dir=tensorboard_dir,
        step_per_epoch=cfg.eval_every,  # 1エポックあたりのステップ数
        write_images=True,  # 訓練中の画像をログに保存
        profile_batch=profile_batch,
        write_steps_per_second=True,
    )

    # 検証用データの1バッチ分をTensorBoardに記録するコールバック
    image_logger_callback = ImageLogger(
        val_data=next(iter(val_loader)),
        log_dir=tensorboard_dir,
        jit_compile=True,
        slice_axes=cfg.image.log_axis,
    )

    # 正解なしテスト入力への推論を記録するコールバック（test.enabled時のみ）
    test_logger_callback = None
    if cfg.test.enabled:
        test_data, test_names = create_test_batch(cfg)
        logging.info(f"Test inputs: {test_names}")
        test_logger_callback = TestImageLogger(
            test_data=test_data,
            names=test_names,
            log_dir=tensorboard_dir,
            jit_compile=True,
            slice_axes=cfg.image.log_axis,
            interval_epochs=cfg.test.interval_epochs,
        )

    # ModelCheckpoint コールバックを設定
    best_checkpoint_callback = ModelCheckpoint(
        filepath=str(cfg.exp_dir / "checkpoints" / "model_best.keras"),
        save_best_only=True,  # 最良モデルのみを保存
        monitor="val_psnr",  # モニタリングする指標。モデルで初期化したMetricsの名前にval_をつけたもの
        mode="max",  # 指標を最小化するか最大にするか（min/max）
        save_weights_only=False,  # モデル全体（オプティマイザの状態を含む）を保存
    )
    latest_model_callback = ModelCheckpoint(
        filepath=str(cfg.exp_dir / "checkpoints" / "model_latest.keras"),
        save_best_only=False,
        save_weights_only=False,
    )

    if cfg.restore:
        # 学習途中のモデルを復元する場合
        assert cfg.restore.exists(), f"restore path not found: {cfg.restore}"
        model = keras.models.load_model(cfg.restore)
        model.cfg = cfg
        # 補助optimizer(discriminator用など)は状態が復元されないので再作成する
        attach_aux_optimizers(model, cfg)
        step = model.optimizer.iterations.numpy()
        logging.info(f"Restoring from {cfg.restore}. (step: {step})")
        initial_epoch = step // cfg.eval_every
    elif cfg.finetune:
        # 事前学習済みモデルをファインチューニングする場合
        assert cfg.finetune.exists(), f"finetune path not found: {cfg.finetune}"
        model.load_weights(cfg.finetune)
        logging.info(f"Finetuning from {cfg.finetune}")
        initial_epoch = 0
    else:
        initial_epoch = 0

    # コールバックを組み立てる
    callbacks = [
        tensorboard_callback,
        image_logger_callback,
        best_checkpoint_callback,
        latest_model_callback,
        TerminateOnNaN(),
    ]
    # テストロガーはEMAより前に置く（on_epoch_end時点ではEMA重みが
    # スワップインされたままなので、検証と同じEMA重みで推論される）
    if test_logger_callback is not None:
        callbacks.append(test_logger_callback)
    # EMAコールバックは検証・チェックポイント保存後に重みを戻す必要があるため、
    # 必ずModelCheckpointより後（末尾）に追加する
    if cfg.ema.enabled:
        callbacks.append(EMACallback(decay=cfg.ema.decay))

    # 学習の実行
    model.fit(
        x=train_loader,
        validation_data=val_loader,
        epochs=math.ceil(cfg.num_train_steps / cfg.eval_every),
        steps_per_epoch=cfg.eval_every,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
    )
