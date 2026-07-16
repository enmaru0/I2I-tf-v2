import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from absl import logging
from keras.api.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.api.optimizers import SGD, AdamW
from keras.api.optimizers.schedules import CosineDecay
from omegaconf import ListConfig, OmegaConf

from callbacks import (
    ConciseProgbarLogger,
    EMACallback,
    ImageLogger,
    TestImageLogger,
    UnifiedTensorBoardLogger,
    WallTimeLimit,
)
from config_utils import load_config_with_extends
from data.dataloader import (
    create_dataloader,
    create_test_batch,
    resolve_target_hdr_path,
)
from optimizer_utils import get_optimizer_iterations
from trainers import (
    MODEL_REGISTRY,
    BaseI2ITrainer,
    attach_aux_optimizers,
    build_trainer,
)


def read_cfg_and_parse_arg():
    # コマンドライン引数と設定ファイルを読み込む関数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/config.yaml"),
        help="設定ファイル。extendsを指定した差分configも利用できます。",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="設定を上書きするフォーマット (例: 'batch_size=12 aug.crop_size_zyx=[64,64,64]')",
    )
    args = parser.parse_args()
    cmd_overrides = args.overrides

    cfg = load_config_with_extends(args.config)

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
    elif target_scale_zyx[0] > 2 and bool(
        cfg.aug.get("require_pre_rescaled_dir", True)
    ):
        raise ValueError(
            "Thickスライスで学習する場合は./utils/rescale_dataset.pyで予めリスケールすることを推奨します"
        )

    # その他cfgのチェック
    assert cfg.image.modality in ["CT", "MR"], cfg.image.modality
    assert cfg.mixed_precision_policy in (
        "float32",
        "mixed_float16",
        "mixed_bfloat16",
    ), cfg.mixed_precision_policy
    # U-Netのmodeに応じてクロップzサイズの割り切れ制約をチェックする
    unet_mode = cfg.model.unet.get("mode", "3d")
    assert unet_mode in ["3d", "mixed", "2d"], unet_mode
    crop_z = int(cfg.aug.crop_size_zyx[0])
    z_div = {"3d": 2**cfg.model.unet.depth, "mixed": 2, "2d": 1}[unet_mode]
    assert crop_z % z_div == 0, (
        f"model.unet.mode={unet_mode}ではcrop_size_zyxのzは{z_div}の倍数にしてください"
        f"（現在: {crop_z}）"
    )
    xy_div = 2**cfg.model.unet.depth
    for axis, crop_size in zip(("y", "x"), cfg.aug.crop_size_zyx[1:]):
        assert int(crop_size) % xy_div == 0, (
            f"crop_size_zyxの{axis}は{xy_div}の倍数にしてください"
        )
    assert 0.0 <= float(cfg.inference.overlap) < 1.0, (
        "inference.overlapは0以上1未満で指定してください"
    )
    assert int(cfg.inference.batch_size) >= 1, (
        "inference.batch_sizeは1以上を指定してください"
    )
    assert int(cfg.num_train_steps) >= 1, "num_train_stepsは1以上を指定してください"
    assert int(cfg.eval_every) >= 1, "eval_everyは1以上を指定してください"
    assert cfg.data.mode in [
        "paired",
        "paired_dir",
        "denoise",
        "self_noise",
        "self_sr",
    ], cfg.data.mode
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))
    assert gradient_accumulation_steps >= 1, (
        "gradient_accumulation_stepsは1以上を指定してください"
    )
    img_interp_order = int(cfg.aug.image_interpolation_order)
    assert 0 <= img_interp_order <= 5, (
        "aug.image_interpolation_orderは0〜5で指定してください"
        f"（現在: {cfg.aug.image_interpolation_order}）"
    )
    crop_exclude_start = float(cfg.aug.crop_exclude_start_slices)
    assert crop_exclude_start.is_integer() and crop_exclude_start >= 0, (
        "aug.crop_exclude_start_slicesは0以上の整数で指定してください"
        f"（現在: {cfg.aug.crop_exclude_start_slices}）"
    )
    assert int(cfg.evaluation.val_patches_per_volume) >= 1, (
        "evaluation.val_patches_per_volumeは1以上を指定してください"
    )
    for axis_name in cfg.evaluation.ssim_axes:
        assert str(axis_name) in ("z", "y", "x"), (
            f"evaluation.ssim_axesはz/y/xで指定してください: {axis_name}"
        )
    if cfg.data.mode in ("denoise", "self_noise"):
        simulation = str(cfg.data.self_noise.get("simulation", "gaussian")).lower()
        assert simulation in ("gaussian", "rician", "kspace", "lowfield"), (
            "data.self_noise.simulationはgaussian/rician/kspace/lowfieldで"
            f"指定してください（現在: {simulation}）"
        )
        if simulation != "gaussian":
            assert cfg.image.modality == "MR", (
                f"{simulation} simulationはMRI magnitude画像向けです。"
                "image.modality=MRを指定してください"
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
        assert int(cfg.data.self_sr.acquisition_order) in (0, 1), (
            "data.self_sr.acquisition_orderは0または1を指定してください"
        )
        assert int(cfg.data.self_sr.val_slice_phase_px) >= 0, (
            "data.self_sr.val_slice_phase_pxは0以上を指定してください"
        )
        for protocol in cfg.data.self_sr.protocols:
            assert str(protocol.axis).upper() in ("AX", "COR", "SAG"), protocol
            assert float(protocol.slice_interval_mm) > 0.0, protocol
            assert float(protocol.get("slice_thickness_mm", 1.0)) > 0.0, protocol
            assert float(protocol.get("weight", 1.0)) > 0.0, protocol
    if cfg.data.mode == "paired_dir":
        assert cfg.data.target_data_dir, (
            "paired_dirモードではdata.target_data_dirを指定してください"
        )
        assert Path(cfg.data.target_data_dir).exists(), (
            f"target_data_dirが存在しません: {cfg.data.target_data_dir}"
        )
        pair_matching = cfg.data.get("pair_matching", {})
        matching_method = str(pair_matching.get("method", "exact")).lower()
        assert matching_method in ("exact", "stem_token"), (
            "data.pair_matching.methodはexact/stem_tokenで指定してください"
        )
        if matching_method == "stem_token":
            assert str(pair_matching.get("delimiter", "_")), (
                "stem_token matchingではdelimiterを空にできません"
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
    if cfg.algorithm.name in ("edm", "edm_karras", "conditional_restoration_ode"):
        cfg_algorithm = cfg.algorithm[cfg.algorithm.name]
        assert int(cfg_algorithm.num_steps) >= 1, (
            f"algorithm.{cfg.algorithm.name}.num_stepsは1以上を指定してください"
        )
        assert int(cfg_algorithm.num_steps_val) >= 1, (
            f"algorithm.{cfg.algorithm.name}.num_steps_valは1以上を指定してください"
        )
        assert 0.0 < float(cfg_algorithm.sigma_min) < float(cfg_algorithm.sigma_max), (
            f"algorithm.{cfg.algorithm.name}は0 < sigma_min < sigma_maxが必要です"
        )
        assert str(cfg_algorithm.get("solver", "heun")) in ("euler", "heun"), (
            f"algorithm.{cfg.algorithm.name}.solverはeuler/heunで指定してください"
        )
        if cfg.algorithm.name == "conditional_restoration_ode":
            assert float(cfg_algorithm.P_std) > 0.0, (
                "conditional_restoration_ode.P_stdは正の値を指定してください"
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
    - paired_dir: source/targetを別フォルダに置き、設定したfilename規則で対応
    - denoise / self_noise / self_sr: クリーン画像のみ（targetファイルは不要。劣化はローダ内で合成）
    targetやマスク類のrawファイルは除外する。
    """
    data_dir = Path(cfg.data_dir)
    target_suffix = cfg.data.target_suffix
    no_target = cfg.data.mode in ("denoise", "self_noise", "self_sr")

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
        max(1, cfg.num_train_steps - warmup_steps),
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
    keras.utils.set_random_seed(int(cfg.reproducibility.seed))
    if cfg.reproducibility.deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    keras.mixed_precision.set_global_policy(cfg.mixed_precision_policy)
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
    # gradient accumulationのtf.cond内でslot variableが初回作成されないよう、
    # jit_compile前にoptimizerを明示的にbuildする。
    if hasattr(optimizer, "build"):
        optimizer.build(model.generator.trainable_variables)

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
        step_per_epoch=cfg.eval_every,  # 1エポックあたりのoptimizer更新数
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
        restored_accumulation_steps = int(
            getattr(model, "gradient_accumulation_steps", 1)
        )
        requested_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))
        assert restored_accumulation_steps == requested_accumulation_steps, (
            "restore時はgradient_accumulation_stepsを変更できません: "
            f"checkpoint={restored_accumulation_steps}, "
            f"config={requested_accumulation_steps}"
        )
        # 補助optimizer(discriminator用など)は状態が復元されないので再作成する
        attach_aux_optimizers(model, cfg)
        optimizer_step = int(get_optimizer_iterations(model.optimizer).numpy())
        processed_micro_steps = optimizer_step * restored_accumulation_steps
        logging.info(
            f"Restoring from {cfg.restore}. "
            f"(optimizer step: {optimizer_step}, "
            f"processed micro steps: {processed_micro_steps})"
        )
        initial_epoch = optimizer_step // cfg.eval_every
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
        ConciseProgbarLogger(),
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
    if float(cfg.max_train_minutes) > 0:
        callbacks.append(WallTimeLimit(cfg.max_train_minutes))
    # EMAコールバックは検証・チェックポイント保存後に重みを戻す必要があるため、
    # 必ずModelCheckpointより後（末尾）に追加する
    if cfg.ema.enabled:
        callbacks.append(EMACallback(decay=cfg.ema.decay))

    # 学習の実行
    model.fit(
        x=train_loader,
        validation_data=val_loader,
        epochs=math.ceil(cfg.num_train_steps / cfg.eval_every),
        steps_per_epoch=(
            cfg.eval_every * int(cfg.get("gradient_accumulation_steps", 1))
        ),
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        verbose=0,
    )
