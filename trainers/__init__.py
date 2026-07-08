import keras

from models import build_patch_discriminator, build_unet

from .base import BaseI2ITrainer
from .edm import EDMTrainer
from .i2i_rfr import I2IRFRTrainer
from .i2i_rfr_x0 import I2IRFRX0Trainer
from .pix2pix import Pix2PixTrainer
from .rectified_flow import RectifiedFlowTrainer
from .regression import RegressionTrainer

# アルゴリズムを追加したらここに登録する
MODEL_REGISTRY = {
    "regression": RegressionTrainer,
    "pix2pix": Pix2PixTrainer,
    "edm": EDMTrainer,
    "rectified_flow": RectifiedFlowTrainer,
    "i2i_rfr": I2IRFRTrainer,
    "i2i_rfr_x0": I2IRFRX0Trainer,
}


def attach_aux_optimizers(trainer: BaseI2ITrainer, cfg) -> None:
    """
    補助ネットワーク用のoptimizerをセットする。
    復元(restore)時にも呼ぶこと（optimizerの状態自体は復元されない点に注意）。
    """
    if cfg.algorithm.name == "pix2pix":
        cfg_d = cfg.algorithm.pix2pix.d_optimizer
        trainer.d_optimizer = keras.optimizers.Adam(
            learning_rate=cfg_d.lr,
            beta_1=cfg_d.beta_1,
            clipvalue=cfg_d.clip_value,
        )


def build_trainer(cfg, input_shape) -> BaseI2ITrainer:
    """
    アルゴリズムに応じてgenerator（＋補助ネットワーク）を構築し、
    トレーナーを返す。cfgもここでセットする。
    """
    name = cfg.algorithm.name

    # アルゴリズムによってgeneratorの入力チャンネル数が変わる
    # edm: [ノイズ入りtarget, source, ノイズレベル] の3チャンネル
    # rectified_flow / i2i_rfr / i2i_rfr_x0: [x_t, source, 時刻t] の3チャンネル
    in_ch = 3 if name in ("edm", "rectified_flow", "i2i_rfr", "i2i_rfr_x0") else 1
    gen_input_shape = tuple(input_shape[:3]) + (in_ch,)
    generator = build_unet(
        gen_input_shape, cfg.model.num_channel, **cfg.model.unet, **cfg.model.renorm
    )

    if name == "regression":
        trainer = RegressionTrainer(generator)
    elif name == "pix2pix":
        discriminator = build_patch_discriminator(
            input_shape, **cfg.algorithm.pix2pix.discriminator
        )
        trainer = Pix2PixTrainer(generator, discriminator)
    elif name == "edm":
        trainer = EDMTrainer(generator)
    elif name == "rectified_flow":
        trainer = RectifiedFlowTrainer(generator)
    elif name == "i2i_rfr":
        trainer = I2IRFRTrainer(generator)
    elif name == "i2i_rfr_x0":
        trainer = I2IRFRX0Trainer(generator)
    else:
        raise NotImplementedError(
            f"Unknown algorithm: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    trainer.cfg = cfg
    attach_aux_optimizers(trainer, cfg)
    return trainer


__all__ = [
    "BaseI2ITrainer",
    "RegressionTrainer",
    "Pix2PixTrainer",
    "EDMTrainer",
    "RectifiedFlowTrainer",
    "I2IRFRTrainer",
    "I2IRFRX0Trainer",
    "MODEL_REGISTRY",
    "build_trainer",
    "attach_aux_optimizers",
]
