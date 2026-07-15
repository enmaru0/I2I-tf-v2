"""DGX Spark向けCAC専用config/mixed U-Net/accumulationのGPU smoke test。"""

from pathlib import Path

import keras
import tensorflow as tf

from config_utils import load_config_with_extends
from optimizer_utils import get_optimizer_iterations
from trainers import build_trainer


def _batch(shape):
    src = tf.random.stateless_uniform(shape, seed=(21, 34), minval=-200.0, maxval=900.0)
    target = src + tf.random.stateless_normal(shape, seed=(55, 89), stddev=10.0)
    mask = tf.ones(shape, tf.float32)
    batch = shape[0]
    minimum = tf.fill((batch,), -1000.0)
    maximum = tf.fill((batch,), 2000.0)
    return {
        "src_imgs": src,
        "tgt_imgs": target,
        "img_msks": mask,
        "src_min_clip_vals": minimum,
        "src_max_clip_vals": maximum,
        "tgt_min_clip_vals": minimum,
        "tgt_max_clip_vals": maximum,
        "sample_seeds": tf.constant([[101, 103]], tf.int32),
    }


def main():
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPUがTensorFlowから見えていません")
    cfg = load_config_with_extends(Path("conf/config_cac.yaml"))
    if tuple(cfg.aug.affine.norm_spacing_zyx) != (3.0, 0.5, 0.5):
        raise AssertionError("CAC native spacing configが読み込まれていません")
    if cfg.model.unet.mode != "mixed":
        raise AssertionError("CAC U-Netはmixed modeである必要があります")

    keras.mixed_precision.set_global_policy(cfg.mixed_precision_policy)
    cfg.gradient_accumulation_steps = 2
    cfg.model.unet.start_ch = 2
    cfg.model.unet.depth = 1
    cfg.model.unet.num_encode_blocks = 1
    cfg.model.unet.num_decode_blocks = 1
    shape = (1, 4, 16, 16, 1)
    trainer = build_trainer(cfg, shape[1:])
    trainer.cfg = cfg
    trainer.compile(optimizer=keras.optimizers.Adam(1e-4), jit_compile=True)
    dataset = tf.data.Dataset.from_tensors(_batch(shape)).repeat()
    trainer.fit(dataset, steps_per_epoch=12, epochs=1, verbose=0)

    updates = int(get_optimizer_iterations(trainer.optimizer).numpy())
    if updates < 1 or updates > 6:
        raise AssertionError(f"CAC optimizer update count is invalid: {updates}")
    prediction = trainer.predict_step(_batch(shape))
    if not bool(tf.reduce_all(tf.math.is_finite(prediction))):
        raise AssertionError("CAC prediction contains NaN/Inf")
    if not all(
        bool(tf.reduce_all(tf.math.is_finite(variable)))
        for variable in trainer.generator.trainable_variables
    ):
        raise AssertionError("CAC training produced non-finite weights")
    print(
        "DGX CAC GPU smoke test: OK",
        f"updates={updates}",
        f"policy={keras.mixed_precision.global_policy().name}",
    )


if __name__ == "__main__":
    main()
