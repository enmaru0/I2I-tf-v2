"""DGX Spark向けshift-tolerant CAC trainerのmixed/XLA GPU smoke test。"""

from pathlib import Path

import keras
import tensorflow as tf

from config_utils import load_config_with_extends
from optimizer_utils import get_optimizer_iterations
from trainers import build_trainer


def _batch(shape):
    source = tf.random.stateless_uniform(
        shape, seed=(31, 47), minval=-200.0, maxval=900.0
    )
    target = source + tf.random.stateless_normal(shape, seed=(71, 97), stddev=10.0)
    batch = shape[0]
    minimum = tf.fill((batch,), -1000.0)
    maximum = tf.fill((batch,), 2000.0)
    return {
        "src_imgs": source,
        "tgt_imgs": target,
        "img_msks": tf.ones(shape, tf.float32),
        "src_min_clip_vals": minimum,
        "src_max_clip_vals": maximum,
        "tgt_min_clip_vals": minimum,
        "tgt_max_clip_vals": maximum,
        "sample_seeds": tf.constant([[107, 109]], tf.int32),
    }


def main():
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPUがTensorFlowから見えていません")
    cfg = load_config_with_extends(Path("conf/config_cac_shift_tolerant.yaml"))
    if cfg.algorithm.name != "cac_shift_tolerant_regression":
        raise AssertionError("shift-tolerant CAC configが読み込まれていません")

    keras.mixed_precision.set_global_policy(cfg.mixed_precision_policy)
    cfg.gradient_accumulation_steps = 2
    cfg.model.unet.start_ch = 2
    cfg.model.unet.depth = 1
    cfg.model.unet.num_encode_blocks = 1
    cfg.model.unet.num_decode_blocks = 1
    shape = (1, 4, 16, 16, 1)
    trainer = build_trainer(cfg, shape[1:])
    trainer.compile(optimizer=keras.optimizers.Adam(1e-4), jit_compile=True)
    dataset = tf.data.Dataset.from_tensors(_batch(shape)).repeat()
    trainer.fit(dataset, steps_per_epoch=6, epochs=1, verbose=0)

    updates = int(get_optimizer_iterations(trainer.optimizer).numpy())
    if updates != 3:
        raise AssertionError(f"optimizer update count is invalid: {updates}")
    prediction = trainer.predict_step(_batch(shape))
    if not bool(tf.reduce_all(tf.math.is_finite(prediction))):
        raise AssertionError("prediction contains NaN/Inf")
    metrics = trainer.test_step(_batch(shape))
    if not all(
        bool(tf.reduce_all(tf.math.is_finite(value))) for value in metrics.values()
    ):
        raise AssertionError("validation metrics contain NaN/Inf")
    print(
        "DGX shift-tolerant CAC GPU smoke test: OK",
        f"updates={updates}",
        f"policy={keras.mixed_precision.global_policy().name}",
    )


if __name__ == "__main__":
    main()
