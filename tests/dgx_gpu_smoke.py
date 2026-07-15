"""DGX Spark向け: 新trainer/gradient accumulation/checkpointのGPU smoke test。"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import tensorflow as tf

from optimizer_utils import get_optimizer_iterations
from trainers import build_trainer


class AttrDict(dict):
    """OmegaConfに依存しない最小attribute-access dict。"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class ConfigNamespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _config(algorithm_name):
    algorithm = AttrDict(
        name=algorithm_name,
        conditional_restoration_ode=AttrDict(
            P_mean=-0.5,
            P_std=1.0,
            sigma_min=0.01,
            sigma_max=2.0,
            degradation_eps=0.01,
            num_steps=2,
            num_steps_val=2,
            solver="heun",
        ),
        edm_karras=AttrDict(
            sigma_data=0.5,
            P_mean=-1.2,
            P_std=1.2,
            sigma_min=0.01,
            sigma_max=2.0,
            rho=3.0,
            num_steps=2,
            num_steps_val=2,
            solver="heun",
        ),
        pix2pix=AttrDict(
            output_mode="residual",
            gan_mode="lsgan",
            l1_weight=10.0,
            gan_weight=1.0,
            discriminator=AttrDict(base_ch=2, num_layers=1),
            d_optimizer=AttrDict(lr=1e-4, beta_1=0.5, clip_value=10.0),
        ),
    )
    return ConfigNamespace(
        algorithm=algorithm,
        gradient_accumulation_steps=2,
        model=SimpleNamespace(
            num_channel=1,
            unet=AttrDict(
                start_ch=2,
                depth=1,
                num_encode_blocks=1,
                num_decode_blocks=1,
                mode="3d",
            ),
            renorm=AttrDict(
                r_max=1.0,
                d_max=0.0,
                warmup_steps=1,
                change_d_steps=2,
                change_r_steps=2,
            ),
        ),
        aug=SimpleNamespace(
            random_normalize=AttrDict(
                prob=0.0,
                random_center_deviation=0.0,
                random_range_deviation=0.0,
            ),
            random_gamma_correction=AttrDict(
                prob=0.0, random_gamma_range=[1.0, 1.0]
            ),
            random_sharpness=AttrDict(prob=0.0, sigma=1.0, alpha_range=[0.0, 0.0]),
            random_gauss_filter=AttrDict(prob=0.0, sigma_range=[0.0, 0.0]),
            random_gauss_noise=AttrDict(prob=0.0, stddev_range=[0.0, 0.0]),
        ),
        data=AttrDict(real_dc=AttrDict(enabled=False)),
        reproducibility=AttrDict(fixed_validation_noise=True),
    )


def _batch():
    shape = (1, 12, 12, 12, 1)
    src = tf.random.stateless_uniform(shape, seed=(1, 2), minval=0.1, maxval=0.9)
    target = tf.clip_by_value(src * 0.9 + 0.05, 0.0, 1.0)
    ones = tf.ones(shape, tf.float32)
    zeros = tf.zeros((1,), tf.float32)
    scalar_ones = tf.ones((1,), tf.float32)
    return {
        "src_imgs": src,
        "tgt_imgs": target,
        "img_msks": ones,
        "src_min_clip_vals": zeros,
        "src_max_clip_vals": scalar_ones,
        "tgt_min_clip_vals": zeros,
        "tgt_max_clip_vals": scalar_ones,
        "sample_seeds": tf.constant([[7, 11]], tf.int32),
    }


def _fit_micro_steps(algorithm_name, micro_steps=2, exact_updates=1):
    cfg = _config(algorithm_name)
    trainer = build_trainer(cfg, (12, 12, 12, 1))
    optimizer = keras.optimizers.Adam(1e-4)
    if hasattr(optimizer, "build"):
        optimizer.build(trainer.generator.trainable_variables)
    trainer.compile(optimizer=optimizer, jit_compile=True)
    dataset = tf.data.Dataset.from_tensors(_batch()).repeat()
    trainer.fit(dataset, epochs=1, steps_per_epoch=micro_steps, verbose=0)
    generator_updates = int(get_optimizer_iterations(trainer.optimizer).numpy())
    if exact_updates is not None and generator_updates != exact_updates:
        raise AssertionError(
            "generator gradient accumulation update mismatch: "
            f"expected={exact_updates}, actual={generator_updates}"
        )
    if algorithm_name == "pix2pix":
        if int(get_optimizer_iterations(trainer.d_optimizer).numpy()) != 1:
            raise AssertionError("discriminator accumulation did not update once")
    else:
        prediction = trainer.predict_step(_batch())
        if not bool(tf.reduce_all(tf.math.is_finite(prediction))):
            raise AssertionError(f"{algorithm_name} prediction contains NaN/Inf")
    return trainer, cfg


def main():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPUがTensorFlowから見えていません")
    print("GPU:", gpus)

    conditional, cfg = _fit_micro_steps("conditional_restoration_ode")
    _fit_micro_steps("edm_karras")
    _fit_micro_steps("pix2pix")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "conditional.keras"
        conditional.save(checkpoint)
        restored = keras.models.load_model(checkpoint, safe_mode=False)
        restored.cfg = cfg
        if restored.gradient_accumulation_steps != 2:
            raise AssertionError("accumulation setting was not restored")
        prediction = restored.predict_step(_batch())
        np.testing.assert_allclose(
            prediction.numpy(), conditional.predict_step(_batch()).numpy(), atol=1e-5
        )

    keras.mixed_precision.set_global_policy("mixed_float16")
    # Dynamic loss scalingでは非有限勾配を検出した更新を意図的にskipする。
    # 複数回のscale低下後に、accumulation経由で実更新できることを確認する。
    mixed, _ = _fit_micro_steps(
        "conditional_restoration_ode", micro_steps=16, exact_updates=None
    )
    if not hasattr(mixed.optimizer, "scale_loss"):
        raise AssertionError("mixed_float16 optimizer is not loss-scaled")
    mixed_updates = int(get_optimizer_iterations(mixed.optimizer).numpy())
    mixed_loss_scale = float(mixed.optimizer.dynamic_scale.numpy())
    if mixed_updates < 1 or mixed_updates > 8:
        raise AssertionError(
            "mixed precision update count is invalid: "
            f"updates={mixed_updates}, loss_scale={mixed_loss_scale}"
        )
    if not all(
        bool(tf.reduce_all(tf.math.is_finite(variable)))
        for variable in mixed.generator.trainable_variables
    ):
        raise AssertionError("mixed precision training produced non-finite weights")
    print(
        "mixed_float16:",
        f"updates={mixed_updates}",
        f"loss_scale={mixed_loss_scale}",
    )
    keras.mixed_precision.set_global_policy("float32")

    print("DGX GPU smoke test: OK")


if __name__ == "__main__":
    main()
