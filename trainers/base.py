import keras
import tensorflow as tf
from keras import Model
from keras.api.metrics import Mean
from keras.src import ops

from data.gpu_aug import (
    apply_random_gaussian_noise,
    apply_random_sharpness_or_gaussian_filter,
    normalize,
    random_gamma_correction,
    random_normalize,
)
from optimizer_utils import get_optimizer_iterations


@keras.saving.register_keras_serializable()
class BaseI2ITrainer(Model):
    """
    image-to-image translation用の学習ループの基底クラス。
    生成ネットワーク(generator)をコンポジションで保持する。
    https://keras.io/guides/custom_train_step_in_tensorflow/
    上記リンクを参考に作成した。GANなど少し複雑なモデルの学習方法も書いてあります。

    アルゴリズムを追加する場合はこのクラスを継承し、
    - METRIC_NAMES
    - train_step
    - _compute_loss_and_metrics
    を実装して trainers/__init__.py の MODEL_REGISTRY と build_trainer に登録する。
    補助ネットワーク(discriminatorなど)は__init__で受け取り、
    get_config/from_configにも追加すること。
    """

    # サブクラスで上書きする
    METRIC_NAMES = ["total_loss", "psnr", "ssim"]

    def __init__(
        self, generator: Model, gradient_accumulation_steps: int = 1, **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.cfg = None  # 学習・推論スクリプト側でセットする（保存されない）
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_stepsは1以上を指定してください")
        self._gradient_accumulators = {}
        self._gradient_accumulation_counters = {}
        self._create_gradient_accumulators(
            "generator", self.generator.trainable_variables
        )

    def call(self, inputs, training=False):
        return self.generator(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config["generator"] = keras.saving.serialize_keras_object(self.generator)
        config["gradient_accumulation_steps"] = self.gradient_accumulation_steps
        return config

    @classmethod
    def from_config(cls, config):
        config["generator"] = keras.saving.deserialize_keras_object(config["generator"])
        return cls(**config)

    @property
    def metrics_dict(self):
        if not hasattr(self, "_metrics_dict") or len(self._metrics_dict) == 0:
            names = list(self.METRIC_NAMES)
            if self._real_dc_enabled():
                names.append("real_dc_loss")
            self._metrics_dict = {name: Mean(name=name) for name in names}
        return self._metrics_dict

    def _prepare_batch(self, data, is_training: bool):
        """
        バッチ辞書からsource/target/有効領域マスクを取り出して正規化する。
        sourceのみ学習時は強度系のデータ拡張を適用する。
        targetは常に決定的に正規化する（教師信号を揺らさないため）。
        """
        img_msks = data["img_msks"]

        if is_training:
            src_imgs = self.gpu_aug(
                data["src_imgs"],
                img_msks,
                data["src_min_clip_vals"],
                data["src_max_clip_vals"],
                self.cfg,
            )
        else:
            src_imgs = normalize(
                data["src_imgs"], data["src_min_clip_vals"], data["src_max_clip_vals"]
            )
            src_imgs = src_imgs * img_msks

        if "tgt_imgs" in data:
            tgt_imgs = normalize(
                data["tgt_imgs"], data["tgt_min_clip_vals"], data["tgt_max_clip_vals"]
            )
            tgt_imgs = tgt_imgs * img_msks
        else:
            # 推論時(predict.py)はtargetが存在しない場合がある
            tgt_imgs = None

        return src_imgs, tgt_imgs, img_msks

    @staticmethod
    def _to_x(img01, img_msks):
        """[0,1]の画像を[-1,1]の作業空間へ（パディングは0になるようマスク）
        EDMやrectified flowなどの生成系アルゴリズムで使う"""
        return (img01 * 2.0 - 1.0) * img_msks

    @staticmethod
    def _to_01(x, img_msks):
        """作業空間[-1,1]から[0,1]の画像へ"""
        return ops.clip((x + 1.0) / 2.0, 0.0, 1.0) * img_msks

    def _validation_sample_seeds(self, data):
        cfg_repro = self.cfg.get("reproducibility", {})
        if not bool(cfg_repro.get("fixed_validation_noise", True)):
            return None
        return data.get("sample_seeds")

    @staticmethod
    def _normal_like(reference, sample_seeds=None, salt: int = 0):
        """症例別seedがあればstateless、なければ通常の正規乱数を返す。"""
        if sample_seeds is None:
            return tf.random.normal(tf.shape(reference), dtype=reference.dtype)

        sample_seeds = tf.cast(sample_seeds, tf.int32)
        sample_shape = tf.shape(reference)[1:]

        def _sample(seed):
            seed = tf.random.experimental.stateless_fold_in(seed, salt)
            return tf.random.stateless_normal(
                sample_shape, seed=seed, dtype=reference.dtype
            )

        return tf.map_fn(
            _sample,
            sample_seeds,
            fn_output_signature=tf.TensorSpec(
                shape=reference.shape[1:], dtype=reference.dtype
            ),
        )

    @staticmethod
    def _scale_loss_for_optimizer(loss, optimizer):
        if hasattr(optimizer, "scale_loss"):
            return optimizer.scale_loss(loss)
        return loss

    def _create_gradient_accumulators(self, key, variables):
        """optimizerごとに勾配バッファとmicro-step counterを作る。"""
        if self.gradient_accumulation_steps == 1 or key in self._gradient_accumulators:
            return
        variables = list(variables)

        def _device(variable):
            # 通常の単一GPU学習ではKerasのnon-trainable weightがCPUへ
            # pinされる場合がある。XLA GPU graphはCPU resourceを参照できないため、
            # GPUがあればaccumulation stateもGPUへ明示配置する。
            logical_gpus = tf.config.list_logical_devices("GPU")
            if logical_gpus:
                return logical_gpus[0].name
            value = getattr(variable, "value", variable)
            return getattr(value, "device", None)

        accumulators = []
        for i, variable in enumerate(variables):
            # XLAでは別device上のresource variableを参照できないため、
            # accumulatorを対応する学習weightと同じdeviceへ明示配置する。
            with tf.device(_device(variable)):
                accumulators.append(
                    self.add_weight(
                        name=f"{key}_gradient_accumulator_{i}",
                        shape=variable.shape,
                        dtype=variable.dtype,
                        initializer="zeros",
                        trainable=False,
                    )
                )
        self._gradient_accumulators[key] = accumulators
        with tf.device(_device(variables[0])):
            self._gradient_accumulation_counters[key] = self.add_weight(
                name=f"{key}_gradient_accumulation_counter",
                shape=(),
                # TensorFlowはint32 resource variableをGPU指定してもCPUへ置く。
                # XLA GPUから参照できるよう、整数値をfloat32で保持する。
                dtype="float32",
                initializer="zeros",
                trainable=False,
            )

    def _apply_gradients(self, optimizer, gradients, variables, key="generator"):
        """
        micro batchの勾配を平均し、設定回数ごとにoptimizerを1回更新する。

        LossScaleOptimizerの場合もscale済み勾配を同じloss scaleのまま平均し、
        apply_gradients側で通常どおりunscaleさせる。
        """
        variables = list(variables)
        gradients = list(gradients)
        if self.gradient_accumulation_steps == 1:
            optimizer.apply_gradients(
                (gradient, variable)
                for gradient, variable in zip(gradients, variables)
                if gradient is not None
            )
            return

        self._create_gradient_accumulators(key, variables)
        accumulators = self._gradient_accumulators[key]
        counter = self._gradient_accumulation_counters[key]
        for accumulator, gradient in zip(accumulators, gradients):
            if gradient is not None:
                accumulator.assign_add(ops.cast(gradient, accumulator.dtype))
        counter.assign_add(1)

        def _apply_and_reset():
            scale = ops.cast(self.gradient_accumulation_steps, accumulators[0].dtype)
            averaged = [accumulator / scale for accumulator in accumulators]
            optimizer.apply_gradients(zip(averaged, variables))
            for accumulator in accumulators:
                accumulator.assign(ops.zeros_like(accumulator))
            counter.assign(0)
            return tf.constant(0)

        tf.cond(
            ops.equal(counter, ops.cast(self.gradient_accumulation_steps, "float32")),
            _apply_and_reset,
            lambda: tf.constant(0),
        )

    def _to_image(self, logits, src_imgs):
        """logitsを[0,1]レンジの画像に変換する（クリップ前）"""
        output_mode = self.cfg.algorithm[self.cfg.algorithm.name].output_mode
        if output_mode == "residual":
            # 残差学習：デノイズ・ぼかし修正のようにsourceとtargetが近いタスク向け
            return src_imgs + logits
        elif output_mode == "direct":
            return ops.sigmoid(logits)
        else:
            raise NotImplementedError(output_mode)

    # ------------------------------------------------------------------
    # 実劣化データのdata consistency損失（data.real_dc、self_sr向け半教師あり）
    # 正解のない実劣化入力に対する予測を微分可能な劣化演算子で再劣化し、
    # 入力自身と比較する。sim2realギャップの縮小に使う。
    # ------------------------------------------------------------------

    def _real_dc_enabled(self) -> bool:
        cfg_dc = self.cfg.data.get("real_dc", None) if self.cfg is not None else None
        return bool(cfg_dc is not None and cfg_dc.enabled)

    def _dc_predict01(self, src01, img_msks):
        """
        DC損失用の安価な予測（1回のネットワーク評価、[0,1]レンジ）。
        直接予測系(regression/pix2pix)はoutput_mode経由のデフォルト実装を使い、
        生成系はサブクラスでオーバーライドする。
        """
        logits = self([src01, img_msks], training=True)
        return ops.clip(self._to_image(logits, src01), 0.0, 1.0) * img_msks

    @staticmethod
    def _axis_gaussian_blur(vol, sigma_px, axis_dim: int):
        """
        指定軸に沿ったサンプル別σのガウシアンぼかし（微分可能・静的shape）。
        ぼかしを(A,A)の帯行列として構築しeinsumで適用する。
        vol: (b, z, y, x, c) / sigma_px: (b,) / axis_dim: 0(z),1(y),2(x)
        """
        length = int(vol.shape[axis_dim + 1])
        idx = ops.arange(0, length, dtype="float32")
        diff = idx[None, :, None] - idx[None, None, :]  # (1, A, A)
        sigma = ops.reshape(ops.maximum(sigma_px, 1e-3), (-1, 1, 1))
        weight = ops.exp(-0.5 * ops.square(diff / sigma))
        weight = weight / (ops.sum(weight, axis=2, keepdims=True) + 1e-8)
        einsum_expr = {
            0: "bij,bjyxc->biyxc",
            1: "bij,bzjxc->bzixc",
            2: "bij,bzyjc->bzyic",
        }[axis_dim]
        return ops.einsum(einsum_expr, weight, vol)

    def _real_dc_loss(self, data):
        """
        実劣化入力へのdata consistency損失を返す: (重み付き損失, 生の損失)。
        劣化演算子は「スライスプロファイル+間引き補間の実効平滑化」を
        軸方向ガウシアン(σはロード時にサンプル別計算)で近似したもの。
        train_stepのGradientTape内で呼ぶこと。
        """
        from data.dataloader import SR_AXIS_TO_DIM

        cfg_dc = self.cfg.data.real_dc
        msk = data["real_msks"]
        real01 = (
            normalize(
                data["real_imgs"],
                data["real_min_clip_vals"],
                data["real_max_clip_vals"],
            )
            * msk
        )
        pred01 = self._dc_predict01(real01, msk)

        axis_dim = SR_AXIS_TO_DIM[str(cfg_dc.axis).upper()]
        degraded = self._axis_gaussian_blur(pred01, data["real_sigma_px"], axis_dim)
        diff = (degraded - real01) * msk
        if cfg_dc.loss == "l1":
            raw = ops.sum(ops.abs(diff)) / (ops.sum(msk) + 1e-6)
        elif cfg_dc.loss == "mse":
            raw = ops.sum(ops.square(diff)) / (ops.sum(msk) + 1e-6)
        else:
            raise NotImplementedError(cfg_dc.loss)

        # warmup: 主損失が形になる前にDCが支配しないよう線形に立ち上げる
        # （step+1で初回ステップから重みが0にならないようにする）
        step = ops.cast(get_optimizer_iterations(self.optimizer), "float32") + 1.0
        warmup = float(max(int(cfg_dc.warmup_steps), 1))
        weight = float(cfg_dc.weight) * ops.minimum(1.0, step / warmup)
        return weight * raw, raw

    def _add_real_dc_loss(self, loss, data):
        """train_stepのtape内から呼ぶ。DC有効時に損失へ加算しメトリクスを更新する"""
        if not self._real_dc_enabled():
            return loss
        weighted, raw = self._real_dc_loss(data)
        self.metrics_dict["real_dc_loss"].update_state(raw)
        return loss + weighted

    def train_step(self, data):
        """
        ここのデータ名であったりselfに渡す引数を変えた場合は、
        callbacks/image_logger.pyのpredict_stepやon_test_batch_endも変更すること
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        """
        raise NotImplementedError

    def test_step(self, data):
        """
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        ./callbacks/image_logger.pyを参考にコールバックを実装する
        """
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=False)
        logits = self([src_imgs, img_msks], training=False)
        _ = self._compute_loss_and_metrics(logits, src_imgs, tgt_imgs, img_msks)
        return self._get_metrics_result()

    def _compute_loss_and_metrics(self, logits, src_imgs, tgt_imgs, img_msks):
        raise NotImplementedError

    def predict_step(self, data, return_aux=False):
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=False)
        logits = self([src_imgs, img_msks], training=False)
        preds = ops.clip(self._to_image(logits, src_imgs), 0, 1) * img_msks

        if return_aux:
            # callbacks/image_logger.pyで必要とするものも返す
            return logits, preds, src_imgs, tgt_imgs, img_msks
        else:
            return preds

    @staticmethod
    def gpu_aug(src_imgs, img_msks, min_clip_vals, max_clip_vals, cfg):
        cfg_aug = cfg.aug

        # 無効な拡張はグラフへ入れない。特に3D Gaussian filterはコストが大きい。
        if float(cfg_aug.random_normalize.prob) > 0.0:
            src_imgs = random_normalize(
                src_imgs, min_clip_vals, max_clip_vals, **cfg_aug.random_normalize
            )
        else:
            src_imgs = normalize(src_imgs, min_clip_vals, max_clip_vals)

        if float(cfg_aug.random_gamma_correction.prob) > 0.0:
            src_imgs = random_gamma_correction(
                src_imgs, **cfg_aug.random_gamma_correction
            )

        if (
            float(cfg_aug.random_sharpness.prob) > 0.0
            or float(cfg_aug.random_gauss_filter.prob) > 0.0
        ):
            src_imgs = apply_random_sharpness_or_gaussian_filter(
                src_imgs,
                cfg_aug.random_sharpness.prob,
                cfg_aug.random_sharpness.sigma,
                cfg_aug.random_sharpness.alpha_range,
                cfg_aug.random_gauss_filter.prob,
                cfg_aug.random_gauss_filter.sigma_range,
            )

        if float(cfg_aug.random_gauss_noise.prob) > 0.0:
            src_imgs = apply_random_gaussian_noise(
                src_imgs, **cfg_aug.random_gauss_noise
            )

        src_imgs = ops.clip(src_imgs, 0, 1)
        src_imgs = src_imgs * img_msks
        return src_imgs

    def _get_metrics_result(self):
        """
        Return the results of all metrics as a dictionary.
        """
        return {metric.name: metric.result() for metric in self.metrics_dict.values()}

    @property
    def metrics(self):
        """
        We list our `Metric` objects here so that `reset_states()` can be
        called automatically at the start of each epoch
        or at the start of `evaluate()`.
        """
        return self.metrics_dict.values()
