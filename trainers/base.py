import keras
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

    def __init__(self, generator: Model, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.cfg = None  # 学習・推論スクリプト側でセットする（保存されない）

    def call(self, inputs, training=False):
        return self.generator(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config["generator"] = keras.saving.serialize_keras_object(self.generator)
        return config

    @classmethod
    def from_config(cls, config):
        config["generator"] = keras.saving.deserialize_keras_object(
            config["generator"]
        )
        return cls(**config)

    @property
    def metrics_dict(self):
        if not hasattr(self, "_metrics_dict") or len(self._metrics_dict) == 0:
            self._metrics_dict = {name: Mean(name=name) for name in self.METRIC_NAMES}
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
        # ランダムに正規化中心と幅を変えながら正規化する
        src_imgs = random_normalize(
            src_imgs, min_clip_vals, max_clip_vals, **cfg.aug.random_normalize
        )
        # ガンマ補正
        src_imgs = random_gamma_correction(src_imgs, **cfg.aug.random_gamma_correction)
        # sharpness or gaussian filter
        src_imgs = apply_random_sharpness_or_gaussian_filter(
            src_imgs,
            cfg.aug.random_sharpness.prob,
            cfg.aug.random_sharpness.sigma,
            cfg.aug.random_sharpness.alpha_range,
            cfg.aug.random_gauss_filter.prob,
            cfg.aug.random_gauss_filter.sigma_range,
        )

        # gaussian noise
        src_imgs = apply_random_gaussian_noise(src_imgs, **cfg.aug.random_gauss_noise)

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
