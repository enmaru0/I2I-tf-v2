import keras

from .edm import EDMTrainer


@keras.saving.register_keras_serializable()
class EDMKarrasTrainer(EDMTrainer):
    """既存EDM実装を明示名 ``edm_karras`` で利用するためのtrainer。"""

    def _cfg_edm(self):
        return self.cfg.algorithm.edm_karras
