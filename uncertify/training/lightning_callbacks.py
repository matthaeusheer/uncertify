from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class SaveHyperParamsCallback(Callback):
    def __init__(self, hyper_params: dict) -> None:
        self._hyper_params = hyper_params

    def on_fit_start(self, trainer: Trainer):
        print('trainer is init now')
        print(trainer.logger.log_dir)
        print(self._hyper_params)
