from typing import Any

from pytorch_lightning.loggers import LightningLoggerBase


class UncertifyLightningLogger(LightningLoggerBase):
    @property
    def experiment(self) -> Any:
        pass
