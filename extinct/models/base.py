from __future__ import annotations
from abc import abstractmethod
from typing import Any

from kit import implements
import pytorch_lightning as pl
from torch import Tensor

from extinct.datamodules import DataBatch, Stage, VisionDataModule

__all__ = ["ModelBase"]


class ModelBase(pl.LightningModule):

    _target: str | None

    @abstractmethod
    def build(self, datamodule: VisionDataModule, trainer: pl.Trainer) -> None:
        ...

    @property
    def target(self) -> str:
        assert self._target is not None
        return self._target

    @target.setter
    def target(self, target: str) -> None:
        self._target = target

    @abstractmethod
    def _inference_step(self, batch: DataBatch, stage: Stage) -> dict[str, Tensor]:
        ...

    @abstractmethod
    def _inference_epoch_end(
        self, output_results: list[dict[str, Tensor]], stage: Stage
    ) -> dict[str, Any]:
        ...

    @implements(pl.LightningModule)
    def validation_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        print(f"17: {self.__class__.__name__=}, {self.on_gpu=}, {self.device=}")
        return self._inference_step(batch=batch, stage="val")

    @implements(pl.LightningModule)
    def validation_epoch_end(self, output_results: list[dict[str, Tensor]]) -> None:
        print(f"18: {self.__class__.__name__=}, {self.on_gpu=}, {self.device=}")
        results_dict = self._inference_epoch_end(output_results=output_results, stage="val")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> dict[str, Tensor]:
        print(f"19: {self.__class__.__name__=}, {self.on_gpu=}, {self.device=}")
        return self._inference_step(batch=batch, stage="test")

    @implements(pl.LightningModule)
    def test_epoch_end(self, output_results: list[dict[str, Tensor]]) -> None:
        print(f"20: {self.__class__.__name__=}, {self.on_gpu=}, {self.device=}")
        results_dict = self._inference_epoch_end(output_results=output_results, stage="test")
        self.log_dict(results_dict)
