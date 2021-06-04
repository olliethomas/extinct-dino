# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DinoModelConf:
    _target_: str = "extinct.models.DinoModel"
    lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class KCBaselineConf:
    _target_: str = "extinct.models.KCBaseline"
    lr: float = MISSING
    weight_decay: float = MISSING
    batch_norm: bool = MISSING
    lr_gamma: float = MISSING


@dataclass
class ErmBaselineConf:
    _target_: str = "extinct.models.ErmBaseline"
    lr: float = MISSING
    weight_decay: float = MISSING
    batch_norm: bool = MISSING
    lr_gamma: float = MISSING


@dataclass
class LaftrBaselineConf:
    _target_: str = "extinct.models.LaftrBaseline"
    lr: float = MISSING
    weight_decay: float = MISSING
    lr_gamma: float = MISSING
    disc_steps: int = MISSING
