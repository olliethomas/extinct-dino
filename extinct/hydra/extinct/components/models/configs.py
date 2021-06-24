# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from extinct.components.models.laftr_baseline import FairnessType
from omegaconf import MISSING


@dataclass
class CelebaKCBaselineConf:
    _target_: str = "extinct.components.models.CelebaKCBaseline"
    lr: float = MISSING
    weight_decay: float = MISSING
    lr_gamma: float = MISSING
    batch_norm: bool = True


@dataclass
class LaftrBaselineConf:
    _target_: str = "extinct.components.models.LaftrBaseline"
    lr: float = MISSING
    weight_decay: float = MISSING
    lr_gamma: float = MISSING
    disc_steps: int = MISSING
    fairness: FairnessType = MISSING
    recon_weight: float = MISSING
    clf_weight: float = MISSING
    adv_weight: float = MISSING


@dataclass
class CelebaErmBaselineConf:
    _target_: str = "extinct.components.models.CelebaErmBaseline"
    lr: float = MISSING
    weight_decay: float = MISSING
    lr_gamma: float = MISSING
    batch_norm: bool = True
