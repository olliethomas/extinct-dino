# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from extinct.components.models.dino.eval import EvalMethod
from extinct.components.models.dino.vit import VitArch
from typing import Optional


@dataclass
class DINOConf:
    _target_: str = "extinct.components.models.dino.DINO"
    lr: float = 0.0005
    warmup_iters: int = 10
    weight_decay: float = 0.04
    min_lr: float = 1e-06
    weight_decay_end: float = 0.4
    freeze_last_layer: int = 1
    arch: VitArch = VitArch.small
    patch_size: int = 16
    out_dim: int = 65536
    norm_last_layer: bool = True
    use_bn_in_head: bool = False
    momentum_teacher: float = 0.996
    teacher_temp: float = 0.04
    warmup_teacher_temp_iters: int = 30
    eval_method: EvalMethod = EvalMethod.lin_clf
    num_eval_blocks: int = 1
    lr_eval: float = 0.0001
    lin_clf_epochs: int = 100
    batch_size_eval: Optional[int] = None
    max_steps: int = -1
    dm_batch_size: int = -1
    local_crops_number: int = -1