from __future__ import annotations
import copy
import math
from typing import Any, TypeVar
import warnings

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

__all__ = ["trunc_normal_", "cosine_scheduler", "get_params_groups"]

T = TypeVar('T')


def _no_grad_trunc_normal_(tensor: Tensor, mean: float, std: float, a: float, b: float) -> Tensor:
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x: float | Tensor) -> float | Tensor:
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_params_groups(model: nn.Module) -> list[dict[str, list[Tensor] | float]]:
    regularized: list[Tensor] = []
    not_regularized: list[Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]


def cosine_scheduler(
    base_value: float,
    final_value: float,
    total_iters: int,
    warmup_iters: int = 0,
    start_warmup_value: float = 0,
) -> np.ndarray:
    assert warmup_iters < total_iters
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule


def deepcopy_with_kwargs(obj: T, **kwargs: Any) -> T:
    obj_cp = copy.deepcopy(obj)
    for attr, value in kwargs.items():
        if not hasattr(obj_cp, attr):
            raise AttributeError(...)
        setattr(obj_cp, attr, value)
    return obj_cp
