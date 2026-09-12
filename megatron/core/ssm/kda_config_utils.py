# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate KDA geometry before allocating parameters or invoking kernels."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def validate_kda_config(config: TransformerConfig, *, tp_size: int, cp_size: int) -> None:
    """Check the equal-head KDA contract against the actual process groups.

    Chunkwise CP partitions the recurrence along time, whereas headwise CP
    partitions heads after TP. Only the latter requires heads divisible by CP.

    Args:
        config: Layer configuration; no model or tensor is constructed.
        tp_size: Size of the supplied tensor-parallel process group.
        cp_size: Size of the supplied context-parallel process group.

    Raises:
        ValueError: Geometry or gate settings cannot describe this KDA layer.
        NotImplementedError: Requested preprocessing fusion is unavailable.
    """
    values = {
        "tp_size": tp_size,
        "cp_size": cp_size,
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_num_value_heads": config.linear_num_value_heads,
        "linear_key_head_dim": config.linear_key_head_dim,
        "linear_value_head_dim": config.linear_value_head_dim,
        "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
    }
    for name, value in values.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"KDA {name} must be a positive integer, got {value!r}.")
    if config.linear_num_key_heads != config.linear_num_value_heads:
        raise ValueError("KDA requires equal key and value head counts.")
    if config.linear_key_head_dim != config.linear_value_head_dim:
        raise ValueError("KDA requires equal key and value head dimensions.")
    if config.linear_cp_mode not in ("headwise", "chunkwise"):
        raise ValueError("KDA linear_cp_mode must be 'headwise' or 'chunkwise'.")
    divisor = tp_size * (cp_size if config.linear_cp_mode == "headwise" else 1)
    if config.linear_num_key_heads % divisor:
        raise ValueError(f"KDA heads must be divisible by the head partition size {divisor}.")
    if cp_size > 1:
        expected = "zigzag" if config.linear_cp_mode == "headwise" else "contiguous"
        if config.linear_cp_layout != expected:
            raise ValueError(f"KDA {config.linear_cp_mode} CP requires layout {expected!r}.")
    if config.kda_safe_gate:
        bound = config.kda_lower_bound
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound >= 0:
            raise ValueError("KDA safe gate requires a finite negative kda_lower_bound.")
    alignment = config.gdn_conv_pad_alignment
    if alignment is not None:
        if type(alignment) is not int or alignment < 1:
            raise ValueError("KDA gdn_conv_pad_alignment must be a positive integer.")
        if config.linear_cp_mode == "chunkwise" and cp_size > 1:
            raise ValueError("KDA convolution padding cannot be used with chunkwise CP.")
    if config.gdn_pre_gated_delta_rule_fusion:
        raise NotImplementedError("KDA pre-gated-delta-rule fusion is not implemented.")
