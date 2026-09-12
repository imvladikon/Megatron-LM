# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-free KDA geometry checks; deliberately avoid importing megatron.core."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_source = Path(__file__).resolve().parents[3] / "megatron/core/ssm/kda_config_utils.py"
_spec = importlib.util.spec_from_file_location("kda_config_utils", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate_kda_config = _module.validate_kda_config


def _config(**overrides):
    values = dict(
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        linear_cp_mode="headwise",
        linear_cp_layout="zigzag",
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        gdn_conv_pad_alignment=None,
        gdn_pre_gated_delta_rule_fusion=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (4, 1), (1, 4), (2, 2)])
def test_random_fixture_geometry_divides_real_head_groups(tp, cp):
    validate_kda_config(_config(), tp_size=tp, cp_size=cp)


def test_chunkwise_cp_does_not_partition_heads():
    config = _config(linear_cp_mode="chunkwise", linear_cp_layout="contiguous")
    validate_kda_config(config, tp_size=4, cp_size=8)


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"linear_num_value_heads": 8}, "equal key and value head counts"),
        ({"linear_value_head_dim": 64}, "equal key and value head dimensions"),
        ({"linear_key_head_dim": None}, "positive integer"),
        ({"linear_num_key_heads": True}, "positive integer"),
        ({"linear_conv_kernel_dim": 0}, "positive integer"),
        ({"linear_cp_mode": "other"}, "linear_cp_mode"),
        ({"kda_lower_bound": None}, "finite negative"),
        ({"kda_lower_bound": float("nan")}, "finite negative"),
        ({"kda_lower_bound": float("-inf")}, "finite negative"),
        ({"kda_lower_bound": 0}, "finite negative"),
        ({"kda_lower_bound": True}, "finite negative"),
        ({"gdn_conv_pad_alignment": 0}, "positive integer"),
    ],
)
def test_invalid_geometry_rejected_before_allocations(changes, error):
    with pytest.raises(ValueError, match=error):
        validate_kda_config(_config(**changes), tp_size=1, cp_size=1)


def test_headwise_cp_cannot_silently_truncate_heads():
    with pytest.raises(ValueError, match="head partition size 8"):
        validate_kda_config(_config(), tp_size=4, cp_size=2)


def test_chunkwise_cp_rejects_wrong_layout_and_convolution_padding():
    with pytest.raises(ValueError, match="requires layout"):
        validate_kda_config(_config(linear_cp_mode="chunkwise"), tp_size=1, cp_size=2)
    with pytest.raises(ValueError, match="padding cannot be used"):
        validate_kda_config(
            _config(linear_cp_mode="chunkwise", linear_cp_layout="contiguous", gdn_conv_pad_alignment=16),
            tp_size=1,
            cp_size=2,
        )


def test_unsupported_fusion_rejected_before_allocations():
    with pytest.raises(NotImplementedError, match="fusion is not implemented"):
        validate_kda_config(_config(gdn_pre_gated_delta_rule_fusion=True), tp_size=1, cp_size=1)
