# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

pytest.importorskip("fla")
pytest.importorskip("triton")

from triton.runtime.autotuner import Autotuner  # noqa: E402

from megatron.core.ssm import fla_autotune  # noqa: E402


def _autotuner(obj):
    while obj is not None and not isinstance(obj, Autotuner) and hasattr(obj, "fn"):
        obj = obj.fn
    return obj


def test_pinning_requested_follows_env_then_torch(monkeypatch):
    monkeypatch.setenv(fla_autotune.FIXED_AUTOTUNE_META_ENV, "1")
    assert fla_autotune.fla_autotune_pinning_requested()
    monkeypatch.setenv(fla_autotune.FIXED_AUTOTUNE_META_ENV, "0")
    assert not fla_autotune.fla_autotune_pinning_requested()
    monkeypatch.delenv(fla_autotune.FIXED_AUTOTUNE_META_ENV)
    assert fla_autotune.fla_autotune_pinning_requested() == torch.are_deterministic_algorithms_enabled()


def test_kda_block_sizes_are_fixed_and_launch_parameters_stay_tunable(monkeypatch):
    from fla.ops.kda import gate

    kernel = _autotuner(gate.kda_gate_fwd_kernel)
    before = list(kernel.configs)
    assert len({tuple(sorted(c.kwargs.items())) for c in before}) > 1

    monkeypatch.setattr(fla_autotune, "_pinned", False)
    try:
        assert fla_autotune.pin_fla_autotune_meta_parameters() > 0
        after = kernel.configs
        assert len({tuple(sorted(c.kwargs.items())) for c in after}) == 1
        assert after[0].kwargs["BT"] == max(c.kwargs["BT"] for c in before)
        # num_warps/num_stages variants of the chosen block size are still autotuned.
        assert len(after) == sum(c.kwargs == after[0].kwargs for c in before) > 1
        assert fla_autotune.pin_fla_autotune_meta_parameters() == 0
    finally:
        kernel.configs = before
