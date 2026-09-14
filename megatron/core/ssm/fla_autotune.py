# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fix the meta-parameters that FLA's Triton autotuners pick, so linear-attention arithmetic is reproducible.

Several FLA kernels (KDA chunk/gate kernels among them) autotune block sizes such as ``BT``, ``BS``,
``BK`` or ``BV``. The block size changes the reduction order, so the selected config changes the result
at bf16 rounding level; downstream MoE/DSA top-k selections turn that into discrete loss differences
between otherwise identical runs, and between ranks that tuned independently. Keeping only the configs
with the largest meta-parameters fixes the arithmetic while ``num_warps``/``num_stages`` stay autotuned.
"""

import importlib
import logging
import os
import pkgutil

import torch

logger = logging.getLogger(__name__)

FIXED_AUTOTUNE_META_ENV = "MCORE_FLA_FIXED_AUTOTUNE_META"

_pinned = False


def fla_autotune_pinning_requested() -> bool:
    """``MCORE_FLA_FIXED_AUTOTUNE_META=1``/``0`` forces it; unset follows torch deterministic algorithms."""
    value = os.environ.get(FIXED_AUTOTUNE_META_ENV, "")
    if value:
        return value == "1"
    return torch.are_deterministic_algorithms_enabled()


def _meta_parameter_rank(config) -> tuple:
    numeric = [v for v in config.kwargs.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(numeric), tuple(sorted((key, str(value)) for key, value in config.kwargs.items()))


def pin_fla_autotune_meta_parameters() -> int:
    """Restrict every FLA Triton autotuner to the configs sharing the largest meta-parameters.

    Must run before the kernels are first called. Idempotent. Returns the number of autotuners changed.
    """
    global _pinned
    if _pinned:
        return 0
    import fla
    from triton.runtime.autotuner import Autotuner

    changed = 0
    for info in pkgutil.walk_packages(fla.__path__, "fla."):
        if not info.name.startswith(("fla.ops.", "fla.modules.")):
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception:  # optional backends of unrelated kernels
            continue
        for obj in list(vars(module).values()):
            # @triton.heuristics wraps the autotuner; follow .fn down to it.
            while obj is not None and not isinstance(obj, Autotuner) and hasattr(obj, "fn"):
                obj = obj.fn
            if not isinstance(obj, Autotuner):
                continue
            if len({_meta_parameter_rank(config)[1] for config in obj.configs}) <= 1:
                continue
            best = max(_meta_parameter_rank(config) for config in obj.configs)
            obj.configs = [config for config in obj.configs if _meta_parameter_rank(config) == best]
            obj.cache.clear()
            changed += 1
    _pinned = True
    logger.info("Fixed the autotuned meta-parameters of %d FLA Triton kernels for reproducible arithmetic.", changed)
    return changed
