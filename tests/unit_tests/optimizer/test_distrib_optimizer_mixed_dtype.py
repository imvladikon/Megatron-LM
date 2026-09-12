# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig


class MixedDtypeParameters(torch.nn.Module):
    """Distinct values expose swapped states even when parameter shapes agree."""

    def __init__(self, fp32_first, same_shape):
        super().__init__()
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
        )
        specs = [
            ("low_a", torch.bfloat16, (3, 5), 1.0),
            ("full", torch.float32, (2, 3), 2.0),
            ("low_b", torch.bfloat16, (4, 2), 3.0),
        ]
        if fp32_first:
            specs = [specs[1], specs[0], specs[2]]
        for name, dtype, shape, value in specs:
            self.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.full((4, 4) if same_shape else shape, value, dtype=dtype, device="cuda")
                ),
            )

    def forward(self):
        return sum((i + 1) * param.float().sum() for i, param in enumerate(self.parameters()))

    def sharded_state_dict(self):
        return {
            name: ShardedTensor.from_rank_offsets(
                name, param, replica_id=(0, 0, parallel_state.get_data_parallel_rank())
            )
            for name, param in self.named_parameters()
        }


@pytest.fixture
def optimizer_process_groups():
    if not torch.cuda.is_available():
        pytest.skip("DistributedOptimizer requires CUDA parameter shards")
    if not torch.distributed.is_initialized():
        # Standard repository torchrun entry point. Owned remote validation
        # preinitializes a private FileStore group before collecting this file.
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_distributed()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        parallel_state.destroy_model_parallel()


@pytest.mark.parametrize("fp32_first", [False, True])
@pytest.mark.parametrize("same_shape", [False, True])
def test_mixed_dtype_checkpoint_parameter_identity(
    optimizer_process_groups, fp32_first, same_shape
):
    model = MixedDtypeParameters(fp32_first, same_shape)
    ddp = DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=True,
        ),
        model,
        pg_collection=optimizer_process_groups,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.01,
            weight_decay=0.0,
            bf16=True,
            use_distributed_optimizer=True,
            clip_grad=0.0,
        ),
        [ddp],
        config_overrides={},
        pg_collection=optimizer_process_groups,
        use_gloo_process_groups=False,
    )
    distributed = (
        optimizer.chained_optimizers[0] if hasattr(optimizer, "chained_optimizers") else optimizer
    )
    assert isinstance(distributed, DistributedOptimizer)
    ddp.zero_grad_buffer()
    ddp().backward()
    ddp.finish_grad_sync()
    success, _, _ = optimizer.step()
    assert success
    ddp.start_param_sync(force_sync=True)

    # The checkpoint lookup must recover the very same shard updated for this
    # model parameter, rather than another same-group parameter's Adam state.
    for param in distributed.model_param_group_index_map:
        param_range = distributed._get_model_param_range_map(param)["param"]
        expected = (
            param.main_param
            if param.dtype == torch.bfloat16
            else param.detach().view(-1)[param_range.start : param_range.end]
        )
        actual = distributed._get_main_param_and_optimizer_states(param)
        assert actual["param"].shape == expected.shape
        torch.testing.assert_close(actual["param"], expected, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            assert actual[key].shape == expected.shape
            assert torch.isfinite(actual[key]).all()

    # Exercise the exact bucket-space export that failed in the VERL SFT save.
    state = distributed.sharded_state_dict(
        model.sharded_state_dict(), metadata={"distrib_optim_sharding_type": "dp_reshardable"}
    )
    assert state
