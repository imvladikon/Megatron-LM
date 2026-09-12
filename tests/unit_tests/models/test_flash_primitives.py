# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tensor contracts; execute on a remote test host, never the work desktop.

The mHC test constructs an empty real HybridStack to isolate its boundary.
The MLA test runs the real projection method with ordinary torch linear/norm
test modules. It does not qualify TE kernels, distributed MLA or DSA/KPool.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F

from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.hyper_connection import learned_output_contract
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(scope="module")
def groups(tmp_path_factory):
    created_world = not dist.is_initialized()
    if created_world:
        rendezvous = tmp_path_factory.mktemp("flash-contract") / "gloo"
        dist.init_process_group("gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1)
        group = dist.group.WORLD
    else:
        # The normal distributed test runner may already own WORLD. Isolate
        # this CPU contract on each rank without replacing that process group.
        group = dist.new_group(
            ranks=[dist.get_rank()],
            backend="gloo",
            use_local_synchronization=True,
        )
    try:
        yield ProcessGroupCollection(tp=group, pp=group, cp=group, tp_cp=group)
    finally:
        dist.destroy_process_group(group)


@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("post_process", [False, True])
@pytest.mark.parametrize("is_mtp_layer", [False, True])
def test_hybrid_mhc_exit_and_checkpoint_schema(groups, learned, post_process, is_mtp_layer):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_learned_output_contract=learned,
        mtp_num_layers=1,
    )
    stack = HybridStack(
        config,
        HybridStackSubmodules(),
        layer_config_list=[],
        pg_collection=groups,
        pre_process=False,
        post_process=post_process,
        post_layer_norm=False,
        is_mtp_layer=is_mtp_layer,
    )
    has_head = learned and post_process and not is_mtp_layer
    head_keys = {"hc_head_fn", "hc_head_base", "hc_head_scale"}
    assert set(stack.state_dict()) == (head_keys if has_head else set())
    x = (torch.arange(3 * 2 * 32).reshape(3, 2, 32).float() / 100).requires_grad_()
    stack.set_input_tensor(x)
    result = stack(None, attention_mask=None)
    collapse = post_process and not is_mtp_layer
    if collapse:
        output, mtp_streams = result
        assert mtp_streams is x
        if learned:
            expected = learned_output_contract(
                x,
                stack.hc_head_fn,
                stack.hc_head_base,
                stack.hc_head_scale,
                4,
                config.layernorm_epsilon,
            )
        else:
            expected = sum(x.split(8, dim=-1)) / 4
    else:
        output = result
        expected = x
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
    (grad,) = torch.autograd.grad(output.square().sum(), x, retain_graph=True)
    (reference_grad,) = torch.autograd.grad(expected.square().sum(), x)
    torch.testing.assert_close(grad, reference_grad, atol=1e-6, rtol=1e-6)


class _TupleLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, *, dtype=None, bias=False, **_kwargs):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, x):
        return F.linear(x, self.weight), None


def _projection_shell(groups, *, q_lora_rank, dtype, fusion):
    # Bypass the attention/TE constructor, not the method under test.
    model = object.__new__(AbsorbedMLASelfAttention)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=3,
        qk_head_dim=4,
        qk_pos_emb_head_dim=0,
        v_head_dim=5,
        sequence_parallel=False,
        apply_rope_fusion=fusion,
        context_parallel_size=1,
        rope_type="yarn",
    )
    model.pg_collection = groups
    model.tp_group = groups.tp
    model.num_attention_heads_per_partition = 2
    model.q_head_dim = 4
    model.rotary_pos_emb = None
    model.recompute_up_proj = False
    model.cache_mla_latents = False
    if q_lora_rank is not None:
        model.linear_q_down_proj = _TupleLinear(12, q_lora_rank, bias=False, dtype=dtype)
        model.q_layernorm = torch.nn.RMSNorm(q_lora_rank, eps=1e-6, dtype=dtype)
        model.linear_q_up_proj = _TupleLinear(q_lora_rank, 8, bias=False, dtype=dtype)
    else:
        model.linear_q_proj = _TupleLinear(12, 8, bias=False, dtype=dtype)
    model.linear_kv_down_proj = _TupleLinear(12, 3, bias=False, dtype=dtype)
    model.kv_layernorm = torch.nn.RMSNorm(3, eps=1e-6, dtype=dtype)
    model.linear_kv_up_proj = _TupleLinear(3, 18, bias=False, dtype=dtype)
    return model


@pytest.mark.parametrize("q_lora_rank", [None, 6])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nope_absorption_matches_dense_mla_gradients(groups, q_lora_rank, packed, fusion, dtype):
    torch.manual_seed(1209)
    model = _projection_shell(groups, q_lora_rank=q_lora_rank, dtype=dtype, fusion=fusion)
    # Copy modules individually: runtime ProcessGroups cannot be deep-copied.
    reference = {name: copy.deepcopy(module) for name, module in model.named_children()}
    x = torch.randn(7, 1 if packed else 2, 12, dtype=dtype, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()
    metadata = None
    if packed:
        cu = torch.tensor([0, 3, 7], dtype=torch.int32)
        metadata = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    q_absorbed, kv_latent, _ = model.get_query_key_value_tensors(x, packed_seq_params=metadata)
    if not packed:
        q_absorbed, kv_latent = q_absorbed.transpose(0, 1), kv_latent.transpose(0, 1)
    else:
        q_absorbed, kv_latent = q_absorbed.unsqueeze(0), kv_latent.unsqueeze(0)

    source = x_reference.transpose(0, 1)
    q_latent = source
    if q_lora_rank is not None:
        q_latent = reference["q_layernorm"](reference["linear_q_down_proj"](source)[0])
    q = reference["linear_q_up_proj" if q_lora_rank is not None else "linear_q_proj"](q_latent)[0]
    q = q.unflatten(-1, (2, 4))
    kv = reference["kv_layernorm"](reference["linear_kv_down_proj"](source)[0])
    full_kv = reference["linear_kv_up_proj"](kv)[0].unflatten(-1, (2, 9))
    key, value = full_kv.split([4, 5], dim=-1)
    causal = torch.ones(7, 7, dtype=torch.bool).tril()
    if packed:
        document = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        causal &= document[:, None] == document[None, :]
    logits = torch.einsum("bsnk,btnk->bnst", q_absorbed, kv_latent) / 2
    dense_logits = torch.einsum("bsnd,btnd->bnst", q, key) / 2
    probability = logits.masked_fill(~causal, -torch.inf).softmax(-1)
    dense_probability = dense_logits.masked_fill(~causal, -torch.inf).softmax(-1)
    latent_output = torch.einsum("bnst,btnk->bsnk", probability, kv_latent)
    # The real helper supplies W_V after the core attention latent output.
    actual = torch.einsum("bsnk,ndk->bsnd", latent_output, model._get_v_up_weight())
    expected = torch.einsum("bnst,btnd->bsnd", dense_probability, value)
    tol = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    cotangent = torch.randn_like(actual)
    (actual * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=tol, atol=tol)
    for name, module in model.named_children():
        for parameter_name, parameter in module.named_parameters():
            reference_parameter = reference[name].get_parameter(parameter_name)
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=tol, atol=tol)


class _TestLayerNorm(torch.nn.LayerNorm):
    def __init__(self, hidden_size, eps, **_kwargs):
        super().__init__(hidden_size, eps=eps)


@pytest.mark.parametrize("rope_type", ["rope", "yarn"])
@pytest.mark.parametrize("packed", [False, True])
def test_nope_indexer_constructor_and_projections(groups, monkeypatch, rope_type, packed):
    def forbidden_rope(*_args, **_kwargs):
        raise AssertionError("NoPE must not construct a rotary embedding")

    monkeypatch.setattr(dsa, "RotaryEmbedding", forbidden_rope)
    monkeypatch.setattr(dsa, "YarnRotaryEmbedding", forbidden_rope)
    config = SimpleNamespace(
        hidden_size=12,
        qk_pos_emb_head_dim=0,
        q_lora_rank=6,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=8,
        rope_type=rope_type,
        rotary_percent=1.0,
        rotary_base=10000,
        rotary_scaling_factor=4.0,
        original_max_position_embeddings=128,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=1.0,
        init_method=None,
        dsa_indexer_k_norm_epsilon=1e-6,
        layernorm_epsilon=1e-5,
        sequence_parallel=False,
        dsa_indexer_k_norm_fp32=False,
        dsa_indexer_rotate_activation=False,
    )
    indexer = dsa.DSAIndexer(
        config,
        dsa.DSAIndexerSubmodules(
            linear_wq_b=_TupleLinear,
            linear_wk=_TupleLinear,
            k_norm=_TestLayerNorm,
            linear_weights_proj=_TupleLinear,
        ),
        pg_collection=groups,
    )
    assert indexer.rotary_pos_emb is None
    assert all(parameter.average_gradients_across_tp_domain for parameter in indexer.parameters())
    x = torch.randn(7, 1 if packed else 2, 12, requires_grad=True)
    qr = torch.randn(*x.shape[:2], 6, requires_grad=True)
    metadata = None
    if packed:
        cu = torch.tensor([0, 3, 7], dtype=torch.int32)
        metadata = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    q, k, weights = indexer.forward_before_topk(x, qr, packed_seq_params=metadata)
    q_ref = F.linear(qr, indexer.linear_wq_b.weight).unflatten(-1, (2, 4))
    k_ref = F.layer_norm(
        F.linear(x, indexer.linear_wk.weight),
        (4,),
        indexer.k_norm.weight,
        indexer.k_norm.bias,
        eps=1e-6,
    )
    w_ref = F.linear(x, indexer.linear_weights_proj.weight) * (2**-0.5) * (4**-0.5)
    actual_loss, reference_loss = 0, 0
    for actual, expected in zip((q, k, weights), (q_ref, k_ref, w_ref), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        cotangent = torch.randn_like(actual)
        actual_loss += (actual * cotangent).sum()
        reference_loss += (expected * cotangent).sum()
    inputs = (x, qr, *indexer.parameters())
    grads = torch.autograd.grad(actual_loss, inputs)
    reference_grads = torch.autograd.grad(reference_loss, inputs)
    for actual, expected in zip(grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
