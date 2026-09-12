# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""KPool tensor and DSA integration tests. Run only on a remote test host.

HF's real indexer is the oracle; the MCore indexer's TE projection/norm
submodules are replaced with small torch test modules for these CPU checks.
"""

import hashlib
import inspect
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kpool import (
    _compress_keys,
    select_kpool_tokens,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_query_positions_and_key_reorder,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig

hf_module = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")


def _reference(dtype=torch.float32, pool_size=4, topk=8):
    config = SimpleNamespace(
        hidden_size=16,
        q_lora_rank=5,
        qk_rope_head_dim=0,
        index_n_heads=3,
        index_head_dim=4,
        index_topk=topk,
        index_kpool=pool_size,
        index_kpool_always_select_tail=True,
    )
    model = hf_module.Glm5NextTextIndexer(config, layer_idx=0).to(dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.4)
    return model


@torch.no_grad()
def _projections(reference, x, qr):
    q = reference.wq_b(qr).unflatten(-1, (reference.n_heads, reference.head_dim))
    k = reference.k_norm(reference.wk(x))
    weights = reference.weights_proj(x)
    gates = F.linear(x, reference.index_kpool_compress_gate)
    return tuple(t.transpose(0, 1) for t in (q, k, weights, gates))


def _valid_set(row):
    values = row[row >= 0].tolist()
    assert len(values) == len(set(values)), "KPool must not duplicate tail/pool tokens"
    return sorted(values)


def _assert_same_tokens(actual, expected):
    assert actual.shape == expected.shape
    for a, e in zip(
        actual.reshape(-1, actual.size(-1)), expected.reshape(-1, expected.size(-1)), strict=True
    ):
        assert _valid_set(a) == _valid_set(e)


def test_hf_reference_is_pinned():
    # Reviewed Transformers 5.16.1 modeling_glm5_next.py. A new oracle revision
    # must be reviewed deliberately, rather than inferred from a version string.
    expected = "2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b"
    actual = hashlib.sha256(Path(inspect.getfile(hf_module)).read_bytes()).hexdigest()
    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pool_size", [2, 4])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("padding", ["none", "left", "right", "all"])
def test_selection_matches_real_hf(dtype, pool_size, tail, padding):
    torch.manual_seed(2907)
    reference = _reference(dtype, pool_size)
    reference.index_kpool_always_select_tail = tail
    x, qr = torch.randn(3, 21, 16, dtype=dtype), torch.randn(3, 21, 5, dtype=dtype)
    valid = torch.ones(3, 21, dtype=torch.bool)
    if padding == "left":
        valid[0, :3], valid[1, :8], valid[2, :] = False, False, False
    elif padding == "right":
        valid[0, 13:], valid[1, 6:], valid[2, :] = False, False, False
    elif padding == "all":
        valid[:] = False
    expected = reference(x, qr, valid, past_key_values=None)
    projections = _projections(reference, x, qr)
    causal = torch.ones(21, 21, dtype=torch.bool).tril()
    visible = causal & valid[:, None, :]
    mask = torch.zeros(3, 21, 21).masked_fill(~visible, -torch.inf)
    actual = select_kpool_tokens(
        *projections,
        reference.index_kpool_compress_ape,
        index_topk=8,
        pool_size=pool_size,
        always_select_tail=tail,
        workspace_bytes=512,
        mask=mask,
    )
    assert not actual.requires_grad and actual.dtype == torch.int32
    _assert_same_tokens(actual, expected)


def test_bf16_compression_rounding_matches_hf_not_fp32_sum():
    torch.manual_seed(681)
    reference = _reference(torch.bfloat16)
    keys, gates = torch.randn(1, 32, 4).bfloat16(), torch.randn(1, 32, 4).bfloat16()
    states = torch.cat((keys, gates, torch.ones(1, 32, 1).bfloat16()), dim=-1)
    expected, _, _ = reference.get_pooled_states(states)
    actual = _compress_keys(keys[0], gates[0], reference.index_kpool_compress_ape, 4, 512)
    torch.testing.assert_close(actual, expected[0], atol=0, rtol=0)
    scores = gates.reshape(8, 4, 4).float() + reference.index_kpool_compress_ape.float()
    wrong = (scores.softmax(1) * keys.reshape(8, 4, 4).float()).sum(1).bfloat16()
    assert torch.count_nonzero(wrong != actual) > 0


@pytest.mark.parametrize("partial_queries", [False, True])
def test_packed_documents_padding_and_cp_query_order(partial_queries):
    torch.manual_seed(413)
    reference = _reference()
    lengths = torch.tensor([3, 0, 9, 6], dtype=torch.int32)
    cu = torch.tensor([0, 4, 4, 16, 24], dtype=torch.int32)
    x, qr = torch.randn(1, 24, 16), torch.randn(1, 24, 5)
    expected = torch.full((1, 24, 11), -1, dtype=torch.int32)
    for start, length in zip(cu[:-1].tolist(), lengths.tolist(), strict=True):
        if length:
            local = reference(
                x[:, start : start + length],
                qr[:, start : start + length],
                torch.ones(1, length, dtype=torch.bool),
                None,
            )
            expected[:, start : start + length] = torch.where(local >= 0, local + start, -1)
    q, k, weights, gates = _projections(reference, x, qr)
    positions = torch.tensor([22, 0, 9, 14, 2, 18, 5, 3]) if partial_queries else torch.arange(24)
    document = torch.bucketize(positions, cu[1:], right=True)
    starts, ends = cu[:-1][document], positions + 1
    actual = select_kpool_tokens(
        q[positions],
        k,
        weights[positions],
        gates,
        reference.index_kpool_compress_ape,
        index_topk=8,
        pool_size=4,
        always_select_tail=True,
        workspace_bytes=512,
        cu_seqlens_kv=cu,
        sequence_lengths=lengths,
        varlen_starts=starts,
        varlen_ends=ends,
    )
    _assert_same_tokens(actual, expected[:, positions])
    # No future token can affect earlier selection, including an incomplete pool.
    changed_k, changed_gates = k.clone(), gates.clone()
    changed_k[10:], changed_gates[10:] = (
        100 * torch.randn_like(k[10:]),
        100 * torch.randn_like(gates[10:]),
    )
    changed = select_kpool_tokens(
        q[positions],
        changed_k,
        weights[positions],
        changed_gates,
        reference.index_kpool_compress_ape,
        index_topk=8,
        pool_size=4,
        always_select_tail=True,
        workspace_bytes=512,
        cu_seqlens_kv=cu,
        sequence_lengths=lengths,
        varlen_starts=starts,
        varlen_ends=ends,
    )
    _assert_same_tokens(changed[:, positions < 10], actual[:, positions < 10])


class _ScoreShapes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.bmm.default:
            self.shapes.append(tuple(result.shape))
        return result


def test_score_workspace_chunks_queries():
    torch.manual_seed(114)
    reference = _reference()
    x, qr = torch.randn(1, 29, 16), torch.randn(1, 29, 5)
    projections = _projections(reference, x, qr)
    with _ScoreShapes() as trace:
        actual = select_kpool_tokens(
            *projections,
            reference.index_kpool_compress_ape,
            index_topk=8,
            pool_size=4,
            always_select_tail=True,
            workspace_bytes=512,
        )
    assert trace.shapes and all(shape[0] <= 2 for shape in trace.shapes)
    expected = reference(x, qr, torch.ones(1, 29, dtype=torch.bool), None)
    _assert_same_tokens(actual, expected)


@pytest.fixture(scope="module")
def groups(tmp_path_factory):
    if dist.is_initialized():
        group = dist.new_group(
            ranks=[dist.get_rank()], backend="gloo", use_local_synchronization=True
        )
    else:
        rendezvous = tmp_path_factory.mktemp("kpool-contract") / "gloo"
        dist.init_process_group("gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1)
        group = dist.group.WORLD
    try:
        yield ProcessGroupCollection(tp=group, cp=group, pp=group, tp_cp=group)
    finally:
        dist.destroy_process_group(group)


class _Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, config, bias=False, **_kwargs):
        super().__init__(in_features, out_features, bias=bias, dtype=config.params_dtype)

    def forward(self, x):
        return F.linear(x, self.weight), None


class _LayerNorm(torch.nn.LayerNorm):
    def __init__(self, hidden_size, eps, config, **_kwargs):
        super().__init__(hidden_size, eps=eps, dtype=config.params_dtype)


def _build_attention(groups, reference, *, tp_size=1, cp_size=1):
    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2 * tp_size,
        kv_channels=4,
        qk_head_dim=4,
        qk_pos_emb_head_dim=0,
        kv_lora_rank=4,
        q_lora_rank=5,
        v_head_dim=4,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_n_heads=3,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=8,
        dsa_indexer_kpool=4,
        dsa_indexer_kpool_always_select_tail=True,
        dsa_indexer_kpool_workspace_bytes=512,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_k_norm_epsilon=1e-6,
        dsa_indexer_loss_coeff=0,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=tp_size > 1,
        cp_comm_type="allgather",
    )
    indexer_spec = ModuleSpec(
        module=DSAIndexer,
        submodules=DSAIndexerSubmodules(
            linear_wq_b=_Linear,
            linear_wk=_Linear,
            k_norm=_LayerNorm,
            linear_weights_proj=_Linear,
        ),
    )
    attention = DSAttention(
        config,
        DSAttentionSubmodules(indexer=indexer_spec),
        1,
        AttnMaskType.causal,
        "self",
        pg_collection=groups,
        cp_comm_type="allgather",
    )
    mapping = {
        "linear_wq_b.weight": "wq_b.weight",
        "linear_wk.weight": "wk.weight",
        "linear_weights_proj.weight": "weights_proj.weight",
        "k_norm.weight": "k_norm.weight",
        "k_norm.bias": "k_norm.bias",
        "index_kpool_compress_ape": "index_kpool_compress_ape",
        "index_kpool_compress_gate": "index_kpool_compress_gate",
    }
    with torch.no_grad():
        for target, source in mapping.items():
            attention.indexer.get_parameter(target).copy_(reference.get_parameter(source))
    return attention


def test_real_dsattention_forward_backward_uses_kpool(groups):
    torch.manual_seed(939)
    reference = _reference()
    attention = _build_attention(groups, reference)
    x, qr = torch.randn(13, 1, 16), torch.randn(13, 1, 5)
    query, key = (
        torch.randn(13, 1, 2, 4, requires_grad=True),
        torch.randn(13, 1, 1, 4, requires_grad=True),
    )
    up_v = torch.randn(2, 4, 4, requires_grad=True)
    output = attention(
        query, key, None, None, x, qr, attn_mask_type=AttnMaskType.causal, up_v_weight=up_v
    )
    indices = reference(
        x.transpose(0, 1), qr.transpose(0, 1), torch.ones(1, 13, dtype=torch.bool), None
    )[0]
    sparse_mask = torch.zeros(13, 13, dtype=torch.bool)
    for row in range(13):
        sparse_mask[row, indices[row][indices[row] >= 0].long()] = True
    scores = torch.einsum("sbhd,tbnd->bhst", query, key) * 0.5
    probabilities = scores.masked_fill(~sparse_mask, -torch.inf).softmax(-1)
    latent = torch.einsum("bhst,tbnd->sbhd", probabilities, key)
    expected = torch.einsum("sbhc,hdc->sbhd", latent, up_v).reshape(13, 1, 8)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
    grad_out = torch.randn_like(output)
    actual_grads = torch.autograd.grad(
        (output * grad_out).sum(), (query, key, up_v), retain_graph=True
    )
    expected_grads = torch.autograd.grad((expected * grad_out).sum(), (query, key, up_v))
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    assert all(parameter.grad is None for parameter in attention.indexer.parameters())
    assert not hasattr(attention.indexer, "_kpool_gate_score")
    # Negative control: the preserved ordinary DSA branch chooses individual
    # tokens. With identical projections it does not implement pool selection.
    attention.config.dsa_indexer_kpool = 1
    attention.indexer.index_kpool = 1
    with torch.no_grad():
        tokenwise = attention(
            query, key, None, None, x, qr, attn_mask_type=AttnMaskType.causal, up_v_weight=up_v
        )
    assert (tokenwise - output.detach()).abs().max() > 1e-3


def _distributed_worker(rank, tp_size, cp_size, rendezvous):
    # MCore's collective helpers allocate on current_device even with Gloo.
    # Redirect only that allocation target inside this isolated CPU worker;
    # keep the actual gather/reduce-scatter and their autograd implementations.
    cpu_allocator = pytest.MonkeyPatch()
    cpu_allocator.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=45),
    )
    try:
        tp_rank, cp_rank = rank % tp_size, rank // tp_size
        tp_groups = [
            dist.new_group(list(range(c * tp_size, (c + 1) * tp_size))) for c in range(cp_size)
        ]
        cp_groups = [
            dist.new_group([c * tp_size + t for c in range(cp_size)]) for t in range(tp_size)
        ]
        groups = ProcessGroupCollection(
            tp=tp_groups[cp_rank], cp=cp_groups[tp_rank], tp_cp=dist.group.WORLD
        )
        torch.manual_seed(2931)
        reference = _reference()
        attention = _build_attention(groups, reference, tp_size=tp_size, cp_size=cp_size)
        physical = torch.tensor([0, 8, 24], dtype=torch.int32)
        logical = torch.tensor([0, 7, 20], dtype=torch.int32)
        cp_positions, _ = build_packed_allgather_cp_query_positions_and_key_reorder(
            cu_seqlens_q=physical,
            cu_seqlens_kv=physical,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=torch.device("cpu"),
            local_output_size=24 // cp_size,
            key_local_output_size=24 // cp_size,
            global_output_size=24,
        )
        x, qr = torch.randn(24, 1, 16), torch.randn(24, 1, 5)
        full_query, full_key = torch.randn(24, 1, 2 * tp_size, 4), torch.randn(24, 1, 1, 4)
        full_up_v = torch.randn(2 * tp_size, 4, 4)
        sp_positions = cp_positions.chunk(tp_size)[tp_rank]
        query = (
            full_query[cp_positions, :, 2 * tp_rank : 2 * (tp_rank + 1)].clone().requires_grad_()
        )
        key = full_key[cp_positions].clone().requires_grad_()
        up_v = full_up_v[2 * tp_rank : 2 * (tp_rank + 1)].clone().requires_grad_()
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=logical,
            cu_seqlens_kv=logical,
            cu_seqlens_q_padded=physical,
            cu_seqlens_kv_padded=physical,
            max_seqlen_q=16,
            max_seqlen_kv=16,
        )
        output = attention(
            query,
            key,
            None,
            None,
            x[sp_positions],
            qr[sp_positions],
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=packed,
            up_v_weight=up_v,
        )
        sparse_mask = torch.zeros(24, 24, dtype=torch.bool)
        for start, length in ((0, 7), (8, 13)):
            indices = reference(
                x[start : start + length].transpose(0, 1),
                qr[start : start + length].transpose(0, 1),
                torch.ones(1, length, dtype=torch.bool),
                None,
            )[0]
            for row in range(length):
                sparse_mask[start + row, indices[row][indices[row] >= 0].long() + start] = True
        reference_q = query.detach().clone().requires_grad_()
        reference_k = full_key.detach().clone().requires_grad_()
        reference_v = up_v.detach().clone().requires_grad_()
        scores = torch.einsum("sbhd,tbnd->bhst", reference_q, reference_k) * 0.5
        visible = sparse_mask[cp_positions]
        valid_rows = visible.any(-1)[None, None, :, None]
        masked_scores = scores.masked_fill(~visible, -torch.inf)
        # Avoid softmax(all -inf) in the independent oracle: masking its NaN
        # output afterward would still produce NaN gradients for padding rows.
        masked_scores = torch.where(valid_rows, masked_scores, 0)
        probabilities = masked_scores.softmax(-1) * valid_rows
        latent = torch.einsum("bhst,tbnd->sbhd", probabilities, reference_k)
        expected = torch.einsum("sbhc,hdc->sbhd", latent, reference_v).reshape(24 // cp_size, 1, 8)
        torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)
        cotangent = torch.randn_like(output)
        (output * cotangent).sum().backward()
        (expected * cotangent).sum().backward()
        # CP gather's adjoint sums all query-rank contributions before selecting
        # the owning key shard. Use a distinct all-reduce oracle for this adjoint.
        dist.all_reduce(reference_k.grad, group=groups.cp)
        torch.testing.assert_close(query.grad, reference_q.grad, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(key.grad, reference_k.grad[cp_positions], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(up_v.grad, reference_v.grad, atol=3e-5, rtol=3e-5)
        assert not hasattr(attention.indexer, "_kpool_gate_score")
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()
        cpu_allocator.undo()


@pytest.mark.parametrize("tp_size,cp_size", [(1, 2), (2, 2)])
def test_kpool_real_tp_cp_forward_and_key_gradients(tmp_path, tp_size, cp_size):
    mp.spawn(
        _distributed_worker,
        args=(tp_size, cp_size, str(tmp_path / "gloo")),
        nprocs=tp_size * cp_size,
        join=True,
    )
