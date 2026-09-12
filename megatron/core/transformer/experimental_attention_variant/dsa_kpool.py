# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Causal KPool selection with bounded score/compression scratch tensors.

Selection is discrete (as in Transformers Glm5NextTextIndexer.forward).
Complete pools are compressed within each document; the current incomplete
pool is appended as raw tokens. Scores are never materialized for the entire
query sequence or between different documents in a packed batch.
"""

from typing import Optional

import torch


def _chunk_rows(workspace_bytes: int, bytes_per_row: int) -> int:
    if type(workspace_bytes) is not int or workspace_bytes < bytes_per_row:
        raise ValueError(
            f"KPool workspace requires at least {bytes_per_row} bytes for one row, "
            f"got {workspace_bytes}. Increase the budget or reduce the sequence/head geometry."
        )
    return max(1, workspace_bytes // bytes_per_row)


def _compress_keys(
    keys: torch.Tensor,
    gates: torch.Tensor,
    ape: torch.Tensor,
    pool_size: int,
    workspace_bytes: int,
) -> torch.Tensor:
    """Compress complete pools, preserving HF's activation-dtype rounding."""
    pools = keys.size(0) // pool_size
    head_dim = keys.size(-1)
    output = keys.new_empty((pools, head_dim))
    rows = _chunk_rows(workspace_bytes, pool_size * head_dim * (8 + 2 * keys.element_size()))
    for begin in range(0, pools, rows):
        end = min(begin + rows, pools)
        grouped_keys = keys[begin * pool_size : end * pool_size].reshape(-1, pool_size, head_dim)
        grouped_gates = gates[begin * pool_size : end * pool_size].reshape(-1, pool_size, head_dim)
        logits = grouped_gates.float() + ape.float().unsqueeze(0)
        # HF rounds probabilities before multiplication, and rounds the product
        # before reduction. An FP32 weighted sum followed by one cast differs.
        probabilities = logits.softmax(dim=1).to(keys.dtype)
        del logits
        output[begin:end] = (probabilities * grouped_keys).sum(dim=1)
    return output


def _mask_bounds(
    mask: torch.Tensor, workspace_bytes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a zero/-inf mask to contiguous visible ranges, one chunk at a time."""
    queries, keys = mask.shape
    starts = torch.zeros(queries, dtype=torch.int64, device=mask.device)
    ends = torch.zeros_like(starts)
    key_valid = torch.zeros(keys, dtype=torch.bool, device=mask.device)
    if keys == 0:
        return starts, ends, key_valid
    rows = _chunk_rows(workspace_bytes, keys * 8)
    positions = torch.arange(keys, device=mask.device)
    for begin in range(0, queries, rows):
        end = min(begin + rows, queries)
        current = mask[begin:end]
        visible = current == 0
        key_valid |= visible.any(dim=0)
        if bool(torch.any(~visible & (current != -torch.inf))):
            raise ValueError("KPool accepts only zero/-inf additive masks")
        first = visible.to(torch.int8).argmax(-1)
        count = visible.sum(-1)
        limit = first + count
        expected = (positions >= first[:, None]) & (positions < limit[:, None])
        if not torch.equal(visible, expected):
            raise ValueError("KPool requires contiguous visible keys within each query row")
        starts[begin:end], ends[begin:end] = first, limit
    return starts, ends, key_valid


@torch.no_grad()
def select_kpool_tokens(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    gate_scores: torch.Tensor,
    ape: torch.Tensor,
    *,
    index_topk: int,
    pool_size: int,
    always_select_tail: bool,
    workspace_bytes: int,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    sequence_lengths: Optional[torch.Tensor] = None,
    query_valid_rows: Optional[torch.Tensor] = None,
    use_relu: bool = True,
) -> torch.Tensor:
    """Select complete causal pools and append each query's incomplete tail.

    Args:
        q: Query projections [queries, batch, heads, head_dim].
        k: Normalized keys [keys, batch, head_dim], in global sequence order.
        weights: Unscaled head-projection outputs [queries, batch, heads].
        gate_scores: Token compression logits, with the same shape as ``k``.
        ape: Per-slot compression bias [pool_size, head_dim].
        index_topk: Token budget for complete pools; must be divisible by pool_size.
        pool_size: Number of tokens in each pool.
        always_select_tail: Append up to pool_size - 1 uncompressed tokens.
        workspace_bytes: Budget for estimated temporary tensor buffers, excluding
            inputs, final indices, pooled keys and backend-internal workspaces.
        mask: Optional zero/-inf additive mask [batch, queries, keys] or [queries, keys].
        varlen_starts: Optional per-query document starts in global key coordinates.
        varlen_ends: Corresponding exclusive causal ends. Bounds may be [queries]
            or [batch, queries]; they support non-monotonic CP query ordering.
        key_positions: Only identity global key order is supported. CP callers
            must gather/reorder keys and gates together before selection.
        cu_seqlens_kv: Physical document boundaries for a packed singleton batch.
        sequence_lengths: Optional logical lengths excluding inter-document padding.
        query_valid_rows: Optional real-token mask [batch, queries].
        use_relu: Apply ReLU to scaled per-head scores before weighting.

    Returns:
        Int32 token indices [batch, queries, index_topk + optional pool_size - 1],
        with -1 for unused slots. No output retains an autograd graph.
    """
    if q.ndim != 4 or k.ndim != 3:
        raise ValueError("KPool expects Q [Q,B,H,D] and K [K,B,D]")
    queries, batch, heads, dim = q.shape
    keys = k.size(0)
    if (
        k.shape[1:] != (batch, dim)
        or gate_scores.shape != k.shape
        or weights.shape != (queries, batch, heads)
        or heads < 1
        or dim < 1
    ):
        raise ValueError("KPool projection shapes disagree")
    if type(pool_size) is not int or pool_size < 1:
        raise ValueError("KPool pool_size must be a positive integer")
    if type(index_topk) is not int or index_topk < 1 or index_topk % pool_size:
        raise ValueError("KPool index_topk must be positive and divisible by pool_size")
    if ape.shape != (pool_size, dim):
        raise ValueError("KPool compression bias must have shape [pool_size, head_dim]")
    if any(t.device != q.device for t in (k, weights, gate_scores, ape)):
        raise ValueError("KPool projections and compression parameters must share a device")
    if mask is not None and (varlen_starts is not None or varlen_ends is not None):
        raise ValueError("KPool mask and varlen bounds are mutually exclusive")
    if (varlen_starts is None) != (varlen_ends is None):
        raise ValueError("KPool requires both varlen starts and ends")
    if key_positions is not None and not torch.equal(
        key_positions.to(device=q.device), torch.arange(keys, device=q.device)
    ):
        raise ValueError("KPool requires keys in identity global order")
    width = index_topk + (pool_size - 1 if always_select_tail else 0)
    result = torch.full((batch, queries, width), -1, dtype=torch.int32, device=q.device)
    if queries == 0 or keys == 0:
        return result
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(batch, -1, -1)
        if mask.shape != (batch, queries, keys) or mask.device != q.device:
            raise ValueError("KPool additive mask shape/device mismatch")
    if query_valid_rows is not None:
        if query_valid_rows.shape != (batch, queries):
            raise ValueError("KPool query validity mask must have shape [batch, queries]")
        query_valid_rows = query_valid_rows.to(device=q.device, dtype=torch.bool)

    segments = None
    if cu_seqlens_kv is not None:
        if batch != 1 or cu_seqlens_kv.ndim != 1:
            raise ValueError("Packed KPool requires a singleton batch and 1D boundaries")
        boundaries = cu_seqlens_kv.tolist()  # one metadata transfer, not one per document
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != keys
            or any(type(x) is not int for x in boundaries)
            or any(a > b for a, b in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError("KPool physical boundaries must cover the key sequence in order")
        lengths = [b - a for a, b in zip(boundaries, boundaries[1:])]
        if sequence_lengths is not None:
            logical = sequence_lengths.tolist()
            if len(logical) != len(lengths) or any(
                type(n) is not int or n < 0 or n > p for n, p in zip(logical, lengths)
            ):
                raise ValueError("KPool logical lengths exceed physical document lengths")
            lengths = logical
        segments = [
            (start, start + length, physical_end)
            for start, length, physical_end in zip(boundaries, lengths, boundaries[1:])
        ]
    elif sequence_lengths is not None:
        raise ValueError("KPool logical lengths require physical document boundaries")

    for batch_idx in range(batch):
        key_valid = None
        if mask is not None:
            starts, ends, key_valid = _mask_bounds(mask[batch_idx], workspace_bytes)
        elif varlen_starts is not None:
            starts = varlen_starts if varlen_starts.ndim == 1 else varlen_starts[batch_idx]
            ends = varlen_ends if varlen_ends.ndim == 1 else varlen_ends[batch_idx]
            starts, ends = starts.to(q.device, torch.int64), ends.to(q.device, torch.int64)
        else:
            if queries != keys:
                raise ValueError("KPool unequal Q/K lengths require explicit causal bounds")
            starts = torch.zeros(queries, dtype=torch.int64, device=q.device)
            ends = torch.arange(1, queries + 1, device=q.device)
            if segments is not None:
                cu = cu_seqlens_kv.to(device=q.device, dtype=torch.int64)
                document = torch.bucketize(ends - 1, cu[1:], right=True)
                starts = cu[:-1].index_select(0, document)
        if (
            starts.shape != (queries,)
            or ends.shape != (queries,)
            or bool(torch.any(starts < 0) or torch.any(ends < starts) or torch.any(ends > keys))
        ):
            raise ValueError("Invalid KPool per-query causal bounds")
        valid = ends > starts
        if key_valid is not None and queries == keys:
            valid = valid & key_valid
        if query_valid_rows is not None:
            valid = valid & query_valid_rows[batch_idx]
        current_segments = segments
        if current_segments is None:
            if not bool(valid.any()):
                continue
            origin = int(starts[valid][0])
            if bool(torch.any(starts[valid] != origin)):
                raise ValueError("Changing KPool document origins require packed boundaries")
            logical_end = int(key_valid.nonzero()[-1]) + 1 if key_valid is not None else keys
            current_segments = [(origin, logical_end, keys)]
        covered = torch.zeros(queries, dtype=torch.bool, device=q.device)
        for start, end, physical_end in current_segments:
            # Rows in alignment padding remain -1 even if the caller only has
            # physical causal bounds and did not attach a real-token row mask.
            covered |= valid & (starts == start) & (ends > end) & (ends <= physical_end)
            selected_rows = valid & (starts == start) & (ends <= end)
            covered |= selected_rows
            query_indices = selected_rows.nonzero().flatten()
            if query_indices.numel() == 0:
                continue
            pooled = _compress_keys(
                k[start:end, batch_idx],
                gate_scores[start:end, batch_idx],
                ape,
                pool_size,
                workspace_bytes,
            ).float()
            pools = pooled.size(0)
            select_k = min(index_topk // pool_size, pools)
            # Per-query score planes, masks, Q cast, top-k and token expansion.
            row_bytes = (
                max(1, pools) * (4 * heads + 16) + 4 * heads * dim + select_k * pool_size * 16
            )
            rows = _chunk_rows(workspace_bytes, row_bytes)
            pool_ends = start + torch.arange(1, pools + 1, device=q.device) * pool_size
            offsets = torch.arange(pool_size, device=q.device)
            tail_offsets = torch.arange(pool_size - 1, device=q.device)
            for begin in range(0, query_indices.numel(), rows):
                indices = query_indices[begin : begin + rows]
                limits = ends.index_select(0, indices)
                if select_k:
                    queries_chunk = q[:, batch_idx].index_select(0, indices).float()
                    scores = torch.matmul(queries_chunk, pooled.t().unsqueeze(0))
                    scores.mul_(dim**-0.5)
                    if use_relu:
                        scores.relu_()
                    head_weights = weights[:, batch_idx].index_select(0, indices).float() * (
                        heads**-0.5
                    )
                    index_scores = torch.matmul(head_weights.unsqueeze(-2), scores).squeeze(-2)
                    del scores, queries_chunk, head_weights
                    visible = pool_ends[None] <= limits[:, None]
                    index_scores.masked_fill_(~visible, torch.finfo(index_scores.dtype).min)
                    selected = index_scores.topk(select_k, dim=-1).indices
                    valid_pools = visible.gather(-1, selected)
                    tokens = start + selected[..., None] * pool_size + offsets
                    tokens.masked_fill_(~valid_pools[..., None], -1)
                    result[batch_idx, indices, : select_k * pool_size] = tokens.flatten(-2).to(
                        torch.int32
                    )
                    del index_scores, visible, selected, valid_pools, tokens
                if always_select_tail:
                    count = (limits - start).remainder(pool_size)
                    tail = (limits - count)[:, None] + tail_offsets
                    tail.masked_fill_(tail_offsets >= count[:, None], -1)
                    result[
                        batch_idx,
                        indices,
                        select_k * pool_size : select_k * pool_size + pool_size - 1,
                    ] = tail.to(torch.int32)
        if bool(torch.any(valid & ~covered)):
            raise ValueError("KPool query bounds cross a document or its logical padding")
    return result
