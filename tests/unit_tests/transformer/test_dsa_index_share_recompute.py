# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest

from megatron.core.recompute import (
    _build_dsa_index_share_recompute_chunks,
    checkpointed_forward,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    _DSA_INDEX_SHARE_STATE,
    dsa_index_share_context,
)


def _layers(*layer_numbers):
    return [SimpleNamespace(layer_number=layer_number) for layer_number in layer_numbers]


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (1, [(0, 1), (1, 2), (2, 6), (6, 10)]),
        (4, [(0, 2), (2, 6), (6, 10)]),
        (8, [(0, 6), (6, 10)]),
    ],
)
def test_glm52_chunks_keep_source_with_consumers(target, expected):
    chunks = _build_dsa_index_share_recompute_chunks(
        _layers(*range(1, 11)), target, skip_topk_offset=3, topk_freq=4
    )

    assert chunks == expected


def test_pipeline_stage_may_start_at_a_source_layer():
    assert _build_dsa_index_share_recompute_chunks(
        _layers(7, 8, 9, 10), 1, skip_topk_offset=3, topk_freq=4
    ) == [(0, 4)]


def test_pipeline_stage_must_not_start_at_a_consumer_layer():
    with pytest.raises(RuntimeError, match="begins with a consumer"):
        _build_dsa_index_share_recompute_chunks(
            _layers(8, 9, 10), 4, skip_topk_offset=3, topk_freq=4
        )


def test_block_recompute_fails_before_running_a_split_share_group():
    block = SimpleNamespace(
        config=SimpleNamespace(
            experimental_attention_variant="dsa",
            dsa_indexer_topk_freq=4,
            recompute_method="block",
        )
    )

    with pytest.raises(ValueError, match="requires recompute_method='uniform'"):
        checkpointed_forward(
            block,
            hidden_states=None,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            attention_bias=None,
            packed_seq_params=None,
            use_inner_quantization_context=False,
        )


def test_index_share_context_is_fresh_and_nested_safe():
    assert _DSA_INDEX_SHARE_STATE.get() is None
    with dsa_index_share_context():
        outer = _DSA_INDEX_SHARE_STATE.get()
        assert outer == ({}, {})
        outer[0][3] = "outer"
        with dsa_index_share_context():
            inner = _DSA_INDEX_SHARE_STATE.get()
            assert inner == ({}, {})
            assert inner is not outer
        assert _DSA_INDEX_SHARE_STATE.get() is outer
    assert _DSA_INDEX_SHARE_STATE.get() is None
