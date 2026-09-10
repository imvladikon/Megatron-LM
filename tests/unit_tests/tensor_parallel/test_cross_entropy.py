# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import numpy as np
import torch

from megatron.core.models.common.language_module import language_module as language_module_module
from megatron.core.tensor_parallel import cross_entropy as cross_entropy_module
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import Utils


class _FakeTPGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


class _IdentityView(torch.autograd.Function):
    """Mirror TP gather's custom-autograd view contract at world size one."""

    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def test_vocab_parallel_cross_entropy_uses_explicit_tp_group(monkeypatch):
    tp_group = _FakeTPGroup()
    all_reduce_groups = []

    def fake_all_reduce(tensor, op=None, group=None):
        all_reduce_groups.append(group)
        return tensor

    def fail_parallel_state_call(*args, **kwargs):
        raise AssertionError("explicit tp_group should avoid parallel_state")

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        cross_entropy_module, "get_tensor_model_parallel_group", fail_parallel_state_call
    )

    vocab_parallel_logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]])
    target = torch.tensor([2, 0])
    expected_output = torch.nn.functional.cross_entropy(
        vocab_parallel_logits.clone(), target, reduction="none"
    )

    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=tp_group)

    torch.testing.assert_close(output, expected_output)
    assert all_reduce_groups == [tp_group, tp_group, tp_group]


def test_vocab_parallel_cross_entropy_accepts_fp32_custom_function_view(monkeypatch):
    tp_group = _FakeTPGroup()
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None, group=None: tensor)

    base_logits = torch.tensor(
        [[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]], requires_grad=True
    )
    logits_view = _IdentityView.apply(base_logits)
    target = torch.tensor([2, 0])
    expected_logits = base_logits.detach().clone().requires_grad_(True)

    output = vocab_parallel_cross_entropy(logits_view, target, tp_group=tp_group)
    expected = torch.nn.functional.cross_entropy(expected_logits, target, reduction="none")
    output.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(base_logits.grad, expected_logits.grad)


def test_language_module_unfused_loss_passes_tp_group(monkeypatch):
    tp_group = _FakeTPGroup()
    captured = {}

    def fake_vocab_parallel_cross_entropy(logits, labels, label_smoothing=0.0, tp_group=None):
        captured["logits"] = logits
        captured["labels"] = labels
        captured["label_smoothing"] = label_smoothing
        captured["tp_group"] = tp_group
        return torch.zeros_like(labels, dtype=logits.dtype)

    monkeypatch.setattr(
        language_module_module.tensor_parallel,
        "vocab_parallel_cross_entropy",
        fake_vocab_parallel_cross_entropy,
    )

    module = SimpleNamespace(
        config=SimpleNamespace(cross_entropy_loss_fusion=False), tp_group=tp_group
    )
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    logits = torch.randn(3, 2, 4)

    loss = language_module_module.LanguageModule.compute_language_model_loss(
        module, labels=labels, logits=logits
    )

    assert captured["logits"] is logits
    assert captured["tp_group"] is tp_group
    assert captured["label_smoothing"] == 0.0
    torch.testing.assert_close(captured["labels"], labels.transpose(0, 1).contiguous())
    assert loss.shape == labels.shape


def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4, 2)
    vocab_parallel_logits = torch.range(0, 7).repeat(16, 4).cuda()
    target = torch.arange(0, 32, 2).cuda()
    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = torch.tensor(
        [
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
        ]
    ).cuda()
    assert torch.equal(torch.round(expected_output), torch.round(output))
    Utils.destroy_model_parallel()
