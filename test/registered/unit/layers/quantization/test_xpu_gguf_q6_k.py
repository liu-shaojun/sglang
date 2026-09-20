"""CPU dispatch coverage for the XPU GGUF Q6_K linear path."""

from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.quantization.gguf as gguf_module
from sglang.srt.layers.quantization.gguf import (
    GGUFLinearXPUMethod,
    GGUFMoEXPUMethod,
    WeightType,
    _xpu_permute_grouped_k,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_xpu_gguf_q6_k_moe_loader_keeps_down_projection_compact(monkeypatch):
    layer = torch.nn.Module()
    layer.w13_qweight = torch.empty((2, 16, 144), dtype=torch.uint8)
    layer.w2_qweight = torch.empty((2, 8, 210), dtype=torch.uint8)
    layer.w13_qweight_type = SimpleNamespace(weight_type=WeightType.Q4_K.value)
    layer.w2_qweight_type = SimpleNamespace(weight_type=WeightType.Q6_K.value)

    def fake_prepare_q4(raw):
        assert raw.shape == (32, 144)
        return (
            torch.empty((32, 128), dtype=torch.uint8),
            torch.empty((32, 16), dtype=torch.uint8),
        )

    def fake_prepare_q6(raw):
        assert raw.shape == (16, 210)
        return (
            torch.empty((16, 256), dtype=torch.int8),
            torch.empty((16, 16), dtype=torch.float16),
        )

    monkeypatch.setattr(gguf_module, "_xpu_prepare_q4_k", fake_prepare_q4)
    monkeypatch.setattr(gguf_module, "_xpu_prepare_q6_k", fake_prepare_q6)

    method = object.__new__(GGUFMoEXPUMethod)
    method.process_weights_after_loading(layer)

    assert layer.w13_xpu_qweight.shape == (2, 16, 128)
    assert layer.w13_xpu_metadata.shape == (2, 16, 16)
    assert layer.w2_xpu_qweight.shape == (2, 8, 256)
    assert layer.w2_xpu_qweight.dtype == torch.int8
    assert layer.w2_xpu_scales.shape == (2, 8, 16)
    assert layer.w2_xpu_scales.dtype == torch.float16
    assert layer.w2_xpu_kind == "q6_k"
    assert not hasattr(layer, "w13_qweight")
    assert not hasattr(layer, "w2_qweight")


def test_xpu_gguf_q6_k_linear_dispatch_uses_group16_scales(monkeypatch):
    layer = torch.nn.Module()
    prefix = "_gguf_xpu_0"
    layer.register_buffer(
        f"{prefix}_weight", torch.empty((1, 72, 256), dtype=torch.int8)
    )
    layer.register_buffer(
        f"{prefix}_scales", torch.empty((1, 72, 16), dtype=torch.float16)
    )
    layer._gguf_xpu_representations = [("q6_k", prefix, None, None)]

    call = {}

    def fake_grouped_mm(output, activations, weights, scales, rows, experts):
        call.update(
            output=output,
            activations=activations,
            weights=weights,
            scales=scales,
            rows=rows,
            experts=experts,
        )
        output.fill_(3)

    monkeypatch.setattr(
        torch.ops.sgl_kernel,
        "gguf_q6_k_grouped_mm",
        fake_grouped_mm,
        raising=False,
    )

    method = object.__new__(GGUFLinearXPUMethod)
    x = torch.ones((2, 3, 256), dtype=torch.bfloat16)
    output = method.apply(layer, x)

    assert output.shape == (2, 3, 72)
    assert output.dtype == torch.bfloat16
    assert torch.all(output == 3)
    assert call["activations"].shape == (6, 256)
    assert call["weights"].shape == (1, 72, 256)
    assert call["weights"].dtype == torch.int8
    assert call["scales"].shape == (1, 72, 16)
    assert call["scales"].dtype == torch.float16
    torch.testing.assert_close(call["rows"], torch.tensor([6], dtype=torch.int32))
    assert call["experts"] == 1


@pytest.mark.parametrize("group_size", [16, 32])
def test_xpu_grouped_k_permutation_keeps_values_and_scales_aligned(group_size):
    rows, ratio, heads, head_dim = 2, 2, 2, 64
    k = ratio * heads * head_dim
    weights = (torch.arange(rows * k).reshape(rows, k) % 127).to(torch.int8)
    scales = torch.arange(rows * k // group_size, dtype=torch.float16).reshape(
        rows, k // group_size
    )

    actual_weights, actual_scales = _xpu_permute_grouped_k(
        weights, scales, (ratio, heads, head_dim), group_size
    )
    expected_weights = (
        weights.reshape(rows, ratio, heads, head_dim)
        .transpose(1, 2)
        .reshape_as(weights)
        .contiguous()
    )
    expected_scales = (
        scales.reshape(rows, ratio, heads, head_dim // group_size)
        .transpose(1, 2)
        .reshape_as(scales)
        .contiguous()
    )

    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual_scales, expected_scales)
