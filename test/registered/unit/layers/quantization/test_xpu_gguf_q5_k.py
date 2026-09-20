"""CPU dispatch coverage for the XPU GGUF Q5_K affine path."""

from types import SimpleNamespace

import torch

import sglang.srt.layers.quantization.gguf as gguf_module
from sglang.srt.layers.quantization.gguf import (
    GGUFLinearXPUMethod,
    GGUFMoEXPUMethod,
    WeightType,
    _xpu_permute_grouped_k_affine,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_xpu_gguf_q5_k_moe_loader_keeps_affine_down_projection(monkeypatch):
    layer = torch.nn.Module()
    layer.w13_qweight = torch.empty((2, 16, 144), dtype=torch.uint8)
    layer.w2_qweight = torch.empty((2, 8, 176), dtype=torch.uint8)
    layer.w13_qweight_type = SimpleNamespace(weight_type=WeightType.Q4_K.value)
    layer.w2_qweight_type = SimpleNamespace(weight_type=WeightType.Q5_K.value)

    def fake_prepare_q4(raw):
        assert raw.shape == (32, 144)
        return (
            torch.empty((32, 128), dtype=torch.uint8),
            torch.empty((32, 16), dtype=torch.uint8),
        )

    def fake_prepare_q5(raw):
        assert raw.shape == (16, 176)
        return (
            torch.empty((16, 256), dtype=torch.int8),
            torch.empty((16, 8), dtype=torch.float16),
            torch.empty((16, 8), dtype=torch.float16),
        )

    monkeypatch.setattr(gguf_module, "_xpu_prepare_q4_k", fake_prepare_q4)
    monkeypatch.setattr(gguf_module, "_xpu_prepare_q5_k", fake_prepare_q5)

    method = object.__new__(GGUFMoEXPUMethod)
    method.process_weights_after_loading(layer)

    assert layer.w2_xpu_qweight.shape == (2, 8, 256)
    assert layer.w2_xpu_scales.shape == (2, 8, 8)
    assert layer.w2_xpu_minimums.shape == (2, 8, 8)
    assert layer.w2_xpu_kind == "q5_k"
    assert not hasattr(layer, "w13_qweight")
    assert not hasattr(layer, "w2_qweight")


def test_xpu_gguf_q5_k_linear_dispatch_passes_scale_and_minimum(monkeypatch):
    layer = torch.nn.Module()
    prefix = "_gguf_xpu_0"
    layer.register_buffer(
        f"{prefix}_weight", torch.empty((1, 72, 256), dtype=torch.int8)
    )
    layer.register_buffer(
        f"{prefix}_scales", torch.empty((1, 72, 8), dtype=torch.float16)
    )
    layer.register_buffer(
        f"{prefix}_minimums", torch.empty((1, 72, 8), dtype=torch.float16)
    )
    layer._gguf_xpu_representations = [("q5_k", prefix, None, None)]

    call = {}

    def fake_grouped_mm(
        output, activations, weights, scales, minimums, rows, experts
    ):
        call.update(
            output=output,
            activations=activations,
            weights=weights,
            scales=scales,
            minimums=minimums,
            rows=rows,
            experts=experts,
        )
        output.fill_(5)

    monkeypatch.setattr(
        torch.ops.sgl_kernel,
        "gguf_q5_k_grouped_mm",
        fake_grouped_mm,
        raising=False,
    )

    method = object.__new__(GGUFLinearXPUMethod)
    x = torch.ones((2, 3, 256), dtype=torch.bfloat16)
    output = method.apply(layer, x)

    assert output.shape == (2, 3, 72)
    assert torch.all(output == 5)
    assert call["activations"].shape == (6, 256)
    assert call["weights"].shape == (1, 72, 256)
    assert call["scales"].shape == (1, 72, 8)
    assert call["minimums"].shape == (1, 72, 8)
    torch.testing.assert_close(call["rows"], torch.tensor([6], dtype=torch.int32))
    assert call["experts"] == 1


def test_xpu_q5_k_gdn_permutation_keeps_affine_metadata_aligned():
    rows, ratio, heads, head_dim, group_size = 2, 2, 2, 64, 32
    k = ratio * heads * head_dim
    weights = (torch.arange(rows * k).reshape(rows, k) % 32).to(torch.int8)
    scales = torch.arange(rows * k // group_size, dtype=torch.float16).reshape(
        rows, k // group_size
    )
    minimums = scales + 100

    actual = _xpu_permute_grouped_k_affine(
        weights, scales, minimums, (ratio, heads, head_dim), group_size
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
    expected_minimums = (
        minimums.reshape(rows, ratio, heads, head_dim // group_size)
        .transpose(1, 2)
        .reshape_as(minimums)
        .contiguous()
    )
    torch.testing.assert_close(actual[0], expected_weights)
    torch.testing.assert_close(actual[1], expected_scales)
    torch.testing.assert_close(actual[2], expected_minimums)
