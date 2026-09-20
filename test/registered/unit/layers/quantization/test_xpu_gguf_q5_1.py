"""CPU dispatch coverage for the XPU GGUF Q5_1 affine path."""

from types import SimpleNamespace

import torch

import sglang.srt.layers.quantization.gguf as gguf_module
from sglang.srt.layers.quantization.gguf import (
    GGUFLinearXPUMethod,
    GGUFMoEXPUMethod,
    WeightType,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_xpu_gguf_q5_1_moe_loader_keeps_affine_down_projection(monkeypatch):
    layer = torch.nn.Module()
    layer.w13_qweight = torch.empty((2, 16, 144), dtype=torch.uint8)
    layer.w2_qweight = torch.empty((2, 8, 24), dtype=torch.uint8)
    layer.w13_qweight_type = SimpleNamespace(weight_type=WeightType.Q4_K.value)
    layer.w2_qweight_type = SimpleNamespace(weight_type=WeightType.Q5_1.value)

    def fake_prepare_q4(raw):
        assert raw.shape == (32, 144)
        return (
            torch.empty((32, 128), dtype=torch.uint8),
            torch.empty((32, 16), dtype=torch.uint8),
        )

    def fake_prepare_q5_1(raw):
        assert raw.shape == (16, 24)
        return (
            torch.empty((16, 32), dtype=torch.int8),
            torch.empty((16, 1), dtype=torch.float16),
            torch.empty((16, 1), dtype=torch.float16),
        )

    monkeypatch.setattr(gguf_module, "_xpu_prepare_q4_k", fake_prepare_q4)
    monkeypatch.setattr(gguf_module, "_xpu_prepare_q5_1", fake_prepare_q5_1)

    method = object.__new__(GGUFMoEXPUMethod)
    method.process_weights_after_loading(layer)

    assert layer.w2_xpu_qweight.shape == (2, 8, 32)
    assert layer.w2_xpu_scales.shape == (2, 8, 1)
    assert layer.w2_xpu_minimums.shape == (2, 8, 1)
    assert layer.w2_xpu_kind == "q5_1"
    assert not hasattr(layer, "w13_qweight")
    assert not hasattr(layer, "w2_qweight")


def test_xpu_gguf_q5_1_linear_dispatch_passes_scale_and_minimum(monkeypatch):
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
    layer._gguf_xpu_representations = [("q5_1", prefix, None, None)]

    call = {}

    def fake_grouped_mm(
        output, activations, weights, scales, minimums, rows, experts
    ):
        call.update(
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
        "gguf_q5_1_grouped_mm",
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
