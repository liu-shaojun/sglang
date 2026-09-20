"""CPU dispatch coverage for the XPU GGUF Q8_0 linear path."""

import torch

from sglang.srt.layers.quantization.gguf import GGUFLinearXPUMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_xpu_gguf_q8_0_linear_dispatch_preserves_compact_layout(monkeypatch):
    layer = torch.nn.Module()
    prefix = "_gguf_xpu_0"
    layer.register_buffer(
        f"{prefix}_weight", torch.empty((1, 72, 256), dtype=torch.int8)
    )
    layer.register_buffer(
        f"{prefix}_scales", torch.empty((1, 72, 8), dtype=torch.float16)
    )
    layer._gguf_xpu_representations = [("q8_0", prefix, None, None)]

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
        output.fill_(2)

    monkeypatch.setattr(
        torch.ops.sgl_kernel,
        "gguf_q8_0_grouped_mm",
        fake_grouped_mm,
        raising=False,
    )

    method = object.__new__(GGUFLinearXPUMethod)
    x = torch.ones((2, 3, 256), dtype=torch.bfloat16)
    output = method.apply(layer, x)

    assert output.shape == (2, 3, 72)
    assert output.dtype == torch.bfloat16
    assert torch.all(output == 2)
    assert call["activations"].shape == (6, 256)
    assert call["weights"].shape == (1, 72, 256)
    assert call["weights"].dtype == torch.int8
    assert call["scales"].shape == (1, 72, 8)
    assert call["scales"].dtype == torch.float16
    torch.testing.assert_close(call["rows"], torch.tensor([6], dtype=torch.int32))
    assert call["experts"] == 1
