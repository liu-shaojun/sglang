import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu, is_xpu


def _static_f32(t: torch.Tensor) -> torch.Tensor:
    """fp32 + contiguous for a *static* parameter, converting at most once.

    ``Tensor.contiguous()`` self-elides when the tensor is already contiguous,
    but ``Tensor.to()`` always goes through the dispatcher, so an unconditional
    ``.float()`` costs a real op per call even when it is a no-op.

    ``dt_bias`` reaches this path in fp16 while the kernel wants fp32, so the
    unconditional ``.float()`` was a real device copy per GDN layer per forward
    (30 copies/forward on this model, ~1.5 ms/forward in the MTP verify step)
    that re-derived the exact same values every time. The parameter is a
    ``[num_v_heads/tp]`` vector, so the converted copy costs ~128 B per layer
    (a few KB model-wide); it is attached to the parameter rather than replacing
    it so the triton fallback and the fused-decode path keep seeing the original
    dtype.
    """
    if t.dtype == torch.float32:
        return t.contiguous()
    cached = getattr(t, "_gdn_verify_f32", None)
    if cached is None:
        cached = t.float().contiguous()
        try:
            t._gdn_verify_f32 = cached
        except AttributeError:
            pass
    return cached


def _as_i32(t: torch.Tensor) -> torch.Tensor:
    """int32 + contiguous, without dispatching when the tensor already is both."""
    return t.contiguous() if t.dtype == torch.int32 else t.to(torch.int32).contiguous()


_VERIFY_DTYPES_LOGGED = False


def _log_one_shot_verify_dtypes(A_log, dt_bias, qsl, cache_idx, inter_idx):
    """One-shot report of the incoming dtypes on the ESIMD verify path.

    Which of these actually need converting decides whether the remaining cost
    is a real device copy (worth fixing at the producer) or just dispatcher
    overhead (already removed by the _as_* helpers above).
    """
    global _VERIFY_DTYPES_LOGGED
    if _VERIFY_DTYPES_LOGGED:
        return
    _VERIFY_DTYPES_LOGGED = True
    import logging

    logging.getLogger(__name__).warning(
        "[gdn_target_verify] incoming dtypes: A_log=%s dt_bias=%s "
        "query_start_loc=%s cache_indices=%s intermediate_state_indices=%s",
        A_log.dtype, dt_bias.dtype, qsl.dtype, cache_idx.dtype, inter_idx.dtype,
    )

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

if is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
elif is_cpu():
    from sgl_kernel.mamba import chunk_gated_delta_rule_cpu

    chunk_gated_delta_rule = chunk_gated_delta_rule_cpu
    fused_sigmoid_gating_delta_rule_update = (
        torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu
    )
elif is_xpu():
    from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    # Triton-XPU 3.7.0 cannot compile the block-pointer-heavy GDN extend
    # kernels (chunk_delta_h / chunk_fwd / chunk_o). Route extend() through a
    # pure-PyTorch reference implementation until the Intel Triton backend
    # supports these patterns. decode() / target_verify() still use Triton,
    # since those kernels compile successfully today.
    from sglang.srt.layers.attention.fla.chunk_torch_xpu import (
        chunk_gated_delta_rule_torch,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_torch


class TritonGDNKernel(LinearAttnKernelBase):
    """Triton-based kernel for GDN (Gated Delta Network) linear attention."""

    supports_packed_decode: bool = not is_cpu() and not is_npu()

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: fuse QKV extraction + gating + recurrent
        update into a single Triton kernel, eliminating intermediate tensors
        and extra kernel launches.

        Args:
            mixed_qkv: [B, qkv_dim] packed projection output after conv1d.
            a, b: [B, HV] gating inputs.
            A_log: [HV] log-space decay parameter.
            dt_bias: [HV] time-step bias.
            scale: attention scale factor (typically head_k_dim ** -0.5).
            ssm_states: [num_slots, HV, V, K] full state pool.
            cache_indices: [B] per-request state slot indices.
            num_v_heads: number of value heads (after TP sharding).
            head_v_dim: dimension per value head.

        Returns:
            output tensor of shape [1, B, HV, V] matching the existing
            decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        # Packed kernel expects output shape [B, 1, HV, V]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )

        # Convert [B, 1, HV, V] → [1, B, HV, V] to match existing output
        # layout. transpose() returns a view — zero cost.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        recurrent_state = ssm_states
        recurrent_state_indices_args = {"initial_state_indices": cache_indices}
        if is_npu() or is_cpu() or is_xpu():
            # These backends don't mutate the state pool in-place; they take
            # only the slice they need, return the new state, and let the
            # caller scatter it back (see gdn_backend.py::forward_extend).
            recurrent_state = ssm_states[cache_indices]
            recurrent_state_indices_args = {}

        # XPU fast path: use the ESIMD chunk_gated_delta_rule_extend kernel
        # when conditions match (head_dim == 128 and H_v % H_k == 0; covers
        # Qwen3.5-0.8B dense GDN with H_k=H_v=16 and Qwen3.5-4B grouped-value
        # GDN with H_k=16, H_v=32). Env-gated so rollback is a one-liner.
        import os as _os
        if (
            is_xpu()
            and _os.environ.get("SGL_XPU_GDN_EXTEND_ESIMD") == "1"
            and q.size(-1) == 128
            and v.size(-1) == 128
            and v.size(-2) % q.size(-2) == 0  # H_v % H_k == 0 (GQA on GDN)
            and hasattr(torch.ops, "eagle_ops")
            and hasattr(torch.ops.eagle_ops, "chunk_gated_delta_rule_extend")
        ):
            scale = float(q.size(-1)) ** -0.5
            # Kernel contract: (q, k, v, g, beta, initial_state, cu_seqlens,
            # scale, h_chunk_size) -> (out, last_state, h).
            # h (per-chunk intermediate states) is only materialised when
            # h_chunk_size > 0, i.e. when this batch actually has to write a
            # mamba track snapshot at a non-chunk-aligned position.
            # g is fp32 log-space decay; kernel expects exactly that.
            # initial_state is IN/OUT: kernel mutates it to last_state.
            h_chunk = int(kwargs.get("intermediate_chunk_size") or 0)
            state_in = recurrent_state.contiguous()
            out, last_state, h = torch.ops.eagle_ops.chunk_gated_delta_rule_extend(
                q.contiguous(), k.contiguous(), v.contiguous(),
                g.contiguous(), beta.contiguous(),
                state_in, query_start_loc.to(torch.int32).contiguous(),
                scale, h_chunk,
            )
            return out, last_state, (h if h_chunk > 0 else None)

        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            **(
                {"intermediate_chunk_size": kwargs["intermediate_chunk_size"]}
                if is_xpu() and kwargs.get("intermediate_chunk_size")
                else {}
            ),
            **recurrent_state_indices_args,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # XPU fast path: the triton GDN recurrence is numerically broken on
        # triton-XPU (same root cause as the decode/extend paths, which already
        # route to ESIMD). gdn_target_verify is the ESIMD port of the very
        # recurrence the decode fast-path uses, plus the two verify-only
        # behaviours this call needs: per-draft-token SSM snapshots into
        # `intermediate_states_buffer`, and no commit to the main SSM pool
        # (mamba_state_scatter does that from the intermediates).
        # It applies the q/k L2 norm and the sigmoid/softplus gating in-kernel,
        # matching use_qk_l2norm_in_kernel=True below. The state pool may be
        # fp32 or fp16/bf16 (SGLANG_MAMBA_SSM_DTYPE); the recurrence always runs
        # in fp32 registers either way. topk=1 (linear chain) only. Env-gated so
        # rollback is a one-liner.
        if (
            is_xpu()
            and envs.SGL_XPU_MTP_GDN_VERIFY.get()
            and q.size(-1) == 128
            and v.size(-1) == 128
            and v.size(-2) % q.size(-2) == 0  # H_v % H_k == 0 (GQA on GDN)
            and hasattr(torch.ops, "eagle_ops")
            and hasattr(torch.ops.eagle_ops, "gdn_target_verify")
        ):
            if ssm_states.dtype != intermediate_states_buffer.dtype:
                raise RuntimeError(
                    "gdn_target_verify needs the SSM state pool and the "
                    "intermediate snapshot buffer to share a dtype, got "
                    f"{ssm_states.dtype} vs {intermediate_states_buffer.dtype}."
                )
            # NOTE: must be a *contiguous* allocation, not empty_like(v): in the
            # verify step v can be a dense-but-permuted view (token-innermost),
            # and empty_like would inherit those strides while the kernel writes
            # its output linearly -- silently scrambling the result.
            out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
            # These five conversions used to run unconditionally, i.e. 5
            # aten::to dispatches per GDN layer per forward (150/forward on a
            # 30-GDN-layer model) even when the tensor already had the required
            # dtype. Tensor.contiguous() self-elides when the tensor is already
            # contiguous, but Tensor.to() does not -- it always dispatches. A
            # python dtype compare is far cheaper, and the parameters/index
            # tensors here are tiny, so there is nothing to cache: the correctly
            # typed ones now cost nothing at all.
            _log_one_shot_verify_dtypes(
                A_log, dt_bias, query_start_loc, cache_indices,
                intermediate_state_indices,
            )
            # q/k/v arrive as strided views carved out of the packed conv
            # output (torch.split on the last dim), so .contiguous() on each is
            # 3 real copies per GDN layer -- 90 per forward at 30 layers.
            # gdn_target_verify_packed reads the same values straight out of
            # mixed_qkv with a row stride + offset: bit-exact with the split
            # call (verified 0 diff) and no copies at all.
            _packed = kwargs.get("packed_qkv")
            if (
                _packed is not None
                and _packed.dim() == 2
                and _packed.stride(1) == 1
                and a.dim() == 2
                and b.dim() == 2
                and a.stride(1) == 1
                and b.stride(1) == 1
                and hasattr(torch.ops.eagle_ops, "gdn_target_verify_packed")
            ):
                # a/b are row-dense but may be halves of a packed [T, 2*Hv]
                # mixed_ba buffer; the kernel takes their real row stride, so
                # no .contiguous() copies here either.
                torch.ops.eagle_ops.gdn_target_verify_packed(
                    out,
                    _packed,
                    a,
                    b,
                    _static_f32(A_log),
                    _static_f32(dt_bias),
                    ssm_states,
                    intermediate_states_buffer,
                    _as_i32(query_start_loc),
                    _as_i32(cache_indices),
                    _as_i32(intermediate_state_indices),
                    int(cache_steps),
                    int(kwargs["num_k_heads"]),
                    int(kwargs["head_k_dim"]),
                    int(kwargs["num_v_heads"]),
                    int(kwargs["head_v_dim"]),
                )
                return out
            torch.ops.eagle_ops.gdn_target_verify(
                out,
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                a.contiguous(),
                b.contiguous(),
                _static_f32(A_log),
                _static_f32(dt_bias),
                ssm_states,
                intermediate_states_buffer,
                _as_i32(query_start_loc),
                _as_i32(cache_indices),
                _as_i32(intermediate_state_indices),
                int(cache_steps),
            )
            return out

        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            # target_verify specific parameters
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )
