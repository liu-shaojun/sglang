from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

# Split-K decode: number of KV splits for the hd512 (gemma4 global) ESIMD kernel.
# The kernel now derives its per-split chunk from the REAL seqlen (seqLens[b]) —
# see splitk_decode.h — so a single high G is optimal across ALL context lengths
# (idle splits early-return; each active split scans ceil(seqlen/G) tokens). G=64
# = 16 q-heads * 64 = 1024 work-items ≈ BMG's ~960 HW-thread ceiling; measured
# near-flat TPOT to 64k (44ms@32k, 50ms@64k) vs G=4/16 which grow with ctx.
# Overridable via env for A/B.
_SPLITK_G = int(os.environ.get("SGLANG_SPLITK_G", "64"))

# Debug gate: force the ESIMD decode fast paths (page_attn_decode + split-K) OFF
# so decode routes through flash_attn_with_kvcache (the fp16->bf16-casting wrapper).
# Used to localize the fp16+graph decode garble.
_DISABLE_ESIMD_DECODE = os.environ.get("SGLANG_DISABLE_ESIMD_DECODE", "0") == "1"

# Debug gate: force the hd256 page_attn_decode path OFF so sliding layers also use
# split-K (which accepts head_dim 256). Both are no-SLM/graph-capturable. Isolates
# whether the graph garble is in page_attn_decode vs split-K.
_DISABLE_PAGE_ATTN = os.environ.get("SGLANG_DISABLE_PAGE_ATTN", "0") == "1"

# Diagnostic: validate the SWA windowed page table against the pool bounds.
_DEBUG_SWA_WINDOW = os.environ.get("SGLANG_DEBUG_SWA_WINDOW", "0") == "1"
_swa_window_reports = 0


def _debug_swa_window(layer_id, page_table_pa, pa_seqlens, max_seq, key_cache, page_size):
    """Check the two ways the windowed page table can break the ESIMD kernel:
    an effective seq_len longer than the windowed table (kernel walks past the
    table) and page indices that fall outside the SWA pool."""
    global _swa_window_reports
    if _swa_window_reports >= 6:
        return
    import logging as _logging

    _log = _logging.getLogger(__name__)
    sl_max = int(pa_seqlens.max().item())
    pt_min = int(page_table_pa.min().item())
    pt_max = int(page_table_pa.max().item())
    pool_slots = int(key_cache.shape[0])
    max_slot = pt_max * page_size + page_size - 1 if pt_max < pool_slots else pt_max
    overrun = sl_max > max_seq
    oob = pt_min < 0 or max_slot >= pool_slots
    if overrun or oob or _swa_window_reports < 2:
        _swa_window_reports += 1
        _log.error(
            "[swa-win] layer=%d sl_max=%d max_seq=%d cols=%d pt_range=[%d,%d] "
            "pool_slots=%d overrun=%s oob=%s seqlens=%s",
            layer_id,
            sl_max,
            max_seq,
            page_table_pa.shape[1],
            pt_min,
            pt_max,
            pool_slots,
            overrun,
            oob,
            pa_seqlens.tolist()[:8],
        )

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionMetadata,
    make_local_attention_virtual_batches,
    merge_state_v2_wrapper,
    prepare_swa_spec_page_table_triton,
)
from sglang.srt.managers.schedule_batch import get_global_server_args
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from custom_esimd_kernels_sglang import eagle_page_attn_decode
from sgl_kernel import flash_mla_decode, flash_mla_get_workspace_size, merge_state_v2

# Proven-correct flat-NHD ESIMD decode kernel (token-granular kv_indptr/kv_indices),
# same op the triton backend's XPU ESIMD fast path uses. On this stack the paged
# eagle_page_attn_decode kernel is numerically wrong for GQA ratio=8 / single KV
# head, so the XPU graph decode path routes here instead. Gated by
# SGL_XPU_DECODE_SGLANG_ATTN (default on); falls back to eagle when unavailable.
_sglang_decode_attn_fn = None
_xpu_create_kv_indices_fn = None
try:
    from custom_esimd_kernels_sglang import sglang_decode_attn as _sglang_decode_attn_fn
    from custom_esimd_kernels_sglang import (
        xpu_create_kv_indices as _xpu_create_kv_indices_fn,
    )
except Exception:
    _sglang_decode_attn_fn = None
    _xpu_create_kv_indices_fn = None
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

_esimd_page_attn_decode = None
_splitk_decode_attention = None
try:
    from custom_esimd_kernels_sglang import eagle_page_attn_decode as _esimd_page_attn_decode
    from custom_esimd_kernels_sglang import splitk_decode_attention as _splitk_decode_attention
except ImportError:
    pass

# PTL-proven fp16 prefill SDPA DPAS kernel (custom_esimd_kernels_sglang, HD=256).
# It implements full causal attention only, so hybrid-SWA layers must stay on
# flash_attn_with_kvcache until the DPAS op accepts a sliding-window bound.
# Lazily resolved; gated by SGL_XPU_PREFILL_DPAS=1. Op namespace is
# custom_esimd_kernels_vllm (unchanged from the ported kernel).
_prefill_dpas_op = None
_prefill_dpas_tried = False


def _get_prefill_dpas_op():
    global _prefill_dpas_op, _prefill_dpas_tried
    if not _prefill_dpas_tried:
        _prefill_dpas_tried = True
        try:
            import custom_esimd_kernels_sglang.custom_esimd_kernels_prefill_dpas  # noqa: F401 — registers the op
            _prefill_dpas_op = torch.ops.custom_esimd_kernels_vllm.esimd_sdpa_prefill_dpas
        except Exception:
            _prefill_dpas_op = None
    return _prefill_dpas_op

try:
    from custom_esimd_kernels_sglang import eagle_page_attn_decode_temp_size
except ImportError:
    # Backward-compatible fallback for older custom_esimd_kernels_sglang builds
    # that export eagle_page_attn_decode but not *_temp_size.
    def eagle_page_attn_decode_temp_size(
        batches: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
    ) -> int:
        hidden_dim_p = ((max_seq_len + 63) // 64) * 64
        hidden_dim_p_max = (max_seq_len + 63) // 64
        reduce_count = (max_seq_len + 1023) // 1024
        gqa_ratio = num_q_heads // num_kv_heads
        sz_p = batches * num_q_heads * hidden_dim_p
        sz_group_max = batches * num_q_heads * hidden_dim_p_max
        sz_global_max = batches * gqa_ratio * num_kv_heads
        sz_global_poll_p = batches * gqa_ratio * num_kv_heads
        sz_out_temp = batches * reduce_count * head_dim * num_q_heads
        sz_global_softmax_sum = batches * reduce_count * num_q_heads
        return (
            sz_p
            + sz_group_max
            + sz_global_max
            + sz_global_poll_p
            + sz_out_temp
            + sz_global_softmax_sum
        )


class XPUAttentionBackend(AttentionBackend):
    """XPU FlashAttention backend, currently based on FlashAttentionBackend, will be refactored later.

    TODO:
    - Prefill and Decode disaggregation, currently only chunked prefill is supported
    - Speculative Decoding support
    - XPU Graph capture/replay for decode paths is supported; extend paths
      still use the eager fallback.
    - MLA Prefill support
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.forward_metadata: FlashAttentionMetadata = None
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.num_attention_heads = (
            model_runner.model_config.hf_text_config.num_attention_heads
        )
        self.tp_size = model_runner.tp_size
        assert self.num_attention_heads % self.tp_size == 0
        self.num_local_heads = self.num_attention_heads // self.tp_size
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.head_dim = getattr(model_runner.model_config.hf_text_config, "head_dim", None)
        if self.head_dim is None:
            self.head_dim = (
                model_runner.model_config.hf_text_config.hidden_size // self.num_attention_heads
            )
        total_kv_heads = getattr(
            model_runner.model_config.hf_text_config, "num_key_value_heads", self.num_local_heads
        )
        self.num_local_kv_heads = max(1, total_kv_heads // self.tp_size)
        self.skip_prefill = skip_prefill
        self.is_hybrid_swa = model_runner.is_hybrid_swa
        self.use_sliding_window_kv_pool = (
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)
            and model_runner.token_to_kv_pool.swa_layer_nums > 0
        )
        if self.use_sliding_window_kv_pool:
            self.token_to_kv_pool = model_runner.token_to_kv_pool
        if self.is_hybrid_swa:
            self.full_to_swa_index_mapping = (
                model_runner.token_to_kv_pool.full_to_swa_index_mapping
            )
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id

        # Local attention settings
        self.attention_chunk_size = (
            model_runner.attention_chunk_size
            if hasattr(model_runner, "attention_chunk_size")
            else None
        )

        # For each layer, the sliding_window_size can be different. This is only used for preparing SWA metadata.
        # We use `layer.sliding_window_size` to decide whether to use SWA for each layer.
        self.sliding_window_size = model_runner.sliding_window_size
        self.has_swa = (
            self.sliding_window_size is not None and self.sliding_window_size > -1
        )
        # XPU graph capture/replay state. Populated by init_cuda_graph_state.
        self._graph_state: dict = {}

    def _build_sglang_decode_attn_inputs_eager(
        self,
        forward_batch: ForwardBatch,
        metadata,
        tp_q_head_num: int,
        head_dim: int,
        sliding_window_size: int = -1,
    ):
        """Build the flat-NHD kv_indptr/kv_indices/temp_p inputs that
        sglang_decode_attn needs, without requiring XPU device-graph capture
        (SGL_XPU_ENABLE_GRAPH=1) to be enabled. init_forward_metadata_capture
        only builds these when graphs are on, which left sglang_decode_attn
        (the proven-correct kernel for GQA ratio=8 / single-KV-head configs)
        unreachable in eager decode -- so decode silently fell back to the
        numerically-wrong eagle_page_attn_decode / flash kernel instead.

        Optimization (P0+P1) for the disable-XPU-graph path: the scratch
        buffers (kv_indptr / kv_indices / temp_p) are kept as a single
        *persistent, grow-only* cache reused across decode steps instead of
        being re-allocated every step, and the per-step
        ``int(kv_indptr[-1].item())`` device->host sync (which stalled the
        CPU-GPU pipeline on every full-attention layer) is removed:
        kv_indices is sized to the upper bound ``bs * max_seq_len_k`` (a CPU
        int already available on ``metadata``, no device readback needed).
        xpu_create_kv_indices only writes into the ranges indexed by
        kv_indptr, so an over-sized buffer is safe, and
        sglang_decode_attn only reads the kv_indptr-delimited ranges."""
        bs = forward_batch.batch_size
        # kv_indptr/kv_indices/temp_p depend only on this step's cache_seqlens
        # (batch-level), not on the attention layer, so they are identical for
        # every full-attention layer in a decode step. ``metadata`` is a fresh
        # object built once per step in init_forward_metadata, so memoizing the
        # built inputs on it lets the first full-attn layer build them and the
        # remaining layers reuse them -- removing ~9x (cumsum + kv_indptr fill +
        # create_flashinfer_kv_indices) dispatched ops per step on the
        # disable-XPU-graph path.
        # Sliding-window layers must NOT reuse the full-attention inputs: their
        # KV lives in the (much smaller) SWA pool and is addressed by translated
        # slot ids, so they get their own memoized variant.
        is_swa = sliding_window_size is not None and sliding_window_size > -1
        swa_index_map = getattr(self, "full_to_swa_index_mapping", None)
        if swa_index_map is None:
            swa_index_map = getattr(
                self.token_to_kv_pool, "full_to_swa_index_mapping", None
            )
        translate_swa = (
            is_swa and self.use_sliding_window_kv_pool and swa_index_map is not None
        )
        cache_attr = "_eager_kv_inputs_swa" if is_swa else "_eager_kv_inputs"
        cached = getattr(metadata, cache_attr, None)
        if cached is not None and cached[0].numel() == bs + 1:
            return cached
        device = metadata.cache_seqlens_int32.device
        _SPLIT_TILE = 64
        _MAX_N_SPLITS = 256
        graph_max_seq = _SPLIT_TILE * _MAX_N_SPLITS  # 16384

        # Upper bound on total kv entries this step: sum(seq_lens) <= bs * max_seq_len_k.
        # max_seq_len_k is a plain python int already computed on the CPU-side
        # seq_lens_cpu in init_forward_metadata, so reading it here costs no
        # device synchronization (unlike kv_indptr[-1].item()).
        max_seq_len_k = getattr(metadata, "max_seq_len_k", None)
        if not isinstance(max_seq_len_k, int) or max_seq_len_k <= 0:
            max_seq_len_k = self.max_context_len
        # For a sliding layer only the last `window` tokens are resident in the
        # SWA pool; everything older was evicted and its full->swa mapping is
        # stale. Walking [0, seq_len) there reads other tokens' KV or, once the
        # full-pool slot ids exceed the SWA pool size, reads out of bounds --
        # which is what poisoned decode with NaN.
        window_tokens = (sliding_window_size + 1) if is_swa else 0
        if is_swa:
            need_kv = max(bs * window_tokens, 1)
        else:
            need_kv = max(bs * max_seq_len_k, 1)
        need_temp = max(bs * tp_q_head_num * _MAX_N_SPLITS * (1 + 1 + 256), 1)

        # Persistent grow-only scratch: allocate once, reuse (and only grow)
        # across decode steps. No per-step torch.empty, no per-bs rebuild.
        scratch_attr = (
            "_sglang_decode_eager_scratch_swa"
            if is_swa
            else "_sglang_decode_eager_scratch"
        )
        cache = getattr(self, scratch_attr, None)
        if (
            cache is None
            or cache["kv_indptr"].numel() < bs + 1
            or cache["kv_indices"].numel() < need_kv
            or cache["temp_p"].numel() < need_temp
        ):
            cache = {
                "kv_indptr": torch.zeros(
                    max(bs + 1, 1), dtype=torch.int32, device=device
                ),
                "kv_indices": torch.empty(
                    need_kv, dtype=torch.int32, device=device
                ),
                "temp_p": torch.empty(
                    need_temp, dtype=torch.float32, device=device
                ),
            }
            setattr(self, scratch_attr, cache)

        kv_indptr = cache["kv_indptr"][: bs + 1]
        kv_indices = cache["kv_indices"]
        temp_p = cache["temp_p"]

        seqlens = metadata.cache_seqlens_int32
        kv_start_idx = None
        if is_swa:
            # Keep only the trailing `window` tokens: length = min(seq_len,
            # window), starting at seq_len - length.
            seqlens = torch.clamp(seqlens, max=window_tokens)
            kv_start_idx = metadata.cache_seqlens_int32 - seqlens

        kv_indptr[0] = 0
        torch.cumsum(
            seqlens,
            dim=0,
            dtype=torch.int32,
            out=kv_indptr[1:],
        )
        _xpu_create_kv_indices_fn(
            self.req_to_token,
            forward_batch.req_pool_indices,
            seqlens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            window_tokens if is_swa else max_seq_len_k,
        )
        if translate_swa:
            # req_to_token holds FULL-pool slot ids; the sliding layers read the
            # SWA pool, so the ids must go through the same full->swa slot map
            # the paged path applies to its page table. Untranslated ids index a
            # different (and for large ids, out-of-range) buffer.
            n_used = min(kv_indices.numel(), max(bs * window_tokens, 1))
            used = kv_indices[:n_used]
            # Entries past kv_indptr[-1] are stale scratch; clamp so the gather
            # below can never run off the mapping table.
            used.clamp_(0, swa_index_map.numel() - 1)
            used.copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(used).to(
                    torch.int32
                )
            )
        # Pass the REAL max sequence length, not the constant graph_max_seq.
        # graph_max_seq is only the scratch-shape bound (SPLIT_TILE*MAX_N_SPLITS);
        # feeding it as max_seq_len made the kernel size its split grid for
        # 16384 tokens regardless of the actual length, so every token past
        # 16384 was silently dropped (and the kernel-side n_splits assert could
        # never fire). The kernel now grows its per-work-item tile instead, so
        # n_splits still stays <= MAX_N_SPLITS and temp_p sizing is unchanged.
        attn_max_seq = min(max_seq_len_k, window_tokens) if is_swa else max_seq_len_k
        result = (kv_indptr, kv_indices, temp_p, int(max(attn_max_seq, 1)))
        try:
            setattr(metadata, cache_attr, result)
        except Exception:
            pass
        return result

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Pre-allocate stable device buffers for XPU graph capture/replay."""
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        device = self.device

        # sglang_decode_attn (flat-NHD, token-granular) scratch sizing.
        # temp_p = batches*Hq*n_splits*(1 + 1 + 256); n_splits capped at 256.
        _SPLIT_TILE = 64
        _MAX_N_SPLITS = 256
        self._sglang_decode_graph_max_seq = _SPLIT_TILE * _MAX_N_SPLITS  # 16384
        _sglang_temp_numel = (
            max_bs * self.num_local_heads * _MAX_N_SPLITS * (1 + 1 + 256)
        )

        self._graph_state = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=device),
            "cu_seqlens_q_decode": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=device
            ),
            "cu_seqlens_k": torch.zeros(max_bs + 1, dtype=torch.int32, device=device),
            "page_table": torch.zeros(
                max_bs, max_num_pages, dtype=torch.int32, device=device
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=device
            ),
            "temp_p": torch.empty(
                eagle_page_attn_decode_temp_size(
                    max_bs,
                    self.num_local_heads,
                    self.num_local_kv_heads,
                    self.head_dim,
                    self.max_context_len,
                ),
                dtype=torch.float32,
                device=device,
            ),
            # Token-granular KV index buffers for sglang_decode_attn. Stable
            # data_ptr across graph replays; filled each step in the replay hook.
            "kv_indptr": torch.zeros(max_bs + 1, dtype=torch.int32, device=device),
            "kv_indices": torch.zeros(
                max_bs * self.max_context_len, dtype=torch.int32, device=device
            ),
            "sglang_temp_p": torch.empty(
                _sglang_temp_numel, dtype=torch.float32, device=device
            ),
            "captured": {},
        }

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        """Prepare forward metadata outside the graph capture region."""
        if not self._graph_state or not forward_batch.forward_mode.is_decode_or_idle():
            self.init_forward_metadata(forward_batch)
            return

        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens[:bs]
        seq_lens_cpu = (
            forward_batch.seq_lens_cpu[:bs]
            if forward_batch.seq_lens_cpu is not None
            else None
        )
        req_pool_indices = forward_batch.req_pool_indices[:bs]

        state = self._graph_state
        metadata = FlashAttentionMetadata()
        metadata.cache_seqlens_int32 = state["cache_seqlens"][:bs]
        metadata.cu_seqlens_q = state["cu_seqlens_q_decode"][: bs + 1]
        metadata.cu_seqlens_k = state["cu_seqlens_k"][: bs + 1]
        metadata.page_table = state["page_table"][:bs, :]
        metadata.max_seq_len_q = 1

        metadata.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))
        torch.cumsum(
            metadata.cache_seqlens_int32,
            dim=0,
            dtype=torch.int32,
            out=metadata.cu_seqlens_k[1:],
        )

        max_len = (
            int(seq_lens_cpu.max().item())
            if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0
            else int(seq_lens.max().item()) if seq_lens.numel() > 0 else 1
        )
        metadata.max_seq_len_k = max(1, max_len)

        max_seq_pages = (metadata.max_seq_len_k + self.page_size - 1) // self.page_size
        if max_seq_pages > 0:
            strided_indices = state["strided_indices"][:max_seq_pages]
            pt = (
                self.req_to_token[
                    req_pool_indices[:, None],
                    strided_indices[None, :],
                ]
                // self.page_size
            )
            metadata.page_table[:bs, :max_seq_pages].copy_(pt)

        # Token-granular kv_indptr/kv_indices for sglang_decode_attn (flat-NHD).
        # Written into pre-allocated graph-stable buffers so their data_ptr is
        # constant across replays. kv_indptr[i+1] = prefix-sum of seq_lens.
        if _sglang_decode_attn_fn is not None:
            kv_indptr = state["kv_indptr"][: bs + 1]
            kv_indptr[0] = 0
            torch.cumsum(
                metadata.cache_seqlens_int32,
                dim=0,
                dtype=torch.int32,
                out=kv_indptr[1:],
            )
            kv_indices = state["kv_indices"]
            _xpu_create_kv_indices_fn(
                self.req_to_token,
                req_pool_indices,
                metadata.cache_seqlens_int32,
                kv_indptr,
                None,
                kv_indices,
                self.max_context_len,
            )
            metadata.kv_indptr = kv_indptr
            metadata.kv_indices = kv_indices

        state["captured"][bs] = metadata
        self.forward_metadata = metadata

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recordable setup hook. XPU uses preallocated metadata buffers."""
        if not self._graph_state:
            self.init_forward_metadata(forward_batch)
            return
        self.forward_metadata = self._graph_state["captured"].get(
            forward_batch.batch_size, self.forward_metadata
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Pre-allocate stable device buffers for XPU graph capture + replay."""
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        device = self.device

        self._graph_state = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=device),
            "cu_seqlens_q": torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            "page_table": torch.zeros(max_bs, max_num_pages, dtype=torch.int32, device=device),
            "strided_indices": torch.arange(0, self.max_context_len, self.page_size, device=device),
        }
        if self.use_sliding_window_kv_pool:
            self._graph_state["swa_page_table"] = torch.zeros(
                max_bs, max_num_pages, dtype=torch.int32, device=device,
            )
            # SWA-pool KV write location (one slot per request per decode step).
            # Must be address-stable for graph capture/replay (the captured
            # set_kv_buffer writes to its capture-time address).
            self._graph_state["swa_out_cache_loc"] = torch.zeros(
                max_bs, dtype=torch.int64, device=device,
            )

    def init_forward_metadata_out_graph(self, forward_batch: ForwardBatch, in_capture: bool = False):
        """Prepare forward_metadata for graph capture or replay (runs outside graph.capture())."""
        if not forward_batch.forward_mode.is_decode_or_idle():
            return
        if not hasattr(self, "_graph_state") or not self._graph_state:
            return

        bs = forward_batch.batch_size
        state = self._graph_state
        seq_lens = forward_batch.seq_lens[:bs]

        metadata = FlashAttentionMetadata()
        metadata.cache_seqlens_int32 = state["cache_seqlens"][:bs]
        metadata.cu_seqlens_q = state["cu_seqlens_q"][:bs + 1]
        metadata.page_table = state["page_table"][:bs, :]
        metadata.max_seq_len_q = 1

        # Populate cache_seqlens from real seq_lens
        metadata.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))
        max_len = int(seq_lens.max().item()) if seq_lens.numel() else 1
        metadata.max_seq_len_k = max_len

        # Populate page_table. NOTE on ordering (must match eager
        # init_forward_metadata): the SWA translation
        # (translate_loc_from_full_to_swa) maps full-pool KV *slot* indices ->
        # SWA-pool slot indices, so it MUST be applied to the RAW req_to_token
        # slot values BEFORE dividing by page_size. The earlier version divided
        # first and then translated page-granularity indices through a slot-index
        # map -> wrong SWA page table -> the 50 sliding layers read the wrong KV
        # -> decode garbled from the first step. Build raw slots first, translate,
        # then stride + //page_size for both full and SWA tables.
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size
        if max_seq_pages > 0:
            req_pool_indices = forward_batch.req_pool_indices[:bs]
            # Raw token-granularity KV slots for [0, max_len).
            raw_slots = self.req_to_token[req_pool_indices, :max_len]

            # Full-pool page table: stride by page_size then divide.
            strided_indices = state["strided_indices"][:max_seq_pages]
            pt = (
                self.req_to_token[req_pool_indices[:, None], strided_indices[None, :]]
                // self.page_size
            )
            metadata.page_table[:bs, :max_seq_pages].copy_(pt)

            # SWA page table: translate RAW slots first, then stride + divide.
            if self.use_sliding_window_kv_pool:
                swa_slots = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    raw_slots
                ).to(torch.int32)
                swa_pt = swa_slots[:, ::self.page_size] // self.page_size
                swa_buf = state["swa_page_table"][:bs, :]
                # Write into the persistent (address-stable) graph buffer so the
                # captured sliding layers read it at a fixed address across replays.
                swa_buf[:, :swa_pt.shape[1]].copy_(swa_pt)
                metadata.swa_page_table = swa_buf

        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            swa_loc = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )
            n = swa_loc.shape[0]
            loc_buf = state["swa_out_cache_loc"][:n]
            loc_buf.copy_(swa_loc)
            metadata.swa_out_cache_loc = loc_buf

        self.forward_metadata = metadata

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """No-op: metadata already set by init_forward_metadata_out_graph."""
        pass

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        import os
        _debug = os.environ.get("SGLANG_HICACHE_DEBUG", "0") == "1"
        if _debug:
            print(f"[XPU_BACKEND] init_forward_metadata: enter, mode={forward_batch.forward_mode}", flush=True)
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device
        if _debug:
            print(f"[XPU_BACKEND] init_forward_metadata: created metadata, batch_size={batch_size}", flush=True)

        if forward_batch.forward_mode.is_decode_or_idle():
            # Draft Decode
            if forward_batch.spec_info is not None:
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]

                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )
                    # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                    cache_loc = forward_batch.out_cache_loc.view(
                        -1, self.speculative_num_steps
                    )
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

            # TODO: we need to test this part for llama 4 eagle case
            self._init_local_attn_metadata(forward_batch, metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + self.speculative_num_draft_tokens
                ).to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    forward_batch.seq_lens_cpu.max().item()
                    + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._init_local_attn_metadata(forward_batch, metadata, device)
            else:
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # create expand page table
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)

                # shift table indices to avoid padding
                # non_masked_page_table [[8, 9, 10],   mask (display with int format) [[1, 0, 0],
                #                        [8, 9, 10],                                   [1, 1, 0],
                #                        [8, 9, 10]]                                   [1, 0, 1]]
                # if masked with padding [[8, 0, 0],   our mask without padding       [[8, 9, 10],
                #                        [8, 9, 0],                                    [8, 9, 10],
                #                        [8, 0, 10]]                                   [8, 10, 9]]
                # note here cache_seqlens_int32 is [1, 2, 2] so extra page indices will be ignored in each row
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                # Build keys: if an entry is valid (mask==True), keep its original index;
                # if not, add self.speculative_num_draft_tokens so that it sorts after all valid entries.
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                self.forward_metadata_spec_decode_expand = metadata_expand

                if self.has_swa:
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand
                    )

        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
            include_draft_extend_v2=True
        ):
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # Setup local attention if enabled
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._init_local_attn_metadata(forward_batch, metadata, device)

        # Encoder metadata for cross attention
        if forward_batch.encoder_lens is not None:
            assert (
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()
            metadata.encoder_page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            metadata.page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]

        # Translate full-pool indices to SWA-pool indices for hybrid models
        if self.use_sliding_window_kv_pool:
            # flash_attn_with_kvcache requires int32 page tables; the SWA index
            # mapping is int64, so cast (matches flashattention_backend.py).
            metadata.swa_page_table = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    metadata.page_table
                ).to(torch.int32)
            )
            if forward_batch.out_cache_loc is not None:
                metadata.swa_out_cache_loc = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        forward_batch.out_cache_loc
                    )
                )

        if self.use_mla:
            workspace_size = flash_mla_get_workspace_size(
                max_seq_len=self.max_context_len,
                num_batches=batch_size,
                num_heads=self.num_local_heads,
                page_size=self.page_size,
                num_kv_splits=-1,
            )
            if (
                not hasattr(self, "workspace")
                or self.workspace.numel() < workspace_size
            ):
                self.workspace = torch.empty(
                    workspace_size, device=self.device, dtype=torch.uint8
                )

        # Translate full-pool indices to SWA-pool indices for hybrid models
        if self.use_sliding_window_kv_pool:
            # flash_attn_with_kvcache requires int32 page tables; the SWA index
            # mapping is int64, so cast (matches flashattention_backend.py).
            metadata.swa_page_table = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    metadata.page_table
                ).to(torch.int32)
            )
            if forward_batch.out_cache_loc is not None:
                metadata.swa_out_cache_loc = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        forward_batch.out_cache_loc
                    )
                )

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )

            if self.use_sliding_window_kv_pool and metadata.swa_page_table is not None:
                metadata.swa_page_table = (
                    metadata.swa_page_table[:, self.strided_indices] // self.page_size
                )

            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if k is None and v is None:
            # Cross-layer KV sharing (Gemma 4): the layer reuses another
            # layer's KV cache. The paged kernel reads K/V directly via
            # page_table, and pool.get_kv_buffer(layer.layer_id) routes
            # to the correct sub-pool because RadixAttention is initialized
            # with layer_id=kv_shared_layer_index for shared layers. No
            # materialization needed; just skip the write path.
            pass
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if self.use_sliding_window_kv_pool:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    )
                elif not self.use_mla:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        is_hybrid_swa = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_hybrid_swa else (-1, -1)

        # currently no FP8 KV cache supported
        k_descale, v_descale = None, None
        # # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # # has corresponding quantization method so that layer.k_scale is not None,
        # # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        # if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
        #     if layer.k_scale is not None:
        #         descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        #         k_descale = layer.k_scale.expand(descale_shape)
        #         v_descale = layer.v_scale.expand(descale_shape)
        #     q = q.to(self.kv_cache_dtype)
        #     q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
        #     k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = not layer.is_cross_attention

        # Check if we should use local attention
        use_local_attn = (
            self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # We do cascade attention for Target Verify with topk > 1
        # We don't use cascade attention for Sliding Window Attention:
        # - Different window sizes should be passed in for each q in the first stage of cascade attention, but FA3 interface doesn't support pass in a list of window sizes.
        # - The overhead of duplicated computation of the common prefix part is small for sliding window layers (seq_len <= window_size), so we can just expand it.
        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and not is_hybrid_swa
        )

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        # Get the appropriate page table based on whether we're using local attention
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif is_hybrid_swa and metadata.swa_spec_metadata is not None:
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
        else:
            page_table = metadata.page_table
            if is_hybrid_swa and self.use_sliding_window_kv_pool:
                if metadata.swa_page_table is not None:
                    page_table = metadata.swa_page_table
                else:
                    page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        metadata.page_table
                    ).to(torch.int32)
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k

        # Use Flash Attention for prefill
        if not self.use_mla:
            # Do multi-head attention
            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            # For full-attention fp16 HD=256 prefill, optionally prefer the
            # PTL-proven DPAS SDPA kernel. The op has no sliding-window argument:
            # routing an SWA layer through it attends [0, q_pos] instead of
            # [q_pos-window, q_pos] and can read evicted SWA mappings.
            _dpas = (
                _get_prefill_dpas_op()
                if (
                    os.environ.get("SGL_XPU_PREFILL_DPAS") == "1"
                    and q.dtype == torch.float16
                    and layer.head_dim == 256
                    and not layer.is_cross_attention
                    and not use_cascade_attn
                    and not is_hybrid_swa
                )
                else None
            )
            if _dpas is not None:
                result = _dpas(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    key_cache,
                    value_cache,
                    cu_seqlens_q.to(torch.int32),
                    cache_seqlens.to(torch.int32),
                    causal,
                    float(layer.scaling),
                    page_table.to(torch.int32),
                )
            else:
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    **kwargs,
                )

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    **kwargs,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                # Do multi-head attention with chunked prefix cache
                if forward_batch.attn_attend_prefix_cache:
                    assert not get_global_server_args().disable_chunked_prefix_cache
                    # MHA for chunked prefix kv cache when running model with MLA
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert forward_batch.mha_return_lse
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                    )
                else:
                    # MHA for extend part of sequence without attending prefix kv cache
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=metadata.cu_seqlens_q,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=forward_batch.mha_return_lse,
                    )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                # Do absorbed multi-latent attention
                kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                    q.dtype
                )
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=None,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result

        out = o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return out

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is None and v is None:
            # Cross-layer KV sharing (Gemma 4): see forward_extend for details.
            pass
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if self.use_sliding_window_kv_pool:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    )
                elif not self.use_mla:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    k_rope_val = (
                        k_rope if k_rope is not None else k[:, :, layer.v_head_dim :]
                    )
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope_val,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # When Spec Decode enabled, forward_decode would be called with two mode:
        # 1. DRAFT_DECODE: we enable cascade attention when top_k > 1
        # 2. IDLE: we don’t need cascade attention, spec_info will be none in this case
        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        causal = not layer.is_cross_attention

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        if not self.use_mla:
            # Do multi-head attention

            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            is_swa_layer = (
                layer.sliding_window_size is not None and layer.sliding_window_size > -1
            )
            page_table = metadata.page_table
            # For SWA layers on hybrid models, use the translated
            # SWA-pool page table so KV reads hit the correct pool.
            if is_swa_layer and self.use_sliding_window_kv_pool:
                if metadata.swa_page_table is not None:
                    page_table = metadata.swa_page_table
                else:
                    page_table = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            metadata.page_table
                        ).to(torch.int32)
                    )

            cache_seqlens = metadata.cache_seqlens_int32
            cu_seqlens_k = metadata.cu_seqlens_k
            max_seqlen_q = metadata.max_seq_len_q
            q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

            if layer.is_cross_attention:
                # Always use non-chunked logic for cross-attention
                o = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            elif use_local_attn:
                # Use chunked (local) attention batching for self-attention
                o = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            elif (
                not use_cascade_attn
                and _sglang_decode_attn_fn is not None
                and layer.head_dim == 256
                and layer.tp_q_head_num % layer.tp_k_head_num == 0
                and os.environ.get("SGL_XPU_DECODE_SGLANG_ATTN", "1") == "1"
            ):
                # Proven-correct flat-NHD token-granular decode kernel. The paged
                # eagle_page_attn_decode is numerically wrong for GQA ratio=8 /
                # single-KV-head on this stack, so route decode here instead.
                # Prefer the graph-captured stable buffers when XPU device graphs
                # are on (SGL_XPU_ENABLE_GRAPH=1); otherwise build the same
                # flat-NHD kv_indptr/kv_indices/temp_p inputs on the fly for
                # eager decode (they were previously only ever built inside
                # init_forward_metadata_capture, which made this branch
                # unreachable without graphs and silently pushed decode into the
                # numerically-wrong kernel below).
                B = forward_batch.batch_size
                k_buf = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buf = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                q_fp16 = q_reshaped.view(B, layer.tp_q_head_num, layer.head_dim)
                if q_fp16.dtype != torch.float16:
                    q_fp16 = q_fp16.to(torch.float16)
                if k_buf.dtype != torch.float16:
                    k_buf = k_buf.to(torch.float16)
                    v_buf = v_buf.to(torch.float16)
                o_fp16 = torch.empty_like(q_fp16)
                if self._graph_state.get("temp_p") is not None and getattr(
                    metadata, "kv_indices", None
                ) is not None:
                    kv_indptr = metadata.kv_indptr
                    kv_indices = metadata.kv_indices
                    temp_p = self._graph_state["sglang_temp_p"]
                    # Real length, not the scratch-shape constant -- see
                    # _build_sglang_decode_attn_inputs_eager for why.
                    _msk = getattr(metadata, "max_seq_len_k", None)
                    if not isinstance(_msk, int) or _msk <= 0:
                        _msk = self._sglang_decode_graph_max_seq
                    graph_max_seq = int(_msk)
                else:
                    (
                        kv_indptr,
                        kv_indices,
                        temp_p,
                        graph_max_seq,
                    ) = self._build_sglang_decode_attn_inputs_eager(
                        forward_batch,
                        metadata,
                        layer.tp_q_head_num,
                        layer.head_dim,
                        layer.sliding_window_size,
                    )
                _sglang_decode_attn_fn(
                    q_fp16,
                    k_buf,
                    v_buf,
                    kv_indptr,
                    kv_indices,
                    o_fp16,
                    float(layer.scaling),
                    temp_p,
                    graph_max_seq,
                )
                o = o_fp16.view(-1, layer.tp_q_head_num, layer.head_dim)
            elif self._graph_state.get("temp_p") is not None and not use_cascade_attn:
                o = torch.empty_like(q_reshaped)
                eagle_page_attn_decode(
                    q_reshaped,
                    key_cache,
                    value_cache,
                    page_table,
                    cache_seqlens,
                    o,
                    max_seqlen_q,
                    metadata.max_seq_len_k,
                    self._graph_state["temp_p"],
                )
            else:
                is_swa_layer = (
                    layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                )

                page_table = metadata.page_table
                # For SWA layers on hybrid models, use the translated
                # SWA-pool page table so KV reads hit the correct pool.
                if is_swa_layer and self.use_sliding_window_kv_pool:
                    if metadata.swa_page_table is not None:
                        page_table = metadata.swa_page_table
                    else:
                        page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                metadata.page_table
                            ).to(torch.int32)
                        )

                cache_seqlens = metadata.cache_seqlens_int32
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                # Sliding-window (SWA) layers: every decode kernel here reads the
                # full [0, seq_len) page table. Once seq_len > window, positions
                # older than the window have been evicted from the SWA pool and
                # their full->swa mapping points at a stale slot, so those reads
                # return another token's KV (or land outside the pool) and poison
                # the output — observed as all-NaN hidden states on the first
                # sliding layer whose garbage happens to overflow.
                #
                # Fix WITHOUT any kernel change: page-align the page table to the
                # LAST `window` tokens and pass the reduced seq_len, so the kernel
                # only ever reads resident, in-window KV. The SWA pool evicts per
                # page, so the window-boundary page is resident whenever it holds
                # any in-window token; floor-aligning the start therefore reads
                # only valid slots (it may attend up to page_size-1 extra,
                # still-resident, slightly-older tokens — a benign superset of
                # FA3's window_size=(sliding_window_size, 0)).
                #
                # This must be applied to BOTH decode kernels: page_attn_decode is
                # absent from some sgl-kernel builds, and the split-K fallback
                # accepts head_dim 256 and would otherwise silently read evicted
                # slots. The windowed table depends only on cache_seqlens + the
                # SWA page table, which are identical across all ~50 SWA layers in
                # a step, so compute it ONCE per step and cache it on the
                # (per-step-fresh) metadata object.
                swa_window_view = None
                if (
                    is_swa_layer
                    and max_seqlen_q == 1
                    and layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                    and metadata.max_seq_len_k > (layer.sliding_window_size + 1)
                ):
                    swa_window_view = getattr(metadata, "_swa_win_cache", None)
                    if swa_window_view is None:
                        window_tokens = layer.sliding_window_size + 1
                        ps = self.page_size
                        n_win_pages = (window_tokens - 1) // ps + 2
                        n_cols = page_table.shape[1]
                        # bs==1 takes a host-side fast path: the window is a
                        # CONTIGUOUS page span, so slice the page table (a
                        # zero-copy view) instead of building an index tensor and
                        # gathering. Microbench: 74us (gather) -> 8us (slice).
                        if forward_batch.batch_size == 1:
                            start_pg = (
                                max(metadata.max_seq_len_k - window_tokens, 0)
                                // ps
                            )
                            end_pg = min(start_pg + n_win_pages, n_cols)
                            pt_win = page_table[:, start_pg:end_pg]
                            sl_win = cache_seqlens - (start_pg * ps)
                        else:
                            start_page = (
                                torch.clamp(cache_seqlens - window_tokens, min=0) // ps
                            )
                            col = start_page.to(torch.int64).unsqueeze(
                                1
                            ) + torch.arange(
                                n_win_pages,
                                device=page_table.device,
                                dtype=torch.int64,
                            )
                            col = col.clamp_(max=n_cols - 1)
                            pt_win = torch.gather(page_table, 1, col)
                            sl_win = (cache_seqlens - start_page * ps).to(torch.int32)
                        swa_window_view = (pt_win, sl_win, pt_win.shape[1] * ps)
                        metadata._swa_win_cache = swa_window_view
                    if _DEBUG_SWA_WINDOW:
                        _debug_swa_window(
                            layer.layer_id,
                            swa_window_view[0],
                            swa_window_view[1],
                            swa_window_view[2],
                            key_cache,
                            self.page_size,
                        )

                # ESIMD page_attn_decode fast path: no SLM, graph-capturable.
                # Fires for head_dim==256 layers, which for gemma4 are the SWA
                # (sliding-window) layers (global layers have head_dim=512 and
                # take the split-K path below). SWA windowing is handled inside
                # the kernel via the `window` param (see the SWA branch below):
                # positions outside the last `window` tokens are masked to -inf,
                # matching FA3's window_size semantics.
                # The kernel's GQA tiling requires q_heads/kv_heads to be a
                # multiple of 4 (it raises "gqaRatio must be a multiple of 4"
                # otherwise, which used to crash the scheduler at decode time
                # instead of falling back -- e.g. Qwen3.6-27B has ratio 6).
                _pa_kv_heads = getattr(layer, "tp_k_head_num", 0) or 0
                _pa_gqa_ok = (
                    _pa_kv_heads > 0
                    and layer.tp_q_head_num % _pa_kv_heads == 0
                    and (layer.tp_q_head_num // _pa_kv_heads) % 4 == 0
                )
                _use_esimd_pa = (
                    not _DISABLE_ESIMD_DECODE
                    and not _DISABLE_PAGE_ATTN
                    and _esimd_page_attn_decode is not None
                    and layer.head_dim == 256
                    and max_seqlen_q == 1
                    and not use_cascade_attn
                    and layer.logit_cap == 0.0
                    and q_reshaped.dtype == torch.float16
                    and _pa_gqa_ok
                )
                if _use_esimd_pa:
                    bs = forward_batch.batch_size
                    # Persistent per-layer buffers: under XPU graph, capture bakes
                    # tensor addresses, so fresh torch.empty()/elementwise temporaries
                    # get unstable addresses across replay and corrupt decode. Reuse
                    # stable buffers + in-place ops so capture and replay match.
                    out_pa = self._decode_buf(
                        ("pa_out", layer.layer_id), (bs, layer.tp_q_head_num, layer.head_dim),
                        torch.float16, q_reshaped.device,
                    )
                    if swa_window_view is not None:
                        page_table_pa, pa_seqlens, max_seq = swa_window_view
                    else:
                        page_table_pa = page_table
                        pa_seqlens = cache_seqlens
                        max_seq = page_table.shape[1] * self.page_size
                    # ESIMD kernel hardcodes matMulQuantCoeff=0.0625 (1/sqrt(256)).
                    # Compensate for the model's actual scaling factor.
                    if layer.scaling != 0.0625:
                        q_scaled = self._decode_buf(
                            ("pa_q", layer.layer_id), q_reshaped.shape,
                            q_reshaped.dtype, q_reshaped.device,
                        )
                        torch.mul(q_reshaped, layer.scaling / 0.0625, out=q_scaled)
                    else:
                        q_scaled = q_reshaped
                    _esimd_page_attn_decode(
                        q_scaled,
                        key_cache,
                        value_cache,
                        page_table_pa,
                        pa_seqlens,
                        out_pa,
                        max_seqlen_q,
                        max_seq,
                        None,
                    )
                    result = out_pa
                else:
                    # Split-K ESIMD decode (no SLM, graph-capturable) for hd512.
                    # Falls back to flash_attn for non-decode or unsupported configs.
                    if (
                        not _DISABLE_ESIMD_DECODE
                        and _splitk_decode_attention is not None
                        and max_seqlen_q == 1
                        and not use_cascade_attn
                        and layer.logit_cap == 0.0
                        and q_reshaped.dtype == torch.float16
                        and layer.head_dim in (256, 512)
                    ):
                        bs = forward_batch.batch_size
                        G = _SPLITK_G  # num KV splits (env SGLANG_SPLITK_G, default 64)
                        nTG = bs * layer.tp_q_head_num * G
                        hd = layer.head_dim
                        # Persistent PER-LAYER scratch + output: under XPU graph the
                        # 10 global layers are all captured; a single shared scratch
                        # (or fresh torch.empty per call) gives unstable/aliased
                        # addresses across replay and corrupts decode. One stable
                        # buffer per layer_id keyed by size.
                        out_sk = self._decode_buf(
                            ("sk_out", layer.layer_id), (bs, layer.tp_q_head_num, hd),
                            torch.float16, q_reshaped.device,
                        )
                        scratch = self._decode_buf(
                            ("sk_scratch", layer.layer_id), (nTG * (hd + 2),),
                            torch.float32, q_reshaped.device,
                        )
                        max_seq = page_table.shape[1] * self.page_size
                        # SWA layers: read only the resident window (see the
                        # swa_window_view comment above). page_attn_decode is
                        # missing from some sgl-kernel builds, so head_dim 256
                        # sliding layers land here and must be windowed too;
                        # without this they read evicted slots and emit NaN.
                        page_table_sk = page_table
                        seqlens_sk = cache_seqlens
                        if swa_window_view is not None:
                            page_table_sk, seqlens_sk, max_seq = swa_window_view
                        # Split-K kernel has built-in 1/sqrt(HD) scaling
                        if layer.scaling != hd ** -0.5:
                            q_scaled = self._decode_buf(
                                ("sk_q", layer.layer_id), q_reshaped.shape,
                                q_reshaped.dtype, q_reshaped.device,
                            )
                            torch.mul(q_reshaped, layer.scaling / (hd ** -0.5), out=q_scaled)
                        else:
                            q_scaled = q_reshaped
                        _splitk_decode_attention(
                            q_scaled, key_cache, value_cache,
                            page_table_sk, seqlens_sk,
                            out_sk, scratch, max_seq, G,
                        )
                        result = out_sk
                    else:
                        result = flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=page_table,
                            cache_seqlens=cache_seqlens,
                            cu_seqlens_q=metadata.cu_seqlens_q,
                            cu_seqlens_k_new=None,
                            max_seqlen_q=max_seqlen_q,
                            softmax_scale=layer.scaling,
                            causal=False if use_cascade_attn else causal,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=use_cascade_attn,
                            **kwargs,
                        )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=None,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            **kwargs,
                        )
                    )
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            # Do absorbed multi-latent attention
            kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
            assert not use_cascade_attn, "Cascade attention is not supported with MLA"

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]

            o = flash_mla_decode(
                q_nope,
                q_rope,
                kv_cache.view(-1, self.page_size, layer.head_dim),
                metadata.cache_seqlens_int32,
                metadata.page_table,
                self.workspace,
                layer.scaling,
            )

        out = o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return out

    def _decode_buf(self, key, shape, dtype, device):
        """Return a persistent, address-stable buffer for the given (key, shape).

        XPU graph capture bakes tensor data pointers; allocating fresh tensors
        (torch.empty) or elementwise temporaries inside the captured decode
        forward yields addresses that differ at replay -> silent corruption.
        Caching one buffer per key (e.g. per layer_id + role) keeps the address
        stable across capture and replay. Reallocates only if a larger shape is
        ever requested.
        """
        if not hasattr(self, "_decode_bufs"):
            self._decode_bufs = {}
        buf = self._decode_bufs.get(key)
        numel = 1
        for s in shape:
            numel *= s
        if buf is None or buf.numel() < numel or buf.dtype != dtype:
            buf = torch.zeros(numel, dtype=dtype, device=device)
            self._decode_bufs[key] = buf
        return buf[:numel].view(*shape)

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def _init_local_attn_metadata(
        self,
        forwardbatch: ForwardBatch,
        metadata: FlashAttentionMetadata,
        device,
    ):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled."""
        if self.attention_chunk_size is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        if self.is_hybrid_swa:
            page_table = self.full_to_swa_index_mapping[metadata.page_table].to(
                torch.int32
            )
        else:
            page_table = metadata.page_table
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        # make_local_attention_virtual_batches expects a page-granularity block table:
        # column p is the logical page number, and the value stored at that column is the
        # physical page index. The raw req_to_token table is token-granularity (column i =
        # the KV slot for token i), so when page_size > 1 we must stride and divide first
        # so that block_starts = k_seqstarts_absolute // page_size correctly indexes the table.
        if self.page_size > 1:
            strided_indices = torch.arange(
                0, page_table.shape[1], self.page_size, device=page_table.device
            )
            page_table = page_table[:, strided_indices] // self.page_size

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata

    def _init_sliding_window_attn_spec_metadata(
        self,
        metadata: FlashAttentionMetadata,
        metadata_expand: FlashAttentionMetadata,
        metadata_swa: Optional[FlashAttentionMetadata] = None,
    ):
        # TODO: support page_size > 1 for swa spec
        assert (
            self.page_size == 1
        ), "FlashAttention backend doesn't support topk > 1 speculative decoding with page size > 1 sliding window attention"

        cache_seqlens_int32 = (
            metadata.cache_seqlens_int32.repeat_interleave(
                self.speculative_num_draft_tokens
            )
            + metadata_expand.cache_seqlens_int32
        )
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        bs = cache_seqlens_int32.shape[0]
        page_table = (
            metadata.page_table.new_zeros(
                (bs, metadata.max_seq_len_k + metadata_expand.page_table.shape[1])
            )
            if metadata_swa is None
            else metadata_swa.page_table
        )

        prepare_swa_spec_page_table_triton(
            page_table,
            metadata.page_table,
            metadata_expand.page_table,
            metadata.cache_seqlens_int32,
            metadata_expand.cache_seqlens_int32,
            self.speculative_num_draft_tokens,
        )

        if metadata_swa is None:
            metadata_swa = FlashAttentionMetadata()
            metadata_swa.max_seq_len_q = 1
            metadata_swa.cu_seqlens_q = metadata_expand.cu_seqlens_q
            metadata_swa.cache_seqlens_int32 = cache_seqlens_int32
            metadata_swa.cu_seqlens_k = cu_seqlens_k
            metadata_swa.page_table = page_table
        else:
            metadata_swa.cache_seqlens_int32.copy_(cache_seqlens_int32)
            metadata_swa.cu_seqlens_k.copy_(cu_seqlens_k)

        metadata.swa_spec_metadata = metadata_swa


class XPUMultiStepDraftBackend:
    """Multi-step EAGLE draft-decode wrapper around XPUAttentionBackend.

    Mirrors FlashAttentionMultiStepBackend: one XPUAttentionBackend per draft
    step, each pinned to its own speculative_step_id so the per-step metadata
    (cache_seqlens = seq_lens + step_id + 1) is built correctly.

    Only topk == 1 is supported: the underlying ESIMD page_attn_decode kernel
    hard-asserts max_query_len == 1, and the topk > 1 cascade path additionally
    needs the two-pass expand metadata that XPU has not been validated for.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        if topk > 1:
            raise ValueError(
                "intel_xpu draft attention backend only supports "
                f"--speculative-eagle-topk 1, got {topk}."
            )
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                XPUAttentionBackend(
                    model_runner,
                    skip_prefill=True,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            encoder_lens=forward_batch.encoder_lens,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_in_graph(forward_batch)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.attn_backends[0].get_cuda_graph_seq_len_fill_value()
