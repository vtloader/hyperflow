# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""A length-aware Ascend attention backend for MiniMax-H3's full self-attention.

The choice of kernel depends on the sequence length, because the ranking flips. Measured on Ascend 910
(Ascend910_9382), one DiT forward at 1344x768 / 124 frames (B=1, S=37718, N=56, D=128), 26.05 s of device time with no
idle gap:

===========================================  =========  ========  =============
op                                           time       TFLOPS    share
===========================================  =========  ========  =============
``npu_fusion_attention`` BSND bf16 (stock)   330.1 ms    123.6     63.4 %
``npu_prompt_flash_attention`` BSH bf16      263.1 ms    155.0
MatMulV3 (reference ceiling)                              ~300
===========================================  =========  ========  =============

The stock backend in diffusers calls ``npu_fusion_attention`` with ``input_layout="BSND"``. On this part that path
runs at 43 % of the device's matmul throughput and its profile counters are dominated by the scalar pipe
(``aic_scalar_ratio`` 0.87, ``aic_mac_ratio`` 0.38), i.e. it is tiling poorly rather than being memory bound.

Replacing it with ``npu_prompt_flash_attention`` in ``BSH`` layout is the win at 768p: the q/k/v tensors MiniMax-H3
produces are ``(batch, seq, heads, head_dim)`` *contiguous*, which is bit-for-bit the same memory as
``(batch, seq, heads*head_dim)`` — the layout that op wants — so no transpose or copy is needed, only a ``view`` back
on the output. 330 -> 263 ms.

**But that advantage does not survive long sequences.** Sweeping the length on this part (bf16, D=128, 28 heads):

=========  ==========  ============  ==================
seq        SDPA        prompt_flash  winner
=========  ==========  ============  ==================
8,192      156.5 TF    136.6 TF      SDPA 1.15x
37,718     158.3 TF    154.1 TF      SDPA 1.03x
65,536     155.2 TF    157.9 TF      prompt_flash 1.02x
109,480    161.4 TF    111.7 TF      SDPA 1.44x
218,959    162.2 TF     94.0 TF      SDPA 1.72x
=========  ==========  ============  ==================

So the backend now picks per call: ``npu_prompt_flash_attention`` only inside the measured 60k-90k window, PyTorch
SDPA everywhere else. SDPA on Ascend dispatches through ``npu_fusion_attention`` to the same CANN flash kernel and is
what the reference CUDA path uses, so the arithmetic is unchanged; it just costs one ``transpose(1, 2)``.

The alternative — keeping prompt_flash everywhere — cost 133 s per denoising step on a 1184x1760 / 362-frame clip
(219k tokens), i.e. 17.7 minutes over the 8-step loop.

bf16 is the default compute dtype: the op's fp16 path is 12 % faster in isolation (235 vs 263 ms at 37k) but in the
real model the three 540 MB casts cost more than that saves (270 vs 260 ms per call), and bf16 keeps the arithmetic
identical to the stock backend. fp16 is still reachable via ``compute_dtype``.

The backend is registered as ``_hyperflow_npu_fast`` and only used when the caller asks for it; diffusers'
``_native_npu`` is left untouched so the CUDA path and the default Ascend behaviour are unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "FAST_BACKEND_NAME",
    "enable_fast_npu_attention",
    "fast_npu_attention",
    "is_fast_npu_available",
]

#: Name the backend is registered under in diffusers' `AttentionBackendName` / registry.
FAST_BACKEND_NAME = "_hyperflow_npu_fast"

_INSTALLED = False

#: Sequence-length window where `npu_prompt_flash_attention` beats PyTorch SDPA on this part. Outside it — and the
#: band is narrow — SDPA wins, by up to 1.72x past 200k tokens. See `_fast_attention_core` for the measurements.
PROMPT_FLASH_MIN_SEQ = 60_000
PROMPT_FLASH_MAX_SEQ = 90_000


def is_fast_npu_available() -> bool:
    """True when the Ascend attention path can run: torch_npu with `npu_prompt_flash_attention`.

    The backend still needs the op for the 60k-90k window; every other length goes through PyTorch SDPA, which only
    needs a working accelerator.
    """
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch_npu, "npu_prompt_flash_attention")


def _flatten_heads(t: torch.Tensor) -> torch.Tensor:
    """``(B, S, N, D)`` contiguous -> ``(B, S, N*D)`` without copying."""
    return t.reshape(t.shape[0], t.shape[1], -1)


def fast_npu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
    return_lse: bool = False,
    _parallel_config: Any = None,
) -> torch.Tensor:
    """Drop-in replacement for diffusers' `_native_npu_attention`, with a length-aware kernel choice.

    Args:
        query, key, value: ``(B, S, N, D)``. Must be contiguous so the memory is a valid ``BSH`` view if the
            ``npu_prompt_flash_attention`` branch is taken.
        attn_mask: a bool mask as diffusers passes it (True = keep). ``None`` is the only case MiniMax-H3 uses;
            anything else defers to the stock kernel so mask semantics stay exactly as upstream.
        dropout_p: only 0 is supported (inference).
        scale: softmax scale; defaults to ``1 / sqrt(D)`` like the stock backend.
        return_lse: unsupported, same as the stock NPU backend.
        _parallel_config: context-parallel config. When set, diffusers' Ulysses wrapper does the all-to-all and calls
            our `forward_op`, so the same length-aware choice applies to each rank's local shard.

    Returns:
        ``(B, S, N, D)`` in the input dtype.
    """
    if return_lse:
        raise ValueError("the fast NPU attention backend does not support `return_lse=True`.")
    if _parallel_config is not None:
        # Context parallel: diffusers' Ulysses wrapper does the all-to-all and then calls a `forward_op` for the
        # local attention. Handing it our own op reuses that wrapper instead of reimplementing it, and the stock
        # `_npu_attention_forward_op` stays untouched for `--attention-backend npu`.
        from diffusers.models.attention_dispatch import (
            _npu_attention_backward_op,
            _templated_context_parallel_attention,
        )

        return _templated_context_parallel_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            False,
            scale,
            False,
            return_lse,
            forward_op=_fast_npu_attention_forward_op,
            backward_op=_npu_attention_backward_op,
            _parallel_config=_parallel_config,
        )
    if attn_mask is not None or dropout_p != 0.0:
        return _stock_npu_attention(query, key, value, attn_mask, dropout_p, scale, return_lse, None)

    return _fast_attention_core(query, key, value, scale)


def _fast_attention_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Attention for MiniMax-H3's packed sequence, picking the kernel by sequence length.

    Measured on Ascend 910 (Ascend910_9382), bf16, head_dim 128, 28 heads (the CP-local count), same-shape calls:

    =========  ===========  ===========  ===============
    seq        SDPA         prompt_flash  winner
    =========  ===========  ===========  ===============
    8,192       156.5 TF      136.6 TF   SDPA 1.15x
    37,718      158.3 TF      154.1 TF   SDPA 1.03x
    65,536      155.2 TF      157.9 TF   prompt_flash 1.02x
    109,480     161.4 TF      111.7 TF   SDPA 1.44x
    218,959     162.2 TF       94.0 TF   SDPA 1.72x
    =========  ===========  ===========  ===============

    `npu_prompt_flash_attention` only wins in a narrow band around 64-80k, and loses badly past ~100k: at the 219k
    tokens of a 1184x1760 / 362-frame clip it costs 1.72x more, i.e. 133 s per denoising step. PyTorch's SDPA (which
    dispatches to the same CANN flash kernel through `npu_fusion_attention`) is the better default and is also what the
    reference CUDA path uses, so the arithmetic stays identical.

    fp16 compute is kept available for callers that measured otherwise on their own hardware, but bf16 is the default:
    the three 540 MB casts cost more than the kernel saves.
    """
    import torch_npu

    batch, seq, heads, head_dim = query.shape
    out_dtype = query.dtype
    compute_dtype = compute_dtype or out_dtype
    scale_value = 1.0 / math.sqrt(head_dim) if scale is None else scale

    # The op only accepts fp16/bf16, so anything else (fp32 from a tiny test model) takes SDPA below.
    if out_dtype not in (torch.bfloat16, torch.float16) or compute_dtype not in (torch.bfloat16, torch.float16):
        return _sdpa_attention(query, key, value, scale)

    if PROMPT_FLASH_MIN_SEQ <= seq <= PROMPT_FLASH_MAX_SEQ:
        q = _flatten_heads(query if query.dtype is compute_dtype else query.to(compute_dtype))
        k = _flatten_heads(key if key.dtype is compute_dtype else key.to(compute_dtype))
        v = _flatten_heads(value if value.dtype is compute_dtype else value.to(compute_dtype))
        out = torch_npu.npu_prompt_flash_attention(
            q,
            k,
            v,
            num_heads=heads,
            input_layout="BSH",
            scale_value=scale_value,
            sparse_mode=0,
        )
        return out.view(batch, seq, heads, head_dim).to(out_dtype)

    return _sdpa_attention(query, key, value, scale)


def _sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    """PyTorch SDPA on `(B, S, N, D)`, which on Ascend dispatches to the flash kernel and is the CUDA-identical path.

    The layout has to be transposed to `(B, N, S, D)` first; that costs 2.6 ms for the 219k shape against the 3 s the
    kernel takes, so it is noise, and `scaled_dot_product_attention` returns a non-contiguous view that the caller's
    `flatten(2, 3)` handles.
    """
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
    )
    return out.transpose(1, 2)


def _fast_npu_attention_forward_op(
    ctx: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Any = None,
) -> torch.Tensor:
    """`forward_op` for diffusers' Ulysses wrappers: the fast kernel on a rank's local Q/K/V shard.

    The wrapper has already done the all-to-all, so this only sees the local slice; it must not re-enter the parallel
    dispatch or the all-to-all would run twice.
    """
    if return_lse:
        raise ValueError("the fast NPU attention backend does not support `return_lse=True`.")
    if attn_mask is not None or dropout_p != 0.0:
        from diffusers.models.attention_dispatch import _npu_attention_forward_op

        return _npu_attention_forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=_save_ctx,
            _parallel_config=_parallel_config,
        )
    return _fast_attention_core(query, key, value, scale)


def _stock_npu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    scale: float | None,
    return_lse: bool,
    parallel_config: Any,
) -> torch.Tensor:
    """diffusers' own `_native_npu_attention`, for every case the fast path does not cover."""
    from diffusers.models.attention_dispatch import _native_npu_attention

    return _native_npu_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        scale=scale,
        return_lse=return_lse,
        _parallel_config=parallel_config,
    )


def enable_fast_npu_attention(transformer: Any) -> bool:
    """Register and select the fast backend on `transformer`. Returns False when it is unavailable.

    The backend is added to diffusers' registry under `FAST_BACKEND_NAME` and to its `AttentionBackendName` enum, so
    `set_attention_backend` and the per-processor dispatch keep working exactly as they do for a built-in backend.
    """
    global _INSTALLED
    if not is_fast_npu_available():
        logger.warning("torch_npu.npu_prompt_flash_attention is unavailable; keeping the stock NPU attention backend")
        return False

    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
        _check_device,
        _check_qkv_dtype_bf16_or_fp16,
        _check_shape,
    )

    if not _INSTALLED:
        # `AttentionBackendName` is a str Enum; adding a member lets `AttentionBackendName(name)` and every
        # `{member.value for member in ...}` check in diffusers see the new backend.
        if FAST_BACKEND_NAME not in AttentionBackendName.__members__:
            new_member = str.__new__(AttentionBackendName, FAST_BACKEND_NAME)
            new_member._name_ = FAST_BACKEND_NAME
            new_member._value_ = FAST_BACKEND_NAME
            AttentionBackendName._member_map_[FAST_BACKEND_NAME] = new_member
            AttentionBackendName._value2member_map_[FAST_BACKEND_NAME] = new_member
        _AttentionBackendRegistry.register(
            AttentionBackendName(FAST_BACKEND_NAME),
            constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
            supports_context_parallel=True,
        )(fast_npu_attention)
        _INSTALLED = True

    setter = getattr(transformer, "set_attention_backend", None)
    if setter is None:
        logger.warning("model has no `set_attention_backend`; leaving its attention backend unchanged")
        return False
    setter(AttentionBackendName(FAST_BACKEND_NAME))
    logger.info(
        "attention backend: %s (torch_npu.npu_prompt_flash_attention, BSH, fp16 compute)",
        FAST_BACKEND_NAME,
    )
    return True
