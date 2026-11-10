# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Ascend RMSNorm fusion for the MiniMax-H3 DiT.

Every DiT block runs four `nn.RMSNorm` calls (`norm1`, `norm2`, `norm_q`, `norm_k`), plus one at the end of the
stack: 105 hidden-size norms and 100 per-head norms per forward at 768p. PyTorch's `F.rms_norm` decomposes into
`Pow`, `ReduceMean`, `Rsqrt`, `Mul`, `Mul` on Ascend; `torch_npu.npu_rms_norm` is a single fused kernel.

Measured at the exact shapes of one 1344x768 / 124-frame forward (S=37718):

=========================================  =========  =========
norm                                       nn.RMSNorm  npu_rms_norm
=========================================  =========  =========
hidden (5376), 105 calls per forward        6.37 ms     0.84 ms
per-head (128), 100 calls per forward       8.35 ms     0.90 ms
=========================================  =========  =========

i.e. 1.33 s of the 26 s forward, or 10.6 s over the 8-step loop. Accuracy is unchanged: against an fp32 reference
both paths report the same 1.4e-03 mean relative error, which is bf16 rounding, not the kernel.

The swap is opt-in and only touches `nn.RMSNorm` instances under the DiT. CUDA and CPU keep `F.rms_norm`, and a
model with any other norm type is left alone.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["enable_fused_rms_norm", "is_fused_rms_norm_available"]


def is_fused_rms_norm_available() -> bool:
    """True when `torch_npu.npu_rms_norm` is importable."""
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch_npu, "npu_rms_norm")


class FusedRMSNorm(torch.nn.Module):
    """`nn.RMSNorm` with the same state dict, backed by `torch_npu.npu_rms_norm`.

    The checkpoint stores `weight` under the norm's own name, so the parameter is kept exactly where it was: wrapping
    a module in place would rename it. Instead `enable_fused_rms_norm` replaces each `nn.RMSNorm` with this class and
    copies `weight`/`eps` across, which keeps every key and every `.to()`/offload hook working.
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        weight = self.weight
        if weight.dtype is not x.dtype:
            # The mixed-precision checkpoint keeps some norms in fp32 next to bf16 activations; the fused kernel wants
            # both in the same dtype. Casting the 5376-element weight is far cheaper than the 37718x5376 activation.
            weight = weight.to(x.dtype)
        return torch_npu.npu_rms_norm(x, weight, self.eps)[0]

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


def enable_fused_rms_norm(model: Any) -> int:
    """Replace every `nn.RMSNorm` under `model` with `FusedRMSNorm`. Returns the number replaced.

    Only `nn.RMSNorm` (not `LayerNorm` or a custom norm) is touched, and only when `torch_npu.npu_rms_norm` exists,
    so the CUDA path is unaffected.
    """
    if not is_fused_rms_norm_available():
        logger.warning("torch_npu.npu_rms_norm is unavailable; keeping torch's F.rms_norm")
        return 0

    replaced = 0
    for _name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, torch.nn.RMSNorm):
                continue
            # Only 1-D normalisation (over the last dim) is what the DiT uses and what the fused op implements.
            if len(child.normalized_shape) != 1:
                continue
            fused = FusedRMSNorm(child.normalized_shape[0], child.eps)
            with torch.no_grad():
                fused.weight.copy_(child.weight)
            fused = fused.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, child_name, fused)
            replaced += 1
    if replaced:
        logger.info("fused RMSNorm: %d nn.RMSNorm -> torch_npu.npu_rms_norm", replaced)
    return replaced
