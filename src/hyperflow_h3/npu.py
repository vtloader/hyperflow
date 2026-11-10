# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Ascend NPU support for the HyperFlow runtime.

This module is additive: nothing here changes the CUDA code path. It provides

* `is_npu_available()` / `default_device()` — device discovery that prefers an Ascend NPU when one is present;
* `device_module()` — the `torch.<backend>` namespace (`torch.cuda`, `torch.npu`, ...) of a device, so call sites do
  not hard-code `torch.cuda`;
* `empty_cache()` / `memory_allocated()` / `max_memory_allocated()` / `synchronize()` / `stream()` — the handful of
  backend calls the example scripts and the offload path make;
* `make_generator()` — a `torch.Generator` on the same backend as the latents (an NPU `randn` refuses a CPU
  generator with ``Expected a 'npu' device type for generator but found 'cpu'``);
* `enable_npu_attention(transformer)` — select diffusers' ``_native_npu`` attention backend, which wraps
  ``torch_npu.npu_fusion_attention``;
* `install_device_aliases()` — an opt-in shim for third-party code that still asks ``tensor.is_cuda``; the alias is
  installed only when `HYPERFLOW_NPU_ALIAS_CUDA=1` is set, so the default behaviour is unchanged.

Import is safe on a CUDA-only or CPU-only host: everything degrades to `torch.cuda` / `cpu`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "default_device",
    "device_module",
    "empty_cache",
    "enable_npu_attention",
    "install_device_aliases",
    "is_npu_available",
    "make_generator",
    "max_memory_allocated",
    "memory_allocated",
    "resolve_device",
    "stream",
    "synchronize",
]

_NPU: Any | None = None
_NPU_CHECKED = False


def _npu() -> Any | None:
    """`torch_npu` when it is importable and reports an available device, else None. Cached."""
    global _NPU, _NPU_CHECKED
    if _NPU_CHECKED:
        return _NPU
    _NPU_CHECKED = True
    try:
        import torch_npu  # noqa: F401
    except Exception:  # ImportError, or an OSError from a mismatched CANN runtime
        _NPU = None
        return None
    try:
        if torch.npu.is_available():
            _NPU = torch.npu
            logger.info("Ascend NPU backend available: %d device(s)", torch.npu.device_count())
        else:
            _NPU = None
    except Exception:
        _NPU = None
    return _NPU


def is_npu_available() -> bool:
    """True when `torch_npu` is installed and at least one Ascend device is visible."""
    return _npu() is not None


def default_device() -> str:
    """``"npu"`` when an Ascend device is available, else ``"cuda"`` when one is, else ``"cpu"``."""
    if is_npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device: str | torch.device | None = None) -> str:
    """``device`` with ``"auto"`` / ``None`` mapped onto `default_device`; anything else passes through."""
    if device is None or device == "auto":
        return default_device()
    return str(device)


def device_module(device: str | torch.device | None = None):
    """The ``torch.<backend>`` namespace of ``device``: ``torch.npu``, ``torch.cuda`` or ``torch`` for CPU."""
    kind = torch.device(resolve_device(device)).type
    if kind == "npu":
        module = _npu()
        if module is None:
            raise RuntimeError("device is 'npu' but torch_npu is not usable; install torch_npu==2.10.0")
        return module
    if kind == "cuda":
        return torch.cuda
    return torch


def _index(device: str | torch.device | None) -> int | None:
    dev = torch.device(resolve_device(device))
    return dev.index


def empty_cache(device: str | torch.device | None = None) -> None:
    module = device_module(device)
    fn = getattr(module, "empty_cache", None)
    if fn is not None:
        fn()


def synchronize(device: str | torch.device | None = None) -> None:
    module = device_module(device)
    fn = getattr(module, "synchronize", None)
    if fn is not None:
        fn()


def memory_allocated(device: str | torch.device | None = None) -> int:
    module = device_module(device)
    fn = getattr(module, "memory_allocated", None)
    if fn is None:
        return 0
    try:
        return int(fn(_index(device)))
    except TypeError:  # a stub (or a test double) that takes no index
        return int(fn())


def max_memory_allocated(device: str | torch.device | None = None) -> int:
    module = device_module(device)
    fn = getattr(module, "max_memory_allocated", None)
    if fn is None:
        return 0
    try:
        return int(fn(_index(device)))
    except TypeError:  # a stub (or a test double) that takes no index
        return int(fn())


def stream(device: str | torch.device | None = None):
    """A new stream on ``device``'s backend, or None when the backend has no streams (CPU)."""
    module = device_module(device)
    cls = getattr(module, "Stream", None)
    return cls() if cls is not None else None


def make_generator(device: str | torch.device | None = None, seed: int | None = None) -> torch.Generator:
    """A generator usable with `torch.randn(..., device=device)`.

    An NPU `randn` rejects a CPU generator (``Expected a 'npu' device type for generator but found 'cpu'``), so on
    Ascend the generator is created on the NPU. On CUDA a CPU generator is kept: switching it would change the
    random stream and break bit-exact reproducibility against the upstream CUDA recipe.
    """
    if torch.device(resolve_device(device)).type == "npu":
        generator = torch.Generator(device="npu")
    else:
        generator = torch.Generator()
    if seed is not None:
        generator = generator.manual_seed(seed)
    return generator


def enable_npu_attention(transformer: Any, backend: str = "_native_npu") -> bool:
    """Point a diffusers model at the NPU attention backend. Returns False when the model has no such setter."""
    setter = getattr(transformer, "set_attention_backend", None)
    if setter is None:
        logger.warning("model has no `set_attention_backend`; leaving its attention backend unchanged")
        return False
    from diffusers.models.attention_dispatch import AttentionBackendName

    try:
        setter(AttentionBackendName(backend))
    except Exception as exc:  # an older diffusers, or a build without the NPU backend
        logger.warning("could not select attention backend %r (%s); using the model default", backend, exc)
        return False
    logger.info("attention backend: %s (torch_npu.npu_fusion_attention)", backend)
    return True


def enable_group_offloading(
    pipe: Any,
    device: str | torch.device,
    *,
    transformer_names: tuple[str, ...] = ("transformer", "transformer_ref"),
    text_encoder_name: str = "text_encoder",
    num_blocks_per_group: int = 1,
    offload_text_encoder: bool = True,
    offload_transformer: bool = True,
    use_stream: bool = True,
) -> dict[str, str]:
    """Stream the two 62 GB components through the device in groups.

    An Ascend 910 has 61.27 GiB of HBM while MiniMax-H3's bf16 text encoder is 62.13 GiB and its transformer 61.70 GiB:
    neither fits, so module-level offloading cannot work — the component never becomes resident in the first place.
    Group offloading keeps the weights on the host and moves one block (transformer) or one leaf (text encoder) at a
    time, overlapping the transfer with compute on a second stream.

    Call this *after* `load_hyperflow_lora` (PEFT swaps the target `nn.Linear` modules, and the hooks must attach to
    the final tree) and *before* the pipeline is first called. Components that are not group offloaded keep the
    manager's own module-level behaviour, so a VAE still moves to the device for its decode and back.

    Args:
        offload_text_encoder: Stream Qwen3-VL. Disable only when it fits on the device (a 141 GB card, or after tensor
            parallelism sharded it).
        offload_transformer: Stream the DiT. Same condition.
        use_stream: Overlap the host-to-device copy with compute on a second stream. Falls back to a synchronous copy
            when the backend has no streams. Worth 3x on the denoise loop (37 s vs 111 s at 640p), so it is on by
            default; its only cost is the pinned-memory copy below.

    Returns a mapping of component name -> offload type, for logging.
    """
    from diffusers.hooks.group_offloading import _is_group_offload_enabled, apply_group_offloading

    target = torch.device(device)
    wanted = []
    if offload_transformer:
        wanted += [(name, "block_level") for name in transformer_names]
    if offload_text_encoder:
        wanted.append((text_encoder_name, "leaf_level"))

    applied: dict[str, str] = {}
    for name, offload_type in wanted:
        module = getattr(pipe, name, None)
        if module is None or not isinstance(module, torch.nn.Module) or _is_group_offload_enabled(module):
            continue
        # Block-level for the DiT (one transformer block per group), leaf-level for Qwen3-VL (one weight per hop:
        # its 50 decoder layers are far too large to move as a unit, and the visual tower is never called).
        kwargs = {"num_blocks_per_group": num_blocks_per_group} if offload_type == "block_level" else {}
        apply_group_offloading(
            module,
            onload_device=target,
            offload_device=torch.device("cpu"),
            offload_type=offload_type,
            non_blocking=use_stream,
            use_stream=use_stream,
            record_stream=use_stream,
            **kwargs,
        )
        applied[name] = offload_type
        logger.info("group offloading: %s (%s) streams through %s", name, offload_type, target)
    return applied


def install_device_aliases() -> bool:
    """Make ``tensor.is_cuda`` report True for NPU tensors.

    Some third-party kernels branch on ``is_cuda``; on Ascend that check silently takes the CPU path. The alias is
    installed only when ``HYPERFLOW_NPU_ALIAS_CUDA=1``: it changes global torch behaviour, so it is opt-in.
    """
    if os.environ.get("HYPERFLOW_NPU_ALIAS_CUDA", "0") != "1":
        return False
    if not is_npu_available():
        return False
    if getattr(torch.Tensor, "_hyperflow_cuda_alias", False):
        return True
    torch.Tensor.is_cuda = property(lambda self: self.device.type == "npu")  # type: ignore[assignment]
    torch.Tensor._hyperflow_cuda_alias = True  # type: ignore[attr-defined]
    logger.warning("HYPERFLOW_NPU_ALIAS_CUDA=1: `Tensor.is_cuda` now reports True for NPU tensors")
    return True
