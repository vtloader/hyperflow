# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
#
# Derived from `MiniMaxH3AttnProcessor` in diffusers
# (`models/transformers/transformer_minimax_h3.py`), Copyright 2025 The MiniMax Team and
# The HuggingFace Team, Apache License 2.0. See THIRD_PARTY_NOTICES.md.
"""Tensor-parallel support for the MiniMax-H3 transformer on non-CUDA accelerators.

diffusers' tensor parallelism shards `Linear` weights column-wise and row-wise with `parallelize_module`. Two things
stop MiniMax-H3 from using it as shipped:

1. `_SUPPORTED_TP_DEVICES` in `diffusers.hooks.tensor_parallel` has no `npu`. That check is a device-type allowlist,
   not a capability probe: DTensor + HCCL sharding works on Ascend (verified in `tests/test_npu_tp.py`), so this
   module registers the missing type.
2. `MiniMaxH3AttnProcessor` reshapes Q/K/V with `unflatten(-1, (attn.heads, -1))`. After column sharding a rank holds
   `inner_dim / tp_degree` columns, so that call infers `head_dim / tp_degree` and the following `RMSNorm(head_dim)`
   raises. `TensorParallelAttnProcessor` reshapes with the fixed `head_dim` instead — the same fix diffusers uses in
   `FluxAttnProcessor` — which makes the processor shard-agnostic.

It also supplies a `_tp_plan` for the model (absent upstream) and lets a sharded model keep its HyperFlow LoRA: PEFT
wraps the *pre-sharding* `nn.Linear`, so sharding must happen first and the adapter is applied to the already-sharded
modules, which keeps the LoRA matrices replicated and numerically exact.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "TP_PLAN",
    "TensorParallelAttnProcessor",
    "enable_npu_tensor_parallel",
    "install_npu_tp_device",
]

#: Names diffusers shards. Every entry is either `colwise` (out-features split) or `rowwise` (in-features split), and
#: the pairs must line up: a colwise projection feeds a rowwise projection, whose all-reduce restores the full width.
#: QK-norms, AdaLN projections, `proj_in`/`proj_out` and the time embedders stay replicated: AdaLN indexes the full
#: hidden size per row, and the norms apply over `head_dim` after the split.
#:
#: ``ff.net.0.proj`` is absent on purpose: it is a fused SwiGLU projection, and neither a plain colwise split (one rank
#: would get all the values and the other all the gates) nor `PackedColwiseParallel` (whose local shard is two
#: concatenated slices while DTensor advertises one contiguous slice, so ``chunk(2)`` reads the wrong halves) can shard
#: it. `enable_npu_tensor_parallel` first rewrites it into two independent `nn.Linear`s, which the last two entries
#: then shard.
TP_PLAN: dict[str, Any] = {
    # main transformer blocks
    "transformer_blocks.*.attn.to_q": "colwise",
    "transformer_blocks.*.attn.to_k": "colwise",
    "transformer_blocks.*.attn.to_v": "colwise",
    "transformer_blocks.*.attn.to_out.0": "rowwise",
    "transformer_blocks.*.ff.net.0.proj_value": "colwise",
    "transformer_blocks.*.ff.net.0.proj_gate": "colwise",
    "transformer_blocks.*.ff.net.2": "rowwise",
    # text token refiner blocks
    "token_refiner.refiner_blocks.*.attn.to_q": "colwise",
    "token_refiner.refiner_blocks.*.attn.to_k": "colwise",
    "token_refiner.refiner_blocks.*.attn.to_v": "colwise",
    "token_refiner.refiner_blocks.*.attn.to_out.0": "rowwise",
    "token_refiner.refiner_blocks.*.ff.net.0.proj_value": "colwise",
    "token_refiner.refiner_blocks.*.ff.net.0.proj_gate": "colwise",
    "token_refiner.refiner_blocks.*.ff.net.2": "rowwise",
}


def install_npu_tp_device() -> bool:
    """Allow `"npu"` in diffusers' tensor-parallel device allowlist. Idempotent; False when already allowed."""
    from diffusers.hooks import tensor_parallel

    supported = tensor_parallel._SUPPORTED_TP_DEVICES
    if "npu" in supported:
        return False
    tensor_parallel._SUPPORTED_TP_DEVICES = (*supported, "npu")
    logger.info("tensor parallelism: registered 'npu' as a supported device type")
    return True


class TensorParallelAttnProcessor:
    """`MiniMaxH3AttnProcessor` that derives the head count from `head_dim`, so it survives column sharding.

    A rank holds `attn.inner_dim / tp_degree` output columns of `to_q`/`to_k`/`to_v`, i.e. `heads / tp_degree` heads.
    Reshaping with `attn.heads` would instead infer a smaller `head_dim` and break the QK-norm. Passing the fixed
    `head_dim` and letting `-1` absorb the head count keeps the processor independent of `tp_degree`.

    Everything else — projections, QK-norm, RoPE, the dispatcher call, the output projection — is the official code
    path, so CUDA/SDPA behaviour and every attention backend stay available.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        head_dim = attn.head_dim
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # `(-1, head_dim)` rather than `(heads, -1)`: a sharded rank has fewer heads, never a smaller head_dim.
        query = query.unflatten(-1, (-1, head_dim))
        key = key.unflatten(-1, (-1, head_dim))
        value = value.unflatten(-1, (-1, head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _swap_processors(module: torch.nn.Module) -> int:
    """Replace `MiniMaxH3AttnProcessor` with `TensorParallelAttnProcessor`, keeping the chosen backend."""
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor

    swapped = 0
    for child in module.modules():
        processor = getattr(child, "processor", None)
        if not isinstance(processor, MiniMaxH3AttnProcessor) or isinstance(processor, TensorParallelAttnProcessor):
            continue
        replacement = TensorParallelAttnProcessor()
        replacement._attention_backend = processor._attention_backend
        replacement._parallel_config = processor._parallel_config
        child.set_processor(replacement)
        swapped += 1
    return swapped


class SplitSwiGLU(torch.nn.Module):
    """`SwiGLU` with the gate and value projections as two separate `nn.Linear`s.

    diffusers' `SwiGLU` fuses them into one `Linear(dim, 2 * inner)` and splits the result with `chunk(2)`. Column
    sharding that fused layer puts the *value* half on one rank and the *gate* half on the other, which cannot produce
    a per-rank `value * silu(gate)`; `PackedColwiseParallel([1, 1])` is no better, because the local shard is two
    concatenated slices while DTensor's `Shard(1)` metadata describes one contiguous slice, so the `chunk(2)` it
    dispatches is wrong.

    Two independent projections shard cleanly: each rank holds `inner / tp_degree` columns of *both*, so
    `value * silu(gate)` is local and the following rowwise `net.2` all-reduces as usual. Weights are copied from the
    fused layer, so the result is the same arithmetic.
    """

    def __init__(self, proj: torch.nn.Linear):
        super().__init__()
        inner = proj.out_features // 2
        self.proj_value = torch.nn.Linear(proj.in_features, inner, bias=proj.bias is not None)
        self.proj_gate = torch.nn.Linear(proj.in_features, inner, bias=proj.bias is not None)
        with torch.no_grad():
            self.proj_value.weight.copy_(proj.weight[:inner])
            self.proj_gate.weight.copy_(proj.weight[inner:])
            if proj.bias is not None:
                self.proj_value.bias.copy_(proj.bias[:inner])
                self.proj_gate.bias.copy_(proj.bias[inner:])
        self.activation = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj_value(hidden_states) * self.activation(self.proj_gate(hidden_states))


def _split_swiglu(module: torch.nn.Module) -> int:
    """Replace every fused `SwiGLU` (and any other `chunk(2)` activation) with [`SplitSwiGLU`]."""
    from diffusers.models.activations import SwiGLU

    replaced = 0
    for _name, child in list(module.named_modules()):
        net = getattr(child, "net", None)
        if not isinstance(net, torch.nn.ModuleList) or len(net) == 0:
            continue
        act = net[0]
        if not isinstance(act, SwiGLU):
            continue
        net[0] = SplitSwiGLU(act.proj)
        replaced += 1
    return replaced


def enable_npu_tensor_parallel(
    transformer: torch.nn.Module,
    *,
    tp_degree: int,
    device: str | torch.device,
    tp_plan: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Shard `transformer` across `tp_degree` ranks of the default process group.

    Must run before `load_hyperflow_lora`: PEFT replaces target `nn.Linear` modules with wrappers, and
    `apply_tensor_parallel` rejects a model holding adapter layers (their `DTensor` sharding would be skipped, leaving
    the adapter unsharded against a sharded base). Sharding first and adapting second keeps every LoRA matrix
    replicated, which is exact: `y = (W + BA)x` shards the same way for the base and the adapter as long as `B`/`A`
    are full-size on every rank.

    Group offloading must be off for this model: each rank already holds only `1 / tp_degree` of the weights, and
    diffusers rejects the combination (both want to decide where a parameter lives).
    """
    from diffusers.hooks.tensor_parallel import apply_tensor_parallel
    from diffusers.models._modeling_parallel import TensorParallelConfig
    from torch.distributed.device_mesh import init_device_mesh

    install_npu_tp_device()
    device = torch.device(device)
    mesh = init_device_mesh(device.type, (tp_degree,))
    config = TensorParallelConfig(tp_degree=tp_degree, mesh=mesh)

    import torch.distributed as dist

    config.setup(dist.get_rank(), dist.get_world_size(), device, mesh)

    swapped = _swap_processors(transformer)
    split = _split_swiglu(transformer)
    plan = dict(TP_PLAN if tp_plan is None else tp_plan)
    apply_tensor_parallel(transformer, config, plan)
    logger.info(
        "tensor parallel: sharded %s over %d %s rank(s); %d attention processor(s) made shard-agnostic, "
        "%d SwiGLU projection(s) split into value/gate",
        type(transformer).__name__,
        tp_degree,
        device.type,
        swapped,
        split,
    )
    return {"tp_degree": tp_degree, "processors": swapped, "swiglu": split, "plan": plan}
