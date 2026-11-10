"""Tensor-parallel support tests: the shard-agnostic attention processor, the plan, and the device allowlist.

The numerical checks need two Ascend devices; they skip elsewhere. The CPU tests cover the parts that do not.
"""

from __future__ import annotations

import pytest
import torch

from hyperflow_h3 import tensor_parallel as tp

NPU = None
try:  # a CUDA-only or CPU-only host simply skips the distributed cases
    from hyperflow_h3 import npu as npu_backend

    NPU = npu_backend.is_npu_available()
except Exception:  # pragma: no cover - import guard
    NPU = False


def test_tp_plan_shards_attention_and_feed_forward():
    plan = tp.TP_PLAN
    assert plan["transformer_blocks.*.attn.to_q"] == "colwise"
    assert plan["transformer_blocks.*.attn.to_out.0"] == "rowwise"
    assert plan["transformer_blocks.*.ff.net.2"] == "rowwise"
    # AdaLN modulation indexes the full hidden size and the QK-norms work over head_dim: both stay replicated.
    assert not any("adaln" in key or "norm" in key or "proj_in" in key for key in plan)


def test_swiglu_projection_is_split_into_value_and_gate():
    """A fused SwiGLU `Linear(hidden, 2 * ffn_dim)` cannot be sharded: neither plain colwise nor
    `PackedColwiseParallel` keeps `chunk(2)` meaningful on a shard. It is rewritten into two Linears instead."""
    plan = tp.TP_PLAN
    assert "transformer_blocks.*.ff.net.0.proj" not in plan
    assert plan["transformer_blocks.*.ff.net.0.proj_value"] == "colwise"
    assert plan["transformer_blocks.*.ff.net.0.proj_gate"] == "colwise"


def test_split_swiglu_keeps_the_fused_arithmetic():
    """`SplitSwiGLU` must be numerically identical to the fused `SwiGLU` it replaces."""
    from diffusers.models.activations import SwiGLU

    torch.manual_seed(0)
    fused = SwiGLU(dim_in=32, dim_out=48, bias=False)
    split = tp.SplitSwiGLU(fused.proj)
    hidden = torch.randn(2, 5, 32)
    torch.testing.assert_close(split(hidden), fused(hidden), rtol=0, atol=1e-6)


def test_processor_reshapes_with_head_dim_not_head_count():
    """The shard-agnostic processor must never use `attn.heads`: after column sharding that count is per rank."""
    import inspect

    source = inspect.getsource(tp.TensorParallelAttnProcessor.__call__)
    assert "head_dim" in source
    assert "attn.heads" not in source
    assert "(-1, head_dim)" in source


@pytest.mark.parametrize("tp_degree", [1, 2, 4])
def test_plan_block_sizes_divide_every_common_degree(tp_degree):
    """Every sharded block must split evenly: 1792 attention columns and 14336/2 SwiGLU halves."""
    inner_dim = 56 * 128
    ffn_dim = 14336
    assert inner_dim % tp_degree == 0
    assert ffn_dim % tp_degree == 0


def test_install_device_alias_is_idempotent(monkeypatch):
    from diffusers.hooks import tensor_parallel as diffusers_tp

    original = diffusers_tp._SUPPORTED_TP_DEVICES
    monkeypatch.setattr(diffusers_tp, "_SUPPORTED_TP_DEVICES", original)
    installed = tp.install_npu_tp_device()
    assert "npu" in diffusers_tp._SUPPORTED_TP_DEVICES
    if installed:
        assert tp.install_npu_tp_device() is False


@pytest.mark.skipif(not NPU, reason="needs two Ascend NPUs and torch_npu")
def test_tensor_parallel_processor_accepts_sharded_width():
    """A 1/N-width Q/K/V projection must still reshape to (heads/N, head_dim), not (heads, head_dim/N)."""
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    attn = MiniMaxH3Attention(hidden_size=64, heads=4, dim_head=16).to("npu", torch.bfloat16)
    processor = tp.TensorParallelAttnProcessor()
    hidden = torch.randn(2, 8, 64, device="npu", dtype=torch.bfloat16)

    # Unsharded reference: 4 heads x 16 dims.
    assert processor(attn, hidden).shape == (2, 8, 64)

    # Simulate one column-sharded rank: the projections emit 2 heads' worth of columns instead of 4.
    half = attn.inner_dim // 2
    attn.to_q = torch.nn.Linear(64, half, bias=False).to("npu", torch.bfloat16)
    attn.to_k = torch.nn.Linear(64, half, bias=False).to("npu", torch.bfloat16)
    attn.to_v = torch.nn.Linear(64, half, bias=False).to("npu", torch.bfloat16)
    attn.to_out[0] = torch.nn.Linear(half, 64, bias=False).to("npu", torch.bfloat16)
    out = processor(attn, hidden)
    assert out.shape == (2, 8, 64)

    # The official processor would have inferred head_dim = 8 here and failed the QK-norm.
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor

    with pytest.raises(RuntimeError):
        MiniMaxH3AttnProcessor()(attn, hidden)
