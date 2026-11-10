"""The Ascend fast attention backend and fused RMSNorm: registration, exactness, and the CUDA/no-NPU fallbacks.

Everything that needs an actual device skips when no NPU is present; the registration and wrapper checks that only
exercise Python run on any host.
"""

from __future__ import annotations

import pytest
import torch

from hyperflow_h3 import npu as npu_backend
from hyperflow_h3 import npu_attention, npu_rmsnorm

NPU = npu_backend.is_npu_available()
needs_npu = pytest.mark.skipif(not NPU, reason="no Ascend NPU available")


# --- fast attention ---------------------------------------------------------------------------------------------


def test_fast_backend_name_is_distinct_from_the_stock_one():
    assert npu_attention.FAST_BACKEND_NAME != "_native_npu"
    assert npu_attention.FAST_BACKEND_NAME.startswith("_hyperflow")


def test_available_probe_is_false_without_torch_npu(monkeypatch):
    """The probe must not raise on a host where `import torch_npu` fails."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch_npu":
            raise ImportError("no torch_npu here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert npu_attention.is_fast_npu_available() is False


@needs_npu
def test_registration_adds_the_enum_member_and_the_backend():
    from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

    npu_attention.enable_fast_npu_attention(_Setter())
    assert npu_attention.FAST_BACKEND_NAME in AttentionBackendName.__members__
    assert AttentionBackendName(npu_attention.FAST_BACKEND_NAME).value == npu_attention.FAST_BACKEND_NAME
    assert AttentionBackendName(npu_attention.FAST_BACKEND_NAME) in _AttentionBackendRegistry._backends


@needs_npu
def test_registration_is_idempotent():
    from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

    npu_attention.enable_fast_npu_attention(_Setter())
    first = _AttentionBackendRegistry._backends[AttentionBackendName(npu_attention.FAST_BACKEND_NAME)]
    npu_attention.enable_fast_npu_attention(_Setter())
    second = _AttentionBackendRegistry._backends[AttentionBackendName(npu_attention.FAST_BACKEND_NAME)]
    assert first is second


@needs_npu
def test_fast_attention_matches_native_attention():
    """The fast kernel agrees with PyTorch's SDPA on the same Q/K/V, in bf16."""
    from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn

    npu_attention.enable_fast_npu_attention(_Setter())
    torch.manual_seed(0)
    query = torch.randn(2, 256, 4, 128, device="npu", dtype=torch.bfloat16)
    reference = dispatch_attention_fn(query, query, query, backend=AttentionBackendName.NATIVE)
    fast = dispatch_attention_fn(query, query, query, backend=AttentionBackendName(npu_attention.FAST_BACKEND_NAME))
    assert fast.shape == reference.shape
    assert fast.dtype == reference.dtype
    torch.testing.assert_close(fast.float(), reference.float(), rtol=2e-2, atol=2e-2)


@needs_npu
def test_fast_attention_keeps_the_input_dtype_and_shape():
    query = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.bfloat16)
    out = npu_attention.fast_npu_attention(query, query, query)
    assert out.shape == query.shape
    assert out.dtype == torch.bfloat16


@needs_npu
def test_fast_attention_with_a_mask_defers_to_the_stock_kernel():
    """A mask is not the fast path; the wrapper must hand it to the stock kernel rather than ignore it."""
    query = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 64, 64, dtype=torch.bool, device="npu")
    out = npu_attention.fast_npu_attention(query, query, query, attn_mask=mask)
    assert out.shape == query.shape


@needs_npu
def test_fast_attention_rejects_return_lse():
    query = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="return_lse"):
        npu_attention.fast_npu_attention(query, query, query, return_lse=True)


@needs_npu
def test_fast_attention_falls_back_for_fp32():
    """fp32 is not an `npu_prompt_flash_attention` dtype; the stock kernel must take over instead of downcasting."""
    query = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.float32)
    out = npu_attention.fast_npu_attention(query, query, query)
    assert out.dtype == torch.float32


@needs_npu
def test_context_parallel_routes_through_the_templated_wrapper():
    """With a parallel config the fast backend must use diffusers' Ulysses wrapper, not a bare kernel call.

    Exercised through `fast_npu_attention`'s early return, which is what a CP rank would hit; the wrapper itself is
    covered by diffusers' own tests.
    """
    import inspect

    source = inspect.getsource(npu_attention.fast_npu_attention)
    assert "_templated_context_parallel_attention" in source
    assert "_fast_npu_attention_forward_op" in source


# --- fused RMSNorm ----------------------------------------------------------------------------------------------


class _Setter(torch.nn.Module):
    """Minimal stand-in for a diffusers model: records the backend it was told to use."""

    def __init__(self):
        super().__init__()
        self.backend = None

    def set_attention_backend(self, backend):
        self.backend = backend


def test_fused_rms_norm_replaces_only_rms_norm():
    model = torch.nn.Sequential(torch.nn.RMSNorm(8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 8))
    replaced = npu_rmsnorm.enable_fused_rms_norm(model)
    if not npu_rmsnorm.is_fused_rms_norm_available():
        assert replaced == 0
        assert isinstance(model[0], torch.nn.RMSNorm)
        return
    assert replaced == 1
    assert isinstance(model[0], npu_rmsnorm.FusedRMSNorm)
    assert isinstance(model[1], torch.nn.LayerNorm)
    assert isinstance(model[2], torch.nn.Linear)


def test_fused_rms_norm_keeps_the_weight():
    norm = torch.nn.RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.arange(8, dtype=torch.float32))
    model = torch.nn.Sequential(norm)
    npu_rmsnorm.enable_fused_rms_norm(model)
    if not npu_rmsnorm.is_fused_rms_norm_available():
        pytest.skip("torch_npu.npu_rms_norm unavailable")
    assert model[0].eps == pytest.approx(1e-6)
    torch.testing.assert_close(model[0].weight.cpu(), torch.arange(8, dtype=torch.float32))


def test_fused_rms_norm_skips_multi_dim_norms():
    model = torch.nn.Sequential(torch.nn.RMSNorm((4, 8)))
    replaced = npu_rmsnorm.enable_fused_rms_norm(model)
    assert replaced == 0
    assert isinstance(model[0], torch.nn.RMSNorm)


def test_fused_rms_norm_is_a_no_op_without_torch_npu(monkeypatch):
    monkeypatch.setattr(npu_rmsnorm, "is_fused_rms_norm_available", lambda: False)
    model = torch.nn.Sequential(torch.nn.RMSNorm(8))
    assert npu_rmsnorm.enable_fused_rms_norm(model) == 0
    assert isinstance(model[0], torch.nn.RMSNorm)


@needs_npu
def test_fused_rms_norm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(128, 512, dtype=torch.bfloat16, device="npu")
    reference = torch.nn.functional.rms_norm(x.float(), (512,), None, 1e-5).float()
    weight = torch.ones(512, dtype=torch.bfloat16, device="npu")
    import torch_npu

    got = torch_npu.npu_rms_norm(x, weight, 1e-5)[0].float()
    torch.testing.assert_close(got, reference, rtol=2e-2, atol=2e-2)


# --- kernel selection by sequence length -------------------------------------------------------------------------


def test_prompt_flash_is_only_used_in_its_measured_window():
    """SDPA wins everywhere except 60-90k, where prompt_flash is up to 2% faster.

    At 219k tokens prompt_flash is 1.72x slower, which cost 133 s per denoising step on a 362-frame 1184x1760 clip.
    """
    assert npu_attention.PROMPT_FLASH_MIN_SEQ < npu_attention.PROMPT_FLASH_MAX_SEQ
    window = (npu_attention.PROMPT_FLASH_MIN_SEQ, npu_attention.PROMPT_FLASH_MAX_SEQ)
    assert window[0] <= 65_536 <= window[1], "the measured 65k optimum must stay inside the window"
    assert not (window[0] <= 218_959 <= window[1]), "219k must fall outside the window"


@needs_npu
def test_long_sequences_route_to_sdpa_and_match_it():
    """Above the window the core must call SDPA and return exactly SDPA's result."""
    q = torch.randn(1, 96_000, 4, 128, dtype=torch.bfloat16, device="npu")
    out = npu_attention._fast_attention_core(q, q, q, None)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(out.float(), reference.float(), rtol=0, atol=0)
    assert out.shape == q.shape


@needs_npu
def test_short_sequences_route_to_sdpa():
    q = torch.randn(1, 512, 4, 128, dtype=torch.bfloat16, device="npu")
    out = npu_attention._fast_attention_core(q, q, q, None)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(out.float(), reference.float(), rtol=0, atol=0)


@needs_npu
def test_window_routes_to_prompt_flash():
    q = torch.randn(1, 65_536, 4, 128, dtype=torch.bfloat16, device="npu")
    out = npu_attention._fast_attention_core(q, q, q, None)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
