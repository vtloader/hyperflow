"""Ascend NPU adaptation tests: device resolution, generator placement, attention backend, and a full
transformer forward on the NPU when one is available.

The tests skip cleanly on a CUDA-only or CPU-only host: everything asserts the CUDA path is unchanged unless an NPU
is actually present.
"""

from __future__ import annotations

import pytest
import torch
from conftest import make_transformer, packed_inputs

from hyperflow_h3 import npu as npu_backend

NPU = npu_backend.is_npu_available()
needs_npu = pytest.mark.skipif(not NPU, reason="no Ascend NPU available")


def forward(model, inputs, timestep):
    """One transformer forward on the packed inputs of `conftest.packed_inputs`."""
    device = timestep.device
    kwargs = dict(
        hidden_states=inputs["latents"][None],
        audio_hidden_states=inputs["audio_latents"][None],
        encoder_hidden_states=inputs["prompt_embeds"],
        timestep=timestep,
        timestep_indices=torch.zeros(inputs["token_tags"].numel(), dtype=torch.long, device=device),
        token_tags=inputs["token_tags"],
        position_ids=inputs["position_ids"],
        video_indices=inputs["video_indices"],
        audio_indices=inputs["audio_indices"],
        text_indices=inputs["text_indices"],
        return_dict=False,
    )
    with torch.no_grad():
        return model(**kwargs)


def test_default_device_prefers_npu_then_cuda_then_cpu():
    expected = "npu" if NPU else ("cuda" if torch.cuda.is_available() else "cpu")
    assert npu_backend.default_device() == expected


def test_resolve_device_maps_auto_and_keeps_explicit_values():
    assert npu_backend.resolve_device("auto") == npu_backend.default_device()
    assert npu_backend.resolve_device(None) == npu_backend.default_device()
    assert npu_backend.resolve_device("cpu") == "cpu"
    assert npu_backend.resolve_device("cuda:1") == "cuda:1"


def test_device_module_follows_the_backend():
    assert npu_backend.device_module("cpu") is torch
    if NPU:
        assert npu_backend.device_module("npu") is torch.npu
        assert npu_backend.device_module("npu:0") is torch.npu


def test_device_module_rejects_npu_without_torch_npu(monkeypatch):
    monkeypatch.setattr(npu_backend, "_NPU", None)
    monkeypatch.setattr(npu_backend, "_NPU_CHECKED", True)
    with pytest.raises(RuntimeError, match="torch_npu"):
        npu_backend.device_module("npu")


def test_generator_is_cpu_on_cpu_and_cuda():
    # A CPU generator is what the upstream CUDA recipe uses; changing it would break bit-exact reproducibility.
    assert npu_backend.make_generator("cpu", 7).device.type == "cpu"
    assert npu_backend.make_generator(None, 7).device.type in ("cpu", "npu")


@needs_npu
def test_generator_is_on_the_npu_for_npu():
    # NPU `randn` refuses a CPU generator, so the generator must live on the device.
    generator = npu_backend.make_generator("npu", 7)
    assert generator.device.type == "npu"
    drawn = torch.randn(8, generator=generator, device="npu")
    again = torch.randn(8, generator=npu_backend.make_generator("npu", 7), device="npu")
    assert drawn.shape == again.shape


@needs_npu
def test_randn_tensor_accepts_the_generator():
    from diffusers.utils.torch_utils import randn_tensor

    generator = npu_backend.make_generator("npu", 3)
    latents = randn_tensor((2, 3, 4), generator=generator, device="npu", dtype=torch.float32)
    assert latents.device.type == "npu" and latents.shape == (2, 3, 4)


@needs_npu
def test_empty_cache_and_memory_queries_work():
    npu_backend.empty_cache("npu")
    npu_backend.synchronize("npu")
    assert npu_backend.memory_allocated("npu") >= 0
    assert npu_backend.max_memory_allocated("npu") >= 0


@needs_npu
def test_npu_stream_is_available():
    assert npu_backend.stream("npu") is not None


@needs_npu
def test_enable_npu_attention_switches_the_backend():
    from diffusers.models.attention import AttentionModuleMixin
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    model = make_transformer()
    saved = _AttentionBackendRegistry._active_backend
    try:
        assert npu_backend.enable_npu_attention(model)
        backends = {
            module.processor._attention_backend
            for module in model.modules()
            if isinstance(module, AttentionModuleMixin) and hasattr(module.processor, "_attention_backend")
        }
        assert len(backends) == 1
        assert next(iter(backends)).value == "_native_npu"
    finally:
        # `set_attention_backend` also changes the process-wide default; restore it so later tests see the original.
        model.reset_attention_backend()
        _AttentionBackendRegistry.set_active_backend(saved)


def test_install_device_aliases_is_opt_in(monkeypatch):
    monkeypatch.delenv("HYPERFLOW_NPU_ALIAS_CUDA", raising=False)
    assert npu_backend.install_device_aliases() is False
    assert not getattr(torch.Tensor, "_hyperflow_cuda_alias", False)


@needs_npu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_transformer_forward_on_npu(dtype):
    """The tiny MiniMax-H3 transformer runs on the NPU and matches the CPU result within dtype tolerance."""
    inputs = packed_inputs(make_transformer())
    cpu_model = make_transformer().eval()
    npu_model = make_transformer().to(dtype).eval()
    npu_model = npu_model.to("npu")

    timestep = torch.tensor([0.3])
    with torch.no_grad():
        cpu_video, cpu_audio = forward(cpu_model, inputs, timestep)
        npu_inputs = {key: value.to("npu") if torch.is_tensor(value) else value for key, value in inputs.items()}
        npu_video, npu_audio = forward(npu_model, npu_inputs, timestep.to("npu"))

    assert npu_video.device.type == "npu" and npu_audio.device.type == "npu"
    # bf16 on Ascend keeps far fewer bits than the fp32 CPU reference; compare at the dtype's own tolerance.
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(npu_video.float().cpu(), cpu_video.float(), rtol=atol, atol=atol)
    torch.testing.assert_close(npu_audio.float().cpu(), cpu_audio.float(), rtol=atol, atol=atol)


@needs_npu
def test_npu_fused_attention_matches_native():
    """`_native_npu` (npu_fusion_attention) agrees with PyTorch SDPA on the same Q/K/V."""
    from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn

    torch.manual_seed(0)
    query = torch.randn(2, 64, 4, 128, device="npu", dtype=torch.bfloat16)
    reference = dispatch_attention_fn(query, query, query, backend=AttentionBackendName.NATIVE)
    fused = dispatch_attention_fn(query, query, query, backend=AttentionBackendName._NATIVE_NPU)
    torch.testing.assert_close(fused.float(), reference.float(), rtol=2e-2, atol=2e-2)
