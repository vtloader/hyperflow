# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Shared plumbing for the example scripts: pipeline construction, GPUs, attention backend, output muxing.

Everything here is the official diffusers loading recipe plus the HyperFlow calls
(`hyperflow_blocks`, `load_hyperflow_lora`, `enable_sol_attention`). Read `generate_fl2va.py` first.

GPUs: by default the DiT runs Ulysses sequence-parallel over up to four GPUs, the degree MiniMax serves the model
with. A script started with plain `python` re-launches itself under `torchrun`. Rank 0 runs the text encoder and
broadcasts the conditioned state (the official two-device split); every rank then runs the rest on its own GPU and
only rank 0 writes the file. `--gpus 1` is the official single-GPU recipe with CPU offload.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from diffusers import ComponentsManager
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
from diffusers.modular_pipelines.modular_pipeline import InsertableDict
from diffusers.utils.export_utils import encode_video
from huggingface_hub.constants import HF_HUB_OFFLINE

from hyperflow_h3 import admit_max_frames, enable_sol_attention, hyperflow_blocks, load_hyperflow_lora
from hyperflow_h3 import npu as npu_backend
from hyperflow_h3.npu_attention import FAST_BACKEND_NAME, enable_fast_npu_attention
from hyperflow_h3.npu_rmsnorm import enable_fused_rms_norm

DEFAULT_MODEL = "MiniMaxAI/MiniMax-H3"
DEFAULT_WEIGHTS = "videorebirth/hyperflow"
OUTPUTS = ["videos", "audio", "sampling_rate"]

#: Attention backends validated with HyperFlow, per device family. ``_native_npu`` wraps
#: ``torch_npu.npu_fusion_attention``; ``_hyperflow_npu_fast`` wraps ``torch_npu.npu_prompt_flash_attention`` in BSH
#: layout with fp16 compute (see `hyperflow_h3.npu_attention`). The CUDA table is the upstream one.
NPU_ATTENTION_BACKENDS = {"sdpa": "native", "npu": "_native_npu", "npu_fast": "_hyperflow_npu_fast"}

# The dense attention backends validated with HyperFlow (1 and 4 GPUs), by short name -> diffusers backend name. Any
# other diffusers backend name is passed through but only with `--gpus 1`: `_flash_3` (source-built FA3) and
# `flash_4_hub` cannot run context parallel in diffusers, and none of the others is tested here.
ATTENTION_BACKENDS = {
    "sdpa": "native",  # PyTorch scaled_dot_product_attention, diffusers' default
    # FlashAttention-3 via `kernels` (kernels-community/flash-attn3, fetched from the Hub on first use; offline runs
    # need LOCAL_KERNELS=kernels-community/flash-attn3=<snapshot dir>). Runs under Ulysses context parallel.
    "fa3": "_flash_3_hub",
}

log = logging.getLogger("hyperflow_h3.examples")


def add_common_args(parser: argparse.ArgumentParser, *, workflow: str) -> None:
    group = parser.add_argument_group("model")
    group.add_argument("--model", default=DEFAULT_MODEL, help="MiniMax-H3 Hub id or local snapshot directory")
    group.add_argument("--weights", default=DEFAULT_WEIGHTS, help="HyperFlow LoRA: Hub id, directory or .safetensors")
    group.add_argument(
        "--weights-filename",
        default=None,
        help="pin one file of the --weights repo / directory instead of the default its hyperflow.json names",
    )
    group.add_argument(
        "--baseline",
        action="store_true",
        help="run the base Diffusers pipeline (no LoRA; --num-inference-steps is the sigma-point count)",
    )
    group.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="baseline only; sigma-grid points (the default 50 means 49 NFE); ignored with HyperFlow",
    )

    group = parser.add_argument_group("generation")
    group.add_argument("--prompt", required=True)
    group.add_argument("--num-frames", type=int, default=124, help="snapped up to 17n+5; 124 is ~5 s at 24 fps")
    group.add_argument("--height", type=int, default=None, help="multiple of 32; default follows the keyframe/16:9")
    group.add_argument("--width", type=int, default=None, help="multiple of 32")
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--output", type=Path, default=Path(f"hyperflow_{workflow}.mp4"))

    group = parser.add_argument_group("attention")
    group.add_argument(
        "--attention-backend",
        default="auto",
        help="dense attention on the DiT: auto (npu_fusion_attention on Ascend, PyTorch SDPA elsewhere), sdpa, npu "
        "(`torch_npu.npu_fusion_attention`), or fa3 (FlashAttention-3 via `pip install kernels`, CUDA only, works "
        "with --gpus N). Any other diffusers backend name (_flash_3, flash_4_hub, flash_hub, ...) is passed through "
        "and needs --gpus 1",
    )
    group.add_argument(
        "--npu-fast-attn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ascend only: use torch_npu.npu_prompt_flash_attention in BSH layout with fp16 inputs instead of "
        "npu_fusion_attention in BSND. Measured 330 -> 236 ms per 768p attention call (28%% of the dominant kernel) "
        "and slightly more accurate than the bf16 path. Defaults to on for NPU. Also applies under --gpus N (the "
        "Ulysses path keeps the stock kernel).",
    )
    group.add_argument(
        "--npu-fused-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ascend only: replace the DiT's 205 nn.RMSNorm calls per forward with torch_npu.npu_rms_norm, a single "
        "fused kernel (7.6-9.3x faster at the DiT's shapes, ~1.3 s per 768p step). Defaults to on for NPU.",
    )
    group.add_argument(
        "--sol-attn", action="store_true", help="enable NVIDIA Sol-Attn on the DiT blocks (SM80+; works with --gpus N)"
    )
    group.add_argument("--dense-steps", type=int, default=None, help="Sol-Attn: first N steps stay dense")
    group.add_argument("--dense-layers", type=int, nargs="*", default=None, help="Sol-Attn: layers that stay dense")
    group.add_argument("--sol-tau", type=float, default=None, help="Sol-Attn: sparsity threshold override")

    group = parser.add_argument_group("gpus")
    group.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="run the DiT context-parallel (Ulysses) over N GPUs, one torchrun rank each; started with plain python "
        "the script re-launches itself under torchrun. Default: min(4, visible GPUs), the degree MiniMax serves "
        "with. 1 = single GPU with CPU offload",
    )
    group.add_argument(
        "--tp",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="shard the DiT weights across N ranks with tensor parallelism instead of splitting the sequence "
        "(`--gpus`). Bare `--tp` uses every rank. Needs Ascend NPU or CUDA: the DiT is sharded after the LoRA is "
        "merged into the base weights, since PEFT layers cannot be sharded. Incompatible with --npu-offload "
        "(each rank already holds 1/N of the weights) and with --sol-attn.",
    )

    group = parser.add_argument_group("memory")
    group.add_argument(
        "--device",
        default="auto",
        help="execution device: auto (npu when torch_npu is usable, else cuda), npu, cuda or cpu; ignored with "
        "--gpus > 1 (each rank takes <device>:LOCAL_RANK)",
    )
    group.add_argument(
        "--no-offload",
        action="store_true",
        help="keep every component on --device (needs ~130 GB for bf16); default streams components via "
        "ComponentsManager auto CPU offload as in the official recipe",
    )
    group.add_argument(
        "--npu-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="group-offload the text encoder and DiT block by block instead of whole components (Ascend only). "
        "Defaults to on for NPU: a 61.27 GiB card cannot hold either 62 GB component, so module-level offload fails. "
        "Use --no-npu-offload to disable (required with --tp, which already shards the weights).",
    )
    group.add_argument(
        "--npu-offload-blocks",
        type=int,
        default=1,
        metavar="N",
        help="transformer blocks per group-offload group (default 1). Raise it to 2-5 for fewer, larger "
        "host-to-device copies when the card has headroom: peak VRAM grows by roughly N x 1.2 GiB.",
    )
    group.add_argument(
        "--memory-reserve-margin",
        default="24GB",
        help="--gpus 1: headroom the ComponentsManager keeps when it decides what to evict. 24GB is the value "
        "validated on H200: under ~18GB it evicts the 10 GB VAE instead of the 62 GB text encoder and starves the "
        "denoiser; below ~100 GB the two never fit together and the margin is irrelevant. With several GPUs nothing "
        "is evicted on nonzero ranks; rank 0 releases the text encoder after each call and reloads it when needed",
    )
    group.add_argument("-v", "--verbose", action="store_true")


def distributed_context() -> tuple[int, int, int]:
    """(rank, local_rank, world_size) from the torchrun environment; (0, 0, 1) outside it."""
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def requested_gpus(args: argparse.Namespace) -> int:
    """Ranks this run uses: WORLD_SIZE inside torchrun, else ``--tp N``, else `--gpus`, else min(4, visible)."""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if getattr(args, "tp", None) is not None:
        # Bare `--tp` (const 0) means "one rank per visible device".
        return args.tp or min(4, _device_count())
    return args.gpus if args.gpus is not None else min(4, _device_count())


def _device_count() -> int:
    """Visible accelerators of the active backend (NPU, else CUDA, else 0)."""
    module = npu_backend.device_module()
    count = getattr(module, "device_count", None)
    return int(count()) if count is not None else 0


def resolve_attention_backend(name: str, *, gpus: int) -> str:
    """Map a `--attention-backend` value onto diffusers' backend name; unvalidated backends need `--gpus 1`."""
    from diffusers.models.attention_dispatch import AttentionBackendName

    if name == "auto":
        # On Ascend the fast backend (npu_prompt_flash_attention, BSH, fp16) is the default: it is 28% faster than
        # npu_fusion_attention on the 768p attention kernel and slightly more accurate. On CUDA/CPU the upstream
        # default (SDPA) is kept.
        name = "npu_fast" if npu_backend.is_npu_available() else "sdpa"

    known_backends = ATTENTION_BACKENDS | NPU_ATTENTION_BACKENDS
    backend = known_backends.get(name, name)
    # `_hyperflow_npu_fast` is registered into diffusers' registry at model setup (see `npu_attention`), so it is not
    # an `AttentionBackendName` member yet when this validation runs.
    known = sorted(member.value for member in AttentionBackendName) + [FAST_BACKEND_NAME]
    if backend not in known:
        raise SystemExit(
            f"unknown --attention-backend {name!r}; use {', '.join(known_backends)} or a diffusers backend name: "
            + ", ".join(known)
        )
    if gpus > 1 and backend not in known_backends.values():
        raise SystemExit(
            f"--attention-backend {name} is only supported with --gpus 1; under context parallel use "
            + " or ".join(known_backends)
        )
    if backend in ("_native_npu", FAST_BACKEND_NAME) and not npu_backend.is_npu_available():
        raise SystemExit(f"--attention-backend {name} needs torch_npu on an Ascend host")
    return backend


def relaunch_under_torchrun_if_needed(args: argparse.Namespace) -> None:
    """Plain `python examples/...py` with more than one device: become `torchrun --nproc_per_node N <same argv>`."""
    if "WORLD_SIZE" in os.environ:  # already a torchrun rank
        return
    gpus = requested_gpus(args)
    if gpus <= 1:
        return
    visible = _device_count()
    if gpus > visible:
        raise SystemExit(f"--gpus {gpus} but only {visible} {npu_backend.default_device()} device(s) are visible")
    # Not `--standalone`: its c10d rendezvous advertises the hostname, which containers often cannot resolve and the
    # workers then retry forever. A static rendezvous on the loopback with a port that is free right now.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1", f"--nproc_per_node={gpus}", "--master_addr=127.0.0.1", f"--master_port={port}",
        *_relaunch_target(), *sys.argv[1:],
    ]  # fmt: skip
    print(f"re-launching under torchrun on {gpus} GPUs: {' '.join(cmd[1:8])} ...", file=sys.stderr, flush=True)
    os.execv(sys.executable, cmd)


def _relaunch_target() -> list[str]:
    """The entry point passed to `torchrun`, followed by the arguments it should receive.

    `torchrun` runs a *script path*, so passing `sys.argv[0]` straight through breaks under `python -m`: argv[0] is
    then the module's own file, and executing it as a script loses the package context
    (`ImportError: attempted relative import with no known parent package`). When the process was started as a module,
    re-launch through `torchrun -m <module>`, which keeps the package intact.
    """
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name:
        return ["-m", spec.name]
    return [str(Path(sys.argv[0]).resolve())]


# Media normalisation (absent in t2va) + Qwen3-VL; everything else is "rest". The official two-device recipe pops
# `text_encoder` alone; `before_encode` needs no model and goes with it so the rest starts at the VAE encoder.
CONDITIONER_BLOCKS = ("before_encode", "text_encoder")


def _is_group_offloaded(module: torch.nn.Module) -> bool:
    """True when `module` streams its own blocks/leaves, so a plain `.to()` would be refused."""
    from diffusers.hooks.group_offloading import _is_group_offload_enabled

    return _is_group_offload_enabled(module)


class _DeviceTensor:
    """A device tensor in the broadcast payload: travels as CPU bytes, lands on the receiving rank's own accelerator.

    Placement is part of the state's contract — `prompt_embeds` is on the device, `text_token_tags` on the CPU where
    `prepare_layout` builds the packed sequence — and a pickled device tensor would unpickle onto ``:0`` on every rank.
    """

    def __init__(self, tensor: torch.Tensor):
        self.cpu = tensor.cpu()


def _map_tensors(obj, fn):
    """`fn` over every tensor / `_DeviceTensor` inside containers and plain objects (objects are edited in place)."""
    if torch.is_tensor(obj) or isinstance(obj, _DeviceTensor):
        return fn(obj)
    if isinstance(obj, list):
        return [_map_tensors(x, fn) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_map_tensors(x, fn) for x in obj)
    if isinstance(obj, dict):
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
        for key, value in vars(obj).items():
            if torch.is_tensor(value) or isinstance(value, list | tuple | dict):
                setattr(obj, key, _map_tensors(value, fn))
    return obj


def _pack_for_broadcast(obj):
    return _map_tensors(obj, lambda t: _DeviceTensor(t) if torch.is_tensor(t) and t.device.type != "cpu" else t)


def _unpack_on(obj, device):
    return _map_tensors(obj, lambda t: t.cpu.to(device) if isinstance(t, _DeviceTensor) else t)


class SplitPipeline:
    """The official two-device split (text encoder apart from the rest), driven across torchrun ranks.

    Rank 0 runs `before_encode` + `text_encoder`, releases the 62 GB text encoder and broadcasts the conditioned
    state; every rank then runs the rest (VAE encode, context-parallel DiT, decoders) on its own GPU. A later call
    reloads the conditioner on rank 0 after parking its resident components on CPU. Ranks other than 0 never load the
    text encoder, so their DiT and VAEs stay resident and nothing is copied back to host RAM.
    """

    def __init__(
        self,
        conditioner,
        conditioner_manager,
        conditioner_factory: Callable[[], tuple[Any, Any]] | None,
        rest,
        rest_manager,
        *,
        rank: int,
        transformer_name: str,
        device: str,
        memory_reserve_margin: str,
    ):
        self.conditioner = conditioner  # None on ranks other than 0
        self._conditioner_manager = conditioner_manager
        self._conditioner_factory = conditioner_factory
        self.rest = rest
        self._rest_manager = rest_manager
        self.rank = rank
        self.transformer_name = transformer_name
        self.device = device
        self._memory_reserve_margin = memory_reserve_margin
        self._rest_inputs = {p.name for p in rest.blocks.inputs if p.name}
        self._conditioner_inputs = {p.name for p in conditioner.blocks.inputs if p.name} if conditioner else set()

    def __getattr__(self, name):  # transformer, vae, audio_vae, ... live on `rest`
        if name == "rest":  # not set yet (copy / unpickling): do not recurse into ourselves
            raise AttributeError(name)
        return getattr(self.rest, name)

    def _warm_transformer(self) -> None:
        # Place the DiT now rather than inside the first denoising step: on ranks other than 0 this overlaps rank 0's
        # text encoding. The offload hook sees the model already on its device and leaves it there. A group offloaded
        # DiT holds one block at a time and refuses `.to()`; its own hook places each block, so there is nothing to do.
        transformer = getattr(self.rest, self.transformer_name)
        if _is_group_offloaded(transformer):
            return
        transformer.to(self.device)

    def _reload_conditioner(self) -> None:
        if self._conditioner_factory is None:
            raise RuntimeError("rank 0 has no conditioner factory; the split pipeline cannot encode another prompt")
        if self._rest_manager is not None:
            # The conditioner has its own manager, so it cannot evict models owned by `rest`. Park those models before
            # loading the next 62 GB text encoder, then re-arm their hooks for the remainder of the workflow.
            self._rest_manager.disable_auto_cpu_offload()
            self._rest_manager.enable_auto_cpu_offload(
                device=self.device, memory_reserve_margin=self._memory_reserve_margin
            )
        self.conditioner, self._conditioner_manager = self._conditioner_factory()

    def __call__(self, output=None, **kwargs):
        box = [None]
        if self.rank == 0:
            if self.conditioner is None:
                self._reload_conditioner()
            started = time.perf_counter()
            state = self.conditioner(**{k: v for k, v in kwargs.items() if k in self._conditioner_inputs})
            # Ship only what the rest reads: not the prompt, the PIL keyframes or the decoded reference clips, which
            # every rank would otherwise unpickle and deepcopy along with the state.
            state.values = _pack_for_broadcast(
                {k: v for k, v in state.values.items() if k in self._rest_inputs or k not in self._conditioner_inputs}
            )
            # Free the text encoder instead of letting its manager copy it back to host memory.
            self.conditioner = self._conditioner_manager = None
            gc.collect()
            npu_backend.empty_cache(self.device)
            log.info(
                "conditioned in %.1fs; text encoder freed (%.1f GiB still allocated)",
                time.perf_counter() - started,
                npu_backend.memory_allocated(self.device) / 2**30,
            )
            self._warm_transformer()
            box[0] = state
        else:
            self._warm_transformer()
        dist.broadcast_object_list(box, src=0)
        state = box[0]
        state.values = _unpack_on(state.values, self.device)
        # Inputs the conditioner consumed are already in the state, some normalised (fl2va's height/width come from
        # the keyframe, ref2va's num_frames from the references); passing them again would put the raw values back.
        rest_kwargs = {k: v for k, v in kwargs.items() if k in self._rest_inputs and k not in state.values}
        return self.rest(state=state, output=output, **rest_kwargs)


def _load_kwargs(args: argparse.Namespace) -> dict:
    # diffusers' sharded-checkpoint loader queries the Hub unless `local_files_only` is passed explicitly;
    # HF_HUB_OFFLINE alone is not enough, so forward it for pre-downloaded models.
    load_kwargs: dict = dict(dtype=torch.bfloat16, local_files_only=HF_HUB_OFFLINE)
    if Path(args.model).is_dir():
        # A snapshot's modular_model_index.json still names the Hub repo for every component;
        # load them from the directory instead so a local `--model` is fully offline and revision-pinned.
        load_kwargs["pretrained_model_name_or_path"] = args.model
    return load_kwargs


def build_pipeline(args: argparse.Namespace, *, workflow: str):
    """Official pipeline for `workflow`, with the HyperFlow blocks + LoRA unless `--baseline`.

    One GPU: the official single-device recipe (ComponentsManager auto offload). Several: `SplitPipeline`.
    """
    # Before the torchrun re-launch, so a bad backend fails once in the parent instead of once per rank.
    if args.npu_fast_attn is False and args.attention_backend == "auto":
        # Explicit opt-out: keep the stock `npu_fusion_attention` path.
        args.attention_backend = "npu"
    elif args.npu_fast_attn is True and args.attention_backend == "auto":
        args.attention_backend = "npu_fast"
    attention_backend = resolve_attention_backend(args.attention_backend, gpus=requested_gpus(args))
    relaunch_under_torchrun_if_needed(args)
    # `--device auto` (the default) resolves to npu when torch_npu is usable, else cuda, else cpu. An explicit value
    # is honoured as given, so a CUDA run behaves exactly as before.
    args.device = npu_backend.resolve_device(args.device)
    if args.npu_offload is None:
        # On by default only for Ascend: a 61.27 GiB card cannot hold either 62 GB component.
        args.npu_offload = torch.device(args.device).type == "npu"
    if args.npu_fused_norm is None:
        # Ascend only; `nn.RMSNorm` stays in place everywhere else.
        args.npu_fused_norm = torch.device(args.device).type == "npu"
    rank, local_rank, world_size = distributed_context()
    if world_size > 1:
        args.device = f"{torch.device(args.device).type}:{local_rank}"
        npu_backend.device_module(args.device).set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                "hccl" if torch.device(args.device).type == "npu" else "nccl",
                device_id=torch.device(args.device),
            )

    level = logging.DEBUG if args.verbose else (logging.INFO if rank == 0 else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()

    if args.baseline:
        blocks = MiniMaxH3Blocks().get_workflow(workflow)
    else:
        blocks = hyperflow_blocks(workflow)
    load_kwargs = _load_kwargs(args)
    transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"

    conditioner = conditioner_manager = None
    conditioner_factory = None
    tp_requested = args.tp is not None and world_size > 1
    if world_size > 1:
        head = InsertableDict()
        for name in CONDITIONER_BLOCKS:
            if name in blocks.sub_blocks:
                head[name] = blocks.sub_blocks.pop(name)
        conditioner_blocks = SequentialPipelineBlocks.from_blocks_dict(head)
        if rank == 0:

            def conditioner_factory():
                loaded_manager = None if args.no_offload else ComponentsManager()
                loaded = conditioner_blocks.init_pipeline(args.model, components_manager=loaded_manager)
                loaded.load_components(**load_kwargs)
                if args.npu_offload or tp_requested:
                    # The rank-0 conditioner holds the 62.13 GiB text encoder, which does not fit in 61.27 GiB of
                    # HBM: stream it per layer instead of letting the manager try to make the whole model resident.
                    # Tensor parallelism shards the DiT, not the conditioner, so it needs this too.
                    npu_backend.enable_group_offloading(loaded, args.device, transformer_names=())
                elif args.no_offload:
                    loaded.to(args.device)
                else:
                    loaded_manager.enable_auto_cpu_offload(device=args.device)
                return loaded, loaded_manager

            conditioner, conditioner_manager = conditioner_factory()

    manager = None if args.no_offload else ComponentsManager()
    pipe = blocks.init_pipeline(args.model, components_manager=manager)
    # A 15 s request snaps to 362 frames (15.083 s) and the stock cap of 15.0 s would reject it, even though 362 is a
    # valid grid point. Raise the cap before any component loads so the whole pipeline (single- and multi-device
    # paths alike) sees it.
    admit_max_frames(pipe)
    pipe.load_components(**load_kwargs)
    log.info("components loaded in %.1fs", time.perf_counter() - started)

    transformer = getattr(pipe, transformer_name)
    if transformer is None:
        # load_components only warns when a component fails to load; fail here with a direct message.
        raise SystemExit(f"{transformer_name} did not load; see the diffusers warning above")

    if not args.baseline:
        # After load_components, before any group offload: PEFT swaps the target Linear modules.
        metadata = load_hyperflow_lora(pipe, args.weights, filename=args.weights_filename)
        log.info(
            "HyperFlow %s on %s: rank %s, gate %.3g, %d-step grid",
            metadata.version,
            transformer_name,
            metadata.lora_rank,
            metadata.gate,
            len(metadata.sigmas) - 1 if metadata.sigmas else -1,
        )

    tp_degree = 0
    if args.tp is not None and world_size > 1:
        tp_degree = world_size
        if args.sol_attn:
            raise SystemExit("--sol-attn replaces the attention processor, which --tp also does; pick one")
        if args.npu_offload:
            raise SystemExit(
                "--tp already keeps 1/N of every weight on each rank, so --npu-offload would fight it for the same "
                "memory; pass --no-npu-offload with --tp"
            )
        if args.no_offload:
            raise SystemExit("--no-offload puts a whole 62 GB component on the device; with --tp use --no-npu-offload")
        if not args.baseline:
            # PEFT cannot be sharded: its wrappers hold full-size LoRA matrices next to 1/N-width base weights.
            # Merging first is exact for inference (`y = (W + s·BA)x`) and leaves plain `nn.Linear` modules.
            from hyperflow_h3.lora import merge_hyperflow_lora

            merged = merge_hyperflow_lora(pipe)
            log.info("HyperFlow adapter merged into %d module(s) ahead of tensor parallelism", merged)

    # Always pinned, so `sdpa` means SDPA even with DIFFUSERS_ATTN_BACKEND set in the environment.
    if attention_backend == FAST_BACKEND_NAME:
        # Registers the backend into diffusers' registry first: it is not one of its built-in members.
        if not enable_fast_npu_attention(transformer):
            raise SystemExit("--attention-backend npu_fast needs torch_npu with npu_prompt_flash_attention")
    else:
        transformer.set_attention_backend(attention_backend)
    if args.npu_fused_norm and torch.device(args.device).type == "npu":
        # 205 nn.RMSNorm calls per forward; the fused kernel is 7.6-9.3x faster at the DiT's shapes.
        enable_fused_rms_norm(transformer)
    # set_attention_backend also switches the global default; the audio VAE is fp32 and flash rejects that.
    audio_vae = getattr(pipe, "audio_vae", None)
    if audio_vae is not None:
        audio_vae.set_attention_backend("native")
    log.info("dense attention backend: %s = %s (audio VAE: native)", args.attention_backend, attention_backend)

    if args.npu_offload or tp_degree:
        # Ascend 910 has 61.27 GiB of HBM; the bf16 text encoder (62.13 GiB) does not fit even alone, so module-level
        # offloading can never make it resident. Stream it in groups instead. With tensor parallelism the DiT is
        # already sharded to 1/N per rank and must *not* be group offloaded (diffusers rejects the combination).
        applied = npu_backend.enable_group_offloading(
            pipe,
            args.device,
            offload_transformer=not tp_degree,
            num_blocks_per_group=max(1, args.npu_offload_blocks),
        )
        if applied:
            log.info("NPU group offloading: %s", ", ".join(f"{k}={v}" for k, v in applied.items()))

    if tp_degree:
        from hyperflow_h3.tensor_parallel import enable_npu_tensor_parallel

        info = enable_npu_tensor_parallel(transformer, tp_degree=tp_degree, device=args.device)
        log.info(
            "tensor parallel: sharded %s over %d rank(s); %d attention processor(s) now derive heads from head_dim",
            transformer_name,
            info["tp_degree"],
            info["processors"],
        )
    elif world_size > 1:
        if getattr(type(transformer), "_cp_plan", None) is None:
            raise SystemExit(
                "this diffusers build predates context parallel for MiniMax-H3 (huggingface/diffusers#14407); "
                "install a newer diffusers or pass --gpus 1"
            )
        from diffusers import ContextParallelConfig

        # Ulysses splits the packed sequence across ranks and exchanges heads inside attention; `ulysses_anything`
        # lifts the "sequence length divisible by the degree" requirement, which a prompt's token count breaks.
        transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True))
        log.info("context parallel: Ulysses degree %d on %s", world_size, transformer_name)

    if args.sol_attn:
        if args.baseline:
            raise SystemExit(
                "--sol-attn is scheduled per HyperFlow step; it needs the HyperFlow blocks (no --baseline)"
            )
        if torch.device(args.device).type == "npu":
            raise SystemExit(
                "Sol-Attn is an NVIDIA CUDA kernel and has no Ascend build; drop --sol-attn on NPU "
                "(use --attention-backend npu for torch_npu.npu_fusion_attention)"
            )
        overrides = {}
        if args.dense_steps is not None:
            overrides["dense_steps"] = args.dense_steps
        if args.dense_layers is not None:
            overrides["dense_layers"] = tuple(args.dense_layers)
        if args.sol_tau is not None:
            overrides["tau"] = args.sol_tau
        recipe = enable_sol_attention(pipe, **overrides)
        log.info("Sol-Attn on: %s", recipe)

    if args.no_offload:
        pipe.to(args.device)
    else:
        # Arm auto offload only now: ComponentsManager.add() re-arms it with the default 3 GB margin whenever a
        # component is registered, so a margin passed before load_components is silently discarded. The strategy
        # runs when a component is first moved to the device and evicts the *smallest* resident set that makes
        # (free - margin) fit it; the margin therefore decides whether the 62 GB text encoder or the 10 GB VAE
        # leaves when the 62 GB DiT arrives, i.e. how much room the denoiser has for activations.
        manager.enable_auto_cpu_offload(device=args.device, memory_reserve_margin=args.memory_reserve_margin)

    if world_size > 1:
        return SplitPipeline(
            conditioner,
            conditioner_manager,
            conditioner_factory,
            pipe,
            manager,
            rank=rank,
            transformer_name=transformer_name,
            device=args.device,
            memory_reserve_margin=args.memory_reserve_margin,
        )
    return pipe


def generation_kwargs(args: argparse.Namespace) -> dict:
    kwargs = dict(
        prompt=args.prompt,
        num_frames=args.num_frames,
        # An NPU (or CUDA) `randn` rejects a CPU generator, so the generator is created on the execution backend.
        generator=npu_backend.make_generator(args.device, args.seed),
        output=OUTPUTS,
    )
    if args.height is not None:
        kwargs["height"] = args.height
    if args.width is not None:
        kwargs["width"] = args.width
    if args.baseline:
        kwargs["num_inference_steps"] = args.num_inference_steps
    return kwargs


def run_and_save(pipe, args: argparse.Namespace, **extra) -> Path | None:
    kwargs = generation_kwargs(args) | extra
    started = time.perf_counter()
    results = pipe(**kwargs)
    elapsed = time.perf_counter() - started
    if torch.device(args.device).type != "cpu":
        peak = npu_backend.max_memory_allocated(args.device) / 2**30
        if peak:
            log.info("peak accelerator memory: %.1f GiB", peak)

    if dist.is_initialized():
        # Every rank holds the same gathered result; let them all finish before rank 0 alone writes the file.
        dist.barrier()
        dist.destroy_process_group()
        if distributed_context()[0] != 0:
            return None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        results["videos"][0],
        fps=24,
        output_path=str(args.output),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    frames = len(results["videos"][0])
    log.info("generated %d frames + audio in %.1fs -> %s", frames, elapsed, args.output)
    return args.output
