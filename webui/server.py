# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""A small browser front end for HyperFlow H3 on Ascend NPU.

Standard library only (``http.server`` + ``email`` for multipart), so the WebUI adds no dependency to the inference
environment. It runs generation as one FIFO worker subprocess at a time — a 2 MP clip already needs the whole card —
and keeps every task's prompt, uploads, log and output in ``webui/runs/<id>/`` so a restart does not lose history.

Endpoints:

    GET  /                     the page
    GET  /api/config           model paths, device list, defaults, limits
    GET  /api/status           device status + queue
    GET  /api/tasks            task history
    GET  /api/tasks/<id>       one task
    GET  /api/tasks/<id>/log   raw log tail
    GET  /api/tasks/<id>/video the produced mp4
    POST /api/plan             resolve a canvas without running anything
    POST /api/tasks            submit (multipart: fields + uploads)
    POST /api/tasks/<id>/cancel
    POST /api/tasks/<id>/retry
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy as email_policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import devices, geometry

WEBUI_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBUI_DIR / "static"
RUNS_DIR = WEBUI_DIR / "runs"
PROJECT_ROOT = WEBUI_DIR.parent

MAX_PROMPT_CHARS = 8000
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCE_TOTAL = 12

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

#: The fastest runtime recipe measured on this box, used as the form default and by the "fastest" reset button.
#:
#: 1344x768 / 124 frames, `pipe()` wall time, same seed:
#:   * 2 cards CP + one block per offload group ............ 123.2 s   <- this
#:   * 2 cards CP + four blocks per group .................. 127.4 s
#:   * 1 card, group offload ............................... 203.9 s
#:   * 2 cards TP .......................................... 175.9 s (and 55.8 GiB peak)
#: 2 MP (1920x1088) keeps the same ranking: CP 452.1 s vs single 844.2 s.
#:
#: Parallelism depends on how many devices exist, so only the device-independent half lives here; the CP/single choice
#: is made per probe in `fastest_defaults`.
FASTEST_BACKEND = "auto"  # -> npu_fast on Ascend
FASTEST_OFFLOAD_BLOCKS = 1  # streams only support one block per group, and larger groups measured slower

#: Workflows the page can request, mapped onto the CLI entry point that implements them.
WORKFLOWS = {
    "t2va": "hyperflow-h3-fl2va",
    "fl2va": "hyperflow-h3-fl2va",
    "ref2va": "hyperflow-h3-ref2va",
}
WORKFLOW_LABELS = {
    "t2va": "文本生成",
    "fl2va": "关键帧生成",
    "ref2va": "参考生成",
}

#: Stage labels keyed by the log fragments the CLI prints.
STAGES: tuple[tuple[str, str, int], ...] = (
    ("loading", "加载模型", 8),
    ("HyperFlow adapter merged", "合并 LoRA", 12),
    ("components loaded", "组件就绪", 15),
    ("HyperFlow 1.0 on", "LoRA 已加载", 18),
    ("dense attention backend", "配置注意力", 20),
    ("group offloading", "配置卸载", 24),
    ("tensor parallel", "张量并行切分", 28),
    ("context parallel", "上下文并行", 28),
    ("conditioned in", "文本编码完成", 34),
    ("it/s", "去噪中", 60),
    ("Encoding video chunks", "编码视频", 92),
    ("generated", "完成", 100),
)

#: Log lines that mean the accelerator fell off the bus. `torch_npu` keeps the worker process alive and retries the
#: runtime call forever, so a task would otherwise sit in "running" with a frozen log until the service is restarted.
#: All of these come from the CANN runtime itself, never from the model code.
DEVICE_LOST_PATTERN = re.compile(
    r"lost heartbeat"
    r"|no valid device"
    r"|device is unavailable"
    r"|aclrtSynchronizeEvent failed"
    r"|ERR00100 PTA call acl api failed",
)

LOG_TAIL_LINES = 400


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------------------
# Device status
# ---------------------------------------------------------------------------------------------------------------
#
# The metrics live in `webui.devices`: it separates the physical HBM total (`npu-smi`) from the framework's
# allocatable budget (`torch.npu.mem_get_info`) instead of conflating them, and reads them from the stable torch API
# first. `read_device_status` is re-exported here because the HTTP layer and the tests use it from this module.

read_device_status = devices.read_device_status


# ---------------------------------------------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class UploadedFile:
    field: str
    filename: str
    data: bytes

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower()

    @property
    def kind(self) -> str:
        if self.suffix in IMAGE_EXT:
            return "image"
        if self.suffix in VIDEO_EXT:
            return "video"
        if self.suffix in AUDIO_EXT:
            return "audio"
        return "unknown"


def _safe_name(filename: str) -> str:
    stem = Path(filename or "upload").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:120] or "upload"


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, list[str]], list[UploadedFile]]:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii", "strict")
    message = BytesParser(policy=email_policy.default).parsebytes(header + body)
    if message.get_content_type() != "multipart/form-data" or not message.is_multipart():
        raise SubmissionError("POST /api/tasks 需要 multipart/form-data")
    fields: dict[str, list[str]] = {}
    uploads: list[UploadedFile] = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            if len(payload) > MAX_UPLOAD_BYTES:
                raise SubmissionError(f"文件 {filename} 超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
            uploads.append(UploadedFile(field=name, filename=_safe_name(filename), data=payload))
        else:
            fields.setdefault(name, []).append(payload.decode("utf-8", "replace"))
    return fields, uploads


class SubmissionError(ValueError):
    """User-facing validation failure (HTTP 400)."""


# ---------------------------------------------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------------------------------------------


def _one(fields: dict[str, list[str]], name: str, default: str = "") -> str:
    values = fields.get(name)
    return values[-1] if values else default


def _checkbox(fields: dict[str, list[str]], name: str, default: bool = False) -> bool:
    raw = _one(fields, name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "on", "yes"}


def _number(fields: dict[str, list[str]], name: str, default: float) -> float:
    raw = _one(fields, name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SubmissionError(f"{name} 必须是数字") from None


def _integer(
    fields: dict[str, list[str]], name: str, default: int | Callable[[], int], minimum: int, maximum: int
) -> int:
    raw = _one(fields, name, "").strip()
    if not raw:
        # `default` may be a callable: a device probe must not run unless the field is actually missing.
        return default() if callable(default) else default
    try:
        value = int(raw)
    except ValueError:
        raise SubmissionError(f"{name} 必须是整数") from None
    if not (minimum <= value <= maximum):
        raise SubmissionError(f"{name} 必须在 {minimum}–{maximum} 之间")
    return value


@dataclass
class TaskRequest:
    mode: str
    workflow: str
    prompt: str
    seed: int
    plan: dict[str, Any]
    devices: int
    parallel: str
    tensor_parallel: bool
    npu_offload: bool
    offload_blocks: int
    baseline: bool
    attention_backend: str
    npu_fused_norm: bool = True
    image: UploadedFile | None = None
    last_image: UploadedFile | None = None
    references: list[UploadedFile] = field(default_factory=list)


def default_devices() -> int:
    """Devices a fresh submission uses: every visible NPU, capped at the four the CLI supports.

    The HTTP layer rarely reaches this: the form always posts `devices`. It is kept for `POST /api/plan` and for
    clients that omit the field, and it is the only place that has to pay for a device probe.
    """
    status = read_device_status(enrich=False)
    return max(1, min(4, status.get("count") or 1))


def validate_request(fields: dict[str, list[str]], uploads: list[UploadedFile]) -> TaskRequest:
    prompt = _one(fields, "prompt", "").strip()
    if not prompt:
        raise SubmissionError("提示词不能为空")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise SubmissionError(f"提示词最多 {MAX_PROMPT_CHARS} 字符")

    mode = _one(fields, "mode", "t2va").strip().lower()
    if mode not in WORKFLOWS:
        raise SubmissionError(f"未知生成模式 {mode!r}")

    aspect = _one(fields, "aspect", geometry.DEFAULT_ASPECT).strip()
    if aspect not in geometry.ASPECTS:
        raise SubmissionError(f"不支持的宽高比 {aspect!r}")
    megapixels = _number(fields, "megapixels", geometry.DEFAULT_MP)
    seconds = _number(fields, "seconds", geometry.MIN_SECONDS)

    override = _checkbox(fields, "override", False)
    if override:
        width = _integer(fields, "width", 0, geometry.CANVAS_MULTIPLE, 8192)
        height = _integer(fields, "height", 0, geometry.CANVAS_MULTIPLE, 8192)
        if not width or not height:
            raise SubmissionError("自定义画布需要同时给出宽和高")
        try:
            geometry.validate_canvas(width, height)
        except ValueError as exc:
            raise SubmissionError(str(exc)) from None
        plan = {
            "width": width,
            "height": height,
            "aspect": aspect,
            "aspect_label": geometry.ASPECT_LABELS[aspect],
            "megapixels": round(width * height / 1024 / 1024, 4),
            "target_megapixels": megapixels,
            "frames": geometry.frames_for_seconds(seconds),
            "fps": geometry.FPS,
            "duration_seconds": round(geometry.frames_for_seconds(seconds) / geometry.FPS, 3),
            "multiple": geometry.CANVAS_MULTIPLE,
            "tokens": geometry.estimate_tokens(width, height, geometry.frames_for_seconds(seconds)),
        }
    else:
        try:
            plan = geometry.plan_canvas(megapixels=megapixels, aspect=aspect, seconds=seconds)
        except ValueError as exc:
            raise SubmissionError(str(exc)) from None

    images = [item for item in uploads if item.field in ("image", "last_image") and item.kind == "image"]
    for item in images:
        if item.field == "image" and item.kind != "image":
            raise SubmissionError(f"{item.filename} 不是图片")
    image = next((item for item in images if item.field == "image"), None)
    last_image = next((item for item in images if item.field == "last_image"), None)

    refs = [item for item in uploads if item.field.startswith("reference")]
    if mode == "ref2va":
        counts = {"image": 0, "video": 0, "audio": 0}
        for item in refs:
            if item.kind == "unknown":
                raise SubmissionError(f"无法识别 {item.filename} 的类型（支持图片 / 视频 / 音频）")
            counts[item.kind] += 1
        if counts["image"] > MAX_REFERENCE_IMAGES:
            raise SubmissionError(f"参考图片最多 {MAX_REFERENCE_IMAGES} 张")
        if counts["video"] > MAX_REFERENCE_VIDEOS:
            raise SubmissionError(f"参考视频最多 {MAX_REFERENCE_VIDEOS} 个")
        if counts["audio"] > MAX_REFERENCE_AUDIOS:
            raise SubmissionError(f"参考音频最多 {MAX_REFERENCE_AUDIOS} 个")
        if len(refs) > MAX_REFERENCE_TOTAL:
            raise SubmissionError(f"参考文件合计最多 {MAX_REFERENCE_TOTAL} 个")
        if not (counts["image"] or counts["video"]):
            raise SubmissionError("参考生成至少需要一张图片或一个视频")
        if counts["audio"] and not (counts["image"] or counts["video"]):
            raise SubmissionError("音频参考必须与图片或视频参考一起使用")
        # Keep the order the user picked; the CLI numbers <Picture n>/<Video n>/<Audio n> by position.
        refs = sorted(refs, key=lambda item: int(re.sub(r"\D", "", item.field) or 0))
    else:
        refs = []
    if mode == "fl2va" and image is None and last_image is None:
        raise SubmissionError("关键帧生成至少需要首帧或尾帧")

    workflow = "ref2va" if mode == "ref2va" else "fl2va"
    if mode == "t2va" and (image or last_image):
        workflow = "fl2va"

    devices = _integer(fields, "devices", default_devices, 1, 8)
    parallel = _one(fields, "parallel", "auto").strip().lower()
    if parallel not in {"auto", "cp", "tp", "single"}:
        raise SubmissionError("并行方式必须是 auto / cp / tp / single")
    if parallel == "single" and devices != 1:
        raise SubmissionError("单卡模式请把设备数设为 1")
    if parallel == "tp" and devices < 2:
        raise SubmissionError("张量并行至少需要 2 个设备")
    if parallel == "cp" and devices < 2:
        raise SubmissionError("上下文并行至少需要 2 个设备")
    if parallel == "auto":
        parallel = "cp" if devices >= 2 else "single"

    npu_offload = _checkbox(fields, "npu_offload", parallel != "tp")
    if parallel == "tp" and npu_offload:
        raise SubmissionError("张量并行已按 rank 切分权重，不能再叠加分组卸载")
    offload_blocks = _integer(fields, "offload_blocks", 1, 1, 10)

    seed_raw = _one(fields, "seed", "").strip()
    if seed_raw in ("", "random", "-1"):
        seed = secrets.randbelow(2**31)
    else:
        try:
            seed = int(seed_raw)
        except ValueError:
            raise SubmissionError("种子必须是整数") from None
        if not (0 <= seed < 2**31):
            raise SubmissionError("种子必须在 0 到 2³¹-1 之间")

    attention = _one(fields, "attention_backend", "auto").strip()
    if attention not in {"auto", "sdpa", "npu", "npu_fast"}:
        raise SubmissionError("注意力后端必须是 auto / sdpa / npu / npu_fast")
    # The DiT has 205 RMSNorm calls per forward; the fused Ascend kernel is 7.6-9.3x faster at its shapes.
    fused_norm = _checkbox(fields, "npu_fused_norm", True)
    baseline = _checkbox(fields, "baseline", False)

    return TaskRequest(
        mode=mode,
        workflow=workflow,
        prompt=prompt,
        seed=seed,
        plan=plan,
        devices=devices,
        parallel=parallel,
        tensor_parallel=parallel == "tp",
        npu_offload=npu_offload,
        offload_blocks=offload_blocks,
        baseline=baseline,
        attention_backend=attention,
        npu_fused_norm=fused_norm,
        image=image,
        last_image=last_image,
        references=refs,
    )


def discover_model_paths() -> tuple[str | None, str | None]:
    """Local base-model snapshot and LoRA file, if the repository has them checked out.

    A WebUI started with no environment configuration should use what is already on disk rather than reaching for the
    Hub: the CLI's defaults are Hub ids, which fail offline. `HYPERFLOW_WEBUI_MODEL` / `HYPERFLOW_WEBUI_WEIGHTS`
    override the detection.
    """
    model = os.environ.get("HYPERFLOW_WEBUI_MODEL")
    weights = os.environ.get("HYPERFLOW_WEBUI_WEIGHTS")

    candidates = [
        PROJECT_ROOT / "models" / "MiniMax-H3",
        PROJECT_ROOT.parent / "models" / "MiniMax-H3",
    ]
    if model is None:
        for candidate in candidates:
            if (candidate / "modular_model_index.json").is_file() or (candidate / "model_index.json").is_file():
                model = str(candidate)
                break

    if weights is None:
        for directory in (PROJECT_ROOT / "models" / "hyperflow", PROJECT_ROOT.parent / "models" / "hyperflow"):
            if not directory.is_dir():
                continue
            files = sorted(directory.glob("*.safetensors"))
            if files:
                weights = str(files[0])
                break
    return model, weights


def build_command(request: TaskRequest, run_dir: Path) -> list[str]:
    """The CLI argv for a validated request. Paths live under the run directory."""
    module = (
        "hyperflow_h3.examples.generate_ref2va"
        if request.workflow == "ref2va"
        else "hyperflow_h3.examples.generate_fl2va"
    )
    cli = [
        sys.executable,
        "-u",
        "-m",
        module,
        "--prompt",
        request.prompt,
        "--width",
        str(request.plan["width"]),
        "--height",
        str(request.plan["height"]),
        "--num-frames",
        str(request.plan["frames"]),
        "--seed",
        str(request.seed),
        "--output",
        str(run_dir / "output.mp4"),
        "--device",
        "npu",
        "--gpus",
        str(request.devices),
        "--attention-backend",
        request.attention_backend,
    ]
    if request.baseline:
        cli.append("--baseline")
    if request.npu_offload:
        cli += ["--npu-offload", "--npu-offload-blocks", str(request.offload_blocks)]
    else:
        cli.append("--no-npu-offload")
    cli += ["--npu-fused-norm"] if request.npu_fused_norm else ["--no-npu-fused-norm"]
    if request.parallel == "tp":
        cli += ["--tp", str(request.devices)]
    if request.mode == "ref2va":
        for reference in request.references:
            cli += ["--ref", f"{reference.kind}:{run_dir / 'inputs' / reference.filename}"]
    else:
        if request.image is not None:
            cli += ["--image", str(run_dir / "inputs" / request.image.filename)]
        if request.last_image is not None:
            cli += ["--last-image", str(run_dir / "inputs" / request.last_image.filename)]
    model, weights = discover_model_paths()
    if model:
        cli += ["--model", model]
    if weights:
        cli += ["--weights", weights]
    return cli


# ---------------------------------------------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Task:
    id: str
    request: TaskRequest
    created_at: str
    status: str = "queued"
    stage: str = "queued"
    stage_label: str = "排队等待"
    progress: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    log_bytes: int = 0
    output_ready: bool = False
    cancel_requested: bool = False
    #: Set when the log shows the accelerator went away; the worker kills the frozen process instead of waiting.
    device_lost: bool = False
    process_started_at: float | None = None
    queue_position: int | None = None
    log_path: str = ""
    run_dir: str = ""

    def to_public(self) -> dict[str, Any]:
        elapsed = self.elapsed_seconds
        if self.status == "running" and self.process_started_at is not None:
            elapsed = time.monotonic() - self.process_started_at
        return {
            "id": self.id,
            "mode": self.request.mode,
            "mode_label": WORKFLOW_LABELS[self.request.mode],
            "workflow": self.request.workflow,
            "status": self.status,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "progress": self.progress,
            "prompt": self.request.prompt,
            "prompt_chars": len(self.request.prompt),
            "seed": self.request.seed,
            "plan": self.request.plan,
            "width": self.request.plan["width"],
            "height": self.request.plan["height"],
            "frames": self.request.plan["frames"],
            "seconds": self.request.plan["duration_seconds"],
            "megapixels": self.request.plan["megapixels"],
            "devices": self.request.devices,
            "parallel": self.request.parallel,
            "tensor_parallel": self.request.tensor_parallel,
            "npu_offload": self.request.npu_offload,
            "offload_blocks": self.request.offload_blocks,
            "baseline": self.request.baseline,
            "attention_backend": self.request.attention_backend,
            "npu_fused_norm": self.request.npu_fused_norm,
            "references": [f"{item.kind}:{item.filename}" for item in self.request.references],
            "image": self.request.image.filename if self.request.image else None,
            "last_image": self.request.last_image.filename if self.request.last_image else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(elapsed, 2),
            "error": self.error,
            "queue_position": self.queue_position,
            "log_bytes": self.log_bytes,
            "output_ready": self.output_ready,
        }


class TaskManager:
    """A FIFO queue, one worker thread, one inference subprocess at a time."""

    def __init__(self, runs_dir: Path = RUNS_DIR) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tasks: dict[str, Task] = {}
        self._queue: deque[str] = deque()
        self._wake = threading.Condition(self._lock)
        self._process: subprocess.Popen | None = None
        self._current: str | None = None
        self._stop = threading.Event()
        self._device_cache: dict[str, Any] = {"at": 0.0, "value": None}
        self._load_history()
        self._worker = threading.Thread(target=self._worker_loop, name="hyperflow-webui-worker", daemon=True)
        self._worker.start()

    # -- persistence -------------------------------------------------------------------------------------------

    def _task_dir(self, task_id: str) -> Path:
        return self.runs_dir / task_id

    def _write(self, task: Task) -> None:
        payload = task.to_public()
        payload["request"] = {
            "mode": task.request.mode,
            "workflow": task.request.workflow,
            "prompt": task.request.prompt,
            "seed": task.request.seed,
            "plan": task.request.plan,
            "devices": task.request.devices,
            "parallel": task.request.parallel,
            "npu_offload": task.request.npu_offload,
            "offload_blocks": task.request.offload_blocks,
            "baseline": task.request.baseline,
            "attention_backend": task.request.attention_backend,
            "npu_fused_norm": task.request.npu_fused_norm,
            "references": [f"{item.kind}:{item.filename}" for item in task.request.references],
            "image": task.request.image.filename if task.request.image else None,
            "last_image": task.request.last_image.filename if task.request.last_image else None,
        }
        path = Path(task.run_dir) / "task.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_history(self) -> None:
        for entry in sorted(self.runs_dir.glob("*/task.json"), key=lambda p: p.stat().st_mtime):
            try:
                payload = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            request = payload.get("request") or {}
            plan = request.get("plan") or {}
            if not plan:
                continue
            task = Task(
                id=payload["id"],
                request=TaskRequest(
                    mode=request.get("mode", "t2va"),
                    workflow=request.get("workflow", "fl2va"),
                    prompt=request.get("prompt", ""),
                    seed=int(request.get("seed", 0)),
                    plan=plan,
                    devices=int(request.get("devices", 1)),
                    parallel=request.get("parallel", "single"),
                    tensor_parallel=request.get("parallel") == "tp",
                    npu_offload=bool(request.get("npu_offload", True)),
                    offload_blocks=int(request.get("offload_blocks", 1)),
                    baseline=bool(request.get("baseline", False)),
                    attention_backend=request.get("attention_backend", "auto"),
                    npu_fused_norm=bool(request.get("npu_fused_norm", True)),
                ),
                created_at=payload.get("created_at", utc_now()),
                status="interrupted",
                stage="interrupted",
                stage_label="已中断（服务重启）",
                elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
                error="WebUI 重启，任务未完成",
                log_path=str(entry.parent / "run.log"),
                run_dir=str(entry.parent),
                output_ready=(entry.parent / "output.mp4").is_file(),
            )
            self._tasks[task.id] = task

    # -- device status -----------------------------------------------------------------------------------------

    def device_status(self, max_age: float = 3.0, wait: bool = True) -> dict[str, Any]:
        """Cached device metrics.

        A probe imports torch_npu in a child process and runs `npu-smi`, so it takes seconds. `wait=False` returns the
        last value (or a "warming up" placeholder) and refreshes in the background, which keeps `/api/config` and the
        first page load fast.
        """
        now = time.monotonic()
        with self._lock:
            cached = self._device_cache
            if cached["value"] is not None:
                age = now - cached["at"]
                # A missing NPU triggers the slow probe; do not retry it on every poll.
                if age < (max_age if cached["value"].get("available") else 30.0):
                    return dict(cached["value"])
            refreshing = cached.get("refreshing", False)
            if not wait and cached["value"] is not None:
                if not refreshing:
                    self._spawn_refresh()
                return dict(cached["value"])
            if not wait and not refreshing:
                self._spawn_refresh()
                return {
                    "available": False,
                    "source": None,
                    "reason": "正在探测 NPU…",
                    "count": 0,
                    "devices": [],
                    "pending": True,
                }
        value = read_device_status()
        with self._lock:
            self._device_cache = {"at": time.monotonic(), "value": value}
        return dict(value)

    def _spawn_refresh(self) -> None:
        with self._lock:
            if self._device_cache.get("refreshing"):
                return
            self._device_cache["refreshing"] = True

        def run() -> None:
            value = read_device_status()
            with self._lock:
                self._device_cache = {"at": time.monotonic(), "value": value, "refreshing": False}

        threading.Thread(target=run, name="hyperflow-webui-device-probe", daemon=True).start()

    def _device_busy(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # -- queue -------------------------------------------------------------------------------------------------

    def queue_snapshot(self) -> dict[str, Any]:
        with self._lock:
            running = [task.to_public() for task in self._tasks.values() if task.status == "running"]
            queued = [task.to_public() for task in self._tasks.values() if task.status == "queued"]
            queued.sort(key=lambda item: item["created_at"])
            for index, item in enumerate(queued, start=1):
                item["queue_position"] = index
            return {
                "active": running[0] if running else None,
                "queued": queued,
                "queued_count": len(queued),
                "running_count": len(running),
            }

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            tasks = sorted(self._tasks.values(), key=lambda item: item.created_at, reverse=True)
            return [task.to_public() for task in tasks[:limit]]

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def read_log(self, task_id: str, tail_lines: int = LOG_TAIL_LINES) -> str:
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        path = Path(task.log_path)
        if not path.is_file():
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=tail_lines))

    def artifact(self, task_id: str, name: str) -> Path | None:
        task = self.get(task_id)
        if task is None:
            return None
        path = Path(task.run_dir) / name
        return path if path.is_file() else None

    # -- submission --------------------------------------------------------------------------------------------

    def submit(self, request: TaskRequest) -> Task:
        # Refuse to queue work when the card cannot run: the run would hang for tens of minutes and then fail with a
        # device error, which is worse than a clear rejection up front.
        status = self.device_status(wait=False)
        if status.get("available") is False and not status.get("pending"):
            raise SubmissionError(f"NPU 当前不可用：{status.get('reason') or '未检测到可用设备'}")
        task_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        run_dir = self._task_dir(task_id)
        (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
        for upload in [request.image, request.last_image, *request.references]:
            if upload is None:
                continue
            (run_dir / "inputs" / upload.filename).write_bytes(upload.data)
        task = Task(
            id=task_id,
            request=request,
            created_at=utc_now(),
            log_path=str(run_dir / "run.log"),
            run_dir=str(run_dir),
        )
        (run_dir / "run.log").touch()
        self._write(task)
        with self._lock:
            self._tasks[task_id] = task
            self._queue.append(task_id)
            self._wake.notify_all()
        return task

    def cancel(self, task_id: str) -> str:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.status in {"done", "failed", "cancelled", "interrupted"}:
                return "already-finished"
            task.cancel_requested = True
            if task.status == "queued":
                try:
                    self._queue.remove(task_id)
                except ValueError:
                    pass
                task.status = "cancelled"
                task.stage = "cancelled"
                task.stage_label = "已取消"
                task.finished_at = utc_now()
                self._write(task)
                return "cancelled"
            process = self._process
        if process is not None and process.poll() is None:
            _terminate_group(process)
            return "terminating"
        return "requested"

    def retry(self, task_id: str) -> Task:
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        with self._lock:
            if task.status in {"queued", "running"}:
                raise SubmissionError("任务仍在队列中，无法重试")
            self._queue.append(task_id)
            task.status = "queued"
            task.stage = "queued"
            task.stage_label = "排队等待"
            task.progress = 0
            task.error = None
            task.cancel_requested = False
            task.finished_at = None
            task.started_at = None
            task.elapsed_seconds = 0.0
            task.output_ready = (Path(task.run_dir) / "output.mp4").is_file()
            Path(task.log_path).write_text("", encoding="utf-8")
            self._write(task)
            self._wake.notify_all()
        return task

    # -- worker ------------------------------------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._queue and not self._stop.is_set():
                    self._wake.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                task_id = self._queue.popleft()
                task = self._tasks[task_id]
                task.status = "running"
                task.stage = "starting"
                task.stage_label = "启动中"
                task.started_at = utc_now()
                task.process_started_at = time.monotonic()
                task.queue_position = None
                self._current = task_id
                self._write(task)
            try:
                self._execute(task)
            except Exception as exc:  # noqa: BLE001 - a worker must never die
                self._fail(task, f"{type(exc).__name__}: {exc}")
            finally:
                with self._lock:
                    task.elapsed_seconds = (
                        time.monotonic() - task.process_started_at if task.process_started_at else 0.0
                    )
                    task.process_started_at = None
                    task.finished_at = utc_now()
                    self._current = None
                    self._process = None
                    if task.status == "running":
                        task.status = "done" if task.output_ready else "failed"
                        if task.status == "done":
                            task.stage = "done"
                            task.stage_label = "完成"
                            task.progress = 100
                        else:
                            task.error = task.error or "进程结束但没有产物，见日志"
                    self._write(task)
                self._device_cache = {"at": 0.0, "value": None}

    def _execute(self, task: Task) -> None:
        run_dir = Path(task.run_dir)
        command = build_command(task.request, run_dir)
        self._set_stage(task, "loading", "加载模型", 5)
        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env["PYTHONUNBUFFERED"] = "1"
        with Path(task.log_path).open("a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                env=env,
                start_new_session=True,
            )
            with self._lock:
                self._process = process
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                with self._lock:
                    task.log_bytes = log.tell()
                    self._advance(task, line)
                    lost = task.device_lost
                if lost:
                    # The runtime has already given up on the device; draining stdout would block forever because the
                    # process is stuck inside a retrying ACL call. Stop it now and report why.
                    log.write(f"\n[webui] 设备已失效，终止卡死的进程（{process.pid}）\n")
                    log.flush()
                    _terminate_group(process, timeout=5.0)
                    break
            returncode = process.wait()
        task.output_ready = (run_dir / "output.mp4").is_file()
        if task.device_lost:
            task.status = "failed"
            task.stage = "failed"
            task.stage_label = "设备失效"
            task.error = "NPU 掉线（lost heartbeat / no valid device），任务已终止；请检查 npu-smi 并重启服务后再试"
            return
        if task.cancel_requested:
            task.status = "cancelled"
            task.stage = "cancelled"
            task.stage_label = "已取消"
            return
        if returncode != 0:
            task.status = "failed"
            task.stage = "failed"
            task.stage_label = "失败"
            task.error = f"进程退出码 {returncode}，见日志"
            return
        if not task.output_ready:
            task.status = "failed"
            task.stage = "failed"
            task.stage_label = "失败"
            task.error = "进程成功退出但没有 output.mp4"
            return
        task.status = "done"
        task.stage = "done"
        task.stage_label = "完成"
        task.progress = 100
        metadata = {
            "task": task.to_public(),
            "command": command,
            "finished_at": utc_now(),
        }
        (run_dir / "output.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _set_stage(self, task: Task, stage: str, label: str, progress: int) -> None:
        with self._lock:
            task.stage = stage
            task.stage_label = label
            task.progress = max(task.progress, progress)
            self._write(task)

    def _advance(self, task: Task, line: str) -> None:
        for fragment, label, progress in STAGES:
            if fragment in line:
                task.stage = fragment.replace(" ", "-").lower()
                task.stage_label = label
                task.progress = max(task.progress, progress)
                break
        # The denoise bar prints `n/8`, so the step count drives the middle of the progress bar.
        step = re.search(r"(\d+)/8 \[", line)
        if step and "it/s" in line:
            task.progress = max(task.progress, min(90, 34 + int(step.group(1)) * 7))
        percent = re.search(r"Encoding video chunks:\s+(\d+)%", line)
        if percent:
            task.progress = max(task.progress, min(99, 90 + int(percent.group(1)) // 10))
        # A device that falls off the bus leaves the worker process alive but frozen: the runtime keeps retrying
        # `aclrtSynchronizeEvent` forever, so `for line in process.stdout` never ends and the task would sit in
        # "running" until someone restarts the service. These lines are terminal; kill it and report the real cause.
        if DEVICE_LOST_PATTERN.search(line):
            task.device_lost = True

    def _fail(self, task: Task, message: str) -> None:
        with self._lock:
            task.status = "failed"
            task.stage = "failed"
            task.stage_label = "失败"
            task.error = message

    def close(self) -> None:
        """Stop the worker and make sure the accelerator is released before the process exits.

        Order matters: set the stop flag first so the worker cannot pick up the next queued task, then kill the
        current process group. Without this the daemon worker would be torn down by the interpreter while a
        `torchrun` child kept the device busy.
        """
        self._stop.set()
        with self._lock:
            process = self._process
            self._wake.notify_all()
        if process is not None and process.poll() is None:
            _terminate_group(process, timeout=8.0)
        # Give the worker a moment to observe the stop flag and write its final task state.
        self._worker.join(timeout=5.0)


def _terminate_group(process: subprocess.Popen, timeout: float = 15.0) -> None:
    """SIGTERM the whole group (the CLI may re-launch itself under torchrun), then SIGKILL.

    A frozen device call can swallow SIGTERM: `aclrtSynchronizeEvent` retries inside the runtime, so the process never
    reaches a Python-level signal handler. `timeout` therefore has to be short for that case, and SIGKILL follows.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    # Reap the direct child so it does not linger as a zombie even when a grandchild outlives it.
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------------------------


def _static(name: str) -> bytes:
    path = (STATIC_DIR / name).resolve()
    if STATIC_DIR not in path.parents and path != STATIC_DIR:
        raise FileNotFoundError(name)
    return path.read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "HyperFlowWebUI/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def manager(self) -> TaskManager:
        return self.server.task_manager  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than the default
        if os.environ.get("HYPERFLOW_WEBUI_ACCESS_LOG"):
            super().log_message(fmt, *args)

    # -- responses ---------------------------------------------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _not_found(self) -> None:
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def _read_body(self, limit: int = MAX_UPLOAD_BYTES * 2) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > limit:
            raise SubmissionError("请求体过大")
        return self.rfile.read(length)

    def _read_json(self) -> dict[str, Any]:
        body = self._read_body(4 * 1024 * 1024)
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            raise SubmissionError("请求体不是合法 JSON") from None
        if not isinstance(payload, dict):
            raise SubmissionError("请求体必须是 JSON 对象")
        return payload

    # -- GET ---------------------------------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, _static("index.html"), "text/html; charset=utf-8")
            return
        if path == "/favicon.ico":
            self._send(HTTPStatus.OK, _static("favicon.svg"), "image/svg+xml")
            return
        if path.startswith("/static/"):
            name = path.removeprefix("/static/")
            try:
                body = _static(name)
            except (FileNotFoundError, OSError):
                self._not_found()
                return
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if content_type.startswith("text/"):
                content_type += "; charset=utf-8"
            self._send(HTTPStatus.OK, body, content_type)
            return
        if path == "/api/config":
            # Non-blocking: a probe takes ~10 s here (torch_npu import + npu-smi), far past a page-load budget.
            status = self.manager.device_status(wait=False)
            self._json(
                HTTPStatus.OK,
                {
                    "workflows": [{"id": key, "label": WORKFLOW_LABELS[key]} for key in WORKFLOWS],
                    "aspects": [
                        {"id": key, "label": geometry.ASPECT_LABELS[key], "ratio": list(value)}
                        for key, value in geometry.ASPECTS.items()
                    ],
                    "defaults": fastest_defaults(status),
                    "limits": {
                        "megapixels": [geometry.MIN_MP, geometry.MAX_MP],
                        "seconds": [geometry.MIN_SECONDS, geometry.MAX_SECONDS],
                        "multiple": geometry.CANVAS_MULTIPLE,
                        "max_prompt_chars": MAX_PROMPT_CHARS,
                        "references": {
                            "images": MAX_REFERENCE_IMAGES,
                            "videos": MAX_REFERENCE_VIDEOS,
                            "audios": MAX_REFERENCE_AUDIOS,
                            "total": MAX_REFERENCE_TOTAL,
                        },
                        "devices": max(1, status["count"]),
                    },
                    "model": _model_config(),
                    "device": status,
                    "fps": geometry.FPS,
                },
            )
            return
        if path == "/api/status":
            self._json(
                HTTPStatus.OK,
                {"device": self.manager.device_status(wait=False), "queue": self.manager.queue_snapshot()},
            )
            return
        if path == "/api/tasks":
            self._json(HTTPStatus.OK, {"tasks": self.manager.history()})
            return
        match = re.fullmatch(r"/api/tasks/([^/]+)(?:/(log|video|metadata))?", path)
        if not match:
            self._not_found()
            return
        task_id = match.group(1)
        task = self.manager.get(task_id)
        if task is None:
            self._not_found()
            return
        suffix = match.group(2)
        if suffix == "log":
            query = parse_qs(parsed.query)
            try:
                lines = int(query.get("lines", [str(LOG_TAIL_LINES)])[0])
            except ValueError:
                lines = LOG_TAIL_LINES
            try:
                text = self.manager.read_log(task_id, tail_lines=max(1, min(lines, 5000)))
            except (KeyError, OSError):
                self._not_found()
                return
            self._send(HTTPStatus.OK, text.encode("utf-8"), "text/plain; charset=utf-8")
            return
        if suffix == "video":
            video = self.manager.artifact(task_id, "output.mp4")
            if video is None:
                self._error(HTTPStatus.NOT_FOUND, "视频尚未生成")
                return
            self._send(
                HTTPStatus.OK,
                video.read_bytes(),
                "video/mp4",
                {"Content-Disposition": f'inline; filename="{task_id}.mp4"'},
            )
            return
        if suffix == "metadata":
            meta = self.manager.artifact(task_id, "output.json")
            if meta is None:
                self._error(HTTPStatus.NOT_FOUND, "元数据尚未生成")
                return
            self._send(HTTPStatus.OK, meta.read_bytes(), "application/json; charset=utf-8")
            return
        self._json(HTTPStatus.OK, task.to_public())

    # -- POST --------------------------------------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/plan":
                payload = self._read_json()
                plan = geometry.plan_canvas(
                    megapixels=float(payload.get("megapixels", geometry.DEFAULT_MP)),
                    aspect=str(payload.get("aspect", geometry.DEFAULT_ASPECT)),
                    seconds=float(payload.get("seconds", geometry.MIN_SECONDS)),
                )
                self._json(HTTPStatus.OK, {"plan": plan})
                return
            match = re.fullmatch(r"/api/tasks/([^/]+)/(cancel|retry)", path)
            if match:
                task_id = match.group(1)
                if self.manager.get(task_id) is None:
                    self._not_found()
                    return
                if match.group(2) == "cancel":
                    result = self.manager.cancel(task_id)
                    self._json(HTTPStatus.ACCEPTED, {"id": task_id, "cancel": result})
                else:
                    task = self.manager.retry(task_id)
                    self._json(HTTPStatus.ACCEPTED, {"task": task.to_public()})
                return
            if path != "/api/tasks":
                self._not_found()
                return
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise SubmissionError("POST /api/tasks 需要 multipart/form-data")
            body = self._read_body()
            fields, uploads = parse_multipart(content_type, body)
            request = validate_request(fields, uploads)
            task = self.manager.submit(request)
        except SubmissionError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except KeyError:
            self._not_found()
            return
        except (OSError, subprocess.SubprocessError) as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._json(HTTPStatus.CREATED, {"task": task.to_public()})

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.do_GET()


def _model_config() -> dict[str, Any]:
    """Which weights a run will use, and whether they were found on disk."""
    model, weights = discover_model_paths()
    return {
        "base": model or "MiniMaxAI/MiniMax-H3",
        "weights": weights or "videorebirth/hyperflow",
        "base_local": model is not None,
        "weights_local": weights is not None,
    }


def fastest_defaults(device_status: dict[str, Any]) -> dict[str, Any]:
    """Form defaults that reproduce the fastest measured run on this host.

    Context parallel over every visible device when there are two or more (Ulysses splits the packed sequence, so it
    also cuts activation memory); a single card otherwise. Group offload is required on a 61 GiB part and stays on;
    its group size is pinned to one block because streaming only overlaps with one block per group and larger groups
    measured slower. The fast attention kernel and the fused RMSNorm are on by default on Ascend.
    """
    count = max(1, min(4, device_status.get("count") or 1))
    return {
        "megapixels": geometry.DEFAULT_MP,
        "aspect": geometry.DEFAULT_ASPECT,
        "seconds": geometry.MIN_SECONDS,
        "devices": count,
        "parallel": "cp" if count >= 2 else "single",
        "npu_offload": True,
        "offload_blocks": FASTEST_OFFLOAD_BLOCKS,
        "npu_fused_norm": True,
        "attention_backend": FASTEST_BACKEND,
        "seed": None,
    }


def _client_config(manager: TaskManager) -> dict[str, Any]:
    status = manager.device_status()
    return {
        "defaults": fastest_defaults(status),
        "device": status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HyperFlow H3 WebUI (Ascend NPU)")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3000")))
    parser.add_argument("--runs-dir", default=os.environ.get("HYPERFLOW_WEBUI_RUNS", str(RUNS_DIR)))
    args = parser.parse_args(argv)

    manager = TaskManager(Path(args.runs_dir))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.task_manager = manager  # type: ignore[attr-defined]
    # Print the device banner before serving; the first request then finds a warm cache.
    status = manager.device_status()

    def _gib(value: int | None) -> str:
        return f"{value / 2**30:.1f}" if value else "?"

    device_text = (
        ", ".join(
            f"{device['label']} 可分配 "
            f"{_gib(device.get('allocated_bytes'))}/{_gib(device.get('allocatable_bytes'))} GiB"
            f"（物理 {_gib(device.get('hbm_total_bytes'))} GiB）"
            for device in status["devices"]
        )
        if status["available"]
        else f"未检测到 NPU（{status['reason']}）"
    )
    print(f"HyperFlow H3 WebUI: http://{args.host}:{args.port}", flush=True)
    print(f"devices: {device_text}", flush=True)
    if status["count"] >= 2:
        print(f"multi-NPU detected ({status['count']}): context parallel is the default", flush=True)

    # `serve_forever()` only returns on KeyboardInterrupt, so a plain `kill` used to skip the cleanup below: the
    # worker thread is a daemon and died with the interpreter while the torchrun child kept running (holding the NPU
    # and showing up later as a stuck device). Turn SIGTERM/SIGINT into a clean shutdown instead.
    def _shutdown(signum: int, _frame: object) -> None:
        print(f"\nreceived signal {signum}, shutting down", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.serve_forever()
    finally:
        print("closing task manager", flush=True)
        manager.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
