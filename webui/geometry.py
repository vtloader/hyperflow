# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Canvas arithmetic for the WebUI, following ComfyUI's `ResolutionSelector`.

ComfyUI's node (`comfy_extras/nodes_resolution.py`) computes a canvas from a target megapixel count and an aspect
ratio:

    total_pixels = megapixels * 1024 * 1024
    scale = sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple

Two details matter and are reproduced here rather than approximated:

* a "megapixel" is 1024*1024 pixels, not 1e6, so 2.0 MP is 2 097 152 pixels;
* both sides are snapped to the nearest multiple by *rounding* (not flooring), which is why 2.0 MP at 16:9 with a
  multiple of 32 lands on 1920x1088 (2 088 960 px) and not on 1920x1080.

MiniMax-H3 needs a multiple of 32, so that is the default here. Everything else (frame grid, aspect list) follows the
model's own constraints, which the pipeline enforces.
"""

from __future__ import annotations

import math
from typing import Any

#: Both canvas sides must be a multiple of this. MiniMax-H3's VAE patch size is 2 and the transformer patches 2x2, so
#: 32 is the value diffusers validates against (`pipeline.canvas_multiple`).
CANVAS_MULTIPLE = 32
FPS = 24
FRAME_STEP = 17
FRAME_OFFSET = 5
MIN_SECONDS = 5.0
#: `MiniMaxH3ModularPipeline.max_duration` is 15.0 s and is checked against the frame count **after** snapping to the
#: `17n + 5` grid. 360 frames (a 15 s request) snaps to 362 = 15.083 s, so the stock cap rejects it; `admit_max_frames`
#: raises the cap to 362 frames, which is what makes a 15 s clip possible. This module therefore accepts requests up to
#: the 15.083 s that 362 frames represents.
MAX_SECONDS = 15.0
MIN_FRAMES = 124  # align_frames(5.2 s) == 124; the CLI floors at 5 s
MAX_FRAMES = 362  # 15.083 s, admitted by `admit_max_frames`
MIN_MP = 0.1
MAX_MP = 8.0
DEFAULT_MP = 2.0
DEFAULT_ASPECT = "16:9"

#: Aspect ratios posted to the API, with the ComfyUI `ResolutionSelector` labels as display names.
ASPECTS: dict[str, tuple[int, int]] = {
    "21:9": (21, 9),
    "16:9": (16, 9),
    "3:2": (3, 2),
    "4:3": (4, 3),
    "1:1": (1, 1),
    "3:4": (3, 4),
    "2:3": (2, 3),
    "9:16": (9, 16),
    "9:21": (9, 21),
}
ASPECT_LABELS = {
    "21:9": "21:9 超宽",
    "16:9": "16:9 宽屏",
    "3:2": "3:2 照片",
    "4:3": "4:3 标准",
    "1:1": "1:1 方形",
    "3:4": "3:4 竖版标准",
    "2:3": "2:3 竖版照片",
    "9:16": "9:16 竖屏",
    "9:21": "9:21 竖版超宽",
}


def align_frames(num_frames: int) -> int:
    """Snap to the nearest ``17n + 5`` frame count, rounding down when the request is not yet on the grid.

    The grid is 5, 22, 39, ... 124, 141. Rounding *up* would turn a 5.2 s request (125 frames) into 141 frames —
    5.88 s — which is not what the user asked for. Rounding to the nearest grid point keeps 5.2 s at 124 frames and
    only moves a duration that sits between two grid points.
    """
    frames = max(int(num_frames), 1)
    if frames <= FRAME_OFFSET:
        return FRAME_OFFSET
    steps = round((frames - FRAME_OFFSET) / FRAME_STEP)
    return FRAME_OFFSET + max(0, steps) * FRAME_STEP


def frames_for_seconds(seconds: float) -> int:
    """Aligned frame count for a duration in seconds."""
    return align_frames(round(float(seconds) * FPS))


def seconds_for_frames(num_frames: int) -> float:
    return align_frames(num_frames) / FPS


def resolve_canvas(megapixels: float, aspect: str, multiple: int = CANVAS_MULTIPLE) -> tuple[int, int]:
    """`(width, height)` for a megapixel target, exactly as ComfyUI's `ResolutionSelector` computes it."""
    if aspect not in ASPECTS:
        raise ValueError(f"unsupported aspect ratio {aspect!r}; choose one of {', '.join(ASPECTS)}")
    if not (MIN_MP <= float(megapixels) <= MAX_MP):
        raise ValueError(f"megapixels must be within [{MIN_MP}, {MAX_MP}]")
    if multiple < 1:
        raise ValueError("multiple must be >= 1")
    w_ratio, h_ratio = ASPECTS[aspect]
    total_pixels = float(megapixels) * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return int(width), int(height)


def validate_canvas(width: int, height: int, multiple: int = CANVAS_MULTIPLE) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("canvas sides must be positive")
    if width % multiple or height % multiple:
        raise ValueError(f"width and height must be multiples of {multiple}, got {width}x{height}")


def plan_canvas(
    *,
    megapixels: float = DEFAULT_MP,
    aspect: str = DEFAULT_ASPECT,
    seconds: float = MIN_SECONDS,
    multiple: int = CANVAS_MULTIPLE,
) -> dict[str, Any]:
    """The full generation plan the WebUI shows before submitting."""
    width, height = resolve_canvas(megapixels, aspect, multiple)
    frames = frames_for_seconds(seconds)
    # Validate the *aligned* frame count, which is what the pipeline checks. Testing the raw input instead would
    # reject a 15.04 s request (361 frames -> 362 = 15.083 s, admitted by `admit_max_frames`) while accepting a
    # 14.9 s one that snaps down to 345 frames anyway.
    if not (MIN_FRAMES <= frames <= MAX_FRAMES):
        raise ValueError(
            f"duration must produce {MIN_FRAMES}–{MAX_FRAMES} frames at {FPS} fps "
            f"({MIN_FRAMES / FPS:.2f}–{MAX_FRAMES / FPS:.2f} s), got {seconds:g} s -> {frames} frames"
        )
    pixels = width * height
    return {
        "width": width,
        "height": height,
        "aspect": aspect,
        "aspect_label": ASPECT_LABELS[aspect],
        "megapixels": round(pixels / 1024 / 1024, 4),
        "target_megapixels": float(megapixels),
        "frames": frames,
        "fps": FPS,
        "duration_seconds": round(frames / FPS, 3),
        "multiple": multiple,
        "tokens": estimate_tokens(width, height, frames),
    }


def estimate_tokens(width: int, height: int, frames: int) -> int:
    """Rows of the packed sequence the transformer attends over; a proxy for runtime cost."""
    aligned = align_frames(frames)
    latent_frames = (aligned - FRAME_OFFSET) // FRAME_STEP * 5 + 2
    video = latent_frames * (height // CANVAS_MULTIPLE) * (width // CANVAS_MULTIPLE)
    audio = round(aligned / FPS * 40) * 2
    return int(video + audio)


__all__ = [
    "ASPECTS",
    "ASPECT_LABELS",
    "CANVAS_MULTIPLE",
    "DEFAULT_ASPECT",
    "DEFAULT_MP",
    "MAX_MP",
    "MAX_SECONDS",
    "MIN_MP",
    "MIN_SECONDS",
    "align_frames",
    "estimate_tokens",
    "frames_for_seconds",
    "plan_canvas",
    "resolve_canvas",
    "seconds_for_frames",
    "validate_canvas",
]
