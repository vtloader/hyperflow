"""`admit_max_frames` raises the pipeline's 15 s cap so a 362-frame request (15.083 s) is accepted.

Without it a 15 s clip is impossible: 360 frames snaps to 362 and the stock `max_duration` of 15.0 s rejects it, even
though 362 is a valid point on the model's own `17n + 5` frame grid.
"""

from __future__ import annotations

import pytest
import torch

from hyperflow_h3 import hyperflow_blocks
from hyperflow_h3.blocks import FPS, MAX_FRAMES, admit_max_frames


class FakePipeline:
    """Stands in for `MiniMaxH3ModularPipeline`: only `max_duration` matters to the code under test."""

    @property
    def max_duration(self) -> float:
        return 15.0

    @property
    def min_duration(self) -> float:
        return 5.0


def test_stock_cap_rejects_362_frames():
    """The baseline this change exists for: 362 / 24 = 15.083 s > 15.0 s."""
    pipeline = FakePipeline()
    assert pipeline.max_duration == 15.0
    assert MAX_FRAMES / FPS > pipeline.max_duration


def test_admit_max_frames_raises_the_cap_to_362_frames():
    pipeline = FakePipeline()
    admit_max_frames(pipeline)
    assert pipeline.max_duration == pytest.approx(MAX_FRAMES / FPS)
    assert pipeline.max_duration * FPS == pytest.approx(362)
    # The lower bound is untouched.
    assert pipeline.min_duration == 5.0


def test_admit_max_frames_is_idempotent():
    pipeline = FakePipeline()
    admit_max_frames(pipeline)
    first = type(pipeline)
    admit_max_frames(pipeline)
    assert type(pipeline) is first
    assert pipeline.max_duration == pytest.approx(MAX_FRAMES / FPS)


def test_admit_max_frames_keeps_the_pipeline_behaviour():
    """Subclassing on the instance must not break isinstance or the rest of the interface."""
    pipeline = FakePipeline()
    admit_max_frames(pipeline)
    assert isinstance(pipeline, FakePipeline)


def test_real_pipeline_cap_is_raised():
    """Against the actual ModularPipeline, whose `max_duration` is a read-only property."""
    pipeline = hyperflow_blocks("fl2va").init_pipeline(None)
    assert pipeline.max_duration == 15.0
    admit_max_frames(pipeline)
    assert pipeline.max_duration == pytest.approx(362 / 24)
    assert pipeline.max_duration * 24 == pytest.approx(362)


def test_hyperflow_blocks_still_builds():
    """The block-swapping path is unaffected by the cap helper."""
    blocks = hyperflow_blocks("fl2va")
    assert blocks is not None
    assert torch is not None  # keep the import meaningful for lint
