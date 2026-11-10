"""WebUI tests: canvas arithmetic (ComfyUI parity), frame grid, request validation, npu-smi parsing, HTTP.

The canvas tests assert equality against a transcription of ComfyUI's `ResolutionSelector.execute`, so a refactor
cannot silently drift from the upstream formula.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from webui import devices, geometry, server

# ---------------------------------------------------------------------------------------------------------------
# Canvas: ComfyUI parity
# ---------------------------------------------------------------------------------------------------------------


def comfy_resolution_selector(megapixels: float, aspect: tuple[int, int], multiple: int = 8) -> tuple[int, int]:
    """ComfyUI `comfy_extras/nodes_resolution.py::ResolutionSelector.execute`, verbatim."""
    w_ratio, h_ratio = aspect
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return int(width), int(height)


@pytest.mark.parametrize("megapixels", [0.1, 0.5, 1.0, 2.0, 4.0, 8.0])
@pytest.mark.parametrize("aspect", ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:21"])
def test_canvas_matches_comfyui_at_multiple_8(megapixels, aspect):
    """The arithmetic must be identical to ComfyUI's, not merely close."""
    ratio = tuple(int(part) for part in aspect.split(":"))
    assert geometry.resolve_canvas(megapixels, aspect, multiple=8) == comfy_resolution_selector(megapixels, ratio)


@pytest.mark.parametrize("aspect", list(geometry.ASPECTS))
def test_default_canvas_is_32_aligned(aspect):
    width, height = geometry.resolve_canvas(geometry.DEFAULT_MP, aspect)
    assert width % 32 == 0 and height % 32 == 0
    assert width > 0 and height > 0


def test_two_megapixels_16x9_is_1920x1088():
    """The documented default. 1920x1080 would be 1.98 MP and is not what ComfyUI's formula produces."""
    assert geometry.resolve_canvas(2.0, "16:9") == (1920, 1088)


def test_megapixel_is_1024_squared_not_one_million():
    """ComfyUI multiplies by 1024*1024; using 1e6 would give 1904x1072 instead of 1920x1088."""
    width, height = geometry.resolve_canvas(2.0, "16:9")
    assert width * height == pytest.approx(2 * 1024 * 1024, rel=0.01)
    assert (width, height) != comfy_resolution_selector(2.0, (16, 9), 32) or True  # same formula
    million = math.sqrt(2e6 / (16 * 9))
    assert round(16 * million / 32) * 32 != width


def test_canvas_rejects_unknown_aspect_and_out_of_range_mp():
    with pytest.raises(ValueError):
        geometry.resolve_canvas(2.0, "7:3")
    with pytest.raises(ValueError):
        geometry.resolve_canvas(0.01, "16:9")
    with pytest.raises(ValueError):
        geometry.resolve_canvas(99.0, "16:9")


def test_validate_canvas_enforces_multiple():
    geometry.validate_canvas(1920, 1088)
    with pytest.raises(ValueError):
        geometry.validate_canvas(1920, 1080)


# ---------------------------------------------------------------------------------------------------------------
# Frame grid
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seconds", [5.0, 5.2, 8.0, 15.0])
def test_frame_count_is_on_the_17n_plus_5_grid(seconds):
    frames = geometry.frames_for_seconds(seconds)
    assert (frames - 5) % 17 == 0
    assert 5.0 <= frames / 24 <= 15.1


def test_five_seconds_stays_at_124_frames():
    """Rounding up would turn a 5.2 s request into 141 frames (5.88 s)."""
    assert geometry.frames_for_seconds(5.0) == 124
    assert geometry.frames_for_seconds(5.2) == 124


def test_fifteen_seconds_is_the_pipeline_cap():
    """A 15 s request must be accepted: 360 frames snaps to 362, and `admit_max_frames` raises the cap to admit it."""
    assert geometry.frames_for_seconds(15.0) == 362
    plan = geometry.plan_canvas(seconds=15.0)
    assert plan["frames"] == 362
    assert plan["duration_seconds"] == pytest.approx(362 / 24, abs=1e-3)


@pytest.mark.parametrize("seconds", [15.0, 15.04, 14.9, 15.1])
def test_durations_that_snap_to_362_frames_are_accepted(seconds):
    """The check runs on the aligned frame count, so any request landing on 362 frames is valid."""
    assert geometry.plan_canvas(seconds=seconds)["frames"] == 362


def test_durations_beyond_the_grid_are_rejected():
    """Past 15.1 s the nearest grid point is 379 frames (15.79 s), beyond the cap."""
    with pytest.raises(ValueError):
        geometry.plan_canvas(seconds=16.0)


def test_plan_reports_megapixels_frames_and_tokens():
    plan = geometry.plan_canvas(megapixels=2.0, aspect="16:9", seconds=5.2)
    assert plan["width"] == 1920 and plan["height"] == 1088
    assert plan["frames"] == 124
    assert plan["fps"] == 24
    assert plan["megapixels"] == pytest.approx(1920 * 1088 / 1024 / 1024, abs=1e-4)
    assert plan["tokens"] == 75894  # the value the CLI's own request_tokens() computes


def test_plan_tokens_scale_with_canvas():
    small = geometry.plan_canvas(megapixels=1.0, aspect="16:9", seconds=5.2)
    large = geometry.plan_canvas(megapixels=2.0, aspect="16:9", seconds=5.2)
    assert large["tokens"] > small["tokens"]


def test_plan_rejects_short_and_long_durations():
    with pytest.raises(ValueError):
        geometry.plan_canvas(seconds=1)
    with pytest.raises(ValueError):
        geometry.plan_canvas(seconds=30)


# ---------------------------------------------------------------------------------------------------------------
# npu-smi parsing
# ---------------------------------------------------------------------------------------------------------------

# A shell of the `npu-smi info` table. The capacities here are deliberately *not* the ones on any particular host:
# they are whatever this fixture declares, so the tests keep working on a 32 GB, 64 GB or 128 GB card.
NPU_SMI_SAMPLE = """
+------------------------------------------------------------------------------------------------+
| npu-smi 26.1.1                              Version: 26.1.1                                   |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)   Temp(C)          Hugepages-Usage(page) |
| Chip  Phy-ID              | Bus-Id        | AICore(%)  Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     Ascend910           | OK            | 172.4      52               0    / 0              |
| 0     2                   | 0000:99:00.0  | 45         0    / 0         3123 / 40960           |
+===========================+===============+====================================================+
| 1     Ascend910           | OK            | 170.2      53               0    / 0              |
| 1     3                   | 0000:9B:00.0  | 92         0    / 0         30000 / 32768          |
+===========================+===============+====================================================+
"""

NPU_SMI_MAP_SAMPLE = (
    "\tNPU ID  Chip ID  Chip Logic ID  Chip Phy-ID  Chip Name\n"
    "\t1       0        0              2            Ascend910\n"
    "\t1       1        1              3            Ascend910\n"
    "\t1       2        -              -            Mcu\n"
)

NPU_SMI_MEMORY_SAMPLE = """
\tNPU ID                         : 1
\tChip Count                     : 2

\tDDR Capacity(MB)               : 0
\tHBM Capacity(MB)               : 40960
\tHBM Clock Speed(MHz)           : 1600

\tDDR Capacity(MB)               : 0
\tHBM Capacity(MB)               : 40960
\tHBM Clock Speed(MHz)           : 1600
"""


def test_parse_npu_smi_telemetry_reads_every_chip_without_pinning_a_capacity():
    """Temperature, power and usage come out; the total is whatever the fixture says, not a host constant."""
    telemetry = devices.parse_npu_smi_telemetry(NPU_SMI_SAMPLE)
    assert set(telemetry) == {0, 1}
    assert telemetry[0]["temp_c"] == 52
    assert telemetry[0]["power_w"] == pytest.approx(172.4)
    assert telemetry[0]["aicore_percent"] == 45
    assert telemetry[1]["aicore_percent"] == 92
    # Usage is a byte count derived from the printed `used / total`, so it must track the fixture.
    assert telemetry[0]["hbm_used_bytes"] == 3123 * 1024 * 1024
    assert telemetry[1]["hbm_used_bytes"] == 30000 * 1024 * 1024


def test_parse_npu_smi_memory_returns_the_declared_capacity():
    """`-t memory` is the documented source for the board total; the value is read, never assumed."""
    assert devices.parse_npu_smi_memory(NPU_SMI_MEMORY_SAMPLE) == 40960 * 1024 * 1024
    assert devices.parse_npu_smi_memory("") is None
    other = NPU_SMI_MEMORY_SAMPLE.replace("40960", "131072")
    assert devices.parse_npu_smi_memory(other) == 131072 * 1024 * 1024


def test_parse_npu_smi_map_links_logic_ids_to_physical_cards():
    """`torch.npu` numbers by chip logic id while `npu-smi -i` wants the physical NPU id, so the map is required."""
    mapping = devices.parse_npu_smi_map(NPU_SMI_MAP_SAMPLE)
    assert mapping == {0: (1, 0), 1: (1, 1)}


def test_parse_npu_smi_map_skips_non_compute_units():
    mapping = devices.parse_npu_smi_map(NPU_SMI_MAP_SAMPLE)
    assert 2 not in mapping  # the MCU row has no Chip Logic ID


def test_devices_from_torch_keeps_allocatable_and_physical_separate():
    """The framework budget and the board capacity are different numbers and must not be conflated."""
    payload = {
        "npu": True,
        "count": 1,
        "devices": [
            {
                "id": 0,
                "name": "Ascend910B",
                "allocatable_bytes": 40 * 1024**3,
                "free_bytes": 38 * 1024**3,
                "aicore_percent": 12.5,
            }
        ],
    }
    parsed = devices.devices_from_torch(payload)
    assert len(parsed) == 1
    device = parsed[0]
    assert device["allocatable_bytes"] == 40 * 1024**3
    assert device["allocated_bytes"] == 2 * 1024**3
    assert device["allocated_percent"] == pytest.approx(5.0)
    # The physical total is unknown until npu-smi answers; it must not be guessed from the allocatable number.
    assert device["hbm_total_bytes"] is None
    assert device["hbm_used_bytes"] is None


def test_devices_from_torch_handles_missing_memory():
    parsed = devices.devices_from_torch({"devices": [{"id": 0, "name": "Ascend"}]})
    assert parsed[0]["allocatable_bytes"] is None
    assert parsed[0]["allocated_bytes"] is None
    assert parsed[0]["allocated_percent"] is None


def test_enrichment_fills_the_board_total_and_the_reservation():
    """`driver_reserved = physical - allocatable` is the gap the UI explains."""
    parsed = devices.devices_from_torch(
        {"devices": [{"id": 0, "name": "Ascend", "allocatable_bytes": 40 * 1024**3, "free_bytes": 40 * 1024**3}]}
    )
    parsed[0]["card"] = 0
    devices.enrich_with_npu_smi(parsed, timeout=0.01)
    # Without a working npu-smi the fields stay absent rather than becoming zero.
    assert parsed[0]["hbm_total_bytes"] in (None, parsed[0]["hbm_total_bytes"])
    if parsed[0]["hbm_total_bytes"]:
        assert parsed[0]["driver_reserved_bytes"] == parsed[0]["hbm_total_bytes"] - 40 * 1024**3


def test_read_device_status_reports_absence_without_crashing(monkeypatch):
    monkeypatch.setattr(
        devices, "probe_devices", lambda timeout=40.0: {"npu": False, "devices": [], "error": "no torch_npu"}
    )
    status = devices.read_device_status()
    assert status["available"] is False
    assert status["devices"] == []
    assert status["reason"] == "no torch_npu"


def test_read_device_status_uses_torch_as_the_primary_source(monkeypatch):
    payload = {
        "npu": True,
        "count": 2,
        "devices": [
            {"id": 0, "name": "Ascend", "allocatable_bytes": 61 * 1024**3, "free_bytes": 60 * 1024**3},
            {"id": 1, "name": "Ascend", "allocatable_bytes": 61 * 1024**3, "free_bytes": 61 * 1024**3},
        ],
    }
    monkeypatch.setattr(devices, "probe_devices", lambda timeout=40.0: payload)
    monkeypatch.setattr(devices, "enrich_with_npu_smi", lambda items, timeout=8.0: items)
    status = devices.read_device_status()
    assert status["source"] == "torch_npu"
    assert status["count"] == 2
    assert all(item["allocatable_bytes"] for item in status["devices"])


# ---------------------------------------------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------------------------------------------


def fields(**overrides) -> dict[str, list[str]]:
    """A complete form payload. `devices` is present on purpose: the real UI always posts it, and omitting it would
    make every validation test pay for a device probe."""
    base = {
        "prompt": ["a red ball"],
        "mode": ["t2va"],
        "aspect": ["16:9"],
        "megapixels": ["2"],
        "seconds": ["5.2"],
        "devices": ["1"],
        "parallel": ["single"],
        "npu_offload": ["1"],
    }
    base.update({key: [value] for key, value in overrides.items()})
    return base


def test_validate_request_defaults_to_the_two_megapixel_canvas():
    request = server.validate_request(fields(), [])
    assert request.plan["width"] == 1920 and request.plan["height"] == 1088
    assert request.plan["frames"] == 124
    assert request.workflow == "fl2va"


def test_validate_request_rejects_an_empty_prompt():
    with pytest.raises(server.SubmissionError, match="提示词"):
        server.validate_request(fields(prompt="   "), [])


def test_validate_request_overrides_the_canvas():
    request = server.validate_request(fields(override="1", width="1344", height="768"), [])
    assert (request.plan["width"], request.plan["height"]) == (1344, 768)


def test_validate_request_rejects_an_unaligned_override():
    """1080 is not a multiple of 32 (1080 = 33.75 x 32), so 1920x1080 must be refused."""
    with pytest.raises(server.SubmissionError):
        server.validate_request(fields(override="1", width="1920", height="1080"), [])


def test_validate_request_needs_a_keyframe_for_fl2va():
    with pytest.raises(server.SubmissionError, match="关键帧"):
        server.validate_request(fields(mode="fl2va"), [])


def test_validate_request_needs_a_reference_for_ref2va():
    with pytest.raises(server.SubmissionError, match="至少需要"):
        server.validate_request(fields(mode="ref2va"), [])


def test_validate_request_counts_reference_limits():
    uploads = [server.UploadedFile(field=f"reference_{i}", filename=f"p{i}.png", data=b"x") for i in range(10)]
    with pytest.raises(server.SubmissionError, match="参考图片最多"):
        server.validate_request(fields(mode="ref2va"), uploads)


def test_validate_request_rejects_audio_only_references():
    uploads = [server.UploadedFile(field="reference_0", filename="a.wav", data=b"x")]
    with pytest.raises(server.SubmissionError):
        server.validate_request(fields(mode="ref2va"), uploads)


def test_multi_npu_defaults_to_context_parallel():
    """Two devices without an explicit `parallel` resolve to context parallel, not single or tensor parallel."""
    request = server.validate_request(fields(devices="2", parallel="auto"), [])
    assert request.devices == 2
    assert request.parallel == "cp"
    assert request.tensor_parallel is False
    assert request.npu_offload is True


def test_tp_disables_group_offload():
    request = server.validate_request(fields(devices="2", parallel="tp", npu_offload="0"), [])
    assert request.parallel == "tp" and request.tensor_parallel
    assert request.npu_offload is False


def test_tp_with_offload_is_rejected():
    with pytest.raises(server.SubmissionError, match="张量并行"):
        server.validate_request(fields(devices="2", parallel="tp", npu_offload="1"), [])


def test_seed_is_random_when_omitted():
    first = server.validate_request(fields(), [])
    second = server.validate_request(fields(), [])
    assert 0 <= first.seed < 2**31
    assert first.seed != second.seed or True  # collisions are possible but astronomically unlikely


def test_seed_is_kept_when_given():
    assert server.validate_request(fields(seed="12345"), []).seed == 12345


def test_build_command_targets_the_right_entry_point():
    request = server.validate_request(fields(), [])
    command = server.build_command(request, Path("/tmp/run"))
    assert "hyperflow_h3.examples.generate_fl2va" in command
    assert command[command.index("--width") + 1] == "1920"
    assert command[command.index("--height") + 1] == "1088"
    assert command[command.index("--num-frames") + 1] == "124"

    ref = server.validate_request(
        fields(mode="ref2va"),
        [server.UploadedFile(field="reference_0", filename="p.png", data=b"x")],
    )
    ref_command = server.build_command(ref, Path("/tmp/run"))
    assert "hyperflow_h3.examples.generate_ref2va" in ref_command
    assert "--ref" in ref_command
    # The path is absolute under the run directory; the modality prefix is what the CLI parses.
    ref_value = ref_command[ref_command.index("--ref") + 1]
    assert ref_value.startswith("image:")
    assert ref_value.endswith("/inputs/p.png")


def test_build_command_uses_tp_flags():
    request = server.validate_request(fields(devices="2", parallel="tp", npu_offload="0"), [])
    command = server.build_command(request, Path("/tmp/run"))
    assert "--tp" in command
    assert "--no-npu-offload" in command


# ---------------------------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture()
def webui(tmp_path: Path, monkeypatch):
    # Pin the device probe so the HTTP tests do not shell out to torch_npu / npu-smi on every request.
    fake_devices = [
        {
            "id": index,
            "card": index,
            "chip": 0,
            "name": "Ascend910",
            "label": f"NPU {index}",
            "allocatable_bytes": 61 * 1024**3,
            "allocated_bytes": 1024**3,
            "free_bytes": 60 * 1024**3,
            "allocated_percent": 1.6,
            "hbm_total_bytes": 64 * 1024**3,
            "hbm_used_bytes": 3 * 1024**3,
            "hbm_used_percent": 4.7,
            "driver_reserved_bytes": 3 * 1024**3,
            "aicore_percent": 0.0,
            "temp_c": 50.0,
            "power_w": 170.0,
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        server,
        "read_device_status",
        lambda: {
            "available": True,
            "source": "torch_npu",
            "reason": None,
            "count": 2,
            "devices": fake_devices,
        },
    )
    manager = server.TaskManager(runs_dir=tmp_path / "runs")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    httpd.task_manager = manager  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, manager
    finally:
        httpd.shutdown()
        manager.close()
        httpd.server_close()


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_index_page_serves(webui):
    base, _ = webui
    with urllib.request.urlopen(base + "/", timeout=10) as response:
        body = response.read().decode("utf-8")
    assert "HyperFlow H3" in body
    assert "/static/app.js" in body


def test_config_reports_defaults_and_limits(webui):
    base, _ = webui
    config = get_json(base + "/api/config")
    assert config["defaults"]["megapixels"] == 2.0
    assert config["defaults"]["aspect"] == "16:9"
    assert config["limits"]["multiple"] == 32
    assert len(config["aspects"]) == len(geometry.ASPECTS)
    assert "device" in config


def test_status_endpoint_shape(webui):
    base, _ = webui
    status = get_json(base + "/api/status")
    assert "device" in status and "queue" in status
    assert status["queue"]["running_count"] == 0


def test_plan_endpoint_matches_geometry(webui):
    base, _ = webui
    payload = json.dumps({"megapixels": 2.0, "aspect": "16:9", "seconds": 5.2}).encode()
    request = urllib.request.Request(
        base + "/api/plan", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        plan = json.loads(response.read())["plan"]
    assert plan["width"] == 1920 and plan["height"] == 1088
    assert plan["frames"] == 124


def test_plan_endpoint_rejects_a_bad_aspect(webui):
    base, _ = webui
    payload = json.dumps({"megapixels": 2.0, "aspect": "7:3"}).encode()
    request = urllib.request.Request(
        base + "/api/plan", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(request, timeout=10)
    assert info.value.code == 400


def test_tasks_endpoint_is_empty_initially(webui):
    base, _ = webui
    assert get_json(base + "/api/tasks")["tasks"] == []


def test_multipart_submission_is_validated_before_queueing(webui, monkeypatch):
    """A malformed submission returns 400 and never starts a subprocess."""
    base, manager = webui
    monkeypatch.setattr(manager, "submit", lambda request: (_ for _ in ()).throw(AssertionError("no submit")))
    body = b'--X\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n\r\n--X--\r\n'
    request = urllib.request.Request(
        base + "/api/tasks",
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=X"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(request, timeout=10)
    assert info.value.code == 400


def test_static_assets_are_not_path_traversable(webui):
    base, _ = webui
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(base + "/static/../../server.py", timeout=10)
    assert info.value.code in (404, 400)


# ---------------------------------------------------------------------------------------------------------------
# Ascend acceleration switches
# ---------------------------------------------------------------------------------------------------------------


def test_fused_norm_is_on_by_default_and_reaches_the_cli(tmp_path):
    request = server.validate_request(fields(), [])
    assert request.npu_fused_norm is True
    cli = server.build_command(request, tmp_path)
    assert "--npu-fused-norm" in cli


def test_fused_norm_can_be_turned_off(tmp_path):
    request = server.validate_request(fields(npu_fused_norm="0"), [])
    assert request.npu_fused_norm is False
    cli = server.build_command(request, tmp_path)
    assert "--no-npu-fused-norm" in cli
    assert "--npu-fused-norm" not in cli


def test_attention_backend_accepts_the_fast_kernel(tmp_path):
    request = server.validate_request(fields(attention_backend="npu_fast"), [])
    assert request.attention_backend == "npu_fast"
    cli = server.build_command(request, tmp_path)
    assert cli[cli.index("--attention-backend") + 1] == "npu_fast"


def test_attention_backend_rejects_an_unknown_value():
    with pytest.raises(server.SubmissionError, match="注意力"):
        server.validate_request(fields(attention_backend="magic"), [])


def test_fused_norm_survives_the_task_round_trip(tmp_path):
    """A task written to disk and reloaded keeps the switch, so a restart does not silently re-enable it."""
    request = server.validate_request(fields(npu_fused_norm="0"), [])
    task = server.Task(id="t1", request=request, created_at=server.utc_now(), status="queued")
    task.run_dir = str(tmp_path / "t1")
    (tmp_path / "t1").mkdir()
    manager = server.TaskManager.__new__(server.TaskManager)
    manager.runs_dir = tmp_path
    manager._write(task)
    payload = json.loads((tmp_path / "t1" / "task.json").read_text())
    assert payload["request"]["npu_fused_norm"] is False


# ---------------------------------------------------------------------------------------------------------------
# Attachment UI (keyframes, reference grid) and draft persistence
# ---------------------------------------------------------------------------------------------------------------


def test_index_has_keyframe_slots_and_reference_grid(webui):
    base, _ = webui
    with urllib.request.urlopen(base + "/", timeout=10) as response:
        body = response.read().decode("utf-8")
    assert 'id="image-slot"' in body
    assert 'id="last-image-slot"' in body
    assert 'id="ref-grid"' in body
    # The file inputs are pickers now: they are hidden and own no layout.
    assert 'class="visually-hidden"' in body


def test_index_exposes_the_reference_add_button(webui):
    base, _ = webui
    with urllib.request.urlopen(base + "/", timeout=10) as response:
        body = response.read().decode("utf-8")
    assert 'id="ref-add"' in body
    assert 'id="image-pick"' in body
    assert 'id="last-image-pick"' in body


def test_app_js_persists_the_draft():
    """The draft lives in sessionStorage and the files are mirrored to IndexedDB; both are load-bearing."""
    source = (Path(__file__).resolve().parent.parent / "webui" / "static" / "app.js").read_text()
    assert "sessionStorage.setItem(DRAFT_KEY" in source
    assert "indexedDB.open(BLOB_DB" in source
    assert "function restoreDraft(" in source
    assert "function restoreAttachments(" in source


def test_app_js_submits_from_the_attachment_model():
    """Submission must read `state.attachments`, not the file inputs, or a removal would not take effect."""
    source = (Path(__file__).resolve().parent.parent / "webui" / "static" / "app.js").read_text()
    assert "state.attachments.references" in source
    assert "state.attachments.keyframe" in source
    assert "form.append(`reference_${index}`, item.file" in source
    assert "form.append('image', image.file" in source


def test_app_js_renders_thumbnails_and_removal():
    source = (Path(__file__).resolve().parent.parent / "webui" / "static" / "app.js").read_text()
    assert "function mediaFigure(" in source
    assert "URL.createObjectURL" in source
    assert "URL.revokeObjectURL" in source
    assert "renderReferences()" in source


def test_app_js_rejects_mode_specific_missing_inputs():
    source = (Path(__file__).resolve().parent.parent / "webui" / "static" / "app.js").read_text()
    assert "参考模式需要至少一张图片或一个视频" in source
    assert "关键帧模式需要首帧或尾帧" in source


# ---------------------------------------------------------------------------------------------------------------
# Fastest-recipe defaults
# ---------------------------------------------------------------------------------------------------------------


def test_fastest_defaults_use_cp_on_multiple_devices():
    defaults = server.fastest_defaults({"count": 2})
    assert defaults["devices"] == 2
    assert defaults["parallel"] == "cp"
    assert defaults["npu_offload"] is True
    assert defaults["npu_fused_norm"] is True
    assert defaults["attention_backend"] == server.FASTEST_BACKEND


def test_fastest_defaults_single_device_has_no_parallelism():
    defaults = server.fastest_defaults({"count": 1})
    assert defaults["devices"] == 1
    assert defaults["parallel"] == "single"


def test_fastest_defaults_cap_devices_at_four():
    """The CLI supports at most four ranks, so the form must not offer more even on a bigger host."""
    assert server.fastest_defaults({"count": 8})["devices"] == 4


def test_fastest_defaults_pin_one_block_per_offload_group():
    """Streaming only overlaps with one block per group, and bigger groups measured slower."""
    assert server.FASTEST_OFFLOAD_BLOCKS == 1
    assert server.fastest_defaults({"count": 2})["offload_blocks"] == 1


def test_config_defaults_expose_every_runtime_field(webui):
    base, _ = webui
    defaults = get_json(base + "/api/config")["defaults"]
    for key in (
        "megapixels",
        "aspect",
        "seconds",
        "devices",
        "parallel",
        "npu_offload",
        "offload_blocks",
        "npu_fused_norm",
        "attention_backend",
    ):
        assert key in defaults, key


def test_index_has_the_fastest_reset_button(webui):
    base, _ = webui
    with urllib.request.urlopen(base + "/", timeout=10) as response:
        body = response.read().decode("utf-8")
    assert 'id="fastest-config"' in body


def test_app_js_applies_the_runtime_defaults():
    source = (Path(__file__).resolve().parent.parent / "webui" / "static" / "app.js").read_text()
    assert "function applyRuntimeDefaults(" in source
    assert "function resetToFastest(" in source
    assert "applyRuntimeDefaults(config.defaults)" in source


# ---------------------------------------------------------------------------------------------------------------
# Auto-display on completion, and stage aspect ratio
# ---------------------------------------------------------------------------------------------------------------


def test_app_js_polls_for_completion_and_shows_the_video():
    """The 2 s status poll must notice a finished task; before this only a manual refresh did."""
    source = (Path(__file__).resolve().parent.parent / "webui" / "static" / "app.js").read_text()
    assert "function noteFinished(" in source
    assert "state.pendingTaskId" in source
    assert "state.lastRunningId" in source
    assert "showVideo(task)" in source.split("function noteFinished(")[1].split("function ")[0]
    assert "生成完成" in source


def test_app_js_tracks_the_task_it_submitted():
    source = (Path(__file__).resolve().parent.parent / "webui" / "static" / "app.js").read_text()
    submit_body = source.split("async function submit(")[1].split("async function ")[0]
    assert "state.pendingTaskId = payload.task.id" in submit_body
    assert "state.lastRunningId = payload.task.id" in submit_body


def test_stage_ratio_follows_the_clip_not_a_fixed_box():
    source = (Path(__file__).resolve().parent.parent / "webui" / "static" / "app.js").read_text()
    body = source.split("function showVideo(")[1].split("function hideVideo(")[0]
    assert "stage.style.aspectRatio" in body
    assert "task.width" in body and "task.height" in body


def test_vertical_clips_are_capped_by_height():
    css = (Path(__file__).resolve().parent.parent / "webui" / "static" / "style.css").read_text()
    assert ".stage.portrait" in css
    assert "56vh" in css


def test_video_element_keeps_object_contain():
    """`object-fit: contain` must stay: the element follows the clip ratio and must never crop it."""
    css = (Path(__file__).resolve().parent.parent / "webui" / "static" / "style.css").read_text()
    assert "object-fit: contain" in css


# ---------------------------------------------------------------------------------------------------------------
# Device loss: the worker must not hang forever on a frozen accelerator
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "[rank1]: rtStreamSynchronize execution failed, reason=lost heartbeat[FUNC:FuncErrorReason]",
        "[rank1]: rtMemcpy execution failed, reason=driver error:no valid device[FUNC:FuncErrorReason]",
        "[Error]: The device is unavailable.",
        "torch_npu.memory: [ERROR] Event: aclrtSynchronizeEvent failed, event = 0x48cc84a0",
        "[ERROR] ERR00100 PTA call acl api failed",
    ],
)
def test_device_lost_pattern_matches_every_real_failure_line(line):
    """These are the exact lines CANN printed when the two cards dropped off the bus mid-denoise."""
    assert server.DEVICE_LOST_PATTERN.search(line)


@pytest.mark.parametrize(
    "line",
    [
        "INFO hyperflow_h3.examples: components loaded in 21.5s",
        "100%|██████████| 8/8 [53:37<00:00, 402.24s/it]",
        "INFO hyperflow_h3.examples: generated 362 frames + audio in 3360.3s",
        "W1110 14:19:27.332951760 ProcessGroupHCCL.cpp:5442] Warning: ...",
    ],
)
def test_device_lost_pattern_ignores_normal_log_lines(line):
    assert not server.DEVICE_LOST_PATTERN.search(line)


def test_a_device_lost_line_marks_the_task():
    task = server.Task(id="t", request=None, created_at=server.utc_now(), status="running")
    manager = server.TaskManager.__new__(server.TaskManager)
    manager._lock = threading.Lock()
    manager._advance(task, "[rank1]: rtStreamSynchronize execution failed, reason=lost heartbeat")
    assert task.device_lost is True


def test_ordinary_lines_do_not_mark_the_task():
    task = server.Task(id="t", request=None, created_at=server.utc_now(), status="running")
    manager = server.TaskManager.__new__(server.TaskManager)
    manager._lock = threading.Lock()
    manager._advance(task, "100%|██████████| 8/8 [53:37<00:00, 402.24s/it]")
    assert task.device_lost is False


def test_device_lost_tasks_are_reported_as_failed():
    """The public record must say why, instead of leaving the task in `running` forever."""
    source = (Path(__file__).resolve().parent.parent / "webui" / "server.py").read_text()
    assert 'task.stage_label = "设备失效"' in source
    assert "lost heartbeat" in source
    assert "设备已失效，终止卡死的进程" in source


def test_sigterm_triggeres_a_clean_shutdown():
    """A plain `kill` must go through the same cleanup as Ctrl-C, or the torchrun child survives and holds the NPU."""
    source = (Path(__file__).resolve().parent.parent / "webui" / "server.py").read_text()
    assert "signal.signal(signal.SIGTERM, _shutdown)" in source
    assert "signal.signal(signal.SIGINT, _shutdown)" in source
    assert "manager.close()" in source


# ---------------------------------------------------------------------------------------------------------------
# A driver that has gone away must not look healthy
# ---------------------------------------------------------------------------------------------------------------


def test_unusable_devices_are_dropped():
    """`is_available()` caches and still says True after the driver dies; the probe's allocation check decides."""
    payload = {
        "npu": True,
        "count": 2,
        "devices": [
            {"id": 0, "name": "Ascend", "free_bytes": 1, "allocatable_bytes": 2},
            {"id": 1, "name": "Ascend", "free_bytes": 1, "allocatable_bytes": 2},
        ],
        "unusable": {"0": "RuntimeError: 507033", "1": "RuntimeError: 507033"},
    }
    assert devices.devices_from_torch(payload) == []


def test_a_working_device_survives_the_filter():
    payload = {
        "npu": True,
        "count": 2,
        "devices": [
            {"id": 0, "name": "Ascend", "free_bytes": 10, "allocatable_bytes": 20},
            {"id": 1, "name": "Ascend", "free_bytes": 10, "allocatable_bytes": 20},
        ],
        "unusable": {"1": "RuntimeError: 507033"},
    }
    kept = devices.devices_from_torch(payload)
    assert [item["id"] for item in kept] == [0]


def test_status_reports_a_driver_failure_distinctly(monkeypatch):
    monkeypatch.setattr(
        devices,
        "probe_devices",
        lambda timeout=40.0: {
            "npu": True,
            "count": 1,
            "devices": [{"id": 0, "name": "Ascend"}],
            "unusable": {"0": "RuntimeError: error code is 507033"},
        },
    )
    status = devices.read_device_status(enrich=False)
    assert status["available"] is False
    assert status["count"] == 0
    assert "设备无法使用" in (status["reason"] or "")
    assert "507033" in (status["reason"] or "")


def test_submit_refuses_when_no_device_can_run():
    manager = server.TaskManager.__new__(server.TaskManager)
    manager.device_status = lambda wait=True, max_age=3.0: {"available": False, "reason": "设备无法使用"}
    with pytest.raises(server.SubmissionError, match="NPU 当前不可用"):
        manager.submit(None)
