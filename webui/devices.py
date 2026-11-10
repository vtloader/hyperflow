# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Device metrics for the WebUI.

An Ascend card exposes two different memory numbers, and a UI that conflates them misleads:

* **physical HBM** — what the board has (`65536 MB` on an Ascend 910). `npu-smi` reports this, and its "used" count
  includes the driver / CANN resident footprint, so it is a few GB even with no process running.
* **framework budget** — what a process may allocate. `torch.npu.mem_get_info()` returns this; on the same card it is
  61.27 GiB, the remaining ~2.73 GiB being reserved by the driver. Its free/used counters describe *this process's*
  allocations only.

The WebUI therefore reports both: the physical bar (matching `npu-smi`, with the driver reservation visible) and the
allocatable bar (what the next generation run actually has to work with). Neither is derived from the other.

`torch_npu` is the primary source because its API is stable, unlike `npu-smi`'s terminal table whose columns and
widths change between CANN releases. `npu-smi` is used only for what torch does not expose (board capacity,
temperature, power) and is parsed by caption offset so a layout change degrades to ``None`` instead of a wrong number.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from typing import Any

#: Read in a child process: importing torch_npu costs ~1 s and a broken CANN can abort the interpreter.
_TORCH_PROBE = r"""
import json
out = {"npu": False, "devices": []}
try:
    import torch
    import torch_npu  # noqa: F401
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
    raise SystemExit(0)
try:
    out["npu"] = bool(torch.npu.is_available())
    out["count"] = int(torch.npu.device_count())
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
    raise SystemExit(0)
# `is_available()` caches its answer, so it still reports True after the driver has gone away and every real call
# fails with `lost heartbeat` / `507033`. Allocate and synchronise once per device to find out whether the card can
# actually run work; that is what a scheduling decision needs.
for index in range(out.get("count", 0)):
    try:
        probe = torch.ones(8, device=f"npu:{index}")
        torch.npu.synchronize(index)
        probe.cpu()
    except Exception as exc:
        out.setdefault("unusable", {})[str(index)] = f"{type(exc).__name__}: {str(exc)[:120]}"
for index in range(out.get("count", 0)):
    record = {"id": index}
    try:
        free, total = torch.npu.mem_get_info(index)
        record["free_bytes"] = int(free)
        record["allocatable_bytes"] = int(total)
    except Exception as exc:
        record["memory_error"] = f"{type(exc).__name__}: {exc}"
    try:
        record["name"] = str(torch.npu.get_device_name(index))
    except Exception:
        record["name"] = "Ascend"
    try:
        record["aicore_percent"] = float(torch.npu.utilization(index))
    except Exception:
        record["aicore_percent"] = None
    try:
        props = torch.npu.get_device_properties(index)
        record["props_total_bytes"] = int(getattr(props, "total_memory", 0) or 0)
        record["cube_core_num"] = int(getattr(props, "cube_core_num", 0) or 0)
        record["vector_core_num"] = int(getattr(props, "vector_core_num", 0) or 0)
        record["l2_cache_bytes"] = int(getattr(props, "L2_cache_size", 0) or 0)
    except Exception:
        pass
    out["devices"].append(record)
print(json.dumps(out))
"""


def probe_devices(timeout: float = 40.0) -> dict[str, Any]:
    """Ask `torch_npu` for device memory and utilisation, in a child process."""
    try:
        result = subprocess.run([sys.executable, "-c", _TORCH_PROBE], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"npu": False, "devices": [], "error": str(exc)[:200]}
    payload = None
    for line in reversed((result.stdout or "").strip().splitlines()):
        try:
            payload = json.loads(line)
            break
        except ValueError:
            continue
    if payload is None:
        stderr = [line for line in (result.stderr or "").strip().splitlines() if line.strip()]
        return {"npu": False, "devices": [], "error": stderr[-1][:200] if stderr else "torch_npu probe failed"}
    return payload


def devices_from_torch(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise the probe records, keeping the two memory numbers separate.

    Devices the probe could not even allocate on are dropped: a card whose driver has gone away still answers
    `is_available()`/`device_count()`, but scheduling work onto it only produces a 30-minute hang.
    """
    unusable = payload.get("unusable") or {}
    devices = []
    for record in payload.get("devices", []):
        index = int(record.get("id", len(devices)))
        if str(index) in unusable:
            continue
        allocatable = record.get("allocatable_bytes")
        free = record.get("free_bytes")
        used = (int(allocatable) - int(free)) if (allocatable and free is not None) else None
        devices.append(
            {
                "id": index,
                "card": index,
                "chip": 0,
                "name": record.get("name") or "Ascend",
                "label": f"NPU {index}",
                # Framework budget (torch): what a generation run may allocate.
                "allocatable_bytes": int(allocatable) if allocatable else None,
                "allocated_bytes": used,
                "free_bytes": int(free) if free is not None else None,
                "allocated_percent": (
                    round(100 * used / allocatable, 1) if used is not None and allocatable else None
                ),
                # Physical HBM (npu-smi): filled in by `enrich_with_npu_smi`.
                "hbm_total_bytes": None,
                "hbm_used_bytes": None,
                "hbm_used_percent": None,
                "driver_reserved_bytes": None,
                "aicore_percent": record.get("aicore_percent"),
                "cube_core_num": record.get("cube_core_num"),
                "vector_core_num": record.get("vector_core_num"),
                "l2_cache_bytes": record.get("l2_cache_bytes"),
            }
        )
    return devices


def parse_npu_smi_map(text: str) -> dict[int, tuple[int, int]]:
    """Chip logic id -> (physical NPU id, chip id), from `npu-smi info -m`.

    `torch.npu` numbers devices 0, 1, ... by *chip logic id*, while `npu-smi -i` takes the *physical* NPU id. On the
    test host the two disagree (logic 0/1 live under physical NPU 1), so the mapping has to be read rather than
    assumed. Rows for non-compute units (the MCU) have no physical id and are skipped.
    """
    mapping: dict[int, tuple[int, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        npu_id, chip_id, logic_id = parts[0], parts[1], parts[2]
        if not (npu_id.isdigit() and chip_id.isdigit() and logic_id.isdigit()):
            continue
        mapping[int(logic_id)] = (int(npu_id), int(chip_id))
    return mapping


def parse_npu_smi_memory(text: str) -> int | None:
    """Physical HBM capacity in bytes from `npu-smi info -t memory`.

    The documented query prints one ``HBM Capacity(MB)`` block per chip on the selected card. The capacity is the
    same for every chip on a board, so the first block is authoritative for the card.
    """
    for line in text.splitlines():
        match = re.search(r"HBM Capacity\(MB\)\s*:\s*(\d+)", line)
        if match:
            return int(match.group(1)) * 1024 * 1024
    return None


def parse_npu_smi_telemetry(text: str) -> dict[int, dict[str, Any]]:
    """Card id -> temperature / power / AI-core, read at the captions' character offsets.

    Decoration only. The caption rows are located first, then each data row is cut at the same offsets, so a different
    column order or an added column degrades to ``None`` rather than to a wrong number.
    """
    lines = text.splitlines()
    captions: dict[str, int] = {}
    for line in lines:
        if not line.startswith("|"):
            continue
        for name in ("Power(W)", "Temp(C)", "AICore(%)", "Memory-Usage(MB)", "HBM-Usage(MB)"):
            index = line.find(name)
            if index != -1 and name not in captions:
                captions[name] = index
    if not captions:
        return {}

    def next_caption(after: str) -> str | None:
        start = captions.get(after)
        if start is None:
            return None
        later = [(pos, name) for name, pos in captions.items() if pos > start]
        return min(later)[1] if later else None

    # `npu-smi` right-aligns numbers, so a caption's character offset does not line up with its digits. The chip rows
    # always print one cell holding `<used>/<total>` twice, memory first and HBM second, in the caption order. Reading
    # the pairs in order is what keeps working when the columns shift.
    def pairs_in_cell(line: str) -> list[tuple[int, int]]:
        boundary = line.rfind("|", 0, line.rfind("|")) if line.count("|") >= 2 else len(line)
        cell = line[boundary + 1 :]
        matches = re.findall(r"(\d+)\s*/\s*(\d+)", cell)
        return [(int(used), int(total)) for used, total in matches]

    def numeric(line: str, name: str) -> float | None:
        start = captions.get(name)
        if start is None:
            return None
        text_slice = line[start:]
        match = re.search(r"-?\d+(?:\.\d+)?", text_slice.split("|")[0])
        return float(match.group(0)) if match else None

    result: dict[int, dict[str, Any]] = {}
    card: int | None = None
    for line in lines:
        if not line.startswith("|"):
            continue
        head = line.split("|")[1].strip() if "|" in line else ""
        card_match = re.fullmatch(r"(\d+)\s+(Ascend\S+)", head)
        if card_match:
            card = int(card_match.group(1))
            result.setdefault(
                card,
                {
                    "power_w": numeric(line, "Power(W)"),
                    "temp_c": numeric(line, "Temp(C)"),
                    "aicore_percent": None,
                    "hbm_used_bytes": None,
                },
            )
            continue
        # Only chip rows (`0  2`) carry AI-core and HBM; the card row above them carries power and temperature.
        if card is None or not re.fullmatch(r"\d+\s+\d+", head):
            continue
        entry = result.setdefault(card, {})
        entry["aicore_percent"] = numeric(line, "AICore(%)")
        # `<memory used>/<memory total>` then `<hbm used>/<hbm total>`; HBM is the second pair.
        pairs = pairs_in_cell(line)
        if len(pairs) >= 2:
            entry["hbm_used_bytes"] = pairs[1][0] * 1024 * 1024
        elif len(pairs) == 1 and captions.get("HBM-Usage(MB)") is None:
            entry["hbm_used_bytes"] = pairs[0][0] * 1024 * 1024
    return result


def enrich_with_npu_smi(devices: list[dict[str, Any]], timeout: float = 8.0) -> list[dict[str, Any]]:
    """Attach physical HBM, temperature and power from `npu-smi`.

    Purely additive: a missing or reshaped table leaves every device untouched, and the allocatable numbers the UI
    reports still come from `torch_npu`.
    """
    if not devices or not shutil.which("npu-smi"):
        return devices

    # Logic id -> physical card, because `torch.npu` and `npu-smi -i` number devices differently.
    mapping: dict[int, tuple[int, int]] = {}
    try:
        info_map = subprocess.run(["npu-smi", "info", "-m"], capture_output=True, text=True, timeout=timeout)
        if info_map.returncode == 0:
            mapping = parse_npu_smi_map(info_map.stdout)
    except (OSError, subprocess.SubprocessError):
        pass

    telemetry: dict[int, dict[str, Any]] = {}
    try:
        info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=timeout)
        if info.returncode == 0:
            telemetry = parse_npu_smi_telemetry(info.stdout)
    except (OSError, subprocess.SubprocessError):
        pass

    capacities: dict[tuple[int, int], int] = {}
    for device in devices:
        card = device["card"]
        location = mapping.get(card)
        if location is None:
            continue
        npu_id, chip_id = location
        if location in capacities:
            continue
        try:
            memory = subprocess.run(
                ["npu-smi", "info", "-t", "memory", "-i", str(npu_id), "-c", str(chip_id)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if memory.returncode == 0:
            capacity = parse_npu_smi_memory(memory.stdout)
            if capacity:
                capacities[location] = capacity

    for device in devices:
        card = device["card"]
        location = mapping.get(card)
        stats = telemetry.get(location[0], {}) if location else {}
        total = capacities.get(location) if location else None
        if total:
            device["hbm_total_bytes"] = total
            used = stats.get("hbm_used_bytes")
            if used is not None:
                device["hbm_used_bytes"] = used
                device["hbm_used_percent"] = round(100 * used / total, 1)
            allocatable = device.get("allocatable_bytes")
            if allocatable:
                # The gap is the driver / CANN reservation, which is why the two totals differ.
                device["driver_reserved_bytes"] = max(0, total - allocatable)
        device["temp_c"] = stats.get("temp_c")
        device["power_w"] = stats.get("power_w")
        if device.get("aicore_percent") is None:
            device["aicore_percent"] = stats.get("aicore_percent")
    return devices


def read_device_status(enrich: bool = True) -> dict[str, Any]:
    """Device list from `torch_npu`, optionally decorated with `npu-smi` board telemetry."""
    payload = probe_devices()
    devices = devices_from_torch(payload)
    if devices and enrich:
        devices = enrich_with_npu_smi(devices)
    error = payload.get("error")
    unusable = payload.get("unusable") or {}
    if devices:
        reason = None
    elif unusable:
        # The driver is loaded but no card can execute: say so instead of a vague "not detected".
        detail = "; ".join(f"NPU {index}: {message}" for index, message in sorted(unusable.items()))
        reason = f"设备无法使用（驱动无响应）：{detail[:180]}"
    else:
        reason = error or "torch_npu 未检测到可用设备"
    return {
        "available": bool(devices),
        "source": "torch_npu" if devices else None,
        "reason": reason,
        "count": len(devices),
        "devices": devices,
    }


__all__ = [
    "devices_from_torch",
    "enrich_with_npu_smi",
    "parse_npu_smi_map",
    "parse_npu_smi_memory",
    "parse_npu_smi_telemetry",
    "probe_devices",
    "read_device_status",
]
