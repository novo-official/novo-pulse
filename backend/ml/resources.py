"""Hardware detection -> recommended training profile.

The recommendation is advisory: the user always chooses the final profile.
"""
from __future__ import annotations

import os
import platform
from typing import Any


def detect_resources() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "ram_gb": None,
        "cuda": False,
        "gpu_name": None,
        "vram_gb": None,
    }

    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        try:  # Linux fallback without psutil
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            info["ram_gb"] = round(pages * page_size / 1e9, 1)
        except (ValueError, OSError, AttributeError):
            pass

    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / 1e9, 1)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpu_name"] = "Apple MPS"
    except ImportError:
        info["torch"] = None

    info["recommended_profile"] = recommend_profile(info)
    info["device"] = "cuda" if info["cuda"] else "cpu"
    return info


def recommend_profile(resources: dict[str, Any]) -> str:
    """Map hardware to demo / competition / full."""
    cpus = resources.get("cpu_count") or 1
    ram = resources.get("ram_gb") or 4
    if resources.get("cuda") and (resources.get("vram_gb") or 0) >= 8 and cpus >= 8:
        return "full"
    if cpus >= 8 and ram >= 16:
        return "competition"
    if cpus >= 4 and ram >= 8:
        return "competition"
    return "demo"


def resource_summary() -> str:
    info = detect_resources()
    gpu = info.get("gpu_name") or "none"
    return (
        f"{info['cpu_count']} CPU / {info.get('ram_gb', '?')} GB RAM / GPU: {gpu} "
        f"-> suggested profile: {info['recommended_profile']}"
    )
