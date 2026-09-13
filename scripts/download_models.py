"""
scripts/download_models.py — Download MediaPipe model files for Magic Mirror.

Run once before launching the application:
    python scripts/download_models.py

Downloads:
    models/holistic_landmarker.task  (~13 MB)
        Used by src/perception/human.py for real-time pose + hand detection.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_MODELS = [
    {
        "filename": "holistic_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "holistic_landmarker/holistic_landmarker/float16/latest/"
            "holistic_landmarker.task"
        ),
        "description": "MediaPipe HolisticLandmarker (pose + hands, float16)",
    },
]


def _download(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} …", end="", flush=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  Downloading {dest.name} … {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    size_kb = dest.stat().st_size // 1024
    print(f"\r  Downloaded  {dest.name}  ({size_kb:,} KB)")


def main() -> None:
    _MODELS_DIR.mkdir(exist_ok=True)
    print(f"Model directory: {_MODELS_DIR}\n")

    for model in _MODELS:
        dest = _MODELS_DIR / model["filename"]
        print(f"[{model['description']}]")
        if dest.exists():
            size_kb = dest.stat().st_size // 1024
            print(f"  Already exists: {dest.name} ({size_kb:,} KB) — skipping.")
        else:
            _download(model["url"], dest)

    print("\nAll models ready.")


if __name__ == "__main__":
    main()
