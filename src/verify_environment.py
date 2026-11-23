"""Fail fast if the project environment cannot run the intended benchmarks."""

from __future__ import annotations

import json
import platform
import shutil
import sys


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "selected_device": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "free_disk_gb": round(shutil.disk_usage(".").free / (1024**3), 1),
    }
    print(json.dumps(report, indent=2))
    if device == "cpu":
        print("WARNING: no accelerator detected; use a Colab or Kaggle GPU for Transformer experiments.")


if __name__ == "__main__":
    main()
