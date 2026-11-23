"""Create a reproducibility manifest for the external OHLCV CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "file": str(args.data),
        "bytes": args.data.stat().st_size,
        "sha256": sha256(args.data),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
