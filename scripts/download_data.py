#!/usr/bin/env python3
"""Download the exact public Yambda inputs using only Python's standard library.

Run before installing dependencies if desired. Every input is checksum-verified,
and incomplete downloads are kept under a .part name until verification passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
BASE_URL = f"https://huggingface.co/datasets/yandex/yambda/resolve/{REVISION}"
FILES = {
    "multi_event.parquet": {
        "remote": "sequential/50m/multi_event.parquet",
        "bytes": 437_501_893,
        "sha256": "2eaf0619bcfadd31a907e2bc88aab7ec34e36cbafca41f4e6ff34188985d990d",
    },
    "artist_item_mapping.parquet": {
        "remote": "artist_item_mapping.parquet",
        "bytes": 38_274_621,
        "sha256": "cf1110aa2413ed70008ec9232d395660b2181ffda731002fb74da54e87b6f25d",
    },
    "album_item_mapping.parquet": {
        "remote": "album_item_mapping.parquet",
        "bytes": 54_528_894,
        "sha256": "559b6576063d96280414574b32292eee7b47ee0b9f2952cb3ae670a5d19125e4",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, spec: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == spec["bytes"]
        and sha256_file(path) == spec["sha256"]
    )


def download_one(path: Path, spec: dict) -> None:
    if verified(path, spec):
        print(f"Verified existing {path}", flush=True)
        return
    temporary = path.with_name(path.name + ".part")
    url = f"{BASE_URL}/{spec['remote']}?download=true"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "recsys-research-reproducibility/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
                digest = hashlib.sha256()
                written = 0
                report_at = 16 << 20
                while chunk := response.read(1 << 20):
                    target.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if written >= report_at:
                        print(f"{path.name}: {written / 1e6:.1f}/{spec['bytes'] / 1e6:.1f} MB", flush=True)
                        report_at += 16 << 20
                target.flush()
                os.fsync(target.fileno())
            if written != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
                raise ValueError(
                    f"Checksum/size mismatch for {path.name}: {written} bytes, {digest.hexdigest()}"
                )
            temporary.replace(path)
            print(f"Downloaded and verified {path}", flush=True)
            return
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            print(f"Attempt {attempt + 1}/3 failed for {path.name}: {error}", flush=True)
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download {path.name}; incomplete file is {temporary}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify pinned local files without network access"
    )
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in FILES.items():
        path = args.raw_dir / name
        if args.verify_only:
            if not verified(path, spec):
                raise SystemExit(f"Missing or invalid input: {path}")
            print(f"Verified {path}", flush=True)
        else:
            download_one(path, spec)
    metadata = args.raw_dir / "huggingface_revision.json"
    if not metadata.exists():
        # The pinned revision and checksums are sufficient for reproducibility;
        # fetching moving Hub metadata is neither necessary nor appropriate.
        metadata.write_text(json.dumps({
            "id": "yandex/yambda", "sha": REVISION,
            "sources": {name: {**spec, "url": f"{BASE_URL}/{spec['remote']}"}
                        for name, spec in FILES.items()},
        }, indent=2) + "\n")
    else:
        actual_revision = json.loads(metadata.read_text()).get("sha")
        if actual_revision != REVISION:
            raise SystemExit(f"Metadata revision mismatch: expected {REVISION}, got {actual_revision}")
    print(f"All inputs verified at revision {REVISION}", flush=True)


if __name__ == "__main__":
    main()
