#!/usr/bin/env python3
"""Preserve and verify frozen V2 weights without model imports or inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return digest_stream(stream)


def sidecar(destination, kind):
    return destination.with_name(destination.name.removesuffix(".tar.gz") + f".{kind}.json")


def verify(destination, study):
    summary = json.loads(sidecar(destination, "archive").read_text())
    if sha(destination) != summary["sha256"]:
        raise ValueError("Archive checksum differs")
    with tarfile.open(destination, "r:gz") as archive:
        manifest = json.load(archive.extractfile("artifacts/v2_frozen_models.manifest.json"))
        if manifest["selection_sha256"] != summary["selection_sha256"]:
            raise ValueError("Archive/manifest selection differs")
        lock = study / "selection_lock.json"
        if lock.exists() and sha(lock) != manifest["selection_sha256"]:
            raise ValueError("Local selection differs from archived models")
        covered = set()
        for member in archive:
            if member.name not in manifest["models"]:
                continue
            if member.name in covered or not member.isfile():
                raise ValueError("Duplicate/non-file model member")
            expected = manifest["models"][member.name]
            if member.size != expected["bytes"] or digest_stream(archive.extractfile(member)) != expected["sha256"]:
                raise ValueError(f"Frozen member differs: {member.name}")
            covered.add(member.name)
        if covered != set(manifest["models"]) or len(covered) != summary["frozen_model_count"]:
            raise ValueError("Archive does not contain the complete frozen model set")
    print(json.dumps({"verified_models": len(covered), "archive_sha256": summary["sha256"],
                      "model_fit_calls": 0, "prediction_calls": 0}), flush=True)


def package(destination, study):
    if destination.exists():
        raise FileExistsError("Archive exists; verify it instead of overwriting")
    selection_path = study / "selection_lock.json"
    selection = json.loads(selection_path.read_text())
    if not (study / "test_results.json").exists():
        raise ValueError("A completed frozen evaluation is required")
    models = {}
    for spec in selection["models"].values():
        if set(spec["model_paths"]) != set(spec["model_sha256"]):
            raise ValueError("Frozen model paths are not exactly covered")
        for name, digest in spec["model_sha256"].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or path.is_symlink() or sha(path) != digest:
                raise ValueError(f"Invalid or changed frozen weight: {name}")
            models[str(path.relative_to(ROOT))] = {"sha256": digest, "bytes": path.stat().st_size}
    manifest = {"selection_sha256": sha(selection_path), "models": dict(sorted(models.items())),
                "scope": "All frozen final weights. Search metrics/logs remain in the report bundle; "
                         "large prepared tables can be regenerated from pinned public inputs."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = sidecar(destination, "manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=6) as archive:
            for name in sorted(models):
                archive.add(ROOT / name, arcname=name, recursive=False)
                metadata = (ROOT / name).with_suffix(".json")
                if metadata.exists():
                    archive.add(metadata, arcname=str(metadata.relative_to(ROOT)), recursive=False)
            archive.add(manifest_path, arcname="artifacts/v2_frozen_models.manifest.json", recursive=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    summary = {"archive": destination.name, "sha256": sha(destination),
               "bytes": destination.stat().st_size, "frozen_model_count": len(models),
               "selection_sha256": manifest["selection_sha256"]}
    sidecar(destination, "archive").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    verify(destination, study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("reports/v2/run"))
    parser.add_argument("--destination", type=Path, default=Path("artifacts/v2_frozen_models.tar.gz"))
    parser.add_argument("--verify", action="store_true", help="Stream-check archive members without extracting")
    args = parser.parse_args()
    destination = args.destination if args.destination.is_absolute() else ROOT / args.destination
    study = args.study_dir if args.study_dir.is_absolute() else ROOT / args.study_dir
    (verify if args.verify else package)(destination, study)


if __name__ == "__main__":
    main()
