#!/usr/bin/env python3
"""Capture supplementary execution provenance after the frozen V2 evaluation.

This does not fit, select, load model weights or regenerate predictions. Backend
flags describe a fresh process using the same runtime, not instrumentation of the
finished training process. The primary evidence remains the frozen ML locks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2.yaml"))
    args = parser.parse_args()
    import torch
    import yaml

    config = yaml.safe_load(args.config.read_text())
    out = Path(config["output_dir"])
    result = json.loads((out / "test_results.json").read_text())
    selection = json.loads((out / "selection_lock.json").read_text())
    if result["selection_sha256"] != sha(out / "selection_lock.json"):
        raise ValueError("Selection lock does not match the completed result")
    if result["predictions_sha256"] != sha(out / "test_predictions.parquet"):
        raise ValueError("Completed predictions have changed")
    query = subprocess.run([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True)
    gpu = [dict(zip(("name", "driver_version", "memory_total_mib", "compute_capability"),
                    (value.strip() for value in row))) for row in csv.reader(io.StringIO(query.stdout))]
    cgroups = {}
    for name in ("memory.max", "cpu.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        cgroups[name] = path.read_text().strip() if path.exists() else None
    jobs = {}
    resource_observations = {}
    for name in ("setup", "download", "prepare", "search", "finish", "postcheck"):
        status = Path(name + ".status.json")
        if status.exists():
            jobs[name] = json.loads(status.read_text())
        resources = Path(name + "_resources.json")
        if resources.exists():
            resource_observations[name] = json.loads(resources.read_text())
    artifacts = {}
    names = ["runtime_environment.json", "requirements-gpu.lock", "gpu_profile.json",
             "registration.json", "preparation_lock.json", "selection_lock.json",
             "prediction_lock.json", "test_results.json", "seed_metrics.json",
             "memory_diagnostic.json", "memory_diagnostic.csv", "audit.json", "audit_initial.json",
             "source_snapshot_note.json", "full_tests.log", "REPORT.md"]
    for name in names:
        path = out / name
        if path.exists():
            artifacts[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    record = {
        "schema": "v2-execution-supplement-1",
        "purpose": "post-evaluation provenance; no training, prediction or selection",
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": gpu,
        "container_cgroups": cgroups, "configured_threads": config.get("threads"),
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "CUBLAS_WORKSPACE_CONFIG", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "fresh_process_backend_defaults": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "backend_scope": "Fresh process only. Training code does not override these flags. "
                         "This is not a promise of bitwise repeat training across hardware/runtimes.",
        "job_walltimes": jobs,
        "stage_resource_observations": resource_observations,
        "cost_scope": "Stage walltimes include I/O and verification; setup/download may overlap. "
                      "They are not summed billing hours or an isolated fit benchmark.",
        "config_sha256": selection["config_sha256"],
        "frozen_sources": selection["sources"],
        "supplementary_artifact_hashes": artifacts,
        "audit_history": "audit_initial.json, when present, preserves the first audit. "
                         "The final audit recognizes binary AppleDouble metadata separately when "
                         "checking unchanged V1 implementation; the complete registered V2 source "
                         "hash set, including metadata, is still attested. No ML choices changed.",
        "script_sha256": sha(Path(__file__)),
    }
    target = out / "execution_provenance.json"
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(target)


if __name__ == "__main__":
    main()
