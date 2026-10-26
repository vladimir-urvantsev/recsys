#!/usr/bin/env python3
"""Measure the same feature engine on nested full-history user samples.

Each size runs in a fresh process, sequentially. No model fitting or quality
selection takes place, and label values are never used for this benchmark.
Run after model training to avoid competing for CPU/RAM.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def worker(config, n):
    import numpy as np
    import pandas as pd
    from recsys_research.features import build_features

    root = Path(config["processed_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    uids = np.array(manifest["sampled_uids"])
    order = np.random.default_rng(941).permutation(uids)
    if not 1 <= n <= len(order):
        raise ValueError(f"Size must be in [1,{len(order)}]")
    chosen = np.sort(order[:n]).tolist()
    t0 = time.perf_counter()
    events = pd.read_parquet(root / "sample_events.parquet", filters=[("uid", "in", chosen)])
    q = pd.read_parquet(root / "queries.parquet", columns=["uid", "item_id", "artist_id", "album_id", "day"])
    q = q.loc[q.uid.isin(chosen)].reset_index(drop=True)
    load_seconds = time.perf_counter() - t0
    x, _, timing = build_features(events, q, config)
    result = {"users": n, "history_events": len(events), "query_rows": len(q),
              "feature_seconds": timing["seconds"], "load_seconds": load_seconds,
              "matrix_bytes": timing["matrix_bytes"], "features": len(x.columns),
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
              "rows_per_second": len(q) / timing["seconds"],
              "events_per_second": len(events) / timing["seconds"]}
    target = Path(config["output_dir"]) / f"scaling_{n}.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 300, 1000])
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.worker is not None:
        worker(config, args.worker)
        return
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.sizes:
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "4"}
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", args.config,
                        "--worker", str(n)], env=env, check=True)
        rows.append(json.loads((output / f"scaling_{n}.json").read_text()))
    (output / "scaling.json").write_text(json.dumps({
        "note": "Sequential fresh processes, nested uniform user samples from main cohort; full histories retained; no labels used for decisions. Timings include feature engine but exclude file loading, reported separately. These measurements do not establish billion-event scale.",
        "platform": platform.platform(), "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
