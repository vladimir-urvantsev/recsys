#!/usr/bin/env python3
"""Measure v2 CUDA training capacity before financing the full experiment.

This is a resource probe, not a model-quality experiment. It reads training
batches only, discards the temporary model, performs no validation/selection,
and never releases confirmation labels or saves a checkpoint. Run after data
preparation on the GPU host, before the full search:

    PYTHONPATH=src python scripts/profile_v2_gpu.py --config configs/v2.yaml

The measured step includes host batch construction, host-to-device copies,
forward/backward, gradient clipping and AdamW. Epoch/schedule numbers are
projections of this short probe; they are not a completed full-run benchmark.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2.yaml"))
    parser.add_argument("--warmup", type=int, default=10, help="Unrecorded warm-up optimizer steps")
    parser.add_argument("--steps", type=int, default=100, help="Measured optimizer steps")
    parser.add_argument("--output", type=Path, help="New JSON file; default output_dir/gpu_profile.json")
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be nonnegative and --steps positive")
    return args


def worst_case_parameters(config):
    """A capacity envelope, never a trial selected by observed model quality."""
    grid = config.get("neural_grid", [])
    if not grid:
        raise ValueError("neural_grid must contain at least one predeclared configuration")
    # Select an existing base for optimizer/dropout settings, then use the
    # componentwise architecture envelope. In the registered grid this is an
    # actual d=128/layers=2/heads=4/pairwise=.1 configuration.
    base = max(grid, key=lambda p: (
        int(p.get("n_layers", 2)) * int(p.get("d_model", 128)) ** 2,
        int(p.get("d_model", 128)), int(p.get("n_heads", 4)),
    ))
    parameters = dict(base)
    parameters.update(
        d_model=max(int(p.get("d_model", 128)) for p in grid),
        n_layers=max(int(p.get("n_layers", 2)) for p in grid),
        n_heads=max(int(p.get("n_heads", 4)) for p in grid),
        pairwise_weight=max(float(p.get("pairwise_weight", 0)) for p in grid),
    )
    if parameters["d_model"] % parameters["n_heads"]:
        raise ValueError("The grid's componentwise capacity envelope has incompatible d_model/n_heads")
    return parameters


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def profile(config, args, output):
    import numpy as np
    import torch
    from torch.nn import functional as F

    from recsys_history_v2.model import EventHistoryModel
    from recsys_history_v2.run import fingerprint, load_prepared
    from recsys_history_v2.trainer import GroupBatches, autocast_context, pairwise_sum
    from recsys_research.data import sha256_file

    device = torch.device("cuda:0")
    seed = int(config.get("model_seeds", [17])[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(config.get("threads", 16)))
    parameters = worst_case_parameters(config)
    loaded = time.perf_counter()
    _, queries, _, preparation = load_prepared(config, features=False)
    load_seconds = time.perf_counter() - loaded
    root = Path(config["processed_dir"])
    normalizer = json.loads((root / "global_normalizer.json").read_text())
    global_dim = len(normalizer["columns"])
    batches = GroupBatches(root, queries, "train", int(config["neural_batch_groups"]), global_dim)
    if not len(batches):
        raise ValueError("Training batches are empty")
    context_length = int(batches.arrays["history_mask"].shape[1])
    expected_context = int(config.get("context_recent", 256)) + int(config.get("context_older", 256))
    if context_length != expected_context:
        raise ValueError("Prepared history length differs from the registered model context length")
    model_config = {**config, **parameters}
    model = EventHistoryModel(model_config, global_dim).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(parameters["learning_rate"]),
        weight_decay=float(config.get("neural_weight_decay", 1e-4)),
    )
    # Identical to fit_history: fixed expected query count preserves row-weighted
    # training across variable-size user-day batches.
    expected_rows = len(batches.rows) / len(batches)
    iterator_epoch = 1
    iterator = iter(batches.batches(shuffle=True, seed=seed * 1000 + iterator_epoch))
    times, valid_queries, groups, candidate_slots = [], [], [], []
    last_finite = True
    warmup_seconds = 0.0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    properties = torch.cuda.get_device_properties(device)
    environment = {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device), "device": str(device),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "gpu_multiprocessors": int(properties.multi_processor_count),
        "visible_cuda_devices": torch.cuda.device_count(),
        "logical_cpu_count": os.cpu_count(), "torch_cpu_threads": torch.get_num_threads(),
        "mixed_precision": config.get("mixed_precision", "fp32"),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    metadata = {
        "status": "running", "purpose": "training capacity probe; no model-quality claim",
        "config_sha256": fingerprint(config),
        "preparation_lock_sha256": sha256_file(root / "preparation_lock.json"),
        "script_sha256": sha256_file(Path(__file__)), "environment": environment,
        "capacity_parameters": parameters,
        "capacity_envelope": "componentwise largest d_model/layers/heads and pairwise weight in neural_grid; not selection",
        "architecture_matches_declared_trial": parameters in config["neural_grid"],
        "model": model.description(), "seed": seed,
        "train_rows": len(batches.rows), "train_user_day_groups": len(batches.groups),
        "train_steps_per_epoch": len(batches), "batch_groups_capacity": batches.batch_groups,
        "context_length": context_length, "global_dim": global_dim,
        "expected_training_rows_per_batch": expected_rows,
        "prepared_data_verification_and_load_seconds": load_seconds,
        "query_splits_used": ["train"], "validation_calls": 0,
        "model_selection_calls": 0, "confirmation_label_reader_calls": 0,
        "checkpoint_files_written": 0,
        "label_sealing_note": "load_prepared verifies stored artifact hashes; it does not release/read confirmation labels as targets",
        "prepared_is_pilot": bool(config.get("is_pilot", False)),
    }
    try:
        for step in range(args.warmup + args.steps):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator_epoch += 1
                iterator = iter(batches.batches(shuffle=True, seed=seed * 1000 + iterator_epoch))
                batch = next(iterator)
            n_groups = len(batch["group_uid"])
            n_valid = int(batch["candidate_mask"].sum())
            n_slots = batch["candidate_mask"].numel()
            batch.pop("row_indices")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config):
                logits = model(batch)
                mask, labels = batch["candidate_mask"], batch["labels"]
                bce = F.binary_cross_entropy_with_logits(
                    logits.float()[mask], labels[mask], reduction="sum"
                )
                loss = bce / expected_rows
                if parameters["pairwise_weight"]:
                    loss = loss + parameters["pairwise_weight"] * pairwise_sum(
                        logits.float(), labels, mask
                    ) / config["neural_batch_groups"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            # The real trainer similarly consumes bce/count on host each step.
            observed_bce = float(bce.detach())
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            last_finite = last_finite and math.isfinite(observed_bce)
            if step < args.warmup:
                warmup_seconds += elapsed
            else:
                times.append(elapsed)
                valid_queries.append(n_valid)
                groups.append(n_groups)
                candidate_slots.append(n_slots)
            if step + 1 == args.warmup or step + 1 == args.warmup + args.steps:
                print(f"Completed {min(step+1,args.warmup)} warm-up / {max(0,step+1-args.warmup)} measured steps", flush=True)
    except torch.cuda.OutOfMemoryError as error:
        metadata.update(
            status="cuda_out_of_memory", error=str(error),
            measured_steps_completed=len(times),
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
            projection=None,
        )
        write_json(output, metadata)
        raise SystemExit(f"CUDA capacity probe ran out of memory; diagnostics saved to {output}") from error
    if not last_finite:
        raise RuntimeError("The resource probe produced non-finite training loss; no timing projection is valid")
    mean_step = statistics.fmean(times)
    max_epochs = int(config["neural_max_epochs"])
    trial_fits = len(config["neural_grid"])
    extra_ensemble_fits = max(len(config.get("model_seeds", [17])) - 1, 0)
    # The registered runner has two controlled ablations: pure/no-encoder.
    ablation_fits = 2 * len(config.get("ablation_seeds", [17]))
    maximum_epoch_schedule = (trial_fits + extra_ensemble_fits + ablation_fits) * max_epochs
    projected_epoch = mean_step * len(batches)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    metadata.update(
        status="completed", warmup_steps=args.warmup, warmup_seconds=warmup_seconds,
        measured_steps=args.steps, measured_seconds=sum(times), mean_step_seconds=mean_step,
        median_step_seconds=statistics.median(times),
        step_seconds_quantiles={str(p): float(np.quantile(times, p)) for p in [0, .1, .5, .9, .95, .99, 1]},
        measured_valid_queries=sum(valid_queries), measured_user_day_groups=sum(groups),
        measured_padded_candidate_slots=sum(candidate_slots),
        candidate_padding_fraction=1 - sum(valid_queries) / sum(candidate_slots),
        valid_queries_per_second=sum(valid_queries) / sum(times),
        user_day_groups_per_second=sum(groups) / sum(times),
        measured_mean_valid_queries_per_batch=statistics.fmean(valid_queries),
        peak_allocated_vram_bytes=peak_allocated, peak_reserved_vram_bytes=peak_reserved,
        cuda_free_bytes_after_probe=int(free_bytes), cuda_total_bytes=int(total_bytes),
        finite_training_loss=True,
        projection={
            "kind": "short-probe extrapolation, not a completed epoch/full-run benchmark or guaranteed bound",
            "training_epoch_seconds": projected_epoch,
            "training_epoch_minutes": projected_epoch / 60,
            "train_steps_per_epoch": len(batches),
            "maximum_epoch_schedule": maximum_epoch_schedule,
            "schedule_components": {"search_trials": trial_fits,
                                    "extra_main_ensemble_fits": extra_ensemble_fits,
                                    "controlled_ablation_fits": ablation_fits,
                                    "max_epochs_per_fit": max_epochs},
            "stress_model_training_only_schedule_hours": projected_epoch * maximum_epoch_schedule / 3600,
            "not_included": ["validation/inference", "checkpoint I/O", "AutoFE/CatBoost", "data/context preparation",
                             "package installation/transfer", "profiling itself", "reruns/debugging"],
            "limitations": ["smaller configurations and no-encoder ablation have different costs",
                            "early stopping can shorten search fits; main seeds use the selected epoch count",
                            "candidate count/padding and filesystem caching vary across batches",
                            "this probe is not an exact total-price or wall-time guarantee",
                            "prepared pilot/subsample measurements cannot establish full-population throughput"],
        },
        measured_step_seconds=times,
    )
    write_json(output, metadata)
    print(f"GPU: {environment['gpu']}; mean step {mean_step:.4f}s; allocated peak {peak_allocated/2**30:.2f} GiB", flush=True)
    print(f"Projected train epoch {projected_epoch/60:.2f} min; {maximum_epoch_schedule}-epoch stress-model training-only scenario "
          f"{projected_epoch*maximum_epoch_schedule/3600:.2f} h (excludes validation/AutoFE/preparation; not a guarantee)", flush=True)
    print(f"Capacity profile written: {output}", flush=True)
    del optimizer, model, batches
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = arguments()  # --help never initializes CUDA or opens prepared data.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this capacity probe. No prepared data or confirmation labels were opened.")
    import yaml
    config_path = args.config if args.config.is_absolute() else REPOSITORY / args.config
    config = yaml.safe_load(config_path.read_text())
    output = args.output or Path(config["output_dir"]) / "gpu_profile.json"
    output = output if output.is_absolute() else REPOSITORY / output
    if output.exists():
        raise SystemExit(f"Profile output already exists: {output}; choose a new --output to retain the previous measurement")
    # Core preparation paths are repository-relative, as in the experiment CLI.
    os.chdir(REPOSITORY)
    profile(config, args, output)


if __name__ == "__main__":
    main()
