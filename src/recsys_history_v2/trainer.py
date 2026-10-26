"""Group-shared history training with a row-weighted supervised objective."""
from __future__ import annotations

import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .model import EventHistoryModel


def global_columns(manual_columns):
    """Exact population marginal block; no user or user-pair aggregates."""
    return [c for c in manual_columns if c == "global_positive_prior" or
            c.startswith(("item__", "artist__", "album__"))]


def prepare_global_features(x, q, groups, output):
    output = Path(output)
    columns = global_columns(groups["manual"])
    raw = x[columns].to_numpy(np.float32)
    train = q.split.eq("train").to_numpy()
    mean = raw[train].mean(axis=0, dtype=np.float64)
    scale = raw[train].std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1
    normalized = np.clip((raw - mean) / scale, -12, 12).astype(np.float32)
    np.save(output / "global_features.npy", normalized)
    metadata = {"columns": columns, "mean": mean.tolist(), "scale": scale.tolist(),
                "clip": [-12, 12], "fitted_rows": int(train.sum()), "fit_split": "train"}
    (output / "global_normalizer.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class GroupBatches:
    """Each user-day history is encoded once for all of that day's candidates."""
    def __init__(self, root, queries, split, batch_groups, global_dim=37):
        root = Path(root)
        self.q = queries
        self.rows = np.flatnonzero(queries.split.eq(split).to_numpy())
        if not len(self.rows):
            raise ValueError(f"No supervised groups for {split}")
        self.batch_groups = batch_groups
        self.global_dim = global_dim
        context = root / "contexts"
        names = ("history_ids", "history_type", "history_ratio", "history_organic",
                 "history_age", "history_mask", "group_uid", "query_group")
        self.arrays = {n: np.load(context / f"{n}.npy", mmap_mode="r") for n in names}
        self.global_features = np.load(root / "global_features.npy", mmap_mode="r")
        codes = self.arrays["query_group"][self.rows]
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1, len(order)]
        self.groups = [self.rows[order[a:b]] for a, b in zip(boundaries[:-1], boundaries[1:])]
        self.group_codes = np.array([codes[order[a]] for a in boundaries[:-1]], np.int64)
        self.query_ids = queries[["item_id", "artist_id", "album_id"]].to_numpy(np.int64)
        self.labels = queries.label.to_numpy(np.float32)
        if split != "test" and not np.isin(self.labels[self.rows], [0., 1.]).all():
            raise ValueError("Development labels must be binary")
        if split == "test" and not np.isnan(self.labels[self.rows]).all():
            raise ValueError("Test labels must still be sealed during prediction")

    def __len__(self):
        return math.ceil(len(self.groups) / self.batch_groups)

    def batches(self, shuffle=False, seed=0):
        indices = np.arange(len(self.groups))
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, len(indices), self.batch_groups):
            selected = indices[start:start + self.batch_groups]
            group_codes = self.group_codes[selected]
            rows = [self.groups[i] for i in selected]
            width = max(map(len, rows))
            batch = {n: torch.from_numpy(np.array(self.arrays[n][group_codes], copy=True)) for n in
                     ("history_ids", "history_type", "history_ratio", "history_organic", "history_age", "history_mask")}
            batch["group_uid"] = torch.from_numpy(np.array(self.arrays["group_uid"][group_codes], dtype=np.int64))
            ids = np.full((len(rows), width, 3), -2, np.int64)
            labels = np.zeros((len(rows), width), np.float32)
            mask = np.zeros((len(rows), width), bool)
            global_values = np.zeros((len(rows), width, self.global_dim), np.float32)
            row_indices = np.full((len(rows), width), -1, np.int64)
            for b, indices_in_q in enumerate(rows):
                n = len(indices_in_q)
                ids[b, :n] = self.query_ids[indices_in_q]
                labels[b, :n] = self.labels[indices_in_q]
                mask[b, :n] = True
                row_indices[b, :n] = indices_in_q
                if self.global_dim:
                    global_values[b, :n] = self.global_features[indices_in_q]
            batch.update(candidate_ids=torch.from_numpy(ids), labels=torch.from_numpy(labels),
                         candidate_mask=torch.from_numpy(mask), global_features=torch.from_numpy(global_values),
                         row_indices=torch.from_numpy(row_indices))
            yield batch


def resolve_device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run preflight on the GPU host")
    return torch.device(requested)


def autocast_context(device, config):
    if device.type == "cuda" and config.get("mixed_precision") == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported; choose mixed_precision: fp32 before registration")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def pairwise_sum(logits, labels, mask, max_pairs=64):
    """Equal-group ranking auxiliary; used only on observed training labels."""
    total = logits.sum() * 0
    for b in range(len(logits)):
        positive = torch.where(mask[b] & (labels[b] == 1))[0]
        negative = torch.where(mask[b] & (labels[b] == 0))[0]
        if not len(positive) or not len(negative):
            continue
        pairs = min(max_pairs, len(positive) * len(negative))
        pos = positive[torch.randint(len(positive), (pairs,), device=logits.device)]
        neg = negative[torch.randint(len(negative), (pairs,), device=logits.device)]
        total = total + F.softplus(logits[b, neg] - logits[b, pos]).mean()
    return total


@torch.no_grad()
def predict_history(model, batches, config, device):
    model.eval()
    result = np.full(len(batches.q), np.nan, np.float64)
    for batch in batches.batches():
        rows = batch.pop("row_indices").numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_context(device, config):
            logits = model(batch)
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        valid = rows >= 0
        result[rows[valid]] = probabilities[valid]
    selected = result[batches.rows]
    if not np.isfinite(selected).all():
        raise ValueError("Missing/non-finite history predictions")
    return selected


def fit_history(root, queries, params, config, seed, checkpoint, epochs=None, pure=False, no_encoder=False):
    """Train only train labels; choose epoch on tune unless explicitly fixed."""
    from recsys_research.run import classification_loss
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("threads", 4))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_config = {**config, **params}
    if no_encoder:
        model_config["n_layers"] = 0
    normalizer = json.loads((Path(root) / "global_normalizer.json").read_text())
    global_dim = 0 if pure else len(normalizer["columns"])
    train = GroupBatches(root, queries, "train", config["neural_batch_groups"], global_dim)
    tune = GroupBatches(root, queries, "tune", config["neural_batch_groups"], global_dim) if epochs is None else None
    model = EventHistoryModel(model_config, global_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                 weight_decay=config.get("neural_weight_decay", 1e-4))
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, failures, best_epoch = float("inf"), 0, 0
    log = []
    started = time.perf_counter()
    # Constant normalization across variable-size user-day batches gives every
    # recorded play the same epoch weight; small days aren't upweighted.
    expected_rows = len(train.rows) / len(train)
    for epoch in range(1, (epochs or config["neural_max_epochs"]) + 1):
        model.train()
        loss_sum, count = 0., 0
        for batch in train.batches(shuffle=True, seed=seed * 1000 + epoch):
            batch.pop("row_indices")
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config):
                logits = model(batch)
                mask, labels = batch["candidate_mask"], batch["labels"]
                bce = F.binary_cross_entropy_with_logits(logits.float()[mask], labels[mask], reduction="sum")
                loss = bce / expected_rows
                if params.get("pairwise_weight", 0):
                    loss = loss + params["pairwise_weight"] * pairwise_sum(logits.float(), labels, mask) / config["neural_batch_groups"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(bce.detach())
            count += int(mask.sum())
        validation = None
        if tune is not None:
            p = predict_history(model, tune, config, device)
            validation = classification_loss(queries.loc[queries.split.eq("tune"), "label"].to_numpy(), p)
        record = {"epoch": epoch, "train_logloss": loss_sum / count, "tune_logloss": validation}
        log.append(record)
        print(f"history seed={seed} epoch={epoch} train={loss_sum/count:.6f} tune={validation}", flush=True)
        improved = tune is None or validation < best - 1e-6
        if improved:
            best = validation if validation is not None else 0.
            best_epoch, failures = epoch, 0
            torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "model_config": model_config, "global_dim": global_dim,
                        "seed": seed, "epoch": epoch}, checkpoint)
        else:
            failures += 1
        if tune is not None and failures >= config["neural_patience"]:
            break
    information = {"seed": seed, "epochs": best_epoch, "params": params, "pure": pure,
                   "no_encoder": no_encoder, "fit_seconds": time.perf_counter() - started,
                   "tune_logloss": best if tune is not None else None, "epoch_log": log,
                   "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
                   "cuda_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "checkpoint": str(checkpoint)}
    checkpoint.with_suffix(".json").write_text(json.dumps(information, indent=2) + "\n")
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return information


def load_history(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = EventHistoryModel(saved["model_config"], saved["global_dim"])
    model.load_state_dict(saved["state"])
    return model.to(device).eval(), saved
