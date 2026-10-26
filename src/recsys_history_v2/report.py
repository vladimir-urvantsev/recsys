"""Render a research report from frozen artifacts, without fit or inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from recsys_research.report import _fmt, _int, _table, _save

LABELS = {"manual": "Manual CatBoost", "selected_auto": "Selected AutoFE", "all_auto": "All AutoFE",
          "history_global": "History + globals", "history_pure": "Pure history",
          "history_no_encoder": "No event encoder", "hybrid": "Logit hybrid"}
COLORS = {"manual": "#111827", "selected_auto": "#2563eb", "history_global": "#ea580c",
          "hybrid": "#059669", "all_auto": "#64748b", "history_pure": "#a78bfa", "history_no_encoder": "#8b5cf6"}


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _optional(out, root, name, default=None):
    return next((_read(p) for p in (out / name, root / name) if p.exists()), default)


def _label(name):
    return LABELS.get(name, name.replace("random_", "Random "))


def _count(value):
    return "—" if value is None else _int(value)


def _mib(value):
    return None if value is None else value / 2**20


def _gib(value):
    return None if value is None else value / 2**30


def _ci(value):
    return "не определён" if value is None else f"[{_fmt(value[0],7)}; {_fmt(value[1],7)}]"


def _load(config):
    out, root = Path(config["output_dir"]), Path(config["processed_dir"])
    result, lock, prediction_lock = [_read(out / n) for n in ("test_results.json", "selection_lock.json", "prediction_lock.json")]
    if lock.get("config") != config or bool(result.get("is_pilot")) != bool(config.get("is_pilot", False)):
        raise ValueError("Report configuration/pilot status differs from the frozen experiment")
    if lock.get("n_confirmation_targets_used_for_selection") != 0 or lock.get("test_labels_opened") is not False:
        raise ValueError("Selection does not certify sealed confirmation targets")
    for key, filename in (("selection_sha256", "selection_lock.json"), ("prediction_lock_sha256", "prediction_lock.json"), ("predictions_sha256", "test_predictions.parquet")):
        if result[key] != _sha(out / filename):
            raise ValueError(f"Frozen artifact checksum differs: {filename}")
    if prediction_lock.get("labels_opened") is not False or prediction_lock["selection_sha256"] != result["selection_sha256"]:
        raise ValueError("Predictions were not locked before label release")
    prep_path = next((p for p in (out / "preparation_lock.json", root / "preparation_lock.json") if p.exists()), None)
    if prep_path is None or lock["preparation_sha256"] != _sha(prep_path):
        raise ValueError("Frozen preparation checksum differs")
    prep = _read(prep_path)
    manifest_path = next(p for p in (out / "manifest.json", root / "manifest.json") if p.exists())
    if prep.get("artifacts", {}).get("manifest.json") not in (None, _sha(manifest_path)):
        raise ValueError("Frozen manifest checksum differs")
    registration = out / "registration.json"
    if registration.exists() and lock.get("registration_sha256") != _sha(registration):
        raise ValueError("Frozen registration checksum differs")
    manifest = _read(manifest_path)
    p = pd.read_parquet(out / "test_predictions.parquet")
    keys = ["uid", "item_id", "day", "group_id"]
    if not {"selected_auto", "history_global"}.issubset(result["models"]):
        raise ValueError("Primary comparison models are absent")
    if not set(keys + ["label", *result["models"]]).issubset(p) or not np.isin(p.label, [0, 1]).all():
        raise ValueError("Final prediction identity/outcome columns are invalid")
    for name, metrics in result["models"].items():
        values = p[name].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any() or metrics.get("n_rows", len(p)) != len(p):
            raise ValueError(f"Saved probability/row count invalid: {name}")
    blind_path = out / "blind_predictions.parquet"
    if blind_path.exists():
        if prediction_lock["blind_predictions_sha256"] != _sha(blind_path):
            raise ValueError("Frozen blind prediction checksum differs")
        blind = pd.read_parquet(blind_path)
        if "label" in blind or not p[keys].equals(blind[keys]):
            raise ValueError("Blind prediction identity/sealing differs")
        if any(n not in blind or not np.array_equal(p[n], blind[n]) for n in result["models"]):
            raise ValueError("Probabilities changed after label release")
    if len(p) != manifest["splits"]["test"]["rows"]:
        raise ValueError("Prediction count differs from the manifest")
    expected = result["models"]["history_global"]["logloss"] - result["models"]["selected_auto"]["logloss"]
    if not np.isclose(expected, result["primary_history_vs_autofe"]["logloss"]["delta"], rtol=0, atol=1e-10):
        raise ValueError("Primary delta contradicts the frozen model metrics")
    return out, root, result, lock, manifest, prep, p


def _figures(out, result, predictions, tabular, trials, context):
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    order = [n for n in LABELS if n in result["models"]] + sorted(set(result["models"]) - set(LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4.8, .36 * len(order) + 1.8)), sharey=True)
    for ax, metric, title in zip(axes, ("logloss", "roc_auc", "user_macro_ndcg_mixed"), ("LogLoss (lower is better)", "ROC-AUC (higher is better)", "User-macro mixed NDCG@10")):
        for position, name in enumerate(order):
            value = result["models"][name].get(metric)
            if value is not None:
                ax.scatter(value, position, color=COLORS.get(name, "#94a3b8"), s=38)
                ax.annotate(f"{value:.4f}", (value, position), xytext=(5, 5), textcoords="offset points", fontsize=7)
        ax.set_title(title)
        ax.grid(axis="x", alpha=.18)
        ax.margins(x=.18)
    axes[0].set_yticks(range(len(order)), [_label(n) for n in order])
    axes[0].invert_yaxis()
    fig.suptitle("Technical pilot only" if result.get("is_pilot") else "Frozen fresh-user confirmation: ensemble probabilities")
    fig.tight_layout()
    _save(fig, folder / "quality.png")

    primary, temporal = result["primary_history_vs_autofe"], result.get("temporal_sensitivity", {})
    intervals = [("Primary: user clusters", primary["logloss"])]
    intervals += [(f"Secondary: {k}-day blocks", {"delta": temporal["delta_logloss"], **v}) for k, v in temporal.get("blocks_days", {}).items()]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    for i, (_, record) in enumerate(intervals):
        if record.get("ci95") is not None:
            axes[0].plot(record["ci95"], [i, i], color="#ea580c", linewidth=2)
            axes[0].scatter([record["delta"]], [i], color="#ea580c")
    axes[0].set_yticks(range(len(intervals)), [n for n, _ in intervals])
    axes[0].set(title="History minus AutoFE: Delta LogLoss", xlabel="Left favors history")
    rank = primary.get("user_macro_ndcg_mixed", {})
    if rank.get("ci95") is not None:
        axes[1].plot(rank["ci95"], [0, 0], color="#2563eb", linewidth=2)
        axes[1].scatter([rank["delta"]], [0], color="#2563eb")
    axes[1].set(yticks=[0], yticklabels=["Secondary: user clusters"], title="Delta user-macro mixed NDCG@10", xlabel="Right favors history")
    for ax in axes:
        ax.axvline(0, color="#111827", linestyle="--", linewidth=1)
        ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    _save(fig, folder / "primary_intervals.png")
    daily = temporal.get("daily_delta", [])
    if daily:
        pd.DataFrame(daily).to_csv(out / "daily_primary_delta.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([r["day"] for r in daily], [np.nan if r["delta_logloss"] is None else r["delta_logloss"] for r in daily], "o-", color="#ea580c")
        ax.axhline(0, color="#111827", linestyle="--")
        ax.axhline(primary["logloss"]["delta"], color="#64748b", linestyle=":", label="Full-period row-weighted delta")
        ax.set(xlabel="Relative observation day", ylabel="History minus AutoFE LogLoss", title="Exploratory temporal heterogeneity")
        ax.grid(alpha=.2)
        ax.legend()
        _save(fig, folder / "daily_primary_delta.png")
    calibration = []
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    y = predictions.label.to_numpy(np.float64)
    for name in ("selected_auto", "history_global", "hybrid"):
        if name not in predictions:
            continue
        scores = predictions[name].to_numpy(float)
        bins = np.minimum((scores * 10).astype(int), 9)
        xs, ys = [], []
        for b in range(10):
            chosen = bins == b
            if chosen.any():
                xs.append(float(scores[chosen].mean()))
                ys.append(float(y[chosen].mean()))
                calibration.append({"family": name, "bin_left": b/10, "bin_right": (b+1)/10, "rows": int(chosen.sum()), "mean_probability": xs[-1], "positive_fraction": ys[-1]})
        ax.plot(xs, ys, "o-", color=COLORS[name], label=_label(name))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed completion fraction", title="Exploratory calibration: fixed probability bins")
    ax.grid(alpha=.2)
    ax.legend()
    _save(fig, folder / "calibration.png")
    pd.DataFrame(calibration).to_csv(out / "calibration.csv", index=False)

    candidates = tabular.get("selection_candidates", [])
    if candidates:
        candidates = sorted(candidates, key=lambda r: len(r["columns"]))
        xs = [len(r["columns"]) for r in candidates]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(xs, [r["logloss"] for r in candidates], "o-", label="Select LogLoss")
        if all("objective" in r for r in candidates):
            ax.plot(xs, [r["objective"] for r in candidates], "s--", label="LogLoss + feature penalty")
        chosen = next((r for r in candidates if r["prefix"] == len(tabular.get("selected_groups", []))), None)
        if chosen:
            ax.scatter([len(chosen["columns"])], [chosen.get("objective", chosen["logloss"])], marker="*", s=170, color="#dc2626", label="Frozen choice", zorder=4)
        ax.set(xlabel="Input feature count", ylabel="Select loss", title="AutoFE: fixed manual backbone")
        ax.grid(alpha=.2)
        ax.legend()
        _save(fig, folder / "autofe_selection.png")
    if any(t.get("epoch_log") for t in trials):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
        epoch_rows = []
        for index, trial in enumerate(trials):
            params = trial.get("params", {})
            name = f"trial {index}: d={params.get('d_model','?')}, pair={params.get('pairwise_weight',0)}"
            for ax, metric in zip(axes, ("train_logloss", "tune_logloss")):
                log = [r for r in trial.get("epoch_log", []) if r.get(metric) is not None]
                if log:
                    ax.plot([r["epoch"] for r in log], [r[metric] for r in log], "o-", label=name)
            epoch_rows += [{"trial": index, **r} for r in trial.get("epoch_log", [])]
        for ax, title in zip(axes, ("Observed train BCE", "Tune LogLoss: checkpoint selection")):
            ax.set(xlabel="Epoch", ylabel="LogLoss", title=title)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, folder / "neural_curves.png")
        pd.DataFrame(epoch_rows).to_csv(out / "neural_epochs.csv", index=False)
    quantiles = context.get("coverage_nonempty_quantiles")
    if quantiles:
        pairs = sorted((float(k), float(v)) for k, v in quantiles.items())
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([q for q, _ in pairs], [100*v for _, v in pairs], "o-", color="#2563eb")
        ax.set(xlabel="Quantile across nonempty user-day histories", ylabel="Retained eligible tokens (%)", title="Context compression across all prepared snapshots")
        ax.grid(alpha=.2)
        _save(fig, folder / "context_coverage.png")
    return order


def _subgroups(out, root, p):
    path = out / "exploratory_subgroups.csv"
    if not (root / "features.parquet").exists() or not (root / "queries.parquet").exists():
        return pd.read_csv(path).to_dict("records") if path.exists() else []
    q = pd.read_parquet(root / "queries.parquet", columns=["uid", "item_id", "day", "split"])
    mask = q.split.eq("test").to_numpy()
    if not q.loc[mask, ["uid", "item_id", "day"]].reset_index(drop=True).equals(p[["uid", "item_id", "day"]]):
        raise ValueError("Prepared diagnostic rows differ from the final predictions")
    columns = ["user__count__all", "user_item__count__all", "user_artist__count__all", "item__count__all"]
    # Four numeric columns only: never load the full multi-GB feature matrix.
    x = pd.read_parquet(root / "features.parquet", columns=columns).loc[mask].reset_index(drop=True)
    n = np.expm1(x.user__count__all.to_numpy())
    masks = [("past_user_listens", "0", n < .5), ("past_user_listens", "1–9", (n >= .5) & (n < 9.5)),
             ("past_user_listens", "10–99", (n >= 9.5) & (n < 99.5)), ("past_user_listens", "100–999", (n >= 99.5) & (n < 999.5)),
             ("past_user_listens", "1000+", n >= 999.5)]
    trained_uid = set(q.loc[q.split.eq("train"), "uid"])
    seen_uid = p.uid.isin(trained_uid).to_numpy()
    masks += [("uid_in_supervised_train", "есть train-примеры", seen_uid),
              ("uid_in_supervised_train", "нет train-примеров", ~seen_uid)]
    for dimension, column in (("user_item", columns[1]), ("user_artist", columns[2]), ("item_in_sample", columns[3])):
        cold = x[column].to_numpy() < .1
        masks += [(dimension, "нет прошлых прослушиваний", cold), (dimension, "есть прошлые прослушивания", ~cold)]
    rows, y = [], p.label.to_numpy(np.float64)
    for dimension, subgroup, chosen in masks:
        if not chosen.any():
            continue
        record = {"dimension": dimension, "subgroup": subgroup, "rows": int(chosen.sum()), "users": int(p.loc[chosen, "uid"].nunique()), "positive_rate": float(y[chosen].mean())}
        for family in ("selected_auto", "history_global"):
            probability = np.clip(p[family].to_numpy()[chosen], 1e-7, 1-1e-7)
            record[family+"_logloss"] = float(np.mean(-(y[chosen]*np.log(probability)+(1-y[chosen])*np.log1p(-probability))))
            record[family+"_auc"] = float(roc_auc_score(y[chosen], probability)) if len(np.unique(y[chosen])) == 2 else None
        record["delta_logloss"] = record["history_global_logloss"] - record["selected_auto_logloss"]
        rows.append(record)
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def _costs(out, lock, tabular, trials):
    rows = []
    for family, spec in lock["models"].items():
        neural = spec.get("kind") == "history"
        fits = spec.get("fits", []) if neural else spec.get("seed_training", [])
        charged = [f for f in fits if not f.get("reused_search_winner", False)] if neural else fits
        rows.append({"family": family, "checkpoints": len(spec.get("model_paths", [])),
                     "additional_fit_seconds": sum(float(f.get("fit_seconds", 0)) for f in charged) if fits else None,
                     "cuda_peak_bytes": max((f.get("cuda_peak_bytes") or 0 for f in fits), default=0) or None,
                     "parameters": max((f.get("parameters", 0) for f in fits), default=0) or None,
                     "epochs_or_trees": spec.get("epochs", spec.get("iterations")), "kind": "history" if neural else "tabular"})
    pd.DataFrame(rows).to_csv(out / "training_costs.csv", index=False)
    usable = [r for r in rows if r["additional_fit_seconds"] is not None]
    if usable:
        fig, ax = plt.subplots(figsize=(9, max(4, .36*len(usable)+1.5)))
        ax.barh([_label(r["family"]) for r in usable], [r["additional_fit_seconds"] for r in usable], color=[COLORS.get(r["family"], "#94a3b8") for r in usable])
        ax.invert_yaxis()
        ax.set(xlabel="Additional fit wall seconds after search/reuse", title="Shared search cost reported separately")
        ax.grid(axis="x", alpha=.2)
        _save(fig, out / "figures" / "training_costs.png")
    neural_final = [r["additional_fit_seconds"] for r in rows if r["kind"] == "history" and r["additional_fit_seconds"] is not None]
    totals = {"tabular_search_fit_seconds": tabular.get("search_fit_seconds"),
              "tabular_additional_final_fit_seconds": tabular.get("additional_final_fit_seconds"),
              "tabular_phase_wall_seconds": tabular.get("total_seconds"),
              "neural_search_fit_seconds": sum(float(t.get("fit_seconds",0)) for t in trials) if trials else None,
              "neural_additional_final_fit_seconds": sum(neural_final) if neural_final else None}
    return rows, totals


def _seeds(out, result, artifact):
    if artifact.get("status") != "completed":
        return "Per-seed test diagnostics не сохранены или не завершены. Разброс оптимизации на confirmation не придумывается: зафиксированные seed используются в ансамбле, а пользовательский bootstrap измеряет другую неопределённость."
    for key in ("selection_sha256", "prediction_lock_sha256", "predictions_sha256"):
        if key in artifact and artifact[key] != result[key]:
            raise ValueError("Seed diagnostic hash refers to a different frozen experiment")
    if any(artifact.get(key, 0) != 0 for key in ("fit_calls", "model_selection_calls", "weight_changes")):
        raise ValueError("Seed diagnostic is not prediction-only")
    rows, series, seed_records, checks = [], [], {}, []
    for family, info in artifact.get("models", {}).items():
        if info.get("ensemble_check", {}).get("passed") is False:
            raise ValueError("Seed diagnostic did not reproduce the frozen ensemble")
        frozen = info.get("frozen_ensemble_metrics", {})
        if family in result["models"] and frozen.get("logloss") is not None and not np.isclose(
                frozen["logloss"], result["models"][family]["logloss"], rtol=0, atol=1e-10):
            raise ValueError("Seed diagnostic belongs to a different frozen ensemble")
        records = [r for r in info.get("seeds", []) if r.get("metrics", {}).get("logloss") is not None]
        if not records:
            continue
        seed_records[family] = {r["seed"]: r for r in records}
        check = info.get("ensemble_check", {})
        if check:
            checks.append([_label(family), _fmt(check.get("max_absolute_probability_error"),12), check.get("tolerance","не записано")])
        values = np.array([r["metrics"]["logloss"] for r in records])
        rows.append([_label(family), len(records), _fmt(values.mean()), _fmt(values.std(ddof=1) if len(records)>1 else None), "; ".join(f"{r['seed']}: {_fmt(r['metrics']['logloss'])}" for r in records)])
        series.append((family, values))
    if not rows:
        return "Завершённый seed artifact не содержит пригодных одиночных LogLoss; фиктивный разброс не строится."
    fig, ax = plt.subplots(figsize=(10, max(4, .4*len(series)+1.5)))
    for i, (family, values) in enumerate(series):
        ax.scatter(values, i+np.linspace(-.12,.12,len(values)), color="#94a3b8", s=28)
        if family in result["models"]:
            ax.scatter([result["models"][family]["logloss"]], [i], marker="D", s=44, color=COLORS.get(family,"#64748b"))
    ax.set_yticks(range(len(series)), [_label(n) for n,_ in series])
    ax.invert_yaxis()
    ax.set(xlabel="Confirmation LogLoss", title="Post-lock seeds: dots=single models, diamond=frozen ensemble")
    ax.grid(axis="x", alpha=.2)
    _save(fig, out / "figures" / "seed_variability.png")
    ablation_rows = []
    reference = seed_records.get("history_global", {})
    for family in ("history_pure", "history_no_encoder"):
        for seed in sorted(set(reference) & set(seed_records.get(family, {}))):
            left = reference[seed]["metrics"]
            right = seed_records[family][seed]["metrics"]
            delta = right["logloss"] - left["logloss"]
            ranking = None if right.get("user_macro_ndcg_mixed") is None or left.get("user_macro_ndcg_mixed") is None else right["user_macro_ndcg_mixed"] - left["user_macro_ndcg_mixed"]
            ablation_rows.append([_label(family),seed,_fmt(delta,7),_fmt(ranking,7)])
    paragraphs = ["Одиночные seed оценены prediction-only аудитом после фиксации результата, без переобучения/выбора. SD — выборочное отклонение по оптимизационным seed, не пользовательский CI; ensemble не обязан совпадать со средним отдельных метрик.", _table(["Семейство","Seed","Средний одиночный LL","SD LL","Одиночные значения"],rows), "![Разброс одиночных seed](figures/seed_variability.png)"]
    if checks:
        paragraphs += ["Повторное GPU-inference допускает записанную floating-point tolerance; нулевое отличие не предполагается заранее.", _table(["Семейство","Max probability reload error","Tolerance"],checks)]
    if ablation_rows:
        paragraphs += ["Дополнительное описательное сравнение **ablation − HistoryGlobal при совпадающей метке seed**, без преимущества трёхмодельного ensemble. Одинаковый номер seed у разных архитектур не означает одинаковые случайные draws. Для одного seed здесь нет оценки неопределённости компонента и нет causal-вывода.", _table(["Абляция","Seed","ΔLogLoss","Δ user-macro NDCG"],ablation_rows)]
    return "\n\n".join(paragraphs)


def _memory(out, result, artifact, n_rows):
    """Describe an optional post-lock input-retention audit, never infer a model."""
    if artifact.get("status") != "completed":
        return ""
    expected = {key: result[key] for key in ("selection_sha256", "prediction_lock_sha256", "predictions_sha256")}
    expected["test_results_sha256"] = _sha(out / "test_results.json")
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("Memory diagnostic hash refers to a different frozen experiment")
    if any(artifact.get(key) != 0 for key in ("fit_calls", "model_selection_calls", "weight_changes")):
        raise ValueError("Memory diagnostic is not a fixed-model input audit")
    entity_labels = {"item": "Трек", "artist": "Исполнитель", "album": "Альбом"}
    bin_labels = {"no_full_prior": "Нет полного прошлого LISTEN",
                  "full_prior_retained": "Прошлый LISTEN сохранён",
                  "full_prior_missing_context": "Прошлый LISTEN отсутствует в context"}
    records, entities = artifact.get("rows", []), artifact.get("entities", {})
    if not records or set(entities) != set(entity_labels):
        raise ValueError("Memory diagnostic lacks the registered entity dimensions")
    for row in records:
        if row.get("entity") not in entity_labels or row.get("bin") not in bin_labels:
            raise ValueError("Memory diagnostic contains an unknown input-only stratum")
        if not isinstance(row.get("rows"), int) or row["rows"] < 0:
            raise ValueError("Memory diagnostic has an invalid row count")
        if row["rows"]:
            values = [row.get(k) for k in ("selected_auto_logloss", "history_global_logloss", "delta_logloss")]
            if any(v is None or not np.isfinite(v) for v in values) or not np.isclose(values[1]-values[0], values[2], rtol=0, atol=1e-10):
                raise ValueError("Memory diagnostic classification loss is inconsistent")
    summary_rows, fractions = [], []
    for entity, label in entity_labels.items():
        summary = entities[entity]
        full, retained, missing = [summary[k] for k in ("full_prior_rows", "retained_prior_rows", "missing_context_prior_rows")]
        no_prior = summary["no_full_prior_rows"]
        if min(full, retained, missing, no_prior) < 0 or retained + missing != full or full + no_prior != n_rows or summary.get("retained_without_full_prior_rows") != 0:
            raise ValueError("Memory diagnostic full-prefix/context counts are inconsistent")
        entity_rows = [row for row in records if row["entity"] == entity]
        if sum(row["rows"] for row in entity_rows) != n_rows or {r["bin"] for r in entity_rows} != set(bin_labels) or len(entity_rows) != 3:
            raise ValueError("Memory diagnostic strata do not partition confirmation rows")
        expected_counts = {"no_full_prior": no_prior, "full_prior_retained": retained, "full_prior_missing_context": missing}
        if any(row["rows"] != expected_counts[row["bin"]] for row in entity_rows) or not 0 <= summary.get("metadata_missing_rows", -1) <= no_prior:
            raise ValueError("Memory diagnostic strata disagree with the retention summary")
        fraction = retained/full if full else None
        saved = summary.get("retention_fraction_given_full_prior")
        if (fraction is None) != (saved is None) or (fraction is not None and not np.isclose(fraction, saved, rtol=0, atol=1e-10)):
            raise ValueError("Memory diagnostic retention fraction has a different denominator")
        fractions.append(fraction)
        summary_rows.append([label, _int(full), _int(retained), _int(missing), _fmt(fraction,4), _count(summary.get("metadata_missing_rows"))])
    # The diagnostic's CSV is an input with its own attested digest. Never
    # normalize/rewrite it while rendering a report, even if the values match.
    csv_path = out / "memory_diagnostic.csv"
    if csv_path.exists():
        expected_csv_sha = artifact.get("csv", {}).get("sha256")
        if expected_csv_sha is not None and _sha(csv_path) != expected_csv_sha:
            raise ValueError("Memory diagnostic CSV checksum differs")
        csv_note = "Исходный CSV сохранён без перезаписи."
    else:
        csv_path = out / "memory_diagnostic_report.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False)
        csv_note = "Исходный CSV отсутствует в bundle; отдельное report-представление восстановлено из JSON, без изменения исходного артефакта."
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.45]})
    for i, fraction in enumerate(fractions):
        if fraction is None:
            axes[0].text(i, 1, "No full prior", ha="center", va="bottom", fontsize=8)
            continue
        axes[0].bar(i, fraction*100, color="#2563eb")
        axes[0].bar(i, (1-fraction)*100, bottom=fraction*100, color="#cbd5e1")
        axes[0].text(i, min(104, fraction*100+2), f"{fraction:.1%}", ha="center", va="bottom", fontsize=8)
    axes[0].set(xticks=range(3), xticklabels=list(entity_labels), ylim=(0,110), ylabel="Test rows with full prior LISTEN (%)", title="Input retention conditional on past match")
    axes[0].legend(handles=[plt.Rectangle((0,0),1,1,color="#2563eb",label="Matching LISTEN retained"),
                            plt.Rectangle((0,0),1,1,color="#cbd5e1",label="Full prior outside memory")], fontsize=7, loc="lower left")
    short_bins = {"no_full_prior": "No full prior", "full_prior_retained": "Prior retained", "full_prior_missing_context": "Prior outside memory"}
    colors = {"no_full_prior": "#94a3b8", "full_prior_retained": "#2563eb", "full_prior_missing_context": "#ea580c"}
    for index, entity in enumerate(entity_labels):
        for offset, bin_name in zip((-.2,0,.2), bin_labels):
            record = next((r for r in records if r["entity"]==entity and r["bin"]==bin_name and r["rows"]), None)
            if record:
                axes[1].scatter(record["delta_logloss"], index+offset, color=colors[bin_name], s=38)
    axes[1].set(yticks=range(3), yticklabels=list(entity_labels), xlabel="History minus AutoFE LogLoss (left favors history)", title="Exploratory association: no subgroup intervals")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#111827", linestyle="--", linewidth=1)
    axes[1].legend(handles=[plt.Line2D([],[],marker="o",linestyle="",color=colors[b],label=short_bins[b]) for b in bin_labels], fontsize=7)
    for ax in axes:
        ax.grid(axis="y" if ax is axes[0] else "x", alpha=.18)
    fig.tight_layout()
    _save(fig, out / "figures" / "memory_diagnostic.png")
    quality_rows = [[entity_labels[r["entity"]], bin_labels[r["bin"]], _int(r["rows"]), _count(r.get("users")),
                     _fmt(r.get("selected_auto_logloss")), _fmt(r.get("history_global_logloss")), _fmt(r.get("delta_logloss"),7),
                     _fmt(r.get("selected_auto_auc")), _fmt(r.get("history_global_auc"))] for r in records]
    return "\n\n".join([
        "### Сохранились ли прошлые прослушивания, связанные с кандидатом",
        "Post-lock анализ связывает **полный eligible prefix** с ограниченной памятью без fit, выбора модели и изменения predictions. Для текущих item/artist/album наличие прошлого LISTEN определяется через полный user–entity count, а сохранность — через exact raw-ID match среди context-токенов типа LISTEN. Целевая метка не участвует в построении групп. Три input-only bins: полного прошлого LISTEN нет; он есть и сохранён; он есть, но ни одного соответствующего LISTEN в context не осталось.",
        "Знаменатель retention — **строки test, у которых в полном прошлом есть соответствующий LISTEN**, отдельно для каждого отношения. Это не доля уникальных событий и не доля всех test-строк. Metadata missing указаны отдельно. `history_counts` в исходном JSON описывает контекст по всем типам событий; matching flags в этой таблице используют только LISTEN, поэтому эти количества не взаимозаменяемы.",
        _table(["Отношение", "С полным прошлым LISTEN", "Сохранено", "Вне context", "Retained/full", "Нет metadata"], summary_rows),
        "![Сохранность релевантных прошлых событий](figures/memory_diagnostic.png)",
        _table(["Отношение", "Input-only группа", "Строк", "UID", "AutoFE LL", "History LL", "ΔLL", "AutoFE AUC", "History AUC"], quality_rows),
        "Это **exploratory ассоциация, не причинный эффект сжатия памяти**: подгруппы различаются активностью, повторяемостью и сложностью примеров, а модель не переобучалась с иной памятью. Разность между строками retained/missing не является оценкой выигрыша от увеличения context. Срезы не использовались для изменения архитектуры/seed/ансамбля; ranking по усечённым candidates не пересчитывается. "
        f"{csv_note} Источники: [`memory_diagnostic.json`](memory_diagnostic.json), [`{csv_path.name}`]({csv_path.name})."
    ])


def _overlap(out, root):
    saved = out / "supervised_uid_overlap.json"
    if (root / "queries.parquet").exists():
        q = pd.read_parquet(root / "queries.parquet", columns=["uid", "split"])
        trained = set(q.loc[q.split.eq("train"), "uid"])
        tested = set(q.loc[q.split.eq("test"), "uid"])
        summary = {"train_users": len(trained), "test_users": len(tested),
                   "test_users_with_train_targets": len(tested & trained),
                   "test_users_without_train_targets": len(tested - trained)}
        saved.write_text(json.dumps(summary, indent=2) + "\n")
    elif saved.exists():
        summary = _read(saved)
    else:
        return "Фактическое пересечение supervised train/test UID не рассчитано: queries недоступны. Наличие raw-прошлого и наличие размеченных train-строк — разные условия."
    return (f"По сохранённым query identity: train содержит {_int(summary['train_users'])} UID; test — {_int(summary['test_users'])}; "
            f"из test UID {_int(summary['test_users_with_train_targets'])} имеют supervised train-строки, "
            f"{_int(summary['test_users_without_train_targets'])} не имеют. Последние всё ещё могут иметь raw-прошлое, например organic-события.")


def _conclusion(primary):
    record = primary["logloss"]
    interval = record.get("ci95")
    if interval is None:
        text = "Первичный пользовательский интервал не определён; утверждение о превосходстве не делается."
    elif interval[1] < 0:
        text = "HistoryGlobal имеет меньший LogLoss, чем выбранный AutoFE: первичный 95% интервал целиком ниже нуля. Вывод условен на данной популяции, месяце и зафиксированной процедуре."
    elif interval[0] > 0:
        text = "HistoryGlobal не превзошёл выбранный AutoFE: его LogLoss выше, и первичный 95% интервал целиком выше нуля. Больше данных и обучаемая история сами по себе не гарантировали замену агрегатов."
    else:
        text = "Надёжное различие HistoryGlobal и выбранного AutoFE не установлено: первичный 95% интервал включает ноль. Точечный сдвиг нельзя выдавать за устойчивую победу."
    text += f" ΔLogLoss (history − AutoFE) = {_fmt(record.get('delta'),7)}, CI {_ci(interval)}."
    rank = primary.get("user_macro_ndcg_mixed", {})
    if rank.get("delta") is not None:
        text += f" Вторичная Δ user-macro mixed NDCG@10 = {_fmt(rank['delta'],7)}, CI {_ci(rank.get('ci95'))}."
    return text


def write_report(config):
    out, root, result, lock, manifest, prep, predictions = _load(config)
    tabular = _optional(out, root, "tabular_search.json", {})
    trials = _optional(out, root, "neural_trials.json", [])
    timing = _optional(out, root, "feature_timing.json", {})
    normalizer = _optional(out, root, "global_normalizer.json", {})
    profile = _optional(out, root, "gpu_profile.json", {})
    context = prep.get("context_summary", {}) or _optional(out, root / "contexts", "context_manifest.json", {})
    if profile and profile.get("config_sha256") not in (None, lock.get("config_sha256")):
        raise ValueError("GPU profile belongs to a different experiment configuration")
    order = _figures(out, result, predictions, tabular, trials, context)
    subgroups = _subgroups(out, root, predictions)
    costs, totals = _costs(out, lock, tabular, trials)
    seeds_text = _seeds(out, result, _optional(out, root, "seed_metrics.json", {}))
    memory_artifact = _optional(out, root, "memory_diagnostic.json", {})
    memory_text = _memory(out, result, memory_artifact, len(predictions))
    overlap_text = _overlap(out, root)
    models, primary = result["models"], result["primary_history_vs_autofe"]
    secondary = result.get("secondary_vs_manual", {})
    secondary_rows = []
    for family in ("selected_auto", "all_auto", "hybrid"):
        if family in secondary:
            values = secondary[family]
            ranking = values.get("user_macro_ndcg_mixed", {})
            secondary_rows.append([_label(family), _fmt(values["logloss"]["delta"],7),
                                   _ci(values["logloss"].get("ci95")), _fmt(ranking.get("delta"),7),
                                   _ci(ranking.get("ci95"))])
    secondary_table = _table(["Модель − Manual", "ΔLogLoss", "95% CI", "Δ user-macro NDCG", "95% CI"], secondary_rows) if secondary_rows else "Вторичные интервалы против Manual не сохранены."
    selected_bounds = secondary.get("selected_auto", {}).get("logloss", {}).get("ci95")
    selection_verdict = ""
    if selected_bounds is not None:
        selection_verdict = ("Интервал Selected AutoFE − Manual включает ноль: отбор не подтвердил преимущество над ручным baseline на этом test." if selected_bounds[0] <= 0 <= selected_bounds[1]
                             else "Интервал Selected AutoFE − Manual целиком ниже нуля: наблюдается вторичное улучшение LogLoss." if selected_bounds[1] < 0
                             else "Интервал Selected AutoFE − Manual целиком выше нуля: отбор ухудшил LogLoss на этом test.")
    plot = lambda name, alt: f"![{alt}](figures/{name})" if (out / "figures" / name).exists() else ""
    quality = _table(["Модель","Checkpoint","LogLoss ↓","ROC-AUC ↑","Brier ↓","AP ↑","User-macro mixed NDCG@10 ↑","Group-macro mixed Recall@10 ↑"], [[_label(n),len(lock["models"].get(n,{}).get("model_paths",[])) if n!="hybrid" else "общие веса", *[_fmt(models[n].get(m)) for m in ("logloss","roc_auc","brier","average_precision","user_macro_ndcg_mixed","recall_mixed")]] for n in order])
    split_rows = [[n,f"{r.get('day_min','—')}–{r.get('day_max','—')}",_count(r.get("users")),_count(r.get("groups")),_int(r["rows"]),_fmt(r.get("positive_rate"),4) if n!="test" else "sealed при подготовке"] for n,r in manifest["splits"].items()]
    select_rows = [[r["prefix"],len(r["columns"]),_fmt(r["logloss"]),_fmt(r.get("objective")),_fmt(r.get("fit_seconds"),2),", ".join(r.get("groups",[])) or "manual"] for r in tabular.get("selection_candidates",[])]
    trial_rows = [[i,json.dumps(t.get("params",{})),t.get("epochs"),len(t.get("epoch_log",[])),_fmt(t.get("tune_logloss")),_fmt(t.get("fit_seconds"),1),_int(t.get("parameters",0)),_fmt(t.get("cuda_peak_bytes")/2**30 if t.get("cuda_peak_bytes") is not None else None,2)] for i,t in enumerate(trials)]
    spec_rows = [[_label(n),json.dumps(s.get("params",{})),s.get("iterations",s.get("epochs","—")),len(s.get("model_paths",[])),"унаследованная абляция" if s.get("controlled_ablation") else "основной/контроль"] for n,s in lock["models"].items()]
    cost_rows = [[_label(r["family"]),r["checkpoints"],_fmt(r["additional_fit_seconds"],1),_int(r["parameters"]) if r["parameters"] else "—",_fmt(r["cuda_peak_bytes"]/2**30 if r["cuda_peak_bytes"] else None,2),r["epochs_or_trees"]] for r in costs]
    context_rows = [[k,context.get(k,"не записано")] for k in ("context_length","recent_capacity","older_capacity","groups","empty_groups","total_available_token_occurrences","total_retained_token_occurrences")]
    temporal = result.get("temporal_sensitivity",{})
    ci_rows = [["Primary: users","LogLoss",_fmt(primary["logloss"]["delta"],7),_ci(primary["logloss"].get("ci95")),primary.get("n_boot","—")]]
    rank = primary.get("user_macro_ndcg_mixed",{})
    if rank.get("delta") is not None:
        ci_rows.append(["Secondary: users","user-macro NDCG",_fmt(rank["delta"],7),_ci(rank.get("ci95")),rank.get("n_valid","—")])
    ci_rows += [[f"Secondary: {k}-day blocks","LogLoss",_fmt(temporal.get("delta_logloss"),7),_ci(v.get("ci95")),v.get("n_boot","—")] for k,v in temporal.get("blocks_days",{}).items()]
    subgroup_table = _table(["Срез","Группа","Строк","Пользователей","P(y=1)","AutoFE LL","History LL","ΔLL"], [[r["dimension"],r["subgroup"],_int(r["rows"]),_int(r["users"]),_fmt(r["positive_rate"],4),_fmt(r["selected_auto_logloss"]),_fmt(r["history_global_logloss"]),_fmt(r["delta_logloss"],7)] for r in subgroups]) if subgroups else "Локальные features и ранее рассчитанные срезы недоступны; значения не придуманы."
    covered = models["selected_auto"]
    coverage = ", ".join(f"{k}: {_count(covered.get(k))}" for k in ("n_rows","n_users","n_groups","n_groups_ge2","n_mixed_groups","n_mixed_users","n_all_positive_groups","n_no_positive_groups","n_singleton_groups"))
    winner, environment = lock.get("neural_winner",{}),lock.get("environment",{})
    pilot = bool(result.get("is_pilot"))
    reproduction_config = "configs/v2_smoke.yaml" if pilot else "configs/v2_reproduce.yaml"
    title = "ТЕХНИЧЕСКИЙ PILOT — не подтверждающий эксперимент" if pilot else "AutoFE против модели истории: новая подтверждающая выборка"
    lead = ("Эта малая выборка проверяет только исполнимость кода. Пользователи взяты из development основной когорты; периоды заканчиваются до основного месяца. Числа pilot нельзя выдавать за результат полного исследования." if pilot else "Все пользователи v1 исключены. Ранние события новой когорты доступны обучению; конфигурации выбираются только по development targets. Confirmation проверяет будущие исходы другой подкогорты с разрешённым ранним участием в train. Не каждый test UID обязательно имеет supervised train-строки. Новизна относится к пользователям и целевым строкам того же наблюдаемого периода, а не к новому календарному режиму.")
    alpha = lock.get("blend",{}).get("auto_weight")
    hybrid_note = "При alpha=1 hybrid не добавляет history-прогноз." if alpha==1 else "При alpha=0 hybrid использует только history-прогноз." if alpha==0 else ""
    profile_text = "Capacity profile отсутствует/не завершён; его throughput, peak VRAM и прогноз времени не подставляются."
    if profile.get("status")=="completed":
        profile_text = f"Train-only capacity probe: {_count(profile.get('measured_steps'))} измеренных шагов; mean step {_fmt(profile.get('mean_step_seconds'),4)} с, valid queries/s {_fmt(profile.get('valid_queries_per_second'),1)}, peak allocated {_fmt(_gib(profile.get('peak_allocated_vram_bytes')),2)} GiB, peak reserved {_fmt(_gib(profile.get('peak_reserved_vram_bytes')),2)} GiB. Конфигурация `{json.dumps(profile.get('capacity_parameters',{}))}`; probe не выбирал модель и не открывал confirmation targets. Полный [`gpu_profile.json`](gpu_profile.json) сохраняет padding и окружение. Его `projection` — экстраполяция короткого probe, **не выполненная эпоха/полный запуск и не гарантия времени или цены**; она исключает validation, AutoFE/CPU, подготовку, I/O и повторные прогоны."
    best = min(order,key=lambda n:models[n]["logloss"])
    report = f"""# {title}

## Результат и граница вывода

{lead}

{_conclusion(primary)}

Победа по LogLoss не означает улучшение порядка треков. Отсутствие значимого вреда не доказывает non-inferiority: для неё нужен заранее обоснованный допустимый margin. Primary-сравнение не заменяется вариантом с лучшим числом в таблице.

## 1. Данные и популяция

Yambda-50M, фиксированная ревизия и SHA256 — в [`manifest.json`](manifest.json). Сохранены **{_int(manifest['retained_events'])} событий {_int(manifest['sampled_users'])} пользователей** с полными предоставленными исходником историями. Development: {_count(manifest.get('development_users',config.get('development_users')))}, confirmation: {_count(manifest.get('confirmation_users'))}. Пересечение с v1: {_count(manifest.get('old_user_overlap'))}; development/confirmation: {_count(manifest.get('development_confirmation_overlap'))}. Пользователи выбираются по ID, без отбора по активности или доле положительного класса. Не все confirmation-ID обязательно дают подходящий target; их не заменяют более активными.

Target: первое за относительный день прослушивание user–item среди всех источников; timestamp/source order разрешают равенства. Затем оставляем `is_organic=0`; `label=1[played_ratio_pct>=50]`. Поэтому ранний organic-запуск не заменяется поздним успешным recommended-запуском. Отрицательные примеры — реальные короткие проигрывания, без negative sampling. Это **условное дослушивание уже начатого трека**, не CTR, full-catalog retrieval или причинный эффект смены policy. [Авторы Yambda](https://huggingface.co/datasets/yandex/yambda/discussions/10) поясняют порог и replay >100%.

Item/artist/album metadata считаются известными при запросе и статичными. Для many-to-many выбирается минимальный ID независимо от outcome; так теряются коллаборации и альбомные связи. Время публикации metadata не дано. Пропуски и множественность сохранены в manifest; это ограничение, а не доказанная историческая доступность.

## 2. Время и запечатанный holdout

Timestamp — относительные секунды, уже округлённые до кратных пяти; повторное умножение на пять ошибочно. Календарные weekday/timezone не восстанавливаются. Warm-up до дня {config.get('warmup_days',30)}, границы `{config.get('split_days')}`. Лимиты supervised-дней `{config.get('days_per_user')}`, seed `{config.get('day_seed')}`: выбранные дни сохраняют все подходящие первые проигрывания, остальные события остаются историей.

{_table(['Split','Дни с примерами','Пользователей','User-day','Строк','Положительная доля'],split_rows)}

Train включает ранние targets обеих новых подкогорт. Tune/select используют только development, test — только confirmation. Это не disjoint-user cold-start benchmark: раннее обучение на confirmation разрешено, но не гарантировано для каждого UID. {overlap_text} Согласование глобального времени и target принципиально для [защиты от утечек](https://arxiv.org/abs/2010.11060) и [sequential evaluation](https://arxiv.org/abs/2507.16289).

```text
listen.available_at = timestamp + ceil(track_length_seconds * played_ratio_pct / 100) + 5
other.available_at  = timestamp + 5
history(query_day d): available_day < d - {config.get('embargo_days',1)}
```

Replay ratio не обрезается при задержке. Train/tune/select-label должен стать доступен до следующей границы; purge до sampling/cohort-фильтра: `{json.dumps(manifest.get('purged_label_rows_by_split_before_cohort_filter',{}))}`. Эти счётчики относятся к исходному пулу, не к уже выбранным дням. Timestamp start/end и wall-clock паузы полностью не документированы: задержка с дневным embargo — консервативное допущение, не доказанная production-механика.

Веса фиксированы, но доступная история обновляется: прежние confirmation-outcome могут стать прошлым последующих дней. Обоим методам задан один cutoff. Текущие ratio, duration, label и source не являются входами. Candidates не attend друг к другу; состав всего будущего slate не нужен для individual score.

Все model paths/хеши и допустимость вероятностей проверяются до label release. **Все** предписанные прогнозы сохранены в `blind_predictions.parquet`, затем `prediction_lock.json`; лишь после этого читается отдельный target-file и проверяются `(uid,item_id,day,group_id)`. Это процедурная защита: raw-log содержит outcomes, а builder обязан видеть разрешённое прошлое. Криптографическая недоступность всех исходных событий не заявляется.

## 3. Сильный baseline и AutoFE

Manual CatBoost: ID и агрегаты user, item, artist, album, user–item, user–artist, user–album; count/rate, recency, historical ratio/organic share, likes/dislikes, user shares и trends. Окна `{config.get('manual_windows')}`. Likes/dislikes — исторические количества, не текущий state после отмены действий.

Грамматика добавляет count/rate для box-окон `{config.get('auto_windows')}` и exponential half-life `{config.get('auto_half_lives')}`. Один kernel образует группу по отношениям. Это ограниченный поиск, связанный с [DFS](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf), [ExploreKit](https://people.eecs.berkeley.edu/~dawnsong/papers/icdm-2016.pdf) и [OpenFE](https://proceedings.mlr.press/v202/zhang23ay.html), а не новый общий алгоритм AutoFE/SQL/LLM.

Константы удаляются только по train. Manual и AllAuto получают grid `{config.get('hyper_grid')}`, максимум {config.get('iterations')} деревьев, early stopping {config.get('early_stopping')}; параметры/деревья выбираются на tune. Importance all-auto модели задаёт порядок групп. Префиксы select, включая полный, используют **фиксированные параметры и число деревьев Manual**. Это устраняет confound смены backbone при оценке новых признаков. Независимо настроенный AllAuto показан отдельно.

Criterion: select LogLoss + `{config.get('selection_feature_penalty',1e-6)}` на дополнительный столбец. Выбрано `{tabular.get('selected_groups',[])}`. Это importance-prefix search, не глобальный оптимум: коррелированные окна делят важность, а множество кандидатов создаёт winner's curse.

{_table(['Групп','Столбцов','Select LL','Objective','Новый fit, с','Группы'],select_rows) if select_rows else 'Prefix-журнал недоступен; значения не восстанавливаются из test.'}

{plot('autofe_selection.png','Выбор групп на select')}

Random subsets `{tabular.get('random_groups',[])}` уникальны, равномерны условно на выбранном K, seed `{config.get('random_seed',8821)}`. Запрошено контролей {_count(tabular.get('n_requested_random_controls'))}, фактически уникальных {_count(tabular.get('n_unique_random_controls'))}. Если конечное пространство меньше трёх, перечисляется всё. Селектор не исключается из контроля; совпадения `{tabular.get('random_control_matches_selector',[])}` не перерисовываются. Совпадающие спецификации могут переиспользовать fit при одинаковом порядке столбцов. Три контроля не являются точной оценкой распределения всех случайных поисков.

## 4. Обучаемая история и информация

История кодируется один раз на user-day. Candidate-query cross-attends к памяти независимо от остальных candidates. Токены содержат raw item/artist/album ID, тип события, прошлые ratio/organic и log1p возраста. Для категорий используются две prime-mixed hash banks, без vocabulary-fit по будущему. Exact-match bias сравнивает исходные неотрицательные IDs, не hash buckets; missing/PAD не создают ложной связи. Исторический ratio кодируется десятипроцентными bins с cap 200%; это сжатие представления, не availability cap.

Transformer видит уже разрешённое прошлое; двустороннее attention внутри него допустимо. Есть null-token для empty history и явная PAD-mask. Это response-архитектура по мотивам [DIN](https://arxiv.org/abs/1706.06978) и [BST](https://arxiv.org/abs/1905.06874), не точное воспроизведение этих моделей, SASRec next-item или HSTU. Последние {config.get('context_recent')} событий дополняются до {config.get('context_older')} равномерно расположенными старыми chronological indices, без дубликатов и выбора по текущему target.

{_table(['Контекст: все подготовленные snapshot','Значение'],context_rows)}

Квантили retained/available по непустым user-day: `{json.dumps(context.get('coverage_nonempty_quantiles'))}`. Это не только test; token-occurrences повторно учитывают события в разных snapshot и не равны числу уникальных сохранённых исходных действий.

{plot('context_coverage.png','Сжатие историй ограниченным контекстом')}

HistoryGlobal получает 37 фиксированных **manual population-marginals**: global prior, item/artist/album count/rate 7/30/all, recency, mean ratio, organic share, likes/dislikes и marginal trends. Нормализатор fit по `{normalizer.get('fit_split','не записано')}` ({normalizer.get('fitted_rows','не записано')} rows), clip `{normalizer.get('clip')}`; собственные user/pair-агрегаты в primary-head не подаются. Фактически записано marginal-столбцов: {len(normalizer.get('columns',[])) if normalizer else 'нет metadata'}.

**Общий raw-log и cutoff не означают идентичной сохранённой информации.** AutoFE агрегирует полный eligible prefix и может добавлять global item/artist/album kernels сверх фиксированного блока. Собственная bounded-history не восстанавливает всю population statistics. Сравниваются законченные представления/компрессия при общем базовом marginal-state, а не идеальная информационно тождественная замена каждого агрегата.

Neural grid `{config.get('neural_grid')}`; AdamW, weight decay `{config.get('neural_weight_decay')}`, максимум {config.get('neural_max_epochs')} эпох, patience {config.get('neural_patience')}. BCE суммируется по valid plays с постоянной expected-rows нормировкой: маленькие user-day не получают лишний вес mean-of-means. Pairwise weight — заранее заданный вариант loss. Выбор epoch/config — по tune LogLoss, не по test NDCG. Gradient clipping 5, precision `{config.get('mixed_precision')}`.

{_table(['Trial','Параметры','Выбрано эпох','Выполнено эпох','Tune LL','Fit, с','Параметров','Peak allocated GiB'],trial_rows) if trial_rows else 'Подробные neural trials отсутствуют.'}

{plot('neural_curves.png','Train и tune-кривые neural trials')}

Выбрано `{json.dumps(winner.get('params',{}))}`, {winner.get('epochs','не записано')} эпох. Выполненные после лучшего checkpoint эпохи не исчезают из затрат. Остальные основные seeds получают фиксированные параметры/число эпох. Pure History и No Event Encoder наследуют их; no-encoder сохраняет age и candidate attention, поэтому это не устранение всей хронологии.

## 5. Зафиксированные модели и качество

{_table(['Семейство','Параметры','Деревья/эпохи','Checkpoint','Роль'],spec_rows)}

Основные seed `{config.get('model_seeds')}`, ансамбль — среднее вероятностей. Абляции используют `{config.get('ablation_seeds')}`; если их меньше, отличие от главного ансамбля не изолирует только удалённый компонент, так как различается ensemble budget. Без одиночного main-seed score это описательный контроль, а не чистый эффект encoder/global features.

Hybrid: `sigmoid(alpha*logit(AutoFE)+(1-alpha)*logit(History))`; alpha={alpha} выбран на select-прогнозах **тех же итоговых ансамблей** по сетке `{config.get('blend_weights')}`. Test не меняет alpha. {hybrid_note}

{quality}

![Качество](figures/quality.png)

Минимальный наблюдённый test LogLoss у {_label(best)} ({_fmt(models[best]['logloss'])}) — описание предписанных контролей, не новое решение. Primary остаётся HistoryGlobal–Selected AutoFE. Покрытие: {coverage}. Mixed ranking исключает singleton, no-positive и тривиальные all-positive группы; user-macro сначала усредняет дни внутри пользователя. Это ретроспективное переупорядочивание реально проигранных треков, не полный каталог и не заранее доступный slate. Ties используют expected gain, а не target-based разрешение.

### Первичная гипотеза и вторичная чувствительность

{_table(['Resampling','Метрика','History − AutoFE','95% CI','Повторов'],ci_rows)}

![Интервалы](figures/primary_intervals.png)

Пользователи пересэмплируются целиком и парно; LogLoss сохраняет веса строк, ranking равномерно взвешивает подходящих пользователей. Bootstrap условен на уже обученных и выбранных моделях, без fit/selection внутри resampling. Доля улучшившихся resamples — не p-value и не posterior probability. Вторичные сравнения не исправлены за multiplicity.

Circular moving-day blocks 3/7 сохраняют временные промежутки, включая дни без строк. Это отдельная sensitivity к общим временным shocks, **не joint user/time bootstrap** и не доказательство нового будущего режима. Больше пользователей не добавляет независимых будущих месяцев и не устраняет общие item/policy shocks.

{plot('daily_primary_delta.png','Ежедневные парные разности')}

### Что дала автоматизация признаков

{secondary_table}

{selection_verdict} Префиксы отбора сохраняют параметры и число деревьев Manual. AllAuto настраивается независимо: его отличие от Manual не изолирует только новые признаки. Hybrid проверяет дополнение двух представлений с весом, выбранным на select. Все интервалы этой таблицы вторичны, без multiplicity correction, и не заменяют primary HistoryGlobal–Selected AutoFE.

Случайные комбинации того же K показаны полностью в таблице качества. Три fixed RNG-контроля не дают основания объявлять общий оптимум грамматики или переизбирать subset по test; они позволяют сопоставить результат селектора с несколькими заранее заданными подмножествами.

### Разброс оптимизационных seed

{seeds_text}

## 6. Анализ и exploratory-срезы

LogLoss оценивает и склонность пользователя к дослушиванию, и калибровку; NDCG оценивает порядок треков внутри user-day. Улучшение одного не гарантирует другого. Возможные объяснения различий: bounded memory, редкие ID/hash-коллизии, global state, выбранные kernels, шум выбора и сдвиг состава активности между периодами. Это гипотезы, а не доказанные механизмы. Абляции/coverage помогают локализовать ограничения; меньший ensemble budget не позволяет приписать весь сдвиг одному компоненту.

{memory_text}

![Калибровка](figures/calibration.png)

Bins шириной 0.1 фиксированы; counts сохранены в `calibration.csv`. Это visual diagnostic, без fit/recalibration по test; малые bins могут быть шумными. Следующие срезы **exploratory**, не новая селекция. Только классификация: ranking по урезанным наборам кандидатов не пересчитывается. Интервалы/multiplicity correction по срезам не заявляются.

{subgroup_table}

`item_in_sample` означает прошлые listens внутри выбранной когорты, не истинный возраст трека. Нет прошлой пары — не то же, что отсутствующий ID при train. User-history count считает listens; raw context дополнительно включает другие типы действий.

## 7. Реальные вычисления и масштаб

Окружение **сохранённого запуска**: `{json.dumps(environment)}`. CatBoost CPU threads: {config.get('threads','не записано')}; batch capacity history: {config.get('neural_batch_groups','не записано')} user-day. В исследовании основной запуск использовал PyTorch 2.8/cu128, локальная разработка — CPU PyTorch 2.14. Точные версии обеих сред зафиксированы отдельно; CUDA-версия записана в runtime metadata и GPU lock.

Вся подготовка: {_fmt(prep.get('seconds'),1)} с; вложенные data sampling/I/O {_fmt(manifest.get('elapsed_seconds'),1)} с, feature engine {_fmt(timing.get('seconds'),1)} с. Матрица {_count(timing.get('n_rows'))} × {_count(timing.get('n_features'))}, DataFrame estimate {_fmt(_mib(timing.get('matrix_bytes')),1)} MiB. Эти таймеры нельзя складывать как независимые стадии.

{_table(['Учтённый компонент','Секунды'],[[k,_fmt(v,1)] for k,v in totals.items()])}

{_table(['Семейство','Checkpoint','Доп. fit после search, с','Параметров сети','Peak allocated GiB','Деревья/эпохи'],cost_rows)}

{plot('training_costs.png','Дополнительные обучения после поиска/reuse')}

Reused search-winner не оплачивается второй раз: CatBoost reuse имеет fit=0, neural copy не прибавляет исходный search-fit повторно. Neural search учитывает все trials и rejected поздние эпохи; остальные main seeds и абляции добавляются отдельно. Это training-таймеры, **не полное end-to-end время и не денежная стоимость**. Neural fit включает epochs/tune/checkpoint после создания модели/optimizer; не все install, transfer, profile, final inference и debugging входят в сумму. Одинаковое число trials не означает одинаковых FLOPs или CPU/GPU-бюджета. Tabular process peak RSS {_fmt(_mib(tabular.get('process_peak_rss_bytes')),1)} MiB включает caller allocations, не изолированную память одного fit.

{profile_text}

History-cache переиспользуется на user-day, сокращая повторное Transformer-кодирование. Attention всё равно растёт квадратично по длине контекста; supervised sampling меняет число групп и candidates. Distributed throughput, online tail latency и промышленный миллиардный масштаб здесь не измерялись.

## 8. Вывод и воспроизведение

{_conclusion(primary)}

Результат относится к выбранной активности Yambda, logged-play target и наблюдаемому периоду. Он не доказывает retrieval/online uplift или универсальную победу Transformer/AutoFE. Следующие варианты потребуют нового подтверждающего набора: этот test не превращается в validation для архитектуры, окна, seed или blend.

```bash
PYTHONPATH=src python -m recsys_history_v2.run --config {reproduction_config} --phase register
PYTHONPATH=src python -m recsys_history_v2.run --config {reproduction_config} --phase prepare
PYTHONPATH=src python scripts/profile_v2_gpu.py --config {reproduction_config}
PYTHONPATH=src python -m recsys_history_v2.run --config {reproduction_config} --phase search
PYTHONPATH=src python -m recsys_history_v2.run --config {reproduction_config} --phase evaluate
PYTHONPATH=src python -m recsys_history_v2.run --config {reproduction_config} --phase report
```

Это команды основного зарегистрированного исследования; pilot использует отдельные YAML/каталоги и более ранние дни. Повторный evaluate при prediction lock запрещён. GPU/веса не требуются для чтения сохранённых результатов: этот генератор не обучает и не предсказывает моделью.

Артефакты: [selection](selection_lock.json), [prediction lock](prediction_lock.json), [результаты](test_results.json), [прогнозы](test_predictions.parquet), [tabular search](tabular_search.json), [neural trials](neural_trials.json), [подготовка/context](preparation_lock.json), [данные](manifest.json), [калибровка](calibration.csv), [срезы](exploratory_subgroups.csv), [затраты](training_costs.csv). Optional `seed_metrics.json` содержит только post-lock диагностику. Проверенная selection SHA256 `{result['selection_sha256']}`; prediction SHA256 `{result['predictions_sha256']}`. При наличии blind parquet дополнительно проверены его hash, отсутствие label и неизменность probabilities после release.

Первичные источники и дизайн: [sequence research](../SEQUENCE_RESEARCH.md), [протокол](../PROTOCOL_REVIEW.md), [v1 как отдельное исследование](../../main/REPORT.md). V2 отделён от замороженного v1, а его просмотренный test не использовался как новый validation.
"""
    target = out / "REPORT.md"
    target.write_text(report, encoding="utf-8")
    provenance = {k:result[k] for k in ("selection_sha256","prediction_lock_sha256","predictions_sha256")}
    provenance.update(model_fit_calls=0,model_prediction_calls=0,primary_delta_definition="history_global minus selected_auto")
    if memory_artifact.get("status") == "completed":
        memory_path = next(p for p in (out / "memory_diagnostic.json", root / "memory_diagnostic.json") if p.exists())
        provenance["memory_diagnostic_sha256"] = _sha(memory_path)
    (out / "report_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(f"Report written: {target}",flush=True)
    return target
