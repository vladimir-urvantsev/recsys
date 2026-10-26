"""Generate a Russian research report from frozen experiment artifacts only."""

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


LABELS = {
    "constant": "Константа train",
    "static": "CatBoost: ID + prior",
    "no_pair_relations": "Manual без парных агрегатов",
    "manual": "Manual CatBoost",
    "all_auto": "Manual + все AutoFE",
    "selected_auto": "Manual + отбор AutoFE",
    "random_auto": "Manual + случайные группы",
}
ORDER = list(LABELS)
EN_LABELS = {
    "constant": "Train prevalence", "static": "IDs + prior",
    "no_pair_relations": "Manual, no pair aggregates", "manual": "Manual",
    "all_auto": "All AutoFE", "selected_auto": "Selected AutoFE",
    "random_auto": "Random AutoFE",
}


def _read(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value, digits=6):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _int(value):
    return f"{int(value):,}".replace(",", " ")


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
    ])


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figures(out, lock, results, importance, predictions):
    destination = out / "figures"
    destination.mkdir(exist_ok=True)
    models = results["models"]
    names = [name for name in ORDER if name in models]
    colors = ["#2563eb" if n == "selected_auto" else "#1e293b" if n == "manual" else "#94a3b8" for n in names]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, metric, title in zip(axes, ["logloss", "roc_auc", "user_macro_ndcg_mixed"], ["LogLoss (lower is better)", "ROC-AUC (higher is better)", "User-macro mixed NDCG@10"]):
        values = [models[n]["metrics"].get(metric) for n in names]
        positions = np.arange(len(names))
        ax.scatter([np.nan if v is None else v for v in values], positions, c=colors, s=48, zorder=3)
        ax.set_yticks(positions, [EN_LABELS[n] for n in names])
        ax.set_title(title)
        finite = np.array([v for v in values if v is not None and np.isfinite(v)])
        if len(finite) and np.ptp(finite) > 0:
            span = np.ptp(finite)
            ax.set_xlim(max(0, finite.min() - .20 * span), min(1.05, finite.max() + .35 * span))
        for position, value in enumerate(values):
            if value is not None:
                ax.annotate(f"{value:.4f}", (value, position), xytext=(4, 5), textcoords="offset points", ha="left", fontsize=7)
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    fig.suptitle("Held-out logged-play task: fixed models, mean seed probabilities")
    fig.tight_layout()
    _save(fig, destination / "comparison.png")

    bootstrap = results.get("bootstrap_vs_manual", {})
    compared = [n for n in ("all_auto", "selected_auto", "random_auto", "no_pair_relations") if n in bootstrap]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, title in zip(axes, ["logloss", "user_macro_ndcg_mixed"], ["Delta LogLoss (left is better)", "Delta user-macro mixed NDCG@10 (right is better)"]):
        for i, name in enumerate(compared):
            record = bootstrap[name].get(metric, {})
            point, interval = record.get("delta"), record.get("ci95")
            if point is None or interval is None:
                continue
            color = "#2563eb" if name == "selected_auto" else "#64748b"
            ax.plot(interval, [i, i], color=color, linewidth=2)
            ax.scatter([point], [i], color=color, zorder=3)
        ax.axvline(0, color="#111827", linewidth=1, linestyle="--")
        ax.set_yticks(np.arange(len(compared)), [EN_LABELS[n] for n in compared])
        ax.set_title(title)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Proposed minus Manual: paired user-cluster bootstrap, percentile 95% CI")
    fig.tight_layout()
    _save(fig, destination / "paired_deltas.png")

    candidates = sorted(lock["selection_candidates"], key=lambda c: len(c["columns"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = [len(c["columns"]) for c in candidates]
    ax.plot(x, [c["logloss"] for c in candidates], "o-", label="Selection LogLoss")
    ax.plot(x, [c["objective"] for c in candidates], "s--", label="LogLoss + feature penalty")
    chosen = min(candidates, key=lambda c: (c["objective"], len(c["columns"])))
    ax.scatter([len(chosen["columns"])], [chosen["objective"]], s=150, marker="*", color="#dc2626", label="Frozen choice", zorder=4)
    ax.set(xlabel="Number of input features", ylabel="Loss", title="Feature selection uses the selection slice only")
    ax.grid(alpha=.2)
    ax.legend()
    _save(fig, destination / "validation_search.png")

    seeded = [n for n in names if models[n].get("seeds")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, name in enumerate(seeded):
        seed_values = [s["metrics"]["logloss"] for s in models[name]["seeds"]]
        offsets = np.linspace(-.12, .12, len(seed_values))
        ax.scatter(i + offsets, seed_values, color="#64748b", alpha=.8, label="Individual seeds" if i == 0 else None)
        ax.scatter([i], [models[name]["metrics"]["logloss"]], marker="D", color="#2563eb", label="Mean probability ensemble" if i == 0 else None)
    ax.set_xticks(range(len(seeded)), [EN_LABELS[n] for n in seeded], rotation=25, ha="right")
    ax.set(ylabel="Test LogLoss", title="Optimization-seed sensitivity; no seed selected using test")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    _save(fig, destination / "seed_variability.png")

    group_scores = importance.get("group_importance", {})
    kernels = sorted(group_scores, key=group_scores.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(kernels[::-1], [group_scores[g] for g in kernels[::-1]], color=["#ea580c" if g.startswith("decay_") else "#2563eb" for g in kernels[::-1]])
    ax.set(xlabel="Sum of CatBoost PredictionValuesChange importance", title="Generated-kernel group importance in the development model")
    ax.grid(axis="x", alpha=.2)
    _save(fig, destination / "kernel_importance.png")

    calibration_rows = []
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    y = predictions.label.to_numpy(dtype=np.float64)
    for name in ("manual", "all_auto", "selected_auto"):
        if name not in predictions:
            continue
        scores = predictions[name].to_numpy()
        bins = np.minimum((scores * 10).astype(int), 9)
        means, fractions = [], []
        for b in range(10):
            mask = bins == b
            if not mask.any():
                continue
            means.append(float(scores[mask].mean()))
            fractions.append(float(y[mask].mean()))
            calibration_rows.append({"model": name, "bin_left": b / 10, "bin_right": (b + 1) / 10, "n": int(mask.sum()), "mean_probability": means[-1], "positive_fraction": fractions[-1]})
        ax.plot(means, fractions, "o-", label=EN_LABELS[name])
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed completion fraction", title="Exploratory calibration: fixed probability bins")
    ax.grid(alpha=.2)
    ax.legend()
    _save(fig, destination / "calibration.png")
    pd.DataFrame(calibration_rows).to_csv(out / "calibration.csv", index=False)


def _subgroups(config, predictions, out):
    """Classification-only diagnostics: never rerank truncated group slates."""
    root = Path(config["processed_dir"])
    if not (root / "features.parquet").exists() or not (root / "queries.parquet").exists():
        existing = out / "exploratory_subgroups.csv"
        return pd.read_csv(existing).to_dict("records") if existing.exists() else []
    q = pd.read_parquet(root / "queries.parquet", columns=["uid", "item_id", "day", "split"])
    test = q.split.eq("test").to_numpy()
    numeric = ["user__count__all", "user_item__count__all", "user_artist__count__all", "item__count__all"]
    features = pd.read_parquet(root / "features.parquet", columns=numeric).loc[test].reset_index(drop=True)
    expected = q.loc[test, ["uid", "item_id", "day"]].reset_index(drop=True)
    actual = predictions[["uid", "item_id", "day"]].reset_index(drop=True)
    if len(features) != len(predictions) or not expected.equals(actual):
        raise ValueError("Prediction rows do not align with test feature rows")
    user_count = np.expm1(features["user__count__all"].to_numpy())
    masks = [
        ("user_history", "0", user_count < .5),
        ("user_history", "1–9", (user_count >= .5) & (user_count < 9.5)),
        ("user_history", "10–99", (user_count >= 9.5) & (user_count < 99.5)),
        ("user_history", "100–999", (user_count >= 99.5) & (user_count < 999.5)),
        ("user_history", "1000+", user_count >= 999.5),
    ]
    for dimension, column in [("user_item", "user_item__count__all"), ("user_artist", "user_artist__count__all"), ("item_in_sample", "item__count__all")]:
        cold = features[column].to_numpy() < .1
        masks.extend([(dimension, "нет прошлых прослушиваний", cold), (dimension, "есть прошлые прослушивания", ~cold)])
    rows = []
    y = predictions.label.to_numpy(dtype=np.float64)
    for dimension, name, mask in masks:
        if not mask.any():
            continue
        record = {"dimension": dimension, "subgroup": name, "rows": int(mask.sum()), "users": int(predictions.loc[mask, "uid"].nunique()), "positive_rate": float(y[mask].mean())}
        for model in ("manual", "selected_auto"):
            p = np.clip(predictions[model].to_numpy()[mask], 1e-7, 1 - 1e-7)
            record[model + "_logloss"] = float(np.mean(-y[mask] * np.log(p) - (1 - y[mask]) * np.log1p(-p)))
            record[model + "_roc_auc"] = float(roc_auc_score(y[mask], p)) if len(np.unique(y[mask])) == 2 else None
        record["delta_logloss"] = record["selected_auto_logloss"] - record["manual_logloss"]
        rows.append(record)
    pd.DataFrame(rows).to_csv(out / "exploratory_subgroups.csv", index=False)
    return rows


def _scaling(out):
    """Optional measured resource curve; no fitted extrapolation is drawn."""
    path = out / "scaling.json"
    if not path.exists():
        return ""
    scaling = _read(path)
    rows = sorted(scaling["rows"], key=lambda r: r["history_events"])
    if not rows:
        raise ValueError("Scaling artifact contains no measured rows")
    required = ["users", "history_events", "query_rows", "feature_seconds", "load_seconds", "peak_rss_bytes", "matrix_bytes"]
    for row in rows:
        if any(key not in row or not np.isfinite(row[key]) or row[key] < 0 for key in required):
            raise ValueError("Scaling artifact has missing or invalid measurements")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.array([r["history_events"] / 1e6 for r in rows])
    axes[0].plot(x, [r["feature_seconds"] for r in rows], "o-", color="#2563eb", label="Feature engine")
    axes[0].plot(x, [r["load_seconds"] for r in rows], "s--", color="#94a3b8", label="File loading")
    axes[0].set(xlabel="Retained history events (millions)", ylabel="Seconds", title="Measured runtime; no extrapolation")
    axes[0].legend()
    axes[1].plot(x, [r["peak_rss_bytes"] / 2**20 for r in rows], "o-", color="#2563eb", label="Process peak RSS")
    axes[1].plot(x, [r["matrix_bytes"] / 2**20 for r in rows], "s--", color="#94a3b8", label="Output DataFrame")
    axes[1].set(xlabel="Retained history events (millions)", ylabel="MiB", title="Measured memory in fresh processes")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    for position, (coordinate, row) in enumerate(zip(x, rows)):
        alignment, offset = ("left", (6, 8)) if position == 0 else ("center", (0, 8))
        axes[0].annotate(f"{row['users']} users", (coordinate, row["feature_seconds"]), xytext=offset, textcoords="offset points", ha=alignment, fontsize=8)
        axes[1].annotate(f"{row['users']} users", (coordinate, row["peak_rss_bytes"] / 2**20), xytext=offset, textcoords="offset points", ha=alignment, fontsize=8)
    fig.tight_layout()
    _save(fig, out / "figures" / "scaling.png")
    table = _table(["Пользователей", "Событий истории", "Query rows", "Feature engine, с", "Загрузка, с", "Peak RSS, MiB", "Матрица, MiB"], [[_int(r["users"]), _int(r["history_events"]), _int(r["query_rows"]), _fmt(r["feature_seconds"], 2), _fmt(r["load_seconds"], 2), _fmt(r["peak_rss_bytes"] / 2**20, 1), _fmt(r["matrix_bytes"] / 2**20, 1)] for r in rows])
    return f"""
### Измеренная размерная кривая

После обучения основных моделей feature engine отдельно запущен на вложенных пользовательских подмножествах: {', '.join(_int(r['users']) for r in rows)} пользователей. Подмножества получены фиксированной случайной перестановкой основной когорты; по выбранным пользователям сохраняются полные доступные истории. Размеры выполняются последовательно, каждый в новом процессе, чтобы не смешивать high-water marks памяти и не конкурировать с обучением. Число запросов зависит от активности выбранных пользователей; их target не используется для решений или подбора.

{table}

![Измеренные время и память по размеру истории](figures/scaling.png)

Feature time не включает чтение файлов, оно показано отдельно. Peak RSS включает интерпретатор, библиотеки, загруженную историю и промежуточные агрегаты; объём выходного DataFrame — другая величина. Эти замеры показывают поведение данной реализации только в измеренном диапазоне. **Экстраполяция на миллиарды событий не проводится**; линейная линия между точками лишь соединяет измерения. Здесь не измеряются качество моделей на разных размерах выборки и distributed throughput. Платформа: `{scaling.get('platform','не записана')}`. Исходные численные результаты: [`scaling.json`](scaling.json).
"""


def _conclusion(models, bootstrap, selected_groups):
    manual = models["manual"]["metrics"]["logloss"]
    selected = models["selected_auto"]["metrics"]["logloss"]
    delta = selected - manual
    relative = -100 * delta / manual
    ci = bootstrap["selected_auto"]["logloss"].get("ci95")
    if not selected_groups:
        claim = "Отбор на validation не сохранил дополнительные группы: автоматический поиск выбрал исходный ручной набор. Это содержательный отрицательный результат, а не основание менять критерий после просмотра теста."
    elif delta < 0:
        if ci is not None and ci[0] <= 0 <= ci[1]:
            claim = f"Надёжное улучшение над manual этим экспериментом не продемонстрировано: точечный test LogLoss выбранного AutoFE ниже на {_fmt(-delta)} ({relative:.3f}% относительно manual)."
        else:
            claim = f"Выбранный AutoFE уменьшил test LogLoss на {_fmt(-delta)} ({relative:.3f}% относительно manual)."
    else:
        claim = f"Выбранный AutoFE не улучшил test LogLoss: изменение составило {_fmt(delta, 7)} ({-relative:.3f}% относительно manual)."
    if ci is not None:
        claim += f" Парный пользовательский bootstrap даёт 95% интервал ΔLogLoss [{_fmt(ci[0], 7)}; {_fmt(ci[1], 7)}]."
        if ci[1] < 0:
            claim += " В этом интервале все значения благоприятны для AutoFE."
        elif ci[0] > 0:
            claim += " В этом интервале все значения благоприятны для manual."
        else:
            claim += " Интервал включает ноль, поэтому направление эффекта при повторной выборке пользователей этой популяции остаётся неопределённым."
    ranking = bootstrap["selected_auto"].get("user_macro_ndcg_mixed", {})
    if ranking.get("delta") is not None:
        claim += f" Изменение вторичной user-macro NDCG@10 на смешанных днях: {_fmt(ranking['delta'],7)}."
        ranking_ci = ranking.get("ci95")
        if ranking_ci is not None:
            claim += f" Её 95% интервал [{_fmt(ranking_ci[0],7)}; {_fmt(ranking_ci[1],7)}]"
            claim += " также включает ноль." if ranking_ci[0] <= 0 <= ranking_ci[1] else "."
    return claim


def write_report(config):
    """Read locked outputs, draw static figures, and write REPORT.md.

    This function never fits models, selects hyperparameters, or changes an
    experiment lock. Missing required numeric artifacts are an error: there is
    deliberately no path that fills a report with simulated results.
    """
    out, processed = Path(config["output_dir"]), Path(config["processed_dir"])
    lock = _read(out / "selection_lock.json")
    results = _read(out / "test_results.json")
    trials = _read(out / "search_trials.json")
    importance = _read(out / "search_importance.json")
    # Lightweight metadata may be bundled beside the final report so a clone
    # can redraw it without checking in the large full event/feature tables.
    def metadata_path(name):
        local = processed / name
        return local if local.exists() else out / name

    manifest = _read(metadata_path("manifest.json"))
    timing = _read(metadata_path("feature_timing.json"))
    groups = _read(metadata_path("feature_groups.json"))
    predictions = pd.read_parquet(out / "test_predictions.parquet")
    models, bootstrap = results["models"], results["bootstrap_vs_manual"]
    required = {"manual", "all_auto", "selected_auto", "random_auto"}
    if not required.issubset(models) or not required.issubset(predictions.columns):
        raise ValueError("Required predeclared comparison models are missing")
    if lock.get("n_test_rows_used") != 0:
        raise ValueError("Search lock must certify that no test rows were used")
    if "config" in lock and lock["config"] != config:
        raise ValueError("Report config differs from the frozen experiment config")
    if results["selection_lock_sha256"] != _sha256(out / "selection_lock.json"):
        raise ValueError("Results do not match the current selection lock")
    if results["test_prediction_sha256"] != _sha256(out / "test_predictions.parquet"):
        raise ValueError("Test prediction checksum differs from the frozen result")
    preparation_path = metadata_path("preparation_lock.json")
    if preparation_path.exists() and "preparation_lock_sha256" in lock:
        if _sha256(preparation_path) != lock["preparation_lock_sha256"]:
            raise ValueError("Preparation lock differs from the frozen selection")
        preparation = _read(preparation_path)
        for name, key in [("manifest.json", "manifest_sha256"), ("feature_groups.json", "feature_groups_sha256")]:
            if key in preparation and _sha256(metadata_path(name)) != preparation[key]:
                raise ValueError(f"Report metadata checksum mismatch: {name}")
    if len(predictions) != manifest["splits"]["test"]["rows"]:
        raise ValueError("Test prediction count differs from the data manifest")
    _figures(out, lock, results, importance, predictions)
    subgroup_rows = _subgroups(config, predictions, out)
    scaling_text = _scaling(out)

    names = [n for n in ORDER if n in models]
    split_rows = [[name, f"{s['day_min']}–{s['day_max']}", _int(s["users"]), _int(s["groups"]), _int(s["rows"]), _fmt(s["positive_rate"], 4), _fmt(s["group_size_median"], 1), _fmt(s["group_size_p90"], 1)] for name, s in manifest["splits"].items()]
    quality_rows = [[LABELS[n], models[n].get("n_features", 0), _fmt(models[n]["metrics"].get("logloss")), _fmt(models[n]["metrics"].get("roc_auc")), _fmt(models[n]["metrics"].get("brier")), _fmt(models[n]["metrics"].get("average_precision")), _fmt(models[n]["metrics"].get("ndcg_mixed")), _fmt(models[n]["metrics"].get("user_macro_ndcg_mixed")), _fmt(models[n]["metrics"].get("recall_mixed"))] for n in names]
    seed_rows = []
    for name in names:
        values = models[name].get("seeds", [])
        if not values:
            continue
        losses = np.array([s["metrics"]["logloss"] for s in values])
        aucs = np.array([s["metrics"]["roc_auc"] for s in values], dtype=float)
        seed_rows.append([LABELS[name], ", ".join(f"{s['seed']}: {s['metrics']['logloss']:.6f}" for s in values), f"{losses.mean():.6f} ± {losses.std(ddof=1) if len(losses)>1 else 0:.6f}", f"{aucs.mean():.6f} ± {aucs.std(ddof=1) if len(aucs)>1 else 0:.6f}"])
    ci_rows = []
    for name, stats in bootstrap.items():
        for metric, label in [("logloss", "LogLoss"), ("user_macro_ndcg_mixed", "user-macro NDCG mixed")]:
            record = stats[metric]
            interval = record.get("ci95")
            ci_rows.append([LABELS.get(name, name), label, _fmt(record.get("delta"), 7), "—" if interval is None else f"[{_fmt(interval[0],7)}; {_fmt(interval[1],7)}]", _fmt(record.get("bootstrap_fraction_improved"), 3)])
    time_rows = [[LABELS[n], _fmt(models[n].get("fit_seconds"), 1), _fmt(models[n].get("prediction_seconds"), 3), _fmt(models[n].get("model_bytes", 0) / 2**20, 2), models[n].get("iterations", "—")] for n in names]
    selected_groups = lock["selected_groups"]
    random_groups = lock["random_groups"]
    candidates = lock["selection_candidates"]
    select_rows = [[c["prefix"], len(c["columns"]), _fmt(c["logloss"]), _fmt(c["objective"]), _fmt(c["fit_seconds"], 1), ", ".join(c["groups"]) or "manual"] for c in candidates]
    kernel_rows = [[g, len([c for c in groups.get(g, []) if c in lock["models"]["all_auto"]["columns"]]), _fmt(importance["group_importance"][g], 3), "да" if g in selected_groups else "нет", "да" if g in random_groups else "нет"] for g in lock["ranked_groups"]]
    important_features = sorted(importance["feature_importance"].items(), key=lambda x: (-x[1], x[0]))[:15]
    important_rows = [[name, _fmt(value, 3)] for name, value in important_features]
    coverage = models["manual"]["metrics"]
    mixed_fraction = coverage["n_mixed_groups"] / coverage["n_groups"] if coverage["n_groups"] else 0
    all_loss, manual_loss = models["all_auto"]["metrics"]["logloss"], models["manual"]["metrics"]["logloss"]
    random_loss, selected_loss = models["random_auto"]["metrics"]["logloss"], models["selected_auto"]["metrics"]["logloss"]
    relation_loss = models.get("no_pair_relations", {}).get("metrics", {}).get("logloss")
    relation_analysis = "" if relation_loss is None else f"Удаление агрегатов user–item, user–artist и user–album из ручной модели изменяет LogLoss на {_fmt(relation_loss-manual_loss,7)}; положительное значение означает пользу этих агрегатов в данном обучении. Категориальные ID остаются, и CatBoost может обучать взаимодействия между ними. Эта абляция наследует параметры manual и не оценивает оптимально перенастроенную модель без парных агрегатов."
    history = manifest["history_lengths"]
    first_source = next(iter(manifest["sources"].values()))
    n_generated = sum(len(v) for k, v in groups.items() if k not in ("manual", "static"))
    train_start = config["warmup_days"]
    a, b, c = config["split_days"]
    max_day = manifest["timestamp_max"] // 86400
    total_final_fit = sum(v.get("fit_seconds", 0) for v in models.values())
    n_tune = sum(r.get("phase") == "tune" for r in trials)
    selected_params = lock["models"]["selected_auto"]
    subgroup_text = _table(["Срез", "Группа", "Строк", "Пользователей", "P(y=1)", "Manual LL", "Selected LL", "ΔLL"], [[r["dimension"], r["subgroup"], _int(r["rows"]), _int(r["users"]), _fmt(r["positive_rate"], 4), _fmt(r["manual_logloss"]), _fmt(r["selected_auto_logloss"]), _fmt(r["delta_logloss"], 7)] for r in subgroup_rows]) if subgroup_rows else "Диагностика не рассчитана: матрица признаков недоступна."
    conclusion = _conclusion(models, bootstrap, selected_groups)
    n_seeds = len(models["manual"].get("seeds", []))
    purged = manifest.get("purged_label_rows_by_split", {})
    purge_text = ", ".join(f"{name}: {_int(value)}" for name, value in purged.items()) or "численности в манифесте не записаны"
    environment = lock.get("environment", lock)
    feature_peak = _fmt(timing.get("peak_rss_bytes", np.nan) / 2**20, 1)
    search_peak = _fmt(lock.get("peak_rss_bytes", np.nan) / 2**20, 1)
    package_versions = environment.get("packages", {})
    chosen_candidate = min(candidates, key=lambda candidate: (candidate["objective"], len(candidate["columns"])))
    manual_candidate = next(candidate for candidate in candidates if candidate["prefix"] == 0)
    selection_delta = chosen_candidate["logloss"] - manual_candidate["logloss"]
    test_delta = selected_loss - manual_loss
    selection_transfer = f"На select изменение LogLoss выбранного набора относительно manual было {_fmt(selection_delta,7)}, на test — {_fmt(test_delta,7)}."
    if selection_delta < 0 and test_delta < 0 and abs(test_delta) < abs(selection_delta):
        selection_transfer += " Точечный выигрыш сократился при переходе к более позднему периоду."
    selection_transfer += " Возможные объяснения — шум выбора лучшего из нескольких префиксов (winner's curse), временное изменение поведения и состава активных пользователей, а также отбор по одному seed при итоговом ансамбле. Это гипотезы, а не установленные причины; разделённые tune/select уменьшают, но не устраняют риск переобучения процедуры на validation."
    select_split, test_split = manifest["splits"].get("select"), manifest["splits"]["test"]
    if select_split is not None:
        selection_transfer += f" В select было {_int(select_split['users'])} пользователей и доля дослушиваний {_fmt(select_split['positive_rate'],4)}, в test — {_int(test_split['users'])} и {_fmt(test_split['positive_rate'],4)}; это свидетельство различия наблюдённых периодов, а не доказательство причины уменьшения эффекта."
    user_importance = sum(value for name, value in importance["feature_importance"].items() if name == "cat_uid" or name.startswith("user__"))
    objective_analysis = f"LogLoss и ranking отвечают на разные вопросы. LogLoss взвешивает строки, включая различия склонности пользователей к дослушиванию; user-macro NDCG рассматривает порядок треков внутри пользователя/дня и равномерно взвешивает покрытых пользователей. В полной development-модели сумма importance `cat_uid` и непарных `user__*` составляет {_fmt(user_importance,2)} из 100. Эти признаки сами по себе постоянны внутри user-day, хотя модель может использовать их взаимодействия с объектами. Это согласуется с гипотезой, что дополнительные временные ядра точнее описывают общую склонность к дослушиванию, не обязательно улучшая порядок треков. Importance не доказывает механизм, а калибровка, discrimination и внутригрупповой порядок не тождественны."
    all_ranking = bootstrap.get("all_auto", {}).get("user_macro_ndcg_mixed", {})
    if all_ranking.get("delta") is not None and all_ranking.get("ci95") is not None:
        interval = all_ranking["ci95"]
        objective_analysis += f" У all_auto Δ user-macro NDCG = {_fmt(all_ranking['delta'],7)}, 95% интервал [{_fmt(interval[0],7)}; {_fmt(interval[1],7)}]."
        if interval[1] < 0:
            objective_analysis += " Весь этот вторичный интервал лежит ниже нуля: более низкий точечный LogLoss здесь сопровождается ухудшением условного ranking. Интервал не скорректирован за множественные вторичные сравнения."
    best_observed = min(models, key=lambda name: models[name]["metrics"]["logloss"])
    posthoc_analysis = f"Минимальный наблюдённый test LogLoss среди предписанных моделей у «{LABELS.get(best_observed,best_observed)}» ({_fmt(models[best_observed]['metrics']['logloss'])}). Это описательное сравнение уже зафиксированных контролей. Итогом процедуры отбора остаётся selected_auto, выбранный на select; менять решение на вариант с лучшим test означало бы подбирать модель по тесту."
    attribution_analysis = ""
    manual_spec = lock["models"]["manual"]
    if manual_spec.get("params") != selected_params.get("params") or manual_spec.get("iterations") != selected_params.get("iterations"):
        attribution_analysis = f"Сравнение AutoFE с manual относится ко всей процедуре, а не к чистому эффекту одного окна: manual использует `{json.dumps(manual_spec['params'], ensure_ascii=False)}` и {manual_spec['iterations']} деревьев, selected_auto — `{json.dumps(selected_params['params'], ensure_ascii=False)}` и {selected_params['iterations']} деревьев. Обе исходные семьи получили одинаковую tune-сетку, но выбранные параметры различаются. Без отдельной абляции manual с параметрами selected_auto нельзя приписать весь наблюдённый сдвиг только дополнительным признакам. Такая абляция после просмотра test не добавлялась."
    collision_summary, collision_detail = "", ""
    if set(selected_groups) == set(random_groups):
        collision_summary = "**Ограничение случайного контроля:** selected_auto и random_auto получили один и тот же набор групп; этот запуск не позволяет показать преимущество отбора над случайным выбором. Подробности — в разделе 4."
        if selected_groups:
            collision_detail = f"**Совпадение случайного контроля.** Фиксированный случайный порядок с seed `8821` случайно дал те же группы, что и отбор: `{', '.join(selected_groups)}`. Контроль и seed не изменялись после обнаружения совпадения."
        else:
            collision_detail = "**Совпадение случайного контроля.** Отбор сохранил ноль дополнительных групп; random-control также использует manual и наследует его параметры и число деревьев."
        selected_spec = lock["models"].get("selected_auto", {})
        random_spec = lock["models"].get("random_auto")
        if random_spec is not None and all(selected_spec.get(key) == random_spec.get(key) for key in ("columns", "params", "iterations")):
            maximum_difference = float(np.max(np.abs(predictions.selected_auto.to_numpy() - predictions.random_auto.to_numpy())))
            collision_detail += f" Совпадают упорядоченные списки столбцов, параметры, число деревьев и фиксированные seeds обучения, поэтому ожидаются одинаковые вероятности; максимальное абсолютное отличие сохранённых прогнозов: `{maximum_difference:.12g}`."
        collision_detail += " Этот запуск не различает качество содержательного отбора и одного случайного выбора. Совпадение также не доказывает, что поиск бесполезен: независимый random-control для такого вывода здесь отсутствует; нужны заранее запланированные дополнительные случайные порядки."
    report = f"""# Автоматический поиск временных признаков для рекомендательных систем

## Результат и исследовательский вопрос

Проверяем, может ли ограниченная автоматически порождённая грамматика временных агрегатов улучшить сильный CatBoost на содержательных ручных признаках. Объект исследования — **вероятность дослушивания уже начатого рекомендованного трека**, а не извлечение объектов из полного каталога и не CTR. Отрицательные примеры — реальные короткие прослушивания; искусственное negative sampling не используется.

{conclusion}

{collision_summary}

Это воспроизводимый CPU-эксперимент на {_int(manifest['sampled_users'])} случайно выбранных пользователях Yambda-50M: сохранено **{_int(manifest['retained_events'])} событий** их доступной истории. Выбор модели завершён до единственного финального сравнения на test. Все числа ниже прочитаны из сохранённых артефактов; код отчёта не обучает и не выбирает модели.

## 1. Почему эти данные и эта постановка

[Yambda](https://arxiv.org/html/2505.22238v1) содержит последовательности прослушиваний, лайков, дизлайков и отмен этих действий; доступны связи track–artist и track–album. В отличие от него, [KION](https://arxiv.org/html/2209.00325) оставляет последний просмотр каждой пары user–item и суммирует длительность по повторным просмотрам. Для исследования истории повторного потребления Yambda даёт более прямую постановку. Выбираем подмножество пользователей Yambda-50M; выводы не переносятся автоматически на всю аудиторию сервиса.

Для каждого user–item в условном дне сначала выбирается **первое** прослушивание среди всех источников трафика: порядок timestamp, затем исходный порядок записи при совпадении timestamp. После этого оставляем только `is_organic == 0`. Поэтому более ранний органический запуск не позволяет выбрать вместо него поздний успешный рекомендованный запуск того же трека. Цель `label = 1[played_ratio_pct >= 50]`. Порог «не менее 50%» и допустимость значений выше 100% подтверждены [авторами датасета](https://huggingface.co/datasets/yandex/yambda/discussions/10). Значения выше 100 означают повторное прослушивание участков, а не автоматически ошибочную запись.

Текущие `played_ratio_pct`, `track_length_seconds`, timestamp внутри дня и текущий outcome **не входят в матрицу модели**. Условие рекомендованного трафика задаёт исследуемую популяцию. История признаков содержит и органические, и рекомендованные события. ID пользователя, трека, выбранного исполнителя и альбома считаем доступными при начале проигрывания.

Для классификации каждая выбранная запись — отдельный пример с естественной долей дослушиваний. Для дополнительного ранжирования кандидатами служат фактически проигранные рекомендованные треки из одного выбранного user-day. Этот набор известен ретроспективно и не является доступным в начале дня рекомендательным slate. Его NDCG характеризует только **условное переупорядочивание наблюдавшихся запусков**. Нет оснований называть его качеством retrieval, будущим online uplift или качеством на всём каталоге. Проблемы sampled-метрик обсуждаются в [Krichene & Rendle, 2020](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/); в нашей постановке выборка отрицательных объектов вообще не конструируется, но остаётся смещение исходной logging policy.

## 2. Выборка и время

Источник: `{manifest['dataset']}`, фиксированная ревизия `{first_source['revision']}`. Из {_int(manifest['source_users'])} пользователей равномерно без возвращения выбираются {_int(manifest['sampled_users'])}, seed `{config['user_seed']}`; активность и целевые значения не участвуют в выборе. По каждому выбранному пользователю сохраняются **все имеющиеся в исходном файле события**. Число объектов в сохранённой истории — {_int(manifest['retained_items'])}. Длины историй: минимум {_int(history['min'])}, медиана {_fmt(history['median'],1)}, p90 {_fmt(history['p90'],1)}, максимум {_int(history['max'])} событий.

Распределение типов событий: {', '.join(f"`{k}`: {_int(v)}" for k,v in manifest['event_type_counts'].items())}. История не урезается по числу последних событий. Для ограничения CPU-бюджета равномерно выбираются **дни supervised-примеров** отдельно для каждого пользователя и каждого временного split, seed `{config['day_seed']}`: {', '.join(f'{k}: до {v}' for k,v in config['days_per_user'].items())}. Выбираются дни, в которых вообще есть подходящие запуски, без обращения к outcome; все подходящие первые проигрывания выбранного дня сохраняются. Это меняет распределение supervised-дней, поэтому оценка относится именно к описанному дизайну выборки.

Timestamp — **относительные секунды, уже округлённые до кратного пяти**, а не число пятисекундных интервалов. Дни вычисляются `timestamp // 86400`. Начало времени анонимизировано: это не календарная дата, не локальный часовой пояс и не реальный день недели. История разогрева — дни `0…{train_start-1}`; train — `{train_start}…{a-1}`; tune — `{a}…{b-1}`; select — `{b}…{c-1}`; test — `{c}…{max_day}`. Разбиение глобально по времени, а не по последнему событию каждого пользователя. Это учитывает риск пересечения временных границ, описанный в [работе об утечках в офлайн-рекомендациях](https://arxiv.org/abs/2010.11060) и [TimeSplit](https://arxiv.org/abs/2507.16289).

{_table(['Split','Дни с примерами','Пользователей','User-day','Строк','Доля y=1','Медиана кандидатов','p90 кандидатов'],split_rows)}

Недостаточно ограничить timestamp признаков: target на границе периода тоже должен успеть стать известным. Поэтому supervised-исходы train/tune/select допускаются только при `available_at < начало следующего split`. Purge выполняется **до выбора supervised-дней**; число удалённых записей из исходного подходящего пула: {purge_text}. Эти счётчики не означают, что столько строк удалили из уже выбранных дней. Исторические события при этом не удаляются, их использование по-прежнему зависит от доступности. Для финального test следующего этапа выбора модели нет.

### Доступность outcome и причинность признаков

Для прослушивания используется консервативная оценка `available_at = timestamp + ceil(track_length_seconds * played_ratio_pct / 100) + 5`; для остальных событий — `timestamp + 5`. При расчёте задержки отношение **не ограничивается 100%**, поскольку повторы требуют времени. `available_day = available_at // 86400`. Запрос дня `d` видит только записи с `available_day < d - {config.get('embargo_days',1)}`: действует дополнительный embargo. Агрегаты не используют цель текущего запроса, события того же дня, равные граничные дни и будущие доступные outcome. В окнах и затухании отсчитывается время доступности, а не неизвестное время публикации результата.

Это явное инженерное предположение о доступности, **не доказательство точной задержки production-логов**. Публичная схема не устанавливает однозначно, обозначает ли timestamp начало или конец прослушивания, а наблюдаемая доля не полностью описывает wall-clock паузы. Рассчитанная длительность и дополнительный дневной разрыв уменьшают этот риск; отдельный эксперимент с другим embargo здесь не проведён.

На test веса моделей зафиксированы, но историческое состояние обновляется: события более ранних test-дней могут попасть в признаки последующих test-дней после выполнения availability и embargo. Это rolling online-history evaluation без переобучения, а не эксперимент с историей, замороженной в конце train. Состояние считается по всем сохранённым событиям, включая дни, которые не выбраны как supervised-примеры.

Метаданные artist/album считаются статическими и известными при запросе; время фактической публикации этих связей не дано. При нескольких значениях выбирается минимальный числовой ID, независимо от популярности и outcome. Это воспроизводимое упрощение many-to-many структуры, которое теряет часть коллабораций и альбомных связей. Отсутствующее значение кодируется `-1` и не создаёт общую ложную аффинность между разными треками. Подробные пропуски и множественность записаны в `manifest.json`.

## 3. Сильный ручной baseline

Модель — [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), binary LogLoss, категориальные ID, `has_time=True`, максимум CTR-комбинаций 1. Пользователь–объект связи не подменяются только популярностью: ручной набор содержит отношения `user`, `item`, `artist`, `album`, `user_item`, `user_artist`, `user_album`.

Для каждого отношения считаются `log1p(count)` и сглаженная доля дослушиваний за 7 дней, 30 дней и всю доступную историю; давность последнего доступного события; средняя историческая доля прослушанного; доля органических прослушиваний; количества лайков и дизлайков. Для парных связей добавляются доли в активности пользователя, для всех отношений — отношение короткой к длинной активности. Общий prior дослушиваний тоже вычисляется только из доступного прошлого; сглаживание `(positive + 5*prior)/(count + 5)` использует фиксированный prior weight 5. В среднем отношении историческая доля прослушивания ограничена 100%; эта нормировка не меняет исходную цель.

Лайки/дизлайки — накопленные количества событий, а не актуальное состояние предпочтения: `unlike`/`undislike` не вычитаются. Они сохранены в исходной истории, но не получают отдельной outcome-статистики. Давность отношения может обновляться по любому из сохранённых типов событий. Это осознанное ограничение грамматики. Нет SVD/нейронных эмбеддингов и аудио: сравнение изолирует генерацию временных агрегатов.

До удаления постоянных на train столбцов в manual {_int(len(groups['manual']))} признаков; в фактически обученном baseline {_int(models['manual']['n_features'])}. Дополнительные контроли: константа с долей положительного класса на train; CatBoost только с ID и историческим global prior; manual без агрегатов трёх парных отношений (с сохранением категориальных ID). Эти модели оцениваются на тех же строках и кандидатах.

## 4. AutoFE: ограниченная грамматика и проверяемый отбор

Предлагаемый вклад — адаптация известной идеи порождения и отбора признаков к строго временному multi-event логу, а не заявление об изобретении AutoFE или causal discovery. [Deep Feature Synthesis](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf) систематизирует relational-примитивы; [ExploreKit](https://people.eecs.berkeley.edu/~dawnsong/papers/icdm-2016.pdf) объединяет генерацию и отбор; [OpenFE](https://proceedings.mlr.press/v202/zhang23ay.html) показывает пользу эффективной генерации и фильтрации табличных признаков. Здесь пространство намеренно мало, задано кодом и полностью просматриваемо.

Грамматика: **7 отношений × 2 статистики × 9 временных ядер = {n_generated} дополнительных столбцов до фильтра постоянных признаков**. Статистики — `log1p(count)` и сглаженная доля дослушиваний. Box-окна: {', '.join(map(str, config['auto_windows']))} дней. Экспоненциальные ядра: периоды полураспада {', '.join(map(str, config['auto_half_lives']))} дней, вес события `2^(-(cutoff-available_day)/half_life)`. Группа — одно ядро сразу по семи отношениям и двум статистикам; это 14 потенциальных признаков и ограничение размера поиска, а не произвольный перебор SQL.

Алгоритм:

1. Удалить признаки, постоянные **на train**. Для manual и manual+all_auto независимо перебрать одинаковую сетку из {len(config['hyper_grid'])} конфигураций на tune. Максимум {config['iterations']} деревьев, early stopping {config['early_stopping']}; seed для поиска `{config['model_seeds'][0]}`. Гиперпараметры: `{json.dumps(config['hyper_grid'], ensure_ascii=False)}`. В журнале сохранено {n_tune} tune-запусков.
2. В выбранной all_auto модели суммировать CatBoost feature importance по группам. Модель обучена на train; её параметры и число деревьев определены tune. Select/test не участвуют в построении этого порядка.
3. На select сравнить manual и заранее заданные префиксы порядка групп. Критерий `LogLoss + {config['selection_feature_penalty']} * число дополнительных признаков`; при равенстве предпочесть меньшее число столбцов. Параметры и число деревьев для непустых префиксов наследуются от all_auto; префиксы не получают отдельную настройку.
4. Сохранить `selection_lock.json`: список столбцов, группы, параметры, seed, конфигурацию и хеши исходников. В зафиксированном журнале `n_test_rows_used = {lock['n_test_rows_used']}`. Только после этого запускать предписанные финальные модели на test. Train не объединяется с tune/select при финальном обучении.

{_table(['Число групп','Столбцов','Select LogLoss','Select objective','Время fit, с','Группы'],select_rows)}

Выбрано: **{', '.join(selected_groups) or 'без дополнительных групп'}**. Итоговые параметры: `{json.dumps(selected_params['params'], ensure_ascii=False)}`, деревьев `{selected_params['iterations']}`. Random-control использует один фиксированный случайный порядок групп, seed `8821`, и такое же число групп: `{', '.join(random_groups) or 'нет'}`. Это **один случайный контроль**, не распределение по многим случайным поискам; число фактически непостоянных столбцов нужно сравнивать по таблице результатов. При непустом наборе его параметры наследованы от all_auto; при нуле групп он наследует manual, включая число деревьев.

{collision_detail}

![Подбор количества признаков](figures/validation_search.png)

Отбор по importance не гарантирует оптимум: коррелированные окна делят важность, префиксный поиск не рассматривает все комбинации, а feature penalty выбран заранее как инженерное ограничение. Больший диапазон пространства и иной бюджет могли бы изменить результат. Все отклонённые префиксы сохранены; нет скрытого поиска по test.

## 5. Итоговое качество на test

Главная метрика — **LogLoss** на наблюдённых запусках. ROC-AUC, Brier и Average Precision дополняют её; они не заменяют основной критерий после просмотра результатов. Для каждой CatBoost-модели обучены {n_seeds} фиксированных seeds `{config['model_seeds']}`; таблица показывает метрики среднего их вероятностей. Константа не ансамблируется. Ансамбль использует одинаковое число запусков для сравниваемых моделей; отдельно ниже показана чувствительность одиночных запусков.

{_table(['Модель','Признаки','LogLoss ↓','ROC-AUC ↑','Brier ↓','AP ↑','NDCG@10 mixed ↑','User-macro NDCG mixed ↑','Recall@10 mixed ↑'],quality_rows)}

![Сравнение качества](figures/comparison.png)

### Покрытие и смысл ранжирования

Test: {_int(coverage['n_rows'])} строк, {_int(coverage['n_users'])} пользователей, {_int(coverage['n_groups'])} user-day. Групп хотя бы с двумя кандидатами: {_int(coverage['n_groups_ge2'])}; смешанных по target: {_int(coverage['n_mixed_groups'])} ({mixed_fraction:.1%} всех групп), пользователей со смешанными группами: {_int(coverage['n_mixed_users'])}. Полностью положительных групп: {_int(coverage['n_all_positive_groups'])}; без положительных: {_int(coverage['n_no_positive_groups'])}; singleton: {_int(coverage['n_singleton_groups'])}. Последние три счётчика относятся к разным явно заданным случаям: all-positive/no-positive считаются только при размере ≥2, singleton отдельно.

`NDCG mixed` и `Recall mixed` сначала усредняются по смешанным user-day. `User-macro NDCG mixed` сначала усредняет дни внутри каждого подходящего пользователя, затем пользователей. Полностью положительные дни имеют тривиальный NDCG=1, поэтому основной ranking-диагностикой служат смешанные дни. Без положительных recall и NDCG не определены и исключаются, а не превращаются в ноль. Дополнительные `ndcg_all_defined`/`recall_all_defined` со всеми положительно определёнными группами сохранены в JSON. Это **условные метрики на покрытой части лога**; отбор mixed-групп не используется для основной классификации и не означает качество на всех пользователях.

NDCG реализован отдельно и проверен тестами на известных примерах и сравнением со scikit-learn. При одинаковых score используется математическое ожидание gain по всем перестановкам tie-блока, включая границу K; порядок строк и target не разрешают ничьи в пользу модели. Для Recall применяется то же правило. Это особенно существенно для константной модели.

### Неопределённость: парный bootstrap по пользователям

{_table(['Модель против manual','Метрика','Δ = модель − manual','95% percentile CI','Доля bootstrap с улучшением'],ci_rows)}

В каждом из {config['bootstrap_samples']} bootstrap-повторов с возвращением выбираются **пользователи целиком**, сохраняя все их test-строки и дни. Для LogLoss сохраняется вес строк, для user-macro NDCG — равный вес подходящих пользователей. При LogLoss улучшение — Δ<0, при NDCG — Δ>0. Доля bootstrap-повторов с улучшением описывает эмпирическое распределение; это **не p-value и не апостериорная вероятность**. Интервалы условны на уже обученных и выбранных моделях: повторного обучения и повторения feature selection внутри bootstrap нет. Они не учитывают весь риск выбора модели и смещения logging policy. Здесь не проводится корректировка за множественные вторичные сравнения.

![Парные интервалы изменений](figures/paired_deltas.png)

### Чувствительность к seed

{_table(['Модель','LogLoss каждого seed','Среднее ± SD LogLoss одиночных моделей','Среднее ± SD ROC-AUC одиночных моделей'],seed_rows)}

SD — выборочное стандартное отклонение по фиксированным seeds оптимизации; это не доверительный интервал по пользователям. Метрика ансамбля не обязана равняться среднему метрик отдельных моделей. Ни один seed не выбран по test.

![Разброс seeds](figures/seed_variability.png)

## 6. Что объясняет результат

Manual+all_auto меняет LogLoss относительно manual на **{_fmt(all_loss-manual_loss,7)}**. Отбор по select меняет его на **{_fmt(selected_loss-manual_loss,7)}**, а выбранный набор относительно случайного контроля — на **{_fmt(selected_loss-random_loss,7)}**. Эти различия надо читать вместе с пользовательскими интервалами и seed-разбросом; одно лучшее число не доказывает универсальное преимущество отбора. {relation_analysis}

{selection_transfer}

{objective_analysis}

{posthoc_analysis}

{attribution_analysis}

Групповые importance ниже получены на этапе development и использовались для порядка поиска. Это model-based важность, а не причинный вклад отдельного временного окна и не результат независимого удаления каждого окна. Сильно коррелированные окна могут заменять друг друга.

{_table(['Группа','Непостоянных столбцов','Сумма importance','Выбрана','В random-control'],kernel_rows)}

![Важность временных ядер](figures/kernel_importance.png)

Пятнадцать ведущих признаков полной development-модели:

{_table(['Признак','CatBoost importance'],important_rows)}

### Исследовательские срезы test

Следующие срезы — **exploratory**, а не дополнительные критерии выбора. Пороговые значения заданы детерминированно по длине доступной истории; статистика вычислена только после фиксации моделей. Здесь показывается только классификация: отбор подмножества кандидатов не используется для пересчёта ranking-метрик, который изменил бы задачу. Интервалы по этим множественным срезам не рассчитаны.

`user_history` — число прошлых доступных прослушиваний пользователя; `user_item` — наличие прошлых прослушиваний этой пары; `user_artist` — прошлые прослушивания выбранного исполнителя данным пользователем (пропуск метаданных также относится к отсутствию наблюдённой связи). `item_in_sample` — наличие прошлых прослушиваний объекта **в выбранной пользовательской выборке**, а не глобальная новизна трека в сервисе. Восстанавливаем counts из `log1p` признака; текущий outcome не участвует в отнесении к срезу.

{subgroup_text}

![Диагностика калибровки](figures/calibration.png)

Калибровка рассчитана в фиксированных интервалах вероятности шириной 0.1; малые интервалы могут быть шумными, численности находятся в `calibration.csv`. Эта визуализация не использовалась для последующей рекалибровки или изменения модели.

## 7. Вычисления и масштабируемость

Вычисление всех признаков: **{_fmt(timing['seconds'],1)} с**, матрица {_int(timing['n_rows'])} × {_int(timing['n_features'])}, оценка памяти DataFrame **{_fmt(timing['matrix_bytes']/2**20,1)} MiB**. Peak RSS всего процесса подготовки: **{feature_peak} MiB**, процесса поиска: **{search_peak} MiB**; это high-water mark процесса, а не память одной операции. Подготовка/сэмплирование/сохранение данных: {_fmt(manifest['elapsed_seconds'],1)} с. Полный feature/hyperparameter search: {_fmt(lock['search_seconds'],1)} с. Суммарное финальное обучение всех моделей и seeds: {_fmt(total_final_fit,1)} с. Время включает реальный локальный CPU-запуск; H100 для этого эксперимента не требуется. GPU-ускорение и размерные кривые на 5k/10k пользователей здесь не измерялись.

{_table(['Модель','Fit всех seeds, с','Predict всех seeds, с','Размер всех моделей, MiB','Деревьев/seed'],time_rows)}

Платформа: `{environment.get('platform','не записана')}`; Python `{environment.get('python','не записан')}`; `thread_count={config['threads']}`. Версии библиотек: `{json.dumps(package_versions, ensure_ascii=False, sort_keys=True)}`. Итоговые времена суммируются по seeds, а не выдаются за latency одного запроса. Время предсказания относится ко всему test-батчу и включает подготовку входа CatBoost. Бенчмарк tail latency и отдельный peak RSS финального обучения не проводились.

{scaling_text}

Feature engine один раз группирует события по entity/day для каждого отношения, хранит кумулятивные суммы отдельно по ключам и получает границы окон через prefix lookup. Для экспоненциальных ядер применяется стабильная рекурсия `S_j = raw_j + 2^(-delta_day/half_life) * S_(j-1)`, со сбросом на границе entity; вычисление реализовано через bidiagonal `solve_banded`. Она не использует выбранную по будущим событиям точку нормировки экспоненты. Добавление box-окна не требует нового прохода по исходным событиям; оно добавляет поиск границ запросов и выходные столбцы. Новое decay-ядро требует прохода по уже сжатым daily-агрегатам. При E событиях, примерно A entity-day агрегатах на отношение, Q запросах, R отношениях и K ядрах практический расход определяют группировки и примерно O(R*K*A + R*K*Q*log A) построение/lookup; память зависит от агрегатов и Q*число признаков. Это оценка структуры реализации, не доказанный distributed benchmark. Для миллиардов событий понадобятся partitioning по ключам, потоковые агрегаты и отдельный store; данная pandas-реализация такую производительность не заявляет.

Бюджеты обучения manual и all_auto одинаковы по числу конфигураций, но AutoFE дополнительно тратит время на матрицу, importance и префиксный поиск. Selected/static/абляции наследуют параметры, а не получают полную независимую настройку. Это сравнение готовых процедур при явно указанном дополнительном бюджете, не утверждение равенства всех вычислительных затрат.

## 8. Ограничения и вывод

{conclusion}

Практический результат — исполняемый генератор агрегатов с проверяемым временным cutoff и честный журнал сравнения. Дополнительные окна и decay можно отбирать без ручного перебора каждого признака; выигрыш или его отсутствие оценивается на едином временном holdout, одинаковых observed-кандидатах и одной доступной истории.

Ограничения: одна случайная пользовательская выборка; только рекомендованный трафик и first-play-per-day постановка; смещение текущей logging policy; условный и ретроспективный ranking-набор; предположение о времени доступности outcome; статические метаданные с упрощёнными many-to-many связями; ограниченная грамматика; один random-control; importance-префиксы вместо глобального поиска; отсутствие отдельной embargo- и learning-curve абляции. Сглаживание, окна baseline и размеры временных блоков фиксированы без полного перебора. Мы не измеряли причинный эффект рекомендаций, качество retrieval, production latency или конкурентоспособность с масштабным sequence transformer. Победа над сильным бустингом не обязательна: зафиксированный отрицательный результат остаётся воспроизводимым результатом исследования.

Следующие разумные проверки — повторить пользовательский sample seed, проверить другой embargo, оценить несколько случайных порядков при том же бюджете, добавить missing/multi-artist-aware агрегаты и при отдельном вычислительном бюджете сравнить с sequence/graph представлениями. Эти эксперименты **не проведены** и не включены в результаты.

## 9. Воспроизведение и аудит

Из корня репозитория после установки зависимостей и загрузки фиксированных исходников:

```bash
python -m pytest -q
PYTHONPATH=src python -m recsys_research.run --config configs/main.yaml --run-name reproduce-01 --phase prepare
PYTHONPATH=src python -m recsys_research.run --config configs/main.yaml --run-name reproduce-01 --phase search
PYTHONPATH=src python -m recsys_research.run --config configs/main.yaml --run-name reproduce-01 --phase evaluate
PYTHONPATH=src python -m recsys_research.run --config configs/main.yaml --run-name reproduce-01 --phase report
PYTHONPATH=src python scripts/audit_results.py --config configs/main.yaml --run-name reproduce-01
```

Точные команды окружения и загрузки — в README. `search` отказывается переписывать существующий selection lock; `evaluate` отказывается повторять уже сохранённую итоговую оценку. При изменении конфигурации нужен новый `processed_dir`/`output_dir`; это не должно использоваться для выбора нового варианта по уже просмотренному test. Признаки и запросы проверяются хешами; исходники фиксируются до оценки. Проверки причинности, временных границ, правил first play и метрик находятся в тестах.

Отдельный [`audit_results.py`](../../scripts/audit_results.py) не обучает модели и не подбирает признаки: проверяет хеши, соответствие строк и метрик, сравнивает NDCG с API scikit-learn и при наличии локальных весов повторяет прогнозы после их перезагрузки. Сохранённый результат основного запуска: [`audit.json`](audit.json); лог полного набора тестов: [`tests.log`](../tests.log).

Артефакты: [`selection_lock.json`](selection_lock.json), [`test_results.json`](test_results.json), [`search_trials.json`](search_trials.json), [`search_importance.json`](search_importance.json), [`test_predictions.parquet`](test_predictions.parquet), [`exploratory_subgroups.csv`](exploratory_subgroups.csv), [`calibration.csv`](calibration.csv). Манифест выборки и `feature_groups.json` создаются в `{config['processed_dir']}`; feature timings — там же. Их лёгкие копии можно хранить рядом с отчётом: [`manifest.json`](manifest.json), [`feature_groups.json`](feature_groups.json), [`feature_timing.json`](feature_timing.json). Хеш selection lock в результате: `{results['selection_lock_sha256']}`. Хеш предсказаний: `{results['test_prediction_sha256']}`. Проверяемые SHA256 каждого исходного файла записаны в manifest; исходные данные не коммитятся.
"""
    target = out / "REPORT.md"
    target.write_text(report, encoding="utf-8")
    print(f"Report written: {target}", flush=True)
    return target
