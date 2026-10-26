"""Synthetic integration artifacts exercise reports, not scientific claims."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from recsys_history_v2.report import write_report
from recsys_history_v2.statistics import temporal_sensitivity
from recsys_research.metrics import evaluate, paired_user_bootstrap


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifacts(tmp_path):
    out, root = tmp_path / "report", tmp_path / "prepared"
    out.mkdir()
    root.mkdir()
    config = {"study": "synthetic report integration", "output_dir": str(out), "processed_dir": str(root),
              "n_users": 6, "development_users": 2, "warmup_days": 30, "split_days": [210,240,270],
              "days_per_user": {"train":12,"tune":4,"select":4,"test":8}, "day_seed":1701,
              "embargo_days":1, "manual_windows":[7,30,None], "auto_windows":[2,14], "auto_half_lives":[10],
              "model_seeds":[17,29,43], "ablation_seeds":[17], "iterations":30, "early_stopping":5,
              "hyper_grid":[{"depth":3,"learning_rate":.1}], "selection_feature_penalty":1e-6,
              "random_seed":8821, "context_recent":2,"context_older":2, "neural_max_epochs":3,
              "neural_patience":2, "neural_weight_decay":.0001,"mixed_precision":"bf16",
              "neural_grid":[{"d_model":16,"pairwise_weight":0.},{"d_model":32,"pairwise_weight":.1}],
              "blend_weights":[0.,.5,1.],"bootstrap_samples":50,"is_pilot":False}
    q = pd.DataFrame({"uid":[1,5,5,6,5,6,1,1,2,2,3,3,4,4],
                      "item_id":list(range(14)), "day":[30,31,210,211,240,241,270,270,271,271,273,273,274,274],
                      "group_id":[1,2,3,4,5,6,10,10,20,20,30,30,40,40],
                      "split":["train"]*2+["tune"]*2+["select"]*2+["test"]*8,
                      "label":[0,1,1,0,0,1]+[np.nan]*8})
    q.to_parquet(root / "queries.parquet",index=False)
    features = {"user__count__all":np.log1p([0]*6+[0,0,10,10,100,100,1000,1000]),
                "user_item__count__all":np.log1p([0]*6+[0,1,0,1,0,1,0,1]),
                "user_artist__count__all":np.log1p([0]*6+[0,0,1,1,3,3,4,4]),
                "item__count__all":np.log1p([0]*6+[0,1,2,3,4,5,6,7])}
    pd.DataFrame(features).to_parquet(root / "features.parquet",index=False)
    p = q.loc[q.split.eq("test"),["uid","item_id","day","group_id"]].reset_index(drop=True)
    y = np.array([0,1,1,0,0,1,1,0],dtype=np.uint8)
    quality = {"manual":.65,"selected_auto":.7,"all_auto":.72,"history_global":.8,
               "history_pure":.6,"history_no_encoder":.75,"random_0":.69,"hybrid":.76}
    for family,value in quality.items():
        p[family] = np.where(y==1,value,1-value)
    p.to_parquet(out / "blind_predictions.parquet",index=False)
    p["label"] = y
    p.to_parquet(out / "test_predictions.parquet",index=False)
    metrics = {family:evaluate(y,p[family],p.uid,p.group_id) for family in quality}
    primary = paired_user_bootstrap(y,p.selected_auto,p.history_global,p.uid,p.group_id,n_boot=50,seed=31)
    context = {"groups":10,"queries":14,"context_length":4,"recent_capacity":2,"older_capacity":2,
               "empty_groups":1,"total_available_token_occurrences":100,"total_retained_token_occurrences":30,
               "coverage_nonempty_quantiles":{"0":.1,"0.1":.1,"0.5":.25,"0.9":1.,"1":1.}}
    manifest = {"dataset":"synthetic fixture","retained_events":100,"sampled_users":6,"development_users":2,
                "confirmation_users":4,"old_user_overlap":0,"development_confirmation_overlap":0,
                "elapsed_seconds":2.,"purged_label_rows_by_split_before_cohort_filter":{"train":0,"tune":0,"select":0},
                "splits":{name:{"rows":int(q.split.eq(name).sum()),"users":int(q.loc[q.split.eq(name),"uid"].nunique()),
                                "groups":int(q.loc[q.split.eq(name),"group_id"].nunique()),
                                "day_min":int(q.loc[q.split.eq(name),"day"].min()),
                                "day_max":int(q.loc[q.split.eq(name),"day"].max()),
                                **({"positive_rate":.5} if name!="test" else {})}
                          for name in ("train","tune","select","test")}}
    _dump(out / "manifest.json",manifest)
    _dump(out / "registration.json",{"synthetic":True})
    prep = {"seconds":5.,"context_summary":context,
            "artifacts":{"manifest.json":_sha(out / "manifest.json")}}
    _dump(out / "preparation_lock.json",prep)
    _dump(out / "feature_timing.json",{"seconds":3.,"n_rows":14,"n_features":4,"matrix_bytes":224})
    _dump(out / "global_normalizer.json",{"columns":[f"global_{i}" for i in range(37)],
                                          "fit_split":"train","fitted_rows":2,"clip":[-12,12]})
    parameters = [{"d_model":16,"pairwise_weight":0.},{"d_model":32,"pairwise_weight":.1}]
    trials = [{"seed":17,"epochs":1,"params":params,"fit_seconds":seconds,"parameters":1000,
               "cuda_peak_bytes":2**30,"checkpoint":f"absent-trial-{i}.pt",
               "tune_logloss":.5+i*.01,"epoch_log":[{"epoch":1,"train_logloss":.6,"tune_logloss":.5+i*.01},
                                                         {"epoch":2,"train_logloss":.5,"tune_logloss":.51+i*.01}]}
              for i,(params,seconds) in enumerate(zip(parameters,[40.,50.]))]
    _dump(out / "neural_trials.json",trials)
    specs = {}
    for family in quality:
        if family=="hybrid":
            continue
        neural = family.startswith("history")
        seeds = [17] if family in ("history_pure","history_no_encoder") else [17,29,43]
        spec = {"model_paths":[f"absent/{family}_{seed}.pt" for seed in seeds],"params":{"depth":3},"iterations":10}
        if neural:
            spec.pop("iterations")
            spec.update(kind="history",params=parameters[0],epochs=1,controlled_ablation=len(seeds)==1)
            spec["fits"] = [{"seed":seed,"fit_seconds":40. if family=="history_global" and seed==17 else 20. if family=="history_global" else 12. if family=="history_pure" else 13.,
                             "reused_search_winner":family=="history_global" and seed==17,
                             "parameters":1000,"cuda_peak_bytes":2**30} for seed in seeds]
        else:
            spec["seed_training"] = [{"seed":seed,"fit_seconds":0. if seed==17 else 1.,"reused_fit":seed==17} for seed in seeds]
        specs[family] = spec
    config_sha = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    lock = {"config":config,"config_sha256":config_sha,"models":specs,"blend":{"auto_weight":.5},
            "n_confirmation_targets_used_for_selection":0,"test_labels_opened":False,
            "preparation_sha256":_sha(out / "preparation_lock.json"),
            "registration_sha256":_sha(out / "registration.json"),"neural_winner":trials[0],
            "environment":{"gpu":"NVIDIA A100 80GB synthetic fixture","torch":"2.8.0+cu128"}}
    _dump(out / "selection_lock.json",lock)
    prediction_lock = {"selection_sha256":_sha(out / "selection_lock.json"),"labels_opened":False,
                       "blind_predictions_sha256":_sha(out / "blind_predictions.parquet")}
    _dump(out / "prediction_lock.json",prediction_lock)
    result = {"models":metrics,"primary_history_vs_autofe":primary,"is_pilot":False,
              "temporal_sensitivity":temporal_sensitivity(y,p.selected_auto,p.history_global,p.day,n_boot=50,seed=31),
              "selection_sha256":_sha(out / "selection_lock.json"),"prediction_lock_sha256":_sha(out / "prediction_lock.json"),
              "predictions_sha256":_sha(out / "test_predictions.parquet")}
    _dump(out / "test_results.json",result)
    _dump(out / "tabular_search.json",{"selected_groups":["window_14"],"random_groups":[["window_2"]],
            "random_control_matches_selector":[],"search_fit_seconds":10.,"additional_final_fit_seconds":6.,"total_seconds":18.,
            "selection_candidates":[{"prefix":0,"columns":["m"],"groups":[],"logloss":.55,"objective":.55,"fit_seconds":0.},
                                    {"prefix":1,"columns":["m","a"],"groups":["window_14"],"logloss":.5,"objective":.500001,"fit_seconds":2.}]})
    _dump(out / "gpu_profile.json",{"status":"completed","config_sha256":config_sha,"measured_steps":3,
            "mean_step_seconds":.02,"valid_queries_per_second":100.,"peak_allocated_vram_bytes":2**31,
            "peak_reserved_vram_bytes":3*2**30,"capacity_parameters":parameters[1],
            "projection":{"stress_model_training_only_schedule_hours":123456.}})
    _dump(out / "seed_metrics.json",{"status":"completed","models":{
        family:{"seeds":[{"seed":s,"metrics":metrics[family],"seconds":.1} for s in spec.get("seeds",[f["seed"] for f in spec.get("fits",[])])],
                "frozen_ensemble_metrics":metrics[family]}
        for family,spec in specs.items()}})
    return config,out,root,result


def test_artifact_only_report_renders_real_schema_figures_costs_and_blinded_identity(artifacts):
    config,out,_,result = artifacts
    target = write_report(config)
    text = target.read_text()
    assert f"{result['primary_history_vs_autofe']['logloss']['delta']:.7f}" in text
    assert "95% интервал целиком ниже нуля" in text
    assert "не каждый test UID" in text or "Не каждый test UID" in text
    assert "новому календарному режиму" in text
    assert "не joint user/time bootstrap" in text
    assert "123456" not in text
    assert "PyTorch 2.8/cu128" in text
    for name in ["quality","primary_intervals","daily_primary_delta","calibration","autofe_selection",
                 "neural_curves","context_coverage","training_costs","seed_variability"]:
        assert (out / "figures" / f"{name}.png").stat().st_size > 1024
    costs = pd.read_csv(out / "training_costs.csv").set_index("family")
    assert costs.loc["history_global","additional_fit_seconds"] == 40.  # reused40 charged only in search
    assert "neural_search_fit_seconds | 90.0" in text
    assert "neural_additional_final_fit_seconds | 65.0" in text
    subgroups = pd.read_csv(out / "exploratory_subgroups.csv")
    assert subgroups.history_global_logloss.ge(0).all()
    assert subgroups.selected_auto_logloss.ge(0).all()  # uint8 labels must not wrap under unary minus
    overlap = json.loads((out / "supervised_uid_overlap.json").read_text())
    assert overlap["test_users_with_train_targets"] == 1
    assert overlap["test_users_without_train_targets"] == 3
    provenance = json.loads((out / "report_provenance.json").read_text())
    assert provenance["model_fit_calls"] == provenance["model_prediction_calls"] == 0


def test_archival_report_without_optional_diagnostics_does_not_invent_timing_or_seed_sd(artifacts):
    config,out,root,_ = artifacts
    for name in ("tabular_search.json","neural_trials.json","feature_timing.json","global_normalizer.json","gpu_profile.json","seed_metrics.json"):
        (out / name).unlink()
    (root / "features.parquet").unlink()
    (root / "queries.parquet").unlink()
    target = write_report(config)
    text = target.read_text()
    assert "Per-seed test diagnostics не сохранены" in text
    assert "Capacity profile отсутствует/не завершён" in text
    assert "Матрица — × —" in text
    assert not (out / "figures" / "seed_variability.png").exists()
    assert not (out / "figures" / "neural_curves.png").exists()


@pytest.mark.parametrize("tampered",["selection_lock.json","prediction_lock.json","test_predictions.parquet"])
def test_report_rejects_changed_frozen_artifacts_before_drawing_any_figures(artifacts,tampered):
    config,out,_,_ = artifacts
    with (out / tampered).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError,match="checksum differs"):
        write_report(config)
    assert not (out / "REPORT.md").exists()
    assert not (out / "figures").exists()


def _memory_artifact(out, result):
    p = pd.read_parquet(out / "test_predictions.parquet")
    assignments = {"item": [0,1,0,1,0,2,0,2], "artist": [0,0,1,1,2,2,2,2], "album": [0]*8}
    bins = ["no_full_prior", "full_prior_retained", "full_prior_missing_context"]
    rows, entities = [], {}
    for entity, assignment in assignments.items():
        assignment = np.array(assignment)
        no_prior, retained, missing = [int((assignment==i).sum()) for i in range(3)]
        full = retained + missing
        entities[entity] = {"full_prior_rows":full, "retained_prior_rows":retained,
                            "missing_context_prior_rows":missing, "no_full_prior_rows":no_prior,
                            "retention_fraction_given_full_prior":retained/full if full else None,
                            "metadata_missing_rows":0, "retained_without_full_prior_rows":0}
        for i, bin_name in enumerate(bins):
            sub = p.loc[assignment==i]
            record = {"entity":entity,"bin":bin_name,"rows":len(sub),"users":int(sub.uid.nunique()),
                      "positive_rate":float(sub.label.mean()) if len(sub) else None}
            for family in ("selected_auto","history_global"):
                metric = evaluate(sub.label,sub[family],sub.uid,sub.group_id) if len(sub) else {}
                record[family+"_logloss"] = metric.get("logloss")
                record[family+"_auc"] = metric.get("roc_auc")
            record["delta_logloss"] = record["history_global_logloss"]-record["selected_auto_logloss"] if len(sub) else None
            rows.append(record)
    artifact = {"status":"completed", "test_results_sha256":_sha(out / "test_results.json"),
                "fit_calls":0,"model_selection_calls":0,"weight_changes":0,"rows":rows,"entities":entities,
                **{k:result[k] for k in ("selection_sha256","prediction_lock_sha256","predictions_sha256")}}
    csv_path = out / "memory_diagnostic.csv"
    # A valid source CSV may use a different field order/format than the report
    # would generate. Its hash and bytes must survive report regeneration.
    pd.DataFrame(rows).iloc[:,::-1].to_csv(csv_path,index=False)
    artifact["csv"] = {"path":str(csv_path),"sha256":_sha(csv_path)}
    _dump(out / "memory_diagnostic.json",artifact)
    return artifact


def test_optional_memory_report_uses_input_bins_denominator_and_handles_no_full_prior(artifacts):
    config,out,_,result = artifacts
    artifact = _memory_artifact(out,result)
    original_csv = (out / "memory_diagnostic.csv").read_bytes()
    original_mtime = (out / "memory_diagnostic.csv").stat().st_mtime_ns
    target = write_report(config)
    text = target.read_text()
    assert "не причинный эффект сжатия памяти" in text
    assert "не является оценкой выигрыша от увеличения context" in text
    assert "Целевая метка не участвует" in text
    assert "строки test, у которых в полном прошлом" in text
    assert (out / "figures" / "memory_diagnostic.png").stat().st_size > 1024
    rows = pd.read_csv(out / "memory_diagnostic.csv")
    assert len(rows) == 9
    assert rows.loc[rows.entity.eq("album") & rows.bin.eq("full_prior_retained"),"history_global_logloss"].isna().all()
    assert rows.loc[rows.rows.gt(0),"delta_logloss"].lt(0).all()
    provenance = json.loads((out / "report_provenance.json").read_text())
    assert provenance["memory_diagnostic_sha256"] == _sha(out / "memory_diagnostic.json")
    assert provenance["model_fit_calls"] == provenance["model_prediction_calls"] == 0
    assert artifact["entities"]["item"]["retention_fraction_given_full_prior"] == .5
    assert (out / "memory_diagnostic.csv").read_bytes() == original_csv
    assert (out / "memory_diagnostic.csv").stat().st_mtime_ns == original_mtime
    assert not (out / "memory_diagnostic_report.csv").exists()


def test_memory_report_missing_source_csv_creates_separate_derived_file(artifacts):
    config,out,_,result = artifacts
    _memory_artifact(out,result)
    (out / "memory_diagnostic.csv").unlink()
    text = write_report(config).read_text()
    assert not (out / "memory_diagnostic.csv").exists()
    assert len(pd.read_csv(out / "memory_diagnostic_report.csv")) == 9
    assert "memory_diagnostic_report.csv" in text
    assert "Исходный CSV отсутствует" in text


@pytest.mark.parametrize("mutation",["wrong_hash","changed_weights","bad_denominator","csv_bytes"])
def test_optional_memory_report_rejects_unbound_or_nonfixed_diagnostic(artifacts,mutation):
    config,out,_,result = artifacts
    artifact = _memory_artifact(out,result)
    if mutation=="wrong_hash":
        artifact["predictions_sha256"] = "different-experiment"
    elif mutation=="changed_weights":
        artifact["weight_changes"] = 1
    elif mutation=="bad_denominator":
        artifact["entities"]["item"]["retention_fraction_given_full_prior"] = .25
    else:
        with (out / "memory_diagnostic.csv").open("ab") as stream:
            stream.write(b" ")
    _dump(out / "memory_diagnostic.json",artifact)
    with pytest.raises(ValueError,match="Memory diagnostic"):
        write_report(config)
    assert not (out / "REPORT.md").exists()
