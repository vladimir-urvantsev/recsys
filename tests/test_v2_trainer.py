"""Synthetic integration and objective checks for the V2 training boundary."""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from recsys_history_v2 import trainer


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    context = root / "contexts"
    context.mkdir(parents=True)
    sizes = [1, 4, 2, 2, 2, 3]
    codes = np.repeat(np.arange(len(sizes)), sizes)
    splits = np.repeat(["train", "train", "train", "tune", "select", "test"], sizes)
    labels = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, np.nan, np.nan, np.nan], np.float32)
    q = pd.DataFrame({"uid": 100 + codes, "item_id": 500 + np.arange(len(codes)),
                      "artist_id": 20 + codes, "album_id": 40 + codes,
                      "day": np.repeat([50, 60, 70, 220, 250, 280], sizes),
                      "group_id": 1000 + codes, "split": splits, "label": labels})
    history_ids = np.full((len(sizes), 3, 3), -2, np.int64)
    history_mask = np.zeros((len(sizes), 3), bool)
    history_type = np.zeros((len(sizes), 3), np.uint8)
    history_ratio = np.full((len(sizes), 3), -1, np.int16)
    history_organic = np.zeros((len(sizes), 3), np.uint8)
    history_age = np.zeros((len(sizes), 3), np.float32)
    for group in range(1, len(sizes)):
        length = group % 3 + 1
        history_mask[group, :length] = True
        for token in range(length):
            history_ids[group, token] = [500 + group + token, 20 + group, 40 + group]
            history_type[group, token] = 1
            history_ratio[group, token] = 100 if token % 2 else 0
            history_age[group, token] = np.log1p(length - token)
    arrays = {"history_ids": history_ids, "history_type": history_type,
              "history_ratio": history_ratio, "history_organic": history_organic,
              "history_age": history_age, "history_mask": history_mask,
              "group_uid": np.arange(100, 100 + len(sizes), dtype=np.int64),
              "query_group": codes.astype(np.int64)}
    for name, array in arrays.items():
        mmap = np.lib.format.open_memmap(context / f"{name}.npy", mode="w+", dtype=array.dtype, shape=array.shape)
        mmap[:] = array
        mmap.flush()
        del mmap
    x = pd.DataFrame({"cat_uid": q.uid.astype(str), "global_positive_prior": .5,
                      "item__count__all": np.arange(len(q), dtype=np.float32),
                      "artist__rate__w7": np.arange(len(q), dtype=np.float32) / 20,
                      "user__count__all": 999., "user_item__rate__all": .9})
    groups = {"manual": x.columns.tolist()}
    trainer.prepare_global_features(x, q, groups, root)
    config = {"device": "cpu", "mixed_precision": "fp32", "threads": 1,
              "neural_batch_groups": 2, "neural_max_epochs": 1, "neural_patience": 1,
              "neural_weight_decay": 0., "context_recent": 3, "context_older": 0,
              "embedding_buckets": {"item": 17, "artist": 11, "album": 19, "user": 13}}
    params = {"d_model": 16, "n_layers": 1, "n_heads": 4, "dropout": 0.,
              "learning_rate": .01, "pairwise_weight": 0.}
    return root, q, x, groups, config, params


def test_global_block_excludes_personal_and_category_features():
    columns = ["cat_item_id", "cat_uid", "user__count__all", "user_item__rate__all",
               "global_positive_prior", "item__count__all", "artist__rate__all", "album__likes"]
    assert trainer.global_columns(columns) == ["global_positive_prior", "item__count__all", "artist__rate__all", "album__likes"]


def test_global_standardizer_fits_train_only_and_ignores_future_features(prepared, tmp_path):
    root, q, x, groups, _, _ = prepared
    first = json.loads((root / "global_normalizer.json").read_text())
    matrix_first = np.load(root / "global_features.npy")
    other = tmp_path / "perturbed"
    other.mkdir()
    changed = x.copy()
    for column in first["columns"]:
        changed.loc[~q.split.eq("train"), column] = 1000000
    second = trainer.prepare_global_features(changed, q, groups, other)
    assert first == second
    assert first["fit_split"] == "train" and first["fitted_rows"] == 7
    assert first["mean"] == pytest.approx([.5, 3., .15])
    assert first["scale"][0] == 1  # Constant prior remains a finite zero column.
    matrix_second = np.load(other / "global_features.npy")
    np.testing.assert_array_equal(matrix_first[q.split.eq("train")], matrix_second[q.split.eq("train")])
    assert np.isfinite(matrix_second).all()
    assert matrix_second.max() <= 12 and matrix_second.min() >= -12


def test_group_batches_preserve_every_row_once_and_context_shared(prepared):
    root, q, *_ = prepared
    batches = trainer.GroupBatches(root, q, "train", batch_groups=2, global_dim=3)
    assert len(batches) == 2
    assert isinstance(batches.arrays["history_ids"], np.memmap)
    all_rows = []
    for batch in batches.batches():
        mask = batch["candidate_mask"].numpy()
        rows = batch["row_indices"].numpy()[mask]
        all_rows.extend(rows.tolist())
        np.testing.assert_array_equal(batch["labels"].numpy()[mask], q.label.to_numpy()[rows])
        np.testing.assert_array_equal(batch["candidate_ids"].numpy()[mask], q[["item_id", "artist_id", "album_id"]].to_numpy()[rows])
        assert batch["history_ids"].shape[0] == batch["candidate_ids"].shape[0]
        assert (batch["candidate_ids"][~batch["candidate_mask"]] == -2).all()
        assert (batch["row_indices"][~batch["candidate_mask"]] == -1).all()
    assert all_rows == list(range(7))
    one_order = [b["row_indices"].tolist() for b in batches.batches(shuffle=True, seed=12)]
    assert one_order == [b["row_indices"].tolist() for b in batches.batches(shuffle=True, seed=12)]


def test_label_boundaries_require_binary_development_and_sealed_test(prepared):
    root, q, *_ = prepared
    poisoned_train = q.copy()
    poisoned_train.loc[q.split.eq("train"), "label"] = np.nan
    with pytest.raises(ValueError, match="Development labels"):
        trainer.GroupBatches(root, poisoned_train, "train", 2, 3)
    opened_test = q.copy()
    opened_test.loc[q.split.eq("test"), "label"] = 1.
    with pytest.raises(ValueError, match="sealed"):
        trainer.GroupBatches(root, opened_test, "test", 2, 3)
    test = trainer.GroupBatches(root, q, "test", 2, global_dim=0)
    batch = next(test.batches())
    assert torch.isnan(batch["labels"][batch["candidate_mask"]]).all()
    assert batch["global_features"].shape[-1] == 0


def test_bce_gradient_normalizer_is_constant_across_different_batch_sizes(prepared, tmp_path, monkeypatch):
    root, q, _, _, config, params = prepared
    original = trainer.F.binary_cross_entropy_with_logits
    coefficients, sizes, sums = [], [], []

    def capture(logits, target, **kwargs):
        assert kwargs.get("reduction") == "sum"
        value = original(logits, target, **kwargs)
        sizes.append(target.numel())
        sums.append(value.detach().item())
        value.register_hook(lambda gradient: coefficients.append(gradient.detach().item()))
        return value

    monkeypatch.setattr(trainer.F, "binary_cross_entropy_with_logits", capture)
    result = trainer.fit_history(root, q, params, config, 3, tmp_path / "normalization.pt", epochs=1)
    assert len(set(sizes)) == 2 and sum(sizes) == 7
    # Seven rows, two user-day minibatches: each BCE sum gets factor 2/7.
    assert coefficients == pytest.approx([2 / 7, 2 / 7])
    assert result["epoch_log"][0]["train_logloss"] == pytest.approx(sum(sums) / 7)


def test_pairwise_loss_has_correct_direction_and_no_padding_contribution():
    good = torch.tensor([[2., -2., 900.]], requires_grad=True)
    labels = torch.tensor([[1., 0., 0.]])
    mask = torch.tensor([[True, True, False]])
    loss = trainer.pairwise_sum(good, labels, mask)
    assert loss.item() == pytest.approx(F.softplus(torch.tensor(-4.)).item())
    loss.backward()
    assert good.grad[0, 0] < 0 and good.grad[0, 1] > 0 and good.grad[0, 2] == 0
    bad = torch.tensor([[-2., 2., 900.]])
    assert trainer.pairwise_sum(bad, labels, mask) > loss


def test_pairwise_empty_or_one_class_groups_have_differentiable_zero():
    logits = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([[1., 1., 1., 1.], [0., 0., 0., 0.], [0., 1., 0., 1.]])
    mask = torch.tensor([[True] * 4, [True] * 4, [False] * 4])
    loss = trainer.pairwise_sum(logits, labels, mask)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_pairwise_sampling_is_capped_and_extreme_logits_stay_finite(monkeypatch):
    original = torch.randint
    sampled_sizes = []

    def tracked(high, size, **kwargs):
        sampled_sizes.append(size)
        return original(high, size, **kwargs)

    monkeypatch.setattr(trainer.torch, "randint", tracked)
    logits = torch.tensor([[-1000.] * 20 + [1000.] * 20], requires_grad=True)
    labels = torch.tensor([[1.] * 20 + [0.] * 20])
    loss = trainer.pairwise_sum(logits, labels, torch.ones_like(labels, dtype=torch.bool), max_pairs=3)
    assert sampled_sizes == [(3,), (3,)]
    assert loss.item() == pytest.approx(2000.) and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_one_epoch_fit_selects_on_tune_and_reload_predicts_sealed_test(prepared, tmp_path, monkeypatch):
    root, q, _, _, config, params = prepared
    requested_splits = []
    original = trainer.GroupBatches

    def tracked(*args, **kwargs):
        requested_splits.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "GroupBatches", tracked)
    # A reader of actual hidden labels would fail; this is never opened by fit.
    secret = root / "test_labels.parquet"
    secret.write_bytes(b"deliberately invalid and inaccessible test labels")
    def forbidden_parquet(*args, **kwargs):
        raise AssertionError("History fitting must not read held-out label files")
    monkeypatch.setattr(trainer.pd, "read_parquet", forbidden_parquet)
    checkpoint = tmp_path / "history.pt"
    info = trainer.fit_history(root, q, params, config, 7, checkpoint)
    assert requested_splits == ["train", "tune"]
    assert info["epochs"] == 1 and np.isfinite(info["tune_logloss"])
    assert np.isfinite(info["epoch_log"][0]["train_logloss"])
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert saved["model_config"]["d_model"] == 16 and saved["global_dim"] == 3
    assert saved["epoch"] == 1 and saved["seed"] == 7
    model, same = trainer.load_history(checkpoint, torch.device("cpu"))
    assert same["epoch"] == saved["epoch"]
    test = original(root, q, "test", 2, global_dim=3)
    first = trainer.predict_history(model, test, config, torch.device("cpu"))
    reloaded, _ = trainer.load_history(checkpoint, torch.device("cpu"))
    second = trainer.predict_history(reloaded, test, config, torch.device("cpu"))
    np.testing.assert_array_equal(first, second)
    assert len(first) == 3 and np.isfinite(first).all() and ((first > 0) & (first < 1)).all()
    assert q.loc[q.split.eq("test"), "label"].isna().all()


def test_fixed_epoch_training_ignores_select_and_test_label_values(prepared, tmp_path, monkeypatch):
    root, q, _, _, config, params = prepared
    requested_splits = []
    original = trainer.GroupBatches
    def train_only(*args, **kwargs):
        requested_splits.append(args[2])
        assert args[2] == "train"
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer, "GroupBatches", train_only)
    one, two = tmp_path / "one.pt", tmp_path / "two.pt"
    first = trainer.fit_history(root, q, params, config, 11, one, epochs=1)
    poisoned = q.copy()
    poisoned.loc[~poisoned.split.eq("train"), "label"] = 777.
    second = trainer.fit_history(root, poisoned, params, config, 11, two, epochs=1)
    assert requested_splits == ["train", "train"]
    assert first["tune_logloss"] is None and second["tune_logloss"] is None
    state1 = torch.load(one, weights_only=True, map_location="cpu")["state"]
    state2 = torch.load(two, weights_only=True, map_location="cpu")["state"]
    assert state1.keys() == state2.keys()
    for name in state1:
        torch.testing.assert_close(state1[name], state2[name], rtol=0, atol=0)


@pytest.mark.parametrize("pure,no_encoder", [(True, False), (False, True)])
def test_fixed_epoch_ablation_checkpoint_records_architecture(prepared, tmp_path, pure, no_encoder):
    root, q, _, _, config, params = prepared
    checkpoint = tmp_path / f"ablation-{pure}-{no_encoder}.pt"
    information = trainer.fit_history(root, q, params, config, 17, checkpoint,
                                      epochs=1, pure=pure, no_encoder=no_encoder)
    model, saved = trainer.load_history(checkpoint, torch.device("cpu"))
    assert model.global_dim == (0 if pure else 3)
    assert model.n_layers == (0 if no_encoder else 1)
    assert information["pure"] == pure and information["no_encoder"] == no_encoder
    assert saved["epoch"] == 1
