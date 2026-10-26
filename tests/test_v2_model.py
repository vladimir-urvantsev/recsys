"""Behavioral checks for the history network, not implementation snapshots."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from recsys_history_v2.model import DualHashEmbedding, EventHistoryModel, exact_relation_matches


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def config():
    return {"d_model": 16, "n_layers": 2, "n_heads": 4, "dropout": 0,
            "context_recent": 8, "context_older": 8,
            "embedding_buckets": {"item": 17, "artist": 11, "album": 19, "user": 13}}


def make_batch(global_dim=3):
    return {
        "history_ids": torch.tensor([[[10, 4, 100], [20, 5, 101], [-2, -2, -2]],
                                     [[30, -1, 102], [-2, -2, -2], [-2, -2, -2]]]),
        "history_type": torch.tensor([[1, 2, 0], [1, 0, 0]], dtype=torch.uint8),
        "history_ratio": torch.tensor([[100, -1, -1], [10, -1, -1]], dtype=torch.int16),
        "history_organic": torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.uint8),
        "history_age": torch.tensor([[2., 1., 0.], [3., 0., 0.]]),
        "history_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "group_uid": torch.tensor([1, 2]),
        "candidate_ids": torch.tensor([[[10, 4, 100], [40, 6, 103], [-2, -2, -2]],
                                       [[30, -1, 102], [50, 7, 104], [-2, -2, -2]]]),
        "global_features": torch.arange(2 * 3 * global_dim, dtype=torch.float32).reshape(2, 3, global_dim) / 10,
        "candidate_mask": torch.tensor([[True, True, False], [True, True, False]]),
    }


def test_output_shape_finite_backward_and_padding_gradients(config):
    model = EventHistoryModel(config, global_dim=3)
    batch = make_batch()
    logits = model(batch)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert torch.equal(logits[:, 2], torch.zeros(2))
    loss = F.binary_cross_entropy_with_logits(logits[batch["candidate_mask"]], torch.tensor([1., 0., 0., 1.]))
    loss.backward()
    assert model.exact_match_bias.grad is not None
    assert torch.isfinite(model.exact_match_bias.grad).all()
    assert model.history_layers[0].self_attn.in_proj_weight.grad is not None
    assert model.history_layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    for embedding in model.id_embeddings.values():
        for table in embedding.tables:
            assert table.weight.grad is not None
            assert torch.isfinite(table.weight.grad).all()
            assert torch.equal(table.weight.grad[0], torch.zeros_like(table.weight.grad[0]))


@pytest.mark.parametrize("length", [0, 3])
def test_empty_history_is_finite_and_trainable(config, length):
    model = EventHistoryModel(config, global_dim=0)
    batch = make_batch(global_dim=0)
    for key in ("history_ids", "history_type", "history_ratio", "history_organic", "history_age", "history_mask"):
        batch[key] = batch[key][:, :length]
    batch["history_mask"][:] = False
    logits = model(batch)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert torch.isfinite(model.null_history.grad).all()


def test_arbitrary_masked_payload_does_not_change_predictions(config):
    torch.manual_seed(2)
    model = EventHistoryModel(config, global_dim=3).eval()
    batch = make_batch()
    changed = deepcopy(batch)
    changed["history_ids"][~changed["history_mask"]] = 987654321
    changed["history_ratio"][~changed["history_mask"]] = 32767
    changed["history_age"][~changed["history_mask"]] = float("nan")
    changed["history_type"][~changed["history_mask"]] = 255
    changed["history_organic"][~changed["history_mask"]] = 255
    changed["candidate_ids"][~changed["candidate_mask"]] = 456789
    changed["global_features"][~changed["candidate_mask"]] = float("nan")
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(changed), rtol=0, atol=0)


def test_inserting_padding_does_not_shift_valid_event_positions(config):
    torch.manual_seed(5)
    model = EventHistoryModel(config, global_dim=3).eval()
    batch = make_batch()
    padded = deepcopy(batch)
    for key in ("history_ids", "history_type", "history_ratio", "history_organic", "history_age", "history_mask"):
        old = batch[key]
        padding = torch.zeros_like(old[:, :1])
        if key == "history_ids":
            padding.fill_(-2)
        if key == "history_ratio":
            padding.fill_(-1)
        padded[key] = torch.cat([padding, old[:, :1], padding, old[:, 1:], padding], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(padded), rtol=1e-5, atol=1e-6)


def test_unrelated_candidates_cannot_change_existing_scores(config):
    torch.manual_seed(9)
    model = EventHistoryModel(config, global_dim=3).eval()
    batch = make_batch()
    one = deepcopy(batch)
    for key in ("candidate_ids", "candidate_mask", "global_features"):
        one[key] = one[key][:, :1]
    with torch.no_grad():
        expected = model(batch)[:, :1]
        torch.testing.assert_close(model(one), expected, rtol=1e-5, atol=1e-6)
        # Reordering other groups cannot couple users through normalization.
        swapped = {key: value.flip(0) for key, value in batch.items()}
        torch.testing.assert_close(model(swapped).flip(0), model(batch), rtol=1e-5, atol=1e-6)


def test_query_outcomes_and_uncontracted_fields_are_ignored(config):
    model = EventHistoryModel(config, global_dim=3).eval()
    batch = make_batch()
    leaked_fields = {**batch, "label": torch.ones(2, 3),
                     "played_ratio_pct": torch.full((2, 3), 1000),
                     "future_events": torch.randn(2, 99, 7),
                     "timestamp": torch.full((2, 3), 99999999)}
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(leaked_fields), rtol=0, atol=0)


def test_no_encoder_is_permutation_invariant_with_age_attached(config):
    config["n_layers"] = 0
    torch.manual_seed(42)
    model = EventHistoryModel(config, global_dim=0).eval()
    assert model.position_embedding is None
    batch = make_batch(global_dim=0)
    permuted = deepcopy(batch)
    for key in ("history_ids", "history_type", "history_ratio", "history_organic", "history_age", "history_mask"):
        permuted[key] = permuted[key][:, [2, 0, 1]]
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(permuted), rtol=1e-5, atol=1e-6)


def test_missing_and_padding_ids_do_not_form_exact_relation_matches():
    candidates = torch.tensor([[[10, -1, -2], [20, 5, 6]]])
    history = torch.tensor([[[10, -1, -2], [20, 5, 7]]])
    expected = torch.tensor([[[[True, False, False], [False, False, False]],
                               [[False, False, False], [True, True, False]]]])
    assert torch.equal(exact_relation_matches(candidates, history), expected)


def test_hash_banks_have_different_collision_classes_and_reserved_tokens():
    embedding = DualHashEmbedding(16, 3)
    ids = torch.arange(3000)
    h0, h1 = embedding.bucket_indices(ids, 0), embedding.bucket_indices(ids, 1)
    # A deterministic affine transform modulo one power-of-two bucket count
    # would put every h0 collision in an h1 collision as well.
    first_bucket = h1[h0 == h0[0]]
    assert len(torch.unique(first_bucket)) > 1
    assert embedding.bucket_indices(torch.tensor([-2, -1, 0]), 0).tolist()[:2] == [0, 1]
    assert torch.equal(embedding(torch.tensor([-2]))[0], torch.zeros(3))
    assert torch.isfinite(embedding(torch.tensor([999999999]))).all()


def test_shared_encoded_and_complete_forward_match(config, tmp_path):
    torch.manual_seed(3)
    model = EventHistoryModel(config, global_dim=3).eval()
    batch = make_batch()
    with torch.no_grad():
        memory, mask, ids = model.encode_history(batch)
        expected = model.score_candidates(memory, mask, ids, batch)
        torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    reloaded = EventHistoryModel(config, global_dim=3).eval()
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(batch), expected, rtol=0, atol=0)
    assert model.description()["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert model.num_parameters() == model.num_parameters(trainable_only=True)


def test_network_can_overfit_a_tiny_logged_response_batch(config):
    """A genuine optimization check catches a detached or untrainable head."""
    torch.manual_seed(13)
    model = EventHistoryModel(config, global_dim=0)
    batch = make_batch(global_dim=0)
    target = torch.tensor([1., 0., 1., 0.])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    before = F.binary_cross_entropy_with_logits(model(batch)[batch["candidate_mask"]], target).item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(batch)[batch["candidate_mask"]], target)
        loss.backward()
        optimizer.step()
    after = F.binary_cross_entropy_with_logits(model(batch)[batch["candidate_mask"]], target).item()
    assert after < 0.2 and after < before / 2


def test_bfloat16_autocast_keeps_logits_and_gradients_finite(config):
    model = EventHistoryModel(config, global_dim=3)
    batch = make_batch()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(
            logits[batch["candidate_mask"]], torch.tensor([1., 0., 1., 0.]))
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert torch.isfinite(model.exact_match_bias.grad).all()


@pytest.mark.parametrize("change", [{"d_model": 15}, {"n_layers": -1}, {"dropout": 1}, {"context_recent": 0, "context_older": 0}])
def test_invalid_architecture_fails_early(config, change):
    with pytest.raises(ValueError):
        EventHistoryModel({**config, **change}, global_dim=0)
