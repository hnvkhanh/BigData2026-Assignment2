import numpy as np
import pandas as pd
import pytest
import torch

from experiments.relgt_weekly.data import NUM_FEATURES, weekly_features
from experiments.relgt_weekly.model import WeeklyGNN, WeeklyNN, WeeklyRelGT

CUTOFF = pd.Timestamp("2026-06-10 16:00:00")


def make_events(rows):
    """rows: (user_id, relation_id, entity_id, repo_id, days_before_cutoff)."""
    frame = pd.DataFrame(
        rows, columns=["user_id", "relation_id", "entity_id", "repo_id", "days_before"]
    )
    frame["action_at"] = CUTOFF - pd.to_timedelta(frame.pop("days_before"), unit="D")
    frame["available_at"] = frame["action_at"]
    frame["entity_available_at"] = frame["action_at"] - pd.Timedelta(days=1)
    frame["global_entity_id"] = frame["entity_id"]
    return frame


def test_events_are_binned_into_weeks_before_cutoff():
    # User 1: 4 pushes this week (2 distinct repos), 2 comments 8 days ago.
    events = make_events(
        [(1, 6, 10, 100, 0.5), (1, 6, 11, 100, 1), (1, 6, 12, 101, 2), (1, 6, 12, 101, 6.9)]
        + [(1, 3, 20, 100, 8), (1, 3, 21, 100, 9.5)]
    )
    users, features = weekly_features(events, CUTOFF, weeks=3, eligibility_days=14)

    assert users.tolist() == [1]
    assert features.shape == (1, 3, NUM_FEATURES)
    push, comment = 5 * 3, 2 * 3  # feature offsets of relation_id 6 and 3
    np.testing.assert_allclose(
        features[0, 0, push : push + 3],
        np.log1p([4, 3, 2]),  # events, distinct entities, distinct repos
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        features[0, 1, comment : comment + 3], np.log1p([2, 2, 1]), rtol=1e-6
    )
    assert features[0, 2].sum() == 0  # nothing three weeks back


def test_ineligible_users_and_future_events_are_excluded():
    events = make_events(
        [(1, 6, 10, 100, d) for d in (1, 2, 3, 4, 5)]  # eligible
        + [(2, 6, 10, 100, d) for d in (1, 2, 3, 4)]  # only four events
        + [(2, 6, 10, 100, -1)]  # after the cutoff: must not count
    )
    users, _ = weekly_features(events, CUTOFF, weeks=2, eligibility_days=14)
    assert users.tolist() == [1]


def test_events_not_yet_available_do_not_leak():
    events = make_events([(1, 6, 10, 100, d) for d in (1, 2, 3, 4, 5)])
    events.loc[0, "available_at"] = CUTOFF + pd.Timedelta(hours=1)
    users, _ = weekly_features(events, CUTOFF, weeks=2, eligibility_days=14)
    assert users.tolist() == []


@pytest.mark.parametrize("user_embedding", [False, True])
def test_model_outputs_six_logits(user_embedding):
    known = np.array([3, 5, 9]) if user_embedding else None
    model = WeeklyRelGT(
        num_weeks=4, num_features=NUM_FEATURES, hidden_dim=16, num_heads=2,
        num_local_layers=1, num_centroids=3, known_user_ids=known,
    )
    features = torch.rand(5, 4, NUM_FEATURES)
    args = (features, torch.tensor([3, 5, 9, 42, 3])) if user_embedding else (features,)
    assert model(*args).shape == (5, 6)


def test_model_rejects_wrong_week_count_and_trains():
    model = WeeklyRelGT(
        num_weeks=4, num_features=NUM_FEATURES, hidden_dim=16, num_heads=2,
        num_local_layers=1, num_centroids=3,
    )
    with pytest.raises(ValueError):
        model(torch.rand(2, 5, NUM_FEATURES))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(torch.rand(2, 4, NUM_FEATURES)), torch.ones(2, 6)
    )
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("model_class", [WeeklyNN, WeeklyGNN])
@pytest.mark.parametrize("user_embedding", [False, True])
def test_nn_and_gnn_output_six_logits_and_train(model_class, user_embedding):
    known = np.array([3, 5, 9]) if user_embedding else None
    model = model_class(
        num_weeks=4, num_features=NUM_FEATURES, hidden_dim=16, num_layers=2,
        known_user_ids=known,
    )
    features = torch.rand(5, 4, NUM_FEATURES)
    args = (features, torch.tensor([3, 5, 9, 42, 3])) if user_embedding else (features,)
    logits = model(*args)
    assert logits.shape == (5, 6)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(5, 6)).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    wrong_weeks = torch.rand(5, 5, NUM_FEATURES)
    with pytest.raises(ValueError, match="Expected 4 weeks"):
        model(wrong_weeks, *args[1:])


def test_weekly_nn_sees_week_order():
    model = WeeklyNN(num_weeks=3, num_features=NUM_FEATURES, hidden_dim=16).eval()
    features = torch.rand(1, 3, NUM_FEATURES)
    assert not torch.allclose(model(features), model(features.flip(1)))
