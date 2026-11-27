"""dual_update: lambda moves the right way and stays inside its bounds."""

import math

import pytest

from hcdpo.constraint import dual_update, kkt_metrics

SETTINGS = dict(epsilon=2.0, dual_lr=0.05, lambda_min=0.01, lambda_max=5.0)


def test_lambda_rises_when_kl_above_epsilon() -> None:
    assert dual_update(0.1, kl=3.0, **SETTINGS) > 0.1


def test_lambda_falls_when_kl_below_epsilon() -> None:
    assert dual_update(0.1, kl=1.0, **SETTINGS) < 0.1


def test_lambda_unchanged_when_kl_equals_epsilon() -> None:
    assert dual_update(0.1, kl=2.0, **SETTINGS) == pytest.approx(0.1)


def test_update_matches_hand_computation() -> None:
    # exponent = 0.05 * (3 - 2) / 2 = 0.025
    assert dual_update(0.1, kl=3.0, **SETTINGS) == pytest.approx(0.1 * math.exp(0.025))


def test_lambda_stays_within_bounds() -> None:
    lam = 0.1
    for _ in range(1000):
        lam = dual_update(lam, kl=100.0, **SETTINGS)
        assert SETTINGS["lambda_min"] <= lam <= SETTINGS["lambda_max"]
    assert lam == SETTINGS["lambda_max"]
    for _ in range(1000):
        lam = dual_update(lam, kl=0.0, **SETTINGS)
        assert SETTINGS["lambda_min"] <= lam <= SETTINGS["lambda_max"]
    assert lam == SETTINGS["lambda_min"]


def test_extreme_kl_does_not_overflow() -> None:
    assert dual_update(0.1, kl=1e9, **SETTINGS) == SETTINGS["lambda_max"]


def test_kkt_metrics() -> None:
    infeasible = kkt_metrics(lam=0.5, kl=3.0, epsilon=2.0)
    assert infeasible == {"constraint_violation": 1.0, "complementary_slackness": 0.5, "feasible": 0.0}
    feasible = kkt_metrics(lam=0.5, kl=1.0, epsilon=2.0)
    assert feasible == {"constraint_violation": 0.0, "complementary_slackness": -0.5, "feasible": 1.0}
