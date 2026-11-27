"""dpo_loss checked against numbers worked out by hand."""

import math

import pytest
import torch

from hcdpo.losses import dpo_loss


def test_dpo_loss_matches_hand_computation() -> None:
    # Pair 1: reward_chosen = 0.5 * (-10 - -11) = 0.5, reward_rejected = 0.5 * (-12 - -11) = -0.5
    #         margin = 1.0, loss = -log(sigmoid(1.0))  = 0.313262
    # Pair 2: reward_chosen = 0.5 * (-5 - -4) = -0.5, reward_rejected = 0.5 * (-3 - -4) = 0.5
    #         margin = -1.0, loss = -log(sigmoid(-1.0)) = 1.313262
    loss, metrics = dpo_loss(
        policy_chosen=torch.tensor([-10.0, -5.0]),
        policy_rejected=torch.tensor([-12.0, -3.0]),
        ref_chosen=torch.tensor([-11.0, -4.0]),
        ref_rejected=torch.tensor([-11.0, -4.0]),
        beta=0.5,
    )
    assert loss.item() == pytest.approx((0.313262 + 1.313262) / 2, abs=1e-5)
    assert metrics["reward_chosen"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["reward_rejected"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["reward_margin"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["preference_accuracy"].item() == pytest.approx(0.5)


def test_policy_equal_to_reference_gives_log2_and_half_accuracy() -> None:
    logprobs = torch.tensor([-7.0, -3.0, -20.0])
    loss, metrics = dpo_loss(logprobs, logprobs - 1.0, logprobs, logprobs - 1.0, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2.0), abs=1e-6)
    assert metrics["reward_margin"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["preference_accuracy"].item() == pytest.approx(0.5)


def test_beta_scales_the_margin() -> None:
    args = (torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-2.0]), torch.tensor([-2.0]))
    _, small = dpo_loss(*args, beta=0.1)
    _, large = dpo_loss(*args, beta=0.5)
    # Log-ratio difference is (1) - (-1) = 2, so the margin is 2 * beta.
    assert small["reward_margin"].item() == pytest.approx(0.2, abs=1e-6)
    assert large["reward_margin"].item() == pytest.approx(1.0, abs=1e-6)
    assert small["preference_accuracy"].item() == pytest.approx(1.0)


def test_gradient_raises_chosen_and_lowers_rejected() -> None:
    policy_chosen = torch.tensor([-4.0], requires_grad=True)
    policy_rejected = torch.tensor([-4.0], requires_grad=True)
    loss, metrics = dpo_loss(policy_chosen, policy_rejected, torch.tensor([-4.0]), torch.tensor([-4.0]), beta=0.5)
    loss.backward()
    # d loss / d policy_chosen = -beta * sigmoid(-margin) = -0.5 * 0.5 at margin 0.
    assert policy_chosen.grad.item() == pytest.approx(-0.25, abs=1e-6)
    assert policy_rejected.grad.item() == pytest.approx(0.25, abs=1e-6)
    assert not metrics["reward_margin"].requires_grad
