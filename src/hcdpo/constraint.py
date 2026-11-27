"""The HC-DPO constraint: dual ascent on the Lagrange multiplier lambda.

HC-DPO enforces KL(policy || reference) <= epsilon. Training uses the ordinary DPO
loss with beta = lambda; this file only decides how lambda moves.
"""

from __future__ import annotations

import math

MAX_EXPONENT = 5.0


def dual_update(
    lam: float, kl: float, epsilon: float, dual_lr: float, lambda_min: float, lambda_max: float
) -> float:
    """One dual-ascent step on lambda. Returns the new lambda.

    - KL above the budget (kl > epsilon): lambda grows, so the DPO loss saturates sooner
      and the policy drifts less.
    - KL below the budget (kl < epsilon): lambda shrinks, allowing more drift.

    The step is multiplicative and uses the error relative to epsilon:
        lambda <- lambda * exp(dual_lr * (kl - epsilon) / epsilon)
    so lambda stays positive and `dual_lr` means the same thing whatever the scale of the
    KL. `kl` is a plain number (a Monte Carlo estimate); no gradient flows through it.
    The result is clipped to [lambda_min, lambda_max]; the positive floor keeps the DPO
    loss from degenerating at lambda = 0.
    """
    exponent = dual_lr * (kl - epsilon) / epsilon
    # Bounds a single step to a factor of e^5, in case of an extreme KL estimate.
    exponent = max(-MAX_EXPONENT, min(MAX_EXPONENT, exponent))
    return min(lambda_max, max(lambda_min, lam * math.exp(exponent)))


def kkt_metrics(lam: float, kl: float, epsilon: float) -> dict[str, float]:
    """KKT quantities for the current lambda and KL estimate.

    - constraint_violation = max(0, kl - epsilon): primal feasibility (0 when satisfied).
    - complementary_slackness = lambda * (kl - epsilon): near 0 at convergence.
    - feasible: 1.0 if kl <= epsilon, else 0.0.
    """
    return {
        "constraint_violation": max(0.0, kl - epsilon),
        "complementary_slackness": lam * (kl - epsilon),
        "feasible": 1.0 if kl <= epsilon else 0.0,
    }
