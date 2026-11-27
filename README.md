# HC-DPO

Comparing standard DPO against a hard-constrained version (HC-DPO) on `Anthropic/hh-rlhf` helpful-base with Qwen2.5-0.5B-Instruct + LoRA.

## The question

In DPO, beta is a soft penalty on how far the policy drifts from the reference. The KL you end up with is only known after training, so hitting a target KL means sweeping beta.

HC-DPO turns the penalty into a budget, KL(policy || reference) <= epsilon, and enforces it with a Lagrange multiplier lambda that is updated during training. Can it hit a chosen KL in one run, without losing reward compared with DPO at the same KL?

## Method

DPO loss, written out in `src/hcdpo/losses.py`:

```
loss = -log sigmoid( beta * [ (log pi(y_c|x) - log pi_ref(y_c|x)) - (log pi(y_r|x) - log pi_ref(y_r|x)) ] )
```

HC-DPO: DPO comes from `max E[r] - beta * KL`. Replace the penalty with the constraint `KL <= epsilon`, form the Lagrangian `E[r] - lambda * (KL - epsilon)`, and for fixed lambda the optimum has the same form as DPO's with `beta = lambda`. So HC-DPO trains with the DPO loss at `beta = lambda` and updates lambda by dual ascent every few steps (`src/hcdpo/constraint.py`):

```
kl     = estimate_kl(policy)                         # Monte Carlo, 32 fixed prompts
lambda = clip(lambda * exp(dual_lr * (kl - epsilon) / epsilon), lambda_min, lambda_max)
```

KL is measured by sampling from the policy and averaging `log pi(y|x) - log pi_ref(y|x)`, in nats per response (<= 128 tokens).

Caveat: DPO has nothing that pulls the policy back toward the reference. A bigger lambda slows further drift but cannot undo an overshoot, so lambda starts at 0.1 and `dual_lr` is small. The overshoot is logged.

## Setup

- Prompts use Qwen's chat template with its default system prompt; responses end with `<|im_end|>`.
- Prompt <= 256 tokens, response <= 256. 10,000 train pairs, 500 held-out pairs, 300 generation prompts, all fixed by seed and saved to disk.
- Sampling is from the exact policy distribution (Qwen's default temperature / top-p / top-k / repetition penalty are all switched off).
- AdamW, lr 5e-5, effective batch 32, 2 epochs, constant lr after warmup.
- Eval: preference accuracy on the 500 pairs; KL and `OpenAssistant/reward-model-deberta-v3-large-v2` score on the 300 generations; paired reward difference vs the base model.

## Experiments

1. DPO sweep: beta in {0.05, 0.1, 0.25, 0.5}.
2. HC-DPO sweep: 4 epsilons spanning the KL range from (1).
3. Seeds 42, 43, 44 for the best DPO beta and for HC-DPO at that run's final KL.

How to run: see `RUNNING.md`.

## Results

| Run | beta / epsilon | Final KL | Reward | Reward vs base | Pref. acc |
|---|---|---|---|---|---|
| Base model | - | 0 | | - | 0.5 |
| DPO | 0.05 | | | | |
| DPO | 0.1 | | | | |
| DPO | 0.25 | | | | |
| DPO | 0.5 | | | | |
| HC-DPO | | | | | |
| HC-DPO | | | | | |
| HC-DPO | | | | | |
| HC-DPO | | | | | |

Matched-KL, 3 seeds (mean ± std):

| Method | Setting | Final KL | Reward | Pref. acc |
|---|---|---|---|---|
| DPO | beta = | | | |
| HC-DPO | epsilon = | | | |

Plots in `notebooks/analysis.ipynb`: calibration (final KL vs epsilon), reward vs KL, KL over training, lambda over training.

## Related work

TODO: DPO (Rafailov et al., 2023); PPO's adaptive KL controller (Schulman et al., 2017); dynamic-beta DPO variants.
