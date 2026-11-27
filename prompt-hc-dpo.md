# Prompts: HC-DPO (Hard-Constrained DPO)

Two prompts. Give Phase 1 first. Give Phase 2 only after Phase 1 runs end to end and you have a baseline beta sweep.

---

## Phase 1 prompt: starter code (standard DPO baseline)

Copy everything below into your coding agent.

You are helping me build a research codebase. Read this whole prompt before doing anything.

### Ground rules
- The only thing you run on my machine is the test suite, inside Docker (`docker build` then `docker run`). Do not install packages on the Mac itself, download models or datasets, or start training. I run training and evaluation on Google Colab (L4 or A100 GPU).
- For anything that runs on Colab, give me the exact command and what output to expect; I will run it and report back.
- Simple, explicit code: plain functions in a few modules. No class hierarchies, no Hydra, no Lightning, and do NOT use TRL's DPOTrainer. I want the DPO loss and training loop written out in plain PyTorch so I can read and modify them. Type hints and short docstrings on every function. Comments only where the reason isn't obvious.
- No bitsandbytes or anything that breaks on macOS. Device order: cuda, then mps, then cpu.
- Precision: bf16 autocast when on CUDA and `torch.cuda.is_bf16_supported()` is true (the A100 case); fp32 everywhere else. No fp16 and no GradScaler.

### What the project is
I will compare standard DPO, whose beta is a soft KL penalty, against HC-DPO, which enforces a hard KL budget KL(policy || reference) <= epsilon by updating a Lagrange multiplier during training. Phase 1 is the standard DPO baseline plus all shared infrastructure: data, log-probabilities, the training loop, KL measurement, evaluation and logging. Phase 2 (later) adds the constraint.

### Stack and versions
- Python 3.12 (what Colab runs), PyTorch, Hugging Face transformers, datasets, peft, wandb, pyyaml, pytest.
- `requirements.txt`: pin exact versions of transformers, peft, datasets, pyyaml and pytest. Give only a lower bound for torch (so Colab keeps its preinstalled CUDA build), and for wandb and sentencepiece (pinning them downgrades protobuf or forces a source build on Colab).
- Set `USE_TF=0` and `USE_FLAX=0` in the package `__init__.py` so transformers never imports Colab's preinstalled TensorFlow.
- You cannot run pip, so your pins are from memory. Pick a combination you are confident was released together and works on Python 3.12. I install them myself on Colab as the first cell of the runner (nothing is installed on my machine by you); if that install fails I will report the error and you fix the pins.
- At the start of every training and evaluation run, log the versions of python, torch, transformers, peft and datasets, and the GPU name.

### Model
- Default `Qwen/Qwen2.5-0.5B-Instruct`; config option `Qwen/Qwen2.5-1.5B-Instruct`.
- LoRA on the policy (r=16, alpha=32, dropout 0.05, attention projections). The reference model is the same base model with the adapter disabled (`with model.disable_adapter():` under `torch.no_grad()`), so only one model is in memory.

### Data (`data.py`)
- `Anthropic/hh-rlhf`, helpful-base subset.
- Split each example into the prompt (the conversation up to and including the final "Assistant:") and the chosen and rejected final responses.
- Parse the "Human:" / "Assistant:" turns into chat messages and build the prompt with the tokenizer's chat template (`add_generation_prompt=True`). Do not pass a system message; let Qwen's template add its default system prompt, and do this identically everywhere (training, KL, evaluation). Document this in the README.
- Each response is the response text followed by the end-of-turn token (`<|im_end|>`). That token is part of the response and is included in the response mask, so the policy learns when to stop and log-probabilities of dataset responses and sampled responses are computed the same way.
- Drop examples where the chosen and rejected prompts are not identical, where either response is empty, or where the two responses are identical.
- Filter by token length (templated prompt <= 256 tokens, response <= 256 tokens).
- Subsample with a fixed seed and save the subsets to disk so every run uses identical data:
  - 10,000 training pairs from the train split.
  - 32 KL-monitoring prompts, drawn from the training pairs.
  - 500 held-out eval pairs from the test split.
  - 300 generation prompts from the test split, disjoint from the 500 pairs.

### Core functions
- `losses.py`
  - `sequence_logprobs(model, input_ids, attention_mask, response_mask)`: sum of log-probabilities over response tokens only. Prompt and padding are masked out.
  - `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)`: returns the loss plus metrics (implicit rewards, reward margin, preference accuracy).
- `kl.py`
  - `sample_responses(model, tokenizer, prompts, num_samples, max_new_tokens, seed)`: the single sampling helper, used by both KL estimation and evaluation. It must sample from the exact policy distribution. Qwen's `generation_config` ships with temperature 0.7, top_p 0.8, top_k 20 and repetition_penalty 1.05, so pass `do_sample=True, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0` explicitly on every call. Left-pad for generation.
  - `estimate_kl(model, tokenizer, prompts, num_samples, max_new_tokens)`: sample responses from the current policy, then re-score each sample with `sequence_logprobs` under the policy (adapter on) and the reference (adapter off), and return the mean and standard error of sum log pi(y|x) - sum log pi_ref(y|x). `num_samples` is samples per prompt. This is a Monte Carlo estimate of sequence-level KL(policy || reference), in nats per response.
  - Put the model in eval mode for sampling and scoring (LoRA dropout off) and restore train mode afterwards. Everything under `torch.no_grad()`.
  - The scored tokens are the generated tokens, including the end-of-turn token when the model produced one. A sample that reaches `max_new_tokens` is scored on the tokens it has. Also return the fraction of truncated samples.
  - Use `max_new_tokens=128` everywhere (training-time KL and evaluation) so KL numbers are comparable.

### Training (`train.py`)
- AdamW, gradient accumulation, gradient clipping, seed from config.
- Learning-rate schedule: linear warmup, then constant by default (`lr_schedule: constant`); `linear` decay is a config option. Constant is the default for both methods so that a KL curve flattening cannot be an artifact of the learning rate going to zero.
- Defaults in `configs/dpo.yaml`: lr 5e-5, batch size 8 with accumulation 4 (effective 32; 16 does not fit on a 22 GB L4), 2 epochs over the 10,000 pairs (about 625 optimizer steps), warmup 30 steps, max grad norm 1.0, beta 0.1.
- Run `estimate_kl` on the 32 saved KL-monitoring prompts (2 samples per prompt) every `kl_eval_every` optimizer steps (default 25), and once at step 0, and log the mean, standard error and truncated fraction. Phase 2 will reuse this.
- Checkpoints: save every `save_every` steps and at the end. A checkpoint contains the LoRA adapter, optimizer state, scheduler state, global step, position in the data order, Python / NumPy / torch RNG states, and the W&B run id. Resuming from the latest checkpoint must continue the same W&B run and the same data order (Colab disconnects).
- CLI: `python -m hcdpo.train --config configs/dpo.yaml --set beta=0.1 seed=42`.

### Evaluation (`evaluate.py`)
For a trained adapter:
1. Preference accuracy on the 500 held-out pairs.
2. Generate one response for each of the 300 generation prompts with `sample_responses` (max_new_tokens 128) and save them to JSONL. Use the same fixed generation seed for every run.
3. KL estimate on those generations (re-score the same samples; do not sample again).
4. Score the generations with the reward model `OpenAssistant/reward-model-deberta-v3-large-v2`. It takes a (question, answer) pair: the question is the last human turn of the prompt, the answer is the generated response, truncated to 512 tokens. Report mean and standard error.
5. Evaluate the untrained base model the same way, once, as the starting point. For each trained run also report the paired difference in reward score against the base model on the same 300 prompts (mean and standard error of the per-prompt differences).

Write all final metrics to a `metrics.json` in the run directory so the analysis notebook does not depend on W&B.

### Weights & Biases
Log config, loss, implicit rewards, margins, preference accuracy, KL estimates over training, final eval metrics, and a table of sample generations. API key from the environment. The code must run with W&B off (`wandb.enabled: false`).

### Repo layout
```
hc-dpo/
  README.md
  requirements.txt
  Dockerfile                 # python:3.12-slim, CPU torch; `docker run` runs the tests
  .dockerignore
  RUNNING.md                 # step-by-step: tests in Docker, sweeps on Colab
  .gitignore                 # data/, outputs/, wandb/
  configs/
    dpo.yaml
    debug.yaml               # 64 pairs, 10 steps, tiny eval: runs on a laptop CPU
  src/hcdpo/
    data.py
    model.py                 # load base model, tokenizer, LoRA
    losses.py
    kl.py
    train.py
    evaluate.py
    utils.py                 # seeding, device, precision, config loading
  scripts/
    run_dpo_sweep.sh         # beta in {0.05, 0.1, 0.25, 0.5}
  notebooks/
    colab_runner.ipynb       # clone, install, mount Drive, set W&B key, run
    analysis.ipynb           # reward vs KL plot, KL over training
  tests/
    test_losses.py           # dpo_loss on hand-computed toy numbers
    test_masking.py          # prompt and padding tokens excluded from log-probs
```
All outputs go to a configurable directory, which on Colab is on Google Drive. The Colab runner assumes an A100 runtime and prints the GPU name and whether bf16 is active.

### README and RUNNING.md
README: short and plain. Research question, method, the data-formatting choices above, experiment list, results placeholders, and a related-work section to be filled in later (DPO, PPO's adaptive KL controller, dynamic-beta DPO variants). RUNNING.md: the step-by-step commands (Docker tests, Colab setup, sweeps, seeds, analysis).

### Order of work
1. Requirements, data and model loading.
2. `sequence_logprobs`, `dpo_loss` and tests.
3. Training loop with logging and checkpointing.
4. `sample_responses`, `estimate_kl` and evaluation.
5. Colab runner, sweep script, analysis notebook, README.

After each step, stop, tell me what you built, and give me the command to check it.

---

## Phase 2 prompt: HC-DPO

Copy everything below into the same agent once Phase 1 works. Same ground rules as before: do not run anything on my machine; keep code simple and explicit.

Extend the repo with HC-DPO. Keep standard DPO working unchanged; HC-DPO is a config option (`method: hcdpo`), not a separate codebase.

### Method
- DPO comes from a KL-penalized objective with penalty weight beta. Replacing the soft penalty with a hard constraint KL(policy || reference) <= epsilon and forming the Lagrangian gives the same optimum with beta replaced by a Lagrange multiplier lambda. So HC-DPO trains with the DPO loss using beta = lambda, and updates lambda by dual ascent.
- Every `dual_update_every` optimizer steps:
  1. Estimate KL with `estimate_kl` on the 32 saved KL-monitoring prompts (same settings as Phase 1).
  2. Update lambda in log space, with the constraint error measured relative to epsilon:
     `lambda = clip(lambda * exp(dual_lr * (kl - epsilon) / epsilon), lambda_min, lambda_max)`.
     The multiplicative form keeps lambda positive and makes `dual_lr` independent of the scale of the KL, which is several nats per sequence while lambda is around 0.01 to 1.
- The lambda update needs no gradient through the KL estimate. Put it in a small, clearly commented function, `dual_update`, in a new file `constraint.py`.
- Config and defaults: `epsilon` (no default, required), `lambda_init: 0.1`, `dual_lr: 0.05`, `lambda_min: 0.01`, `lambda_max: 5.0`, `dual_update_every: 10`. In hcdpo mode the KL estimate from the dual update is the one that gets logged; do not run a second KL estimate on the `kl_eval_every` schedule.
- Save lambda in every checkpoint and restore it on resume.

### Known limitation to handle and document
The DPO loss has no term that pulls the policy back toward the reference. Raising lambda only makes the loss saturate sooner, which slows further drift; it does not reduce a KL that is already above epsilon. Two consequences:
- At step 0 the KL is exactly 0 (LoRA starts as the identity), so lambda falls at first. If it falls too fast the policy overshoots epsilon and cannot come back. That is why `lambda_init` starts at a normal DPO beta and `dual_lr` is small. Do not add other mechanisms; `dual_lr` is the knob I will tune.
- Log the overshoot so it is visible: the running maximum of `kl - epsilon` over training, and the fraction of dual updates that were infeasible.
State this limitation in the README.

### KKT monitoring (log at every dual update)
- `kl_estimate` (with standard error) and `epsilon`
- `lambda`
- `constraint_violation = max(0, kl - epsilon)` (primal feasibility)
- `complementary_slackness = lambda * (kl - epsilon)` (should be near 0 at convergence)
- `feasible` flag (kl <= epsilon)
- `max_overshoot` (running maximum of kl - epsilon)

### What the experiment claims
If lambda converges to some value, HC-DPO is equivalent to standard DPO at beta equal to that value, so in theory both methods lie on the same reward-vs-KL curve. The claim under test is control: HC-DPO reaches a chosen KL in one run, where DPO needs a beta sweep to find it. So:
- The calibration plot (final KL vs target epsilon) is the main result.
- The reward-vs-KL plot is the check that this control costs no reward.
Write the README and the analysis notebook around this framing.

### Experiments
- `scripts/run_hcdpo_sweep.sh` over 4 epsilon values (nats per response, same units as the KL estimate). Leave them as config placeholders: I will choose them to span the KL range the Phase 1 beta sweep produced.
- Seeds: one seed (42) for the sweeps. Then 3 seeds (42, 43, 44) for:
  - the best DPO beta, meaning the sweep run with the highest mean reward-model score, and
  - the HC-DPO epsilon set equal to that run's final evaluation KL, so the two are compared at matched KL.
- Add `scripts/run_seeds.sh` that takes the beta and epsilon as arguments.

### Analysis notebook additions
1. Calibration: final KL vs target epsilon for HC-DPO, with the y = x line (the main result).
2. Reward vs final KL for every DPO and HC-DPO run on one plot, with standard-error bars.
3. KL over training steps: HC-DPO runs should level off near epsilon, DPO runs are wherever their beta puts them.
4. Lambda over training.
5. Matched-KL table: reward, KL and preference accuracy, mean and standard deviation over the 3 seeds, DPO vs HC-DPO.

### Tests
- `test_constraint.py`: lambda rises when KL > epsilon, falls when KL < epsilon, is unchanged when KL = epsilon, and stays within [lambda_min, lambda_max].

Update the README with the method (including the Lagrangian derivation in a few lines), the limitation above, the experiment design, and results placeholders. Stop after each piece and give me the command to check it.
