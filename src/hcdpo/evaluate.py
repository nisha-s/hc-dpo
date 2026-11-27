"""Evaluate a trained adapter, or the untrained base model.

    python -m hcdpo.evaluate --config configs/dpo.yaml --base
    python -m hcdpo.evaluate --config configs/dpo.yaml --set beta=0.1 seed=42

Use the same --config and --set values as the training run; they identify the run
directory. Writes `metrics.json` and `generations.jsonl` into that directory (the
base model uses <output_dir>/base_model). Evaluate the base model first so trained
runs can report their paired reward difference against it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from hcdpo.data import check_data_matches_config, load_subsets, tokenize_pairs
from hcdpo.kl import sample_responses, score_samples, summarize_kl
from hcdpo.losses import pair_logprobs
from hcdpo.model import load_policy, load_tokenizer
from hcdpo.train import default_run_name, init_wandb
from hcdpo.utils import get_device, load_config, print_environment, read_json, read_jsonl, write_json, write_jsonl

BASE_RUN_NAME = "base_model"
REWARD_MAX_TOKENS = 512


def mean_and_stderr(values: list[float]) -> tuple[float, float]:
    """Mean and standard error of the mean."""
    tensor = torch.tensor(values, dtype=torch.double)
    stderr = (tensor.std(unbiased=True) / len(values) ** 0.5).item() if len(values) > 1 else 0.0
    return tensor.mean().item(), stderr


@torch.no_grad()
def preference_accuracy(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Fraction of held-out pairs whose DPO margin is positive (a tie counts as half), with its standard error."""
    ids = tokenize_pairs(tokenizer, pairs)
    margins = []
    for start in range(0, len(pairs), batch_size):
        rows = slice(start, start + batch_size)
        policy_chosen, policy_rejected, ref_chosen, ref_rejected = pair_logprobs(
            model, ids["prompt"][rows], ids["chosen"][rows], ids["rejected"][rows], tokenizer.pad_token_id, device
        )
        margins.append(((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)).float().cpu())
    margin = torch.cat(margins)
    correct = (margin > 0).float() + 0.5 * (margin == 0).float()
    return mean_and_stderr(correct.tolist())


@torch.no_grad()
def score_rewards(
    questions: list[str], answers: list[str], reward_model_name: str, device: torch.device, batch_size: int
) -> list[float]:
    """Score (question, answer) pairs with the reward model; higher is better."""
    tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device).eval()
    scores: list[float] = []
    for start in range(0, len(questions), batch_size):
        inputs = tokenizer(
            questions[start : start + batch_size],
            answers[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=REWARD_MAX_TOKENS,
            return_tensors="pt",
        ).to(device)
        scores.extend(model(**inputs).logits[:, 0].float().cpu().tolist())
    return scores


def paired_difference_vs_base(rows: list[dict[str, Any]], base_dir: Path) -> dict[str, float]:
    """Mean and standard error of per-prompt (this run's reward - base model's reward), or {} if the base was not evaluated."""
    base_path = base_dir / "generations.jsonl"
    if not base_path.exists():
        print(f"No base-model evaluation at {base_path}; skipping the paired difference. Run with --base first.")
        return {}
    base_reward = {row["id"]: row["reward"] for row in read_jsonl(base_path)}
    mean, stderr = mean_and_stderr([row["reward"] - base_reward[row["id"]] for row in rows])
    return {"reward_diff_vs_base_mean": mean, "reward_diff_vs_base_stderr": stderr}


def log_to_wandb(
    config: dict[str, Any], run_name: str, run_dir: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Log final metrics and a table of sample generations, into the training run when there is one."""
    id_path = run_dir / "wandb_run_id.txt"
    wandb_run = init_wandb(config, run_name, id_path.read_text().strip() if id_path.exists() else None)
    if wandb_run is None:
        return
    import wandb

    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            wandb_run.summary[f"eval/{name}"] = value
    table = wandb.Table(columns=["question", "response", "reward", "log_ratio", "truncated"])
    for row in rows[: config["eval"]["n_table_rows"]]:
        table.add_data(row["last_human"], row["response"], row["reward"], row["log_ratio"], row["truncated"])
    wandb_run.log({"eval/generations": table})
    wandb_run.finish()


def evaluate(config: dict[str, Any], base: bool = False, force: bool = False) -> dict[str, Any]:
    """Run the full evaluation and write metrics.json and generations.jsonl. Returns the metrics."""
    output_dir = Path(config["output_dir"])
    run_name = BASE_RUN_NAME if base else (config["run_name"] or default_run_name(config))
    run_dir = output_dir / run_name
    adapter_dir = None if base else run_dir / "final_adapter"
    if adapter_dir is not None and not adapter_dir.exists():
        raise FileNotFoundError(f"No trained adapter at {adapter_dir}. Train this run first (same --config and --set).")
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"{metrics_path} already exists (use --force to re-evaluate).")
        return read_json(metrics_path)

    device = get_device()
    print_environment(device)
    check_data_matches_config(config)
    subsets = load_subsets(config["data"]["dir"])
    tokenizer = load_tokenizer(config["model_name"])
    # For the base model this attaches a fresh adapter, which is exactly the identity.
    model = load_policy(config, device, adapter_dir=adapter_dir)
    model.eval()
    batch_size = config["eval"]["batch_size"]
    metrics: dict[str, Any] = {
        "run_name": run_name,
        "is_base": base,
        "method": "base" if base else config["method"],
        "beta": None if base else config["beta"],
        "epsilon": None if base else config.get("epsilon"),
        "seed": None if base else config["seed"],
    }

    # 1. Preference accuracy on the held-out pairs.
    metrics["preference_accuracy"], metrics["preference_accuracy_stderr"] = preference_accuracy(
        model, tokenizer, subsets["eval_pairs"], batch_size, device
    )
    print(f"Preference accuracy: {metrics['preference_accuracy']:.4f} +- {metrics['preference_accuracy_stderr']:.4f}")

    # 2. One sampled response per generation prompt, with a seed shared by every run.
    gen_prompts = subsets["gen_prompts"]
    samples = sample_responses(
        model,
        tokenizer,
        [example["prompt"] for example in gen_prompts],
        num_samples=1,
        max_new_tokens=config["max_new_tokens"],
        seed=config["eval"]["gen_seed"],
        batch_size=batch_size,
    )

    # 3. KL estimate on those same samples.
    policy_logprobs, reference_logprobs = score_samples(model, samples, tokenizer.pad_token_id, batch_size)
    kl = summarize_kl(samples, policy_logprobs, reference_logprobs)
    metrics.update({"kl_mean": kl["mean"], "kl_stderr": kl["stderr"], "kl_truncated_fraction": kl["truncated_fraction"]})
    print(f"KL(policy || reference): {kl['mean']:.4f} +- {kl['stderr']:.4f} nats per response")

    responses = [tokenizer.decode(s["response_ids"], skip_special_tokens=True).strip() for s in samples]
    log_ratios = (policy_logprobs - reference_logprobs).tolist()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Reward-model scores on (last human turn, generated response).
    rewards = score_rewards(
        [example["last_human"] for example in gen_prompts], responses, config["eval"]["reward_model"], device, batch_size
    )
    metrics["reward_mean"], metrics["reward_stderr"] = mean_and_stderr(rewards)
    print(f"Reward: {metrics['reward_mean']:.4f} +- {metrics['reward_stderr']:.4f}")

    rows = [
        {
            "id": example["id"],
            "last_human": example["last_human"],
            "prompt": example["prompt"],
            "response": response,
            "truncated": sample["truncated"],
            "log_ratio": log_ratio,
            "reward": reward,
        }
        for example, sample, response, log_ratio, reward in zip(gen_prompts, samples, responses, log_ratios, rewards)
    ]

    # 5. Paired comparison with the base model on the same prompts.
    if not base:
        metrics.update(paired_difference_vs_base(rows, output_dir / BASE_RUN_NAME))
        if "reward_diff_vs_base_mean" in metrics:
            print(f"Reward vs base: {metrics['reward_diff_vs_base_mean']:+.4f} +- {metrics['reward_diff_vs_base_stderr']:.4f}")

    write_jsonl(run_dir / "generations.jsonl", rows)
    write_json(metrics_path, metrics)
    print(f"Wrote {metrics_path}")
    log_to_wandb(config, run_name, run_dir, metrics, rows)
    return metrics


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a trained run or the base model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", help="key=value overrides")
    parser.add_argument("--base", action="store_true", help="evaluate the untrained base model")
    parser.add_argument("--force", action="store_true", help="re-evaluate even if metrics.json exists")
    args = parser.parse_args()
    evaluate(load_config(args.config, args.overrides), base=args.base, force=args.force)


if __name__ == "__main__":
    main()
