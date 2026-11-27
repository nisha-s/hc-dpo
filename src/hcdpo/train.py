"""DPO and HC-DPO training loop in plain PyTorch.

    python -m hcdpo.train --config configs/dpo.yaml --set beta=0.1 seed=42
    python -m hcdpo.train --config configs/hcdpo.yaml --set epsilon=2.0 seed=42

HC-DPO is the same loop with beta replaced by a multiplier lambda that
`constraint.dual_update` adjusts every `dual_update_every` steps from the measured KL.

A run lives in <output_dir>/<run_name>/:
    config.yaml          the resolved config
    train_log.jsonl      every logged metric (the analysis notebook reads this)
    checkpoints/step_N/  LoRA adapter + trainer_state.pt (optimizer, scheduler, RNG, lambda)
    final_adapter/       the adapter after the last step
Rerunning the same command resumes from the latest checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import PreTrainedTokenizerBase

from hcdpo.constraint import dual_update, kkt_metrics
from hcdpo.data import check_data_matches_config, load_subsets, tokenize_pairs
from hcdpo.kl import estimate_kl
from hcdpo.losses import dpo_loss, pair_logprobs
from hcdpo.model import load_policy, load_tokenizer
from hcdpo.utils import get_device, load_config, print_environment, read_jsonl, set_seed, write_jsonl

CHECKPOINTS_TO_KEEP = 2


def default_run_name(config: dict[str, Any]) -> str:
    """Run name, e.g. `dpo_beta0.1_seed42` or `hcdpo_eps2.0_seed42`."""
    if config["method"] == "hcdpo":
        return f"hcdpo_eps{config['epsilon']}_seed{config['seed']}"
    return f"dpo_beta{config['beta']}_seed{config['seed']}"


def lr_multiplier(step: int, warmup_steps: int, total_steps: int, schedule: str) -> float:
    """Learning-rate factor at optimizer step `step` (0-based): linear warmup, then constant or linear decay."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    if schedule == "linear":
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    raise ValueError(f"Unknown lr_schedule '{schedule}' (use 'constant' or 'linear')")


def training_order(num_examples: int, num_epochs: int, seed: int) -> list[int]:
    """Example indices for the whole run: a fresh shuffle per epoch, fixed by the seed so a resumed run sees the same order."""
    generator = torch.Generator().manual_seed(seed)
    return [i for _ in range(num_epochs) for i in torch.randperm(num_examples, generator=generator).tolist()]


def run_kl_eval(
    model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, kl_prompts: list[str], config: dict[str, Any]
) -> dict[str, float]:
    """KL estimate on the fixed monitoring prompts, as metrics ready to log."""
    result = estimate_kl(
        model,
        tokenizer,
        kl_prompts,
        num_samples=config["kl_num_samples"],
        max_new_tokens=config["max_new_tokens"],
        batch_size=config["eval"]["batch_size"],
    )
    return {"kl/mean": result["mean"], "kl/stderr": result["stderr"], "kl/truncated_fraction": result["truncated_fraction"]}


def hcdpo_dual_step(dual_state: dict[str, Any], kl: float, config: dict[str, Any]) -> dict[str, float]:
    """Update lambda in `dual_state` from a KL estimate and return the KKT metrics to log."""
    epsilon = config["epsilon"]
    metrics = kkt_metrics(dual_state["lambda"], kl, epsilon)
    dual_state["lambda"] = dual_update(
        dual_state["lambda"], kl, epsilon, config["dual_lr"], config["lambda_min"], config["lambda_max"]
    )
    dual_state["num_updates"] += 1
    dual_state["num_infeasible"] += int(kl > epsilon)
    dual_state["max_overshoot"] = max(dual_state["max_overshoot"], kl - epsilon)
    logged = {f"hcdpo/{name}": value for name, value in metrics.items()}
    logged["hcdpo/lambda"] = dual_state["lambda"]  # the value used from this step on
    logged["hcdpo/epsilon"] = epsilon
    logged["hcdpo/max_overshoot"] = dual_state["max_overshoot"]
    logged["hcdpo/infeasible_fraction"] = dual_state["num_infeasible"] / dual_state["num_updates"]
    return logged


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Most recent complete checkpoint, or None. A checkpoint is complete once trainer_state.pt exists."""
    root = run_dir / "checkpoints"
    if not root.exists():
        return None
    complete = sorted(d for d in root.iterdir() if (d / "trainer_state.pt").exists())
    return complete[-1] if complete else None


def save_checkpoint(
    run_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    wandb_run_id: str | None,
    dual_state: dict[str, Any],
) -> None:
    """Save the adapter and everything needed to resume exactly; keep only the newest checkpoints."""
    directory = run_dir / "checkpoints" / f"step_{step:06d}"
    model.save_pretrained(str(directory))
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "wandb_run_id": wandb_run_id,
        "dual_state": dict(dual_state),
    }
    # Written last: its presence marks the checkpoint as complete.
    torch.save(state, directory / "trainer_state.pt")
    complete = sorted(d for d in (run_dir / "checkpoints").iterdir() if (d / "trainer_state.pt").exists())
    for old in complete[:-CHECKPOINTS_TO_KEEP]:
        shutil.rmtree(old)


def restore_rng(state: dict[str, Any]) -> None:
    """Restore Python, NumPy and torch RNG states from a checkpoint."""
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])


def init_wandb(config: dict[str, Any], run_name: str, run_id: str | None) -> Any:
    """Start (or resume, when `run_id` is given) a W&B run. Returns None when W&B is off."""
    if not config["wandb"]["enabled"]:
        return None
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("wandb.enabled is true but WANDB_API_KEY is not set (or pass --set wandb.enabled=false)")
    import wandb

    return wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=run_name,
        id=run_id,
        resume="allow",
        config=config,
    )


def log_metrics(log_path: Path, wandb_run: Any, step: int, metrics: dict[str, float]) -> None:
    """Append metrics to train_log.jsonl and, if W&B is on, log them there too."""
    with open(log_path, "a") as f:
        f.write(json.dumps({"step": step, **metrics}) + "\n")
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def prepare_run_dir(run_dir: Path, config: dict[str, Any]) -> None:
    """Create the run directory and save the config; refuse to resume with a different config."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return
    with open(config_path) as f:
        if yaml.safe_load(f) != config:
            raise ValueError(f"{run_dir} holds a run with a different config; use another run_name or delete it")


def train(config: dict[str, Any]) -> Path:
    """Train one run to completion (resuming if a checkpoint exists). Returns the run directory."""
    if config["method"] not in ("dpo", "hcdpo"):
        raise ValueError(f"Unknown method '{config['method']}' (use 'dpo' or 'hcdpo')")
    is_hcdpo = config["method"] == "hcdpo"
    if is_hcdpo and config["epsilon"] is None:
        raise ValueError("method hcdpo needs a KL budget: pass --set epsilon=<value>")

    device = get_device()
    environment = print_environment(device)
    run_name = config["run_name"] or default_run_name(config)
    run_dir = Path(config["output_dir"]) / run_name
    prepare_run_dir(run_dir, config)
    log_path = run_dir / "train_log.jsonl"

    check_data_matches_config(config)
    subsets = load_subsets(config["data"]["dir"])
    tokenizer = load_tokenizer(config["model_name"])
    train_ids = tokenize_pairs(tokenizer, subsets["train"])
    kl_prompts = [example["prompt"] for example in subsets["kl_prompts"]]

    # One optimizer step = grad_accum_steps batches.
    batch_size, grad_accum = config["batch_size"], config["grad_accum_steps"]
    order = training_order(len(subsets["train"]), config["num_epochs"], config["seed"])
    total_steps = len(order) // (batch_size * grad_accum)
    if config["max_steps"] is not None:
        total_steps = min(total_steps, config["max_steps"])

    latest = find_latest_checkpoint(run_dir)
    set_seed(config["seed"])
    model = load_policy(config, device, adapter_dir=latest)
    model.print_trainable_parameters()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, config["warmup_steps"], total_steps, config["lr_schedule"])
    )

    global_step = 0
    wandb_run_id = None
    dual_state: dict[str, Any] = {}
    if is_hcdpo:
        dual_state = {"lambda": config["lambda_init"], "num_updates": 0, "num_infeasible": 0, "max_overshoot": 0.0}
    if latest is not None:
        # weights_only=False: the file holds RNG states (tuples, NumPy arrays), and we wrote it ourselves.
        state = torch.load(latest / "trainer_state.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state)
        global_step = state["step"]
        wandb_run_id = state["wandb_run_id"]
        dual_state = state.get("dual_state", dual_state)
        # Drop log lines written after the checkpoint, so resumed steps are not logged twice.
        write_jsonl(log_path, [row for row in read_jsonl(log_path) if row["step"] <= global_step])
        print(f"Resuming from step {global_step} ({latest})")
    if global_step >= total_steps:
        print(f"Run {run_name} is already complete ({global_step}/{total_steps} steps). Nothing to do.")
        return run_dir

    wandb_run = init_wandb(config, run_name, wandb_run_id)
    if wandb_run is not None:
        wandb_run_id = wandb_run.id
        wandb_run.config.update({"environment": environment}, allow_val_change=True)
        (run_dir / "wandb_run_id.txt").write_text(wandb_run_id)

    # In HC-DPO the multiplier lambda takes the place of beta and changes during training.
    beta = dual_state["lambda"] if is_hcdpo else config["beta"]
    kl_interval = config["dual_update_every"] if is_hcdpo else config["kl_eval_every"]
    print(f"Run {run_name}: {total_steps} optimizer steps, effective batch size {batch_size * grad_accum}, beta {beta}")
    if is_hcdpo:
        print(f"HC-DPO: epsilon {config['epsilon']}, dual update every {kl_interval} steps")

    if global_step == 0:
        kl_metrics = run_kl_eval(model, tokenizer, kl_prompts, config)
        log_metrics(log_path, wandb_run, 0, kl_metrics)
        print(f"step 0 | KL {kl_metrics['kl/mean']:.4f} +- {kl_metrics['kl/stderr']:.4f}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    while global_step < total_steps:
        step_metrics: dict[str, float] = {}
        for micro in range(grad_accum):
            start = (global_step * grad_accum + micro) * batch_size
            indices = order[start : start + batch_size]
            policy_chosen, policy_rejected, ref_chosen, ref_rejected = pair_logprobs(
                model,
                [train_ids["prompt"][i] for i in indices],
                [train_ids["chosen"][i] for i in indices],
                [train_ids["rejected"][i] for i in indices],
                tokenizer.pad_token_id,
                device,
            )
            loss, metrics = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
            (loss / grad_accum).backward()
            metrics["loss"] = loss.detach()
            for name, value in metrics.items():
                step_metrics[f"train/{name}"] = step_metrics.get(f"train/{name}", 0.0) + value.item() / grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"])
        step_metrics["train/grad_norm"] = grad_norm.item()
        step_metrics["train/lr"] = scheduler.get_last_lr()[0]
        step_metrics["train/beta"] = beta
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        is_last = global_step == total_steps

        if global_step % config["log_every"] == 0 or is_last:
            log_metrics(log_path, wandb_run, global_step, step_metrics)
            print(
                f"step {global_step}/{total_steps} | loss {step_metrics['train/loss']:.4f} | "
                f"acc {step_metrics['train/preference_accuracy']:.3f} | "
                f"margin {step_metrics['train/reward_margin']:.4f} | "
                f"lr {step_metrics['train/lr']:.2e} | {time.time() - start_time:.0f}s"
            )
        if global_step % kl_interval == 0 or is_last:
            kl_metrics = run_kl_eval(model, tokenizer, kl_prompts, config)
            kl_line = f"step {global_step} | KL {kl_metrics['kl/mean']:.4f} +- {kl_metrics['kl/stderr']:.4f}"
            if is_hcdpo and global_step % kl_interval == 0:
                kl_metrics.update(hcdpo_dual_step(dual_state, kl_metrics["kl/mean"], config))
                beta = dual_state["lambda"]
                kl_line += f" | epsilon {config['epsilon']} | lambda -> {beta:.4f}"
            log_metrics(log_path, wandb_run, global_step, kl_metrics)
            print(kl_line)
        if global_step % config["save_every"] == 0 or is_last:
            save_checkpoint(run_dir, global_step, model, optimizer, scheduler, wandb_run_id, dual_state)
            print(f"step {global_step} | saved checkpoint")

    model.save_pretrained(str(run_dir / "final_adapter"))
    print(f"Done. Final adapter saved to {run_dir / 'final_adapter'}")
    if wandb_run is not None:
        wandb_run.finish()
    return run_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train DPO or HC-DPO.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", help="key=value overrides")
    args = parser.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
