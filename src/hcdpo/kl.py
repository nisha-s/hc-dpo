"""Sampling from the policy and Monte Carlo estimation of KL(policy || reference).

KL(pi || pi_ref) = E_{y ~ pi}[ log pi(y|x) - log pi_ref(y|x) ], so we sample responses
from the policy and average the log-ratio. Units: nats per response, where a response
is at most `max_new_tokens` tokens.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from hcdpo.data import build_batch, end_of_turn_id, tokenize_prompt
from hcdpo.losses import sequence_logprobs
from hcdpo.utils import autocast


def left_pad(sequences: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad token id lists into (input_ids, attention_mask); generation needs prompts right-aligned."""
    max_length = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, max_length - len(sequence) :] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, max_length - len(sequence) :] = 1
    return input_ids, attention_mask


@torch.no_grad()
def sample_responses(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    seed: int | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Sample `num_samples` responses per prompt from the model's exact distribution.

    Returns one dict per sample: `prompt_index`, `prompt_ids`, `response_ids` (ending with
    the end-of-turn token unless truncated) and `truncated` (hit `max_new_tokens`).
    The model is put in eval mode (dropout off) and restored afterwards.
    """
    device = next(model.parameters()).device
    end_id = end_of_turn_id(tokenizer)
    if seed is not None:
        torch.manual_seed(seed)
    prompt_ids = [tokenize_prompt(tokenizer, prompt) for prompt in prompts]
    jobs = [(index, ids) for index, ids in enumerate(prompt_ids) for _ in range(num_samples)]

    samples: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start : start + batch_size]
            input_ids, attention_mask = left_pad([ids for _, ids in chunk], tokenizer.pad_token_id)
            with autocast(device):
                output = model.generate(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    max_new_tokens=max_new_tokens,
                    eos_token_id=end_id,
                    pad_token_id=tokenizer.pad_token_id,
                    # The model's generation_config sets temperature 0.7, top_p 0.8, top_k 20 and
                    # repetition_penalty 1.05. Each is switched off here so samples come from the
                    # same distribution that sequence_logprobs scores.
                    do_sample=True,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                )
            new_tokens = output[:, input_ids.shape[1] :].tolist()
            for (prompt_index, ids), tokens in zip(chunk, new_tokens):
                truncated = end_id not in tokens
                # Rows that finished early are padded after the end-of-turn token; cut there.
                response = tokens if truncated else tokens[: tokens.index(end_id) + 1]
                samples.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt_ids": ids,
                        "response_ids": response,
                        "truncated": truncated,
                    }
                )
    finally:
        model.train(was_training)
    return samples


@torch.no_grad()
def score_samples(
    model: torch.nn.Module, samples: list[dict[str, Any]], pad_token_id: int, batch_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of each sampled response under the policy and the reference.

    Returns (policy_logprobs, reference_logprobs), each shape (num_samples,), on the CPU.
    """
    device = next(model.parameters()).device
    policy_chunks, reference_chunks = [], []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = build_batch(
                [s["prompt_ids"] for s in chunk], [s["response_ids"] for s in chunk], pad_token_id
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast(device):
                policy_chunks.append(sequence_logprobs(model, **batch).cpu())
                with model.disable_adapter():
                    reference_chunks.append(sequence_logprobs(model, **batch).cpu())
    finally:
        model.train(was_training)
    return torch.cat(policy_chunks), torch.cat(reference_chunks)


def summarize_kl(
    samples: list[dict[str, Any]], policy_logprobs: torch.Tensor, reference_logprobs: torch.Tensor
) -> dict[str, float]:
    """Mean and standard error of the KL estimate from scored samples.

    Samples of the same prompt are averaged first, so the standard error is over
    prompts (the independent units), not over individual samples.
    """
    log_ratios = (policy_logprobs - reference_logprobs).double()
    prompt_index = torch.tensor([s["prompt_index"] for s in samples])
    num_prompts = int(prompt_index.max().item()) + 1
    sums = torch.zeros(num_prompts, dtype=torch.double).index_add_(0, prompt_index, log_ratios)
    counts = torch.zeros(num_prompts, dtype=torch.double).index_add_(
        0, prompt_index, torch.ones_like(log_ratios)
    )
    per_prompt = sums / counts
    stderr = (per_prompt.std(unbiased=True) / num_prompts**0.5).item() if num_prompts > 1 else 0.0
    return {
        "mean": per_prompt.mean().item(),
        "stderr": stderr,
        "truncated_fraction": sum(s["truncated"] for s in samples) / len(samples),
        "num_prompts": num_prompts,
        "num_samples": len(samples),
    }


def estimate_kl(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    seed: int | None = None,
    batch_size: int = 32,
) -> dict[str, float]:
    """Monte Carlo estimate of sequence-level KL(policy || reference) on the given prompts.

    Samples `num_samples` responses per prompt from the current policy, re-scores them
    under the policy (adapter on) and the reference (adapter off), and returns the mean
    and standard error of the log-ratio plus the fraction of truncated samples.
    """
    samples = sample_responses(model, tokenizer, prompts, num_samples, max_new_tokens, seed, batch_size)
    policy_logprobs, reference_logprobs = score_samples(model, samples, tokenizer.pad_token_id, batch_size)
    return summarize_kl(samples, policy_logprobs, reference_logprobs)
