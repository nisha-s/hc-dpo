"""Sequence log-probabilities and the DPO loss, in plain PyTorch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hcdpo.data import build_batch
from hcdpo.utils import autocast


def response_token_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> torch.Tensor:
    """Vocabulary logits only at the positions that predict a response token.

    `prediction_mask` is boolean (batch, seq_len - 1): True where position t predicts a
    response token at t + 1. Returns (num_response_tokens, vocab).

    A (batch, seq_len, vocab) logit tensor over a 152k vocabulary is several GB, and the
    prompt and padding rows would be thrown away. So for a PEFT-wrapped Hugging Face model
    we run the transformer body, pick the hidden states we need, and apply the output
    layer to those alone. LoRA lives inside the body, so adapter on/off behaves as usual.
    """
    if hasattr(model, "get_base_model"):
        causal_lm = model.get_base_model()
        hidden = causal_lm.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return causal_lm.get_output_embeddings()(hidden[:, :-1, :][prediction_mask])
    # Any other model (e.g. the tiny fake model in the tests): full logits, then select.
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return logits[:, :-1, :][prediction_mask]


def sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum of log p(token | previous tokens) over response tokens only.

    All inputs are (batch, seq_len). `response_mask` is 1 on response tokens and 0 on
    prompt and padding tokens, so those contribute nothing. Returns shape (batch,).
    Gradients flow if the caller has them enabled.
    """
    # Position t predicts token t+1, so drop the last position and the first target.
    # The first token is always a prompt token, so no response token is lost.
    targets = input_ids[:, 1:]
    mask = response_mask[:, 1:].bool()
    response_logits = response_token_logits(model, input_ids, attention_mask, mask).float()
    token_logprobs = -F.cross_entropy(response_logits, targets[mask], reduction="none")
    row_of_token = mask.nonzero()[:, 0]
    totals = torch.zeros(input_ids.shape[0], dtype=token_logprobs.dtype, device=token_logprobs.device)
    return totals.index_add(0, row_of_token, token_logprobs)


def pair_logprobs(
    model: torch.nn.Module,
    prompt_ids: list[list[int]],
    chosen_ids: list[list[int]],
    rejected_ids: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-probabilities of chosen and rejected responses under the policy and the reference.

    Returns (policy_chosen, policy_rejected, ref_chosen, ref_rejected), each (batch,).
    The reference is the same model with the LoRA adapter disabled and never gets gradients.
    The policy terms carry gradients if the caller has them enabled.
    """
    chosen = {k: v.to(device) for k, v in build_batch(prompt_ids, chosen_ids, pad_token_id).items()}
    rejected = {k: v.to(device) for k, v in build_batch(prompt_ids, rejected_ids, pad_token_id).items()}
    with autocast(device):
        with torch.no_grad(), model.disable_adapter():
            ref_chosen = sequence_logprobs(model, **chosen)
            ref_rejected = sequence_logprobs(model, **rejected)
        policy_chosen = sequence_logprobs(model, **chosen)
        policy_rejected = sequence_logprobs(model, **rejected)
    return policy_chosen, policy_rejected, ref_chosen, ref_rejected


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DPO loss: -log sigmoid(beta * [(pi_c - ref_c) - (pi_r - ref_r)]), averaged over the batch.

    Inputs are sequence log-probabilities of shape (batch,). Returns the scalar loss and
    a dict of detached scalar metrics.
    """
    # Implicit reward of a response: beta * log(pi(y|x) / pi_ref(y|x)).
    reward_chosen = beta * (policy_chosen - ref_chosen)
    reward_rejected = beta * (policy_rejected - ref_rejected)
    margin = reward_chosen - reward_rejected
    loss = -F.logsigmoid(margin).mean()

    # A tie counts as half correct, so an untrained policy (margin exactly 0) scores 0.5, not 0.
    correct = (margin > 0).float() + 0.5 * (margin == 0).float()
    metrics = {
        "reward_chosen": reward_chosen.detach().mean(),
        "reward_rejected": reward_rejected.detach().mean(),
        "reward_margin": margin.detach().mean(),
        "preference_accuracy": correct.mean(),
    }
    return loss, metrics
