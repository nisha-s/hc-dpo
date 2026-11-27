"""Prompt and padding tokens must be excluded from sequence log-probabilities.

Uses a tiny fake model (next-token logits looked up from a fixed random table), so no
download is needed and the expected values can be computed by hand.
"""

from types import SimpleNamespace

import pytest
import torch

from hcdpo.data import build_batch
from hcdpo.losses import sequence_logprobs

VOCAB_SIZE = 11
PAD_ID = 0


class BigramModel(torch.nn.Module):
    """Fake language model: the logits for the next token depend only on the current token."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.table = torch.randn(VOCAB_SIZE, VOCAB_SIZE, generator=generator)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.table[input_ids])


class PeftLikeBigramModel(torch.nn.Module):
    """The same bigram model, laid out like a PEFT-wrapped Hugging Face model.

    The body returns one-hot hidden states and the output layer maps them through the
    table, so this exercises the path that applies the output layer to response positions only.
    """

    def __init__(self) -> None:
        super().__init__()
        self.table = BigramModel().table

    def get_base_model(self) -> "PeftLikeBigramModel":
        return self

    def base_model(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        hidden = torch.nn.functional.one_hot(input_ids, VOCAB_SIZE).float()
        return SimpleNamespace(last_hidden_state=hidden)

    def get_output_embeddings(self):
        return lambda hidden: hidden @ self.table

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        raise AssertionError("the full-logits path must not be used for a PEFT-style model")


def manual_response_logprob(model: BigramModel, prompt: list[int], response: list[int]) -> float:
    """Sum of log p(token | previous token) over the response tokens, one token at a time."""
    sequence = prompt + response
    total = 0.0
    for position in range(len(prompt), len(sequence)):
        log_probs = torch.log_softmax(model.table[sequence[position - 1]], dim=-1)
        total += log_probs[sequence[position]].item()
    return total


def score(model: BigramModel, prompts: list[list[int]], responses: list[list[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    batch = build_batch(prompts, responses, pad_id)
    return sequence_logprobs(model, batch["input_ids"], batch["attention_mask"], batch["response_mask"])


def test_build_batch_masks() -> None:
    batch = build_batch([[1, 2, 3], [4]], [[5, 6], [7, 8, 9]], PAD_ID)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 5, 6], [4, 7, 8, 9, PAD_ID]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert batch["response_mask"].tolist() == [[0, 0, 0, 1, 1], [0, 1, 1, 1, 0]]


def test_logprobs_cover_response_tokens_only() -> None:
    model = BigramModel()
    prompts = [[1, 2, 3], [4]]
    responses = [[5, 6], [7, 8, 9]]
    result = score(model, prompts, responses)
    for row in range(2):
        expected = manual_response_logprob(model, prompts[row], responses[row])
        assert result[row].item() == pytest.approx(expected, abs=1e-5)


def test_padding_does_not_change_logprobs() -> None:
    model = BigramModel()
    alone = score(model, [[1, 2]], [[3, 4]])
    # Same row next to a much longer one, so it is heavily padded; and with a different pad token.
    padded = score(model, [[1, 2], [5, 6, 7, 8]], [[3, 4], [9, 10, 9, 10, 9]])
    other_pad = score(model, [[1, 2], [5, 6, 7, 8]], [[3, 4], [9, 10, 9, 10, 9]], pad_id=7)
    assert padded[0].item() == pytest.approx(alone[0].item(), abs=1e-6)
    assert other_pad[0].item() == pytest.approx(alone[0].item(), abs=1e-6)


def test_prompt_tokens_do_not_contribute() -> None:
    model = BigramModel()
    # Two prompts that differ everywhere except the last token. In a bigram model the
    # response depends only on that last token, so any difference would come from
    # (wrongly) counting prompt tokens.
    first = score(model, [[1, 2, 3]], [[4, 5]])
    second = score(model, [[6, 7, 8, 9, 3]], [[4, 5]])
    assert first.item() == pytest.approx(second.item(), abs=1e-6)


def test_peft_style_path_matches_full_logits_path() -> None:
    prompts = [[1, 2, 3], [4], [5, 6, 7, 8]]
    responses = [[5, 6], [7, 8, 9], [9, 10, 9, 10, 9]]
    full = score(BigramModel(), prompts, responses)
    selected = score(PeftLikeBigramModel(), prompts, responses)
    assert torch.allclose(full, selected, atol=1e-5)


def test_empty_response_mask_gives_zero() -> None:
    model = BigramModel()
    batch = build_batch([[1, 2, 3]], [[4, 5]], PAD_ID)
    empty_mask = torch.zeros_like(batch["response_mask"])
    result = sequence_logprobs(model, batch["input_ids"], batch["attention_mask"], empty_mask)
    assert result.item() == 0.0
