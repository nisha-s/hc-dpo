"""Build, save and load the fixed data subsets, and turn text into model batches.

Source: Anthropic/hh-rlhf, helpful-base. Each raw example is two transcripts,
`chosen` and `rejected`, that share a conversation and differ in the final
assistant response.

Formatting choices (also in the README):
- "Human:" / "Assistant:" turns become chat messages and the prompt is built with
  the tokenizer's chat template, ending in the assistant generation prompt.
- No system message is passed, so Qwen's template adds its default system prompt.
- A response is its text followed by the end-of-turn token, which counts as a
  response token.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from hcdpo.model import load_tokenizer
from hcdpo.utils import load_config, read_json, read_jsonl, write_json, write_jsonl

DATASET_NAME = "Anthropic/hh-rlhf"
DATASET_SUBSET = "helpful-base"
ASSISTANT_MARKER = "\n\nAssistant:"
TURN_PATTERN = re.compile(r"\n\n(Human|Assistant):")
END_OF_TURN = "<|im_end|>"

SUBSET_FILES = {
    "train": "train.jsonl",
    "kl_prompts": "kl_prompts.jsonl",
    "eval_pairs": "eval_pairs.jsonl",
    "gen_prompts": "gen_prompts.jsonl",
}


# ---------------------------------------------------------------------------
# Parsing raw transcripts
# ---------------------------------------------------------------------------


def split_final_response(transcript: str) -> tuple[str, str] | None:
    """Split a transcript into (conversation up to and including the final "Assistant:", final response)."""
    index = transcript.rfind(ASSISTANT_MARKER)
    if index == -1:
        return None
    cut = index + len(ASSISTANT_MARKER)
    return transcript[:cut], transcript[cut:].strip()


def parse_turns(conversation: str) -> list[dict[str, str]] | None:
    """Parse a conversation ending in an empty "Assistant:" turn into chat messages.

    Returns None unless the turns strictly alternate user/assistant, start and end
    with the user, and are all non-empty.
    """
    parts = TURN_PATTERN.split(conversation)
    if parts[0].strip():
        return None
    messages = [
        {"role": "user" if speaker == "Human" else "assistant", "content": content.strip()}
        for speaker, content in zip(parts[1::2], parts[2::2])
    ]
    if not messages or messages[-1] != {"role": "assistant", "content": ""}:
        return None
    messages = messages[:-1]
    if len(messages) % 2 == 0:
        return None
    for i, message in enumerate(messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected_role or not message["content"]:
            return None
    return messages


def build_example(
    row: dict[str, str], example_id: str, tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any] | None:
    """Turn one raw row into an example dict, or None if the row should be dropped."""
    chosen_split = split_final_response(row["chosen"])
    rejected_split = split_final_response(row["rejected"])
    if chosen_split is None or rejected_split is None:
        return None
    conversation, chosen = chosen_split
    rejected_conversation, rejected = rejected_split
    if conversation != rejected_conversation:
        return None
    if not chosen or not rejected or chosen == rejected:
        return None
    messages = parse_turns(conversation)
    if messages is None:
        return None
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {
        "id": example_id,
        "messages": messages,
        "prompt": prompt,
        "last_human": messages[-1]["content"],
        "chosen": chosen,
        "rejected": rejected,
    }


def process_split(
    rows: Any,
    split_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> list[dict[str, Any]]:
    """Parse every row of a split and keep the examples that pass the token-length filter."""
    examples = []
    for index, row in enumerate(rows):
        example = build_example(row, f"{split_name}-{index}", tokenizer)
        if example is not None:
            examples.append(example)

    def token_counts(texts: list[str]) -> list[int]:
        return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]

    prompt_lengths = token_counts([e["prompt"] for e in examples])
    # +1 for the end-of-turn token appended to every response.
    chosen_lengths = [n + 1 for n in token_counts([e["chosen"] for e in examples])]
    rejected_lengths = [n + 1 for n in token_counts([e["rejected"] for e in examples])]

    kept = [
        example
        for example, n_prompt, n_chosen, n_rejected in zip(
            examples, prompt_lengths, chosen_lengths, rejected_lengths
        )
        if n_prompt <= max_prompt_tokens
        and n_chosen <= max_response_tokens
        and n_rejected <= max_response_tokens
    ]
    print(f"{split_name}: {len(rows)} raw -> {len(examples)} parsed -> {len(kept)} within token limits")
    return kept


# ---------------------------------------------------------------------------
# Building, saving and loading the fixed subsets
# ---------------------------------------------------------------------------


def prompt_only(example: dict[str, Any]) -> dict[str, Any]:
    """Drop the responses from an example, keeping what generation and scoring need."""
    return {key: example[key] for key in ("id", "messages", "prompt", "last_human")}


def build_subsets(
    data_config: dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> dict[str, list[dict[str, Any]]]:
    """Download hh-rlhf helpful-base, filter it, and draw the four fixed subsets."""
    dataset = load_dataset(DATASET_NAME, data_dir=DATASET_SUBSET)
    limits = (data_config["max_prompt_tokens"], data_config["max_response_tokens"])
    train_pool = process_split(dataset["train"], "train", tokenizer, *limits)
    test_pool = process_split(dataset["test"], "test", tokenizer, *limits)

    n_train, n_kl = data_config["n_train"], data_config["n_kl_prompts"]
    n_eval, n_gen = data_config["n_eval_pairs"], data_config["n_gen_prompts"]
    if len(train_pool) < n_train:
        raise ValueError(f"Need {n_train} training pairs, only {len(train_pool)} pass the filters")
    if len(test_pool) < n_eval + n_gen:
        raise ValueError(f"Need {n_eval + n_gen} test examples, only {len(test_pool)} pass the filters")

    rng = random.Random(data_config["seed"])
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    train = train_pool[:n_train]
    return {
        "train": train,
        "kl_prompts": [prompt_only(e) for e in rng.sample(train, n_kl)],
        "eval_pairs": test_pool[:n_eval],
        # Sliced after the eval pairs, so the two test subsets are disjoint.
        "gen_prompts": [prompt_only(e) for e in test_pool[n_eval : n_eval + n_gen]],
    }


def save_subsets(
    subsets: dict[str, list[dict[str, Any]]], data_dir: str | Path, meta: dict[str, Any]
) -> None:
    """Write each subset to JSONL plus a meta.json describing how it was built."""
    data_dir = Path(data_dir)
    for name, filename in SUBSET_FILES.items():
        write_jsonl(data_dir / filename, subsets[name])
    write_json(data_dir / "meta.json", meta)


def subsets_exist(data_dir: str | Path) -> bool:
    """True if every subset file and meta.json are on disk."""
    data_dir = Path(data_dir)
    names = list(SUBSET_FILES.values()) + ["meta.json"]
    return all((data_dir / name).exists() for name in names)


def load_subsets(data_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load the saved subsets. Run `python -m hcdpo.data` first if they are missing."""
    if not subsets_exist(data_dir):
        raise FileNotFoundError(
            f"No prepared data in {data_dir}. Run: python -m hcdpo.data --config <config>"
        )
    return {name: read_jsonl(Path(data_dir) / filename) for name, filename in SUBSET_FILES.items()}


def check_data_matches_config(config: dict[str, Any]) -> None:
    """Fail loudly if the data on disk was built with a different model or data config."""
    meta = read_json(Path(config["data"]["dir"]) / "meta.json")
    if meta["model_name"] != config["model_name"] or meta["data_config"] != config["data"]:
        raise ValueError(
            f"Data in {config['data']['dir']} was built with a different model or data config. "
            "Rebuild it with: python -m hcdpo.data --config <config> --force"
        )


# ---------------------------------------------------------------------------
# Text -> token ids -> padded batches
# ---------------------------------------------------------------------------


def end_of_turn_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Token id of the end-of-turn marker that closes every assistant response."""
    token_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer has no {END_OF_TURN} token; this code assumes a ChatML model")
    return token_id


def tokenize_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str) -> list[int]:
    """Token ids of an already-templated prompt (the template supplies all special tokens)."""
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def tokenize_response(tokenizer: PreTrainedTokenizerBase, response: str) -> list[int]:
    """Token ids of a response followed by the end-of-turn token."""
    return tokenizer(response, add_special_tokens=False)["input_ids"] + [end_of_turn_id(tokenizer)]


def build_batch(
    prompt_ids: list[list[int]], response_ids: list[list[int]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Concatenate prompt + response per row and right-pad to a batch.

    Returns `input_ids`, `attention_mask` and `response_mask`, all (batch, seq_len).
    `response_mask` is 1 exactly on response tokens: 0 on prompt and padding.
    """
    lengths = [len(p) + len(r) for p, r in zip(prompt_ids, response_ids)]
    max_length = max(lengths)
    batch_size = len(prompt_ids)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    for row, (p, r) in enumerate(zip(prompt_ids, response_ids)):
        input_ids[row, : len(p) + len(r)] = torch.tensor(p + r, dtype=torch.long)
        attention_mask[row, : len(p) + len(r)] = 1
        response_mask[row, len(p) : len(p) + len(r)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "response_mask": response_mask}


def tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase, examples: list[dict[str, Any]]
) -> dict[str, list[list[int]]]:
    """Tokenize preference pairs once, up front: lists of prompt, chosen and rejected token ids."""
    return {
        "prompt": [tokenize_prompt(tokenizer, e["prompt"]) for e in examples],
        "chosen": [tokenize_response(tokenizer, e["chosen"]) for e in examples],
        "rejected": [tokenize_response(tokenizer, e["rejected"]) for e in examples],
    }


def encode_pairs(
    tokenizer: PreTrainedTokenizerBase, prompts: list[str], responses: list[str]
) -> dict[str, torch.Tensor]:
    """Tokenize (prompt, response) text pairs and build a padded batch."""
    prompt_ids = [tokenize_prompt(tokenizer, prompt) for prompt in prompts]
    response_ids = [tokenize_response(tokenizer, response) for response in responses]
    return build_batch(prompt_ids, response_ids, tokenizer.pad_token_id)


# ---------------------------------------------------------------------------
# CLI: python -m hcdpo.data --config configs/dpo.yaml
# ---------------------------------------------------------------------------


def main() -> None:
    """Build the subsets once and save them; later runs reuse the files."""
    parser = argparse.ArgumentParser(description="Prepare the fixed hh-rlhf subsets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    parser.add_argument("--force", action="store_true", help="rebuild even if the files exist")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    data_dir = config["data"]["dir"]

    if subsets_exist(data_dir) and not args.force:
        print(f"Data already prepared in {data_dir} (use --force to rebuild).")
    else:
        tokenizer = load_tokenizer(config["model_name"])
        subsets = build_subsets(config["data"], tokenizer)
        meta = {
            "dataset": f"{DATASET_NAME}/{DATASET_SUBSET}",
            "model_name": config["model_name"],
            "data_config": config["data"],
            "sizes": {name: len(rows) for name, rows in subsets.items()},
        }
        save_subsets(subsets, data_dir, meta)
        print(f"Saved subsets to {data_dir}")

    check_data_matches_config(config)
    subsets = load_subsets(data_dir)
    print("Sizes: " + ", ".join(f"{name}={len(rows)}" for name, rows in subsets.items()))
    example = subsets["train"][0]
    print("\n--- first training example ---")
    print("PROMPT:\n" + example["prompt"])
    print("CHOSEN:   " + example["chosen"][:200])
    print("REJECTED: " + example["rejected"][:200])


if __name__ == "__main__":
    main()
