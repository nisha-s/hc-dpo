"""Load the tokenizer and the LoRA policy.

There is one model in memory. The policy is the base model with the LoRA adapter
on; the reference is the same model inside `with model.disable_adapter():`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer with right padding (generation code left-pads itself)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy(config: dict[str, Any], device: torch.device, adapter_dir: str | Path | None = None) -> PeftModel:
    """Load the base model plus LoRA: a fresh adapter, or saved adapter weights when `adapter_dir` is given.

    Weights stay fp32; bf16 is applied by autocast where supported, not by casting the model.
    """
    base = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.float32).to(device)
    if adapter_dir is not None:
        return PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base, peft_config)
