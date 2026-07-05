from __future__ import annotations

from typing import Any

import torch

from minivla.configuration_minivla import MiniVLAConfig
from minivla.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    SUBTASK_ATTENTION_MASK,
    SUBTASK_LABEL,
    SUBTASK_TOKENS,
)


class MiniVLAProcessor:
    """Small tokenizer/device processor for MiniVLA batches.

    This intentionally stays lightweight. Normalization can still be handled by
    LeRobot processors before calling this processor.
    """

    def __init__(self, config: MiniVLAConfig):
        self.config = config
        self.tokenizer = None

    def _load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def metadata(self) -> dict[str, Any]:
        tokenizer = self.tokenizer
        return {
            "processor": type(self).__name__,
            "tokenizer_name": self.config.tokenizer_name,
            "tokenizer_max_length": self.config.tokenizer_max_length,
            "pad_token_id": self.config.pad_token_id if tokenizer is None else tokenizer.pad_token_id,
            "add_newline_to_task": self.config.add_newline_to_task,
        }

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        if isinstance(text, str):
            text = [text]
        tokenizer = self._load_tokenizer()
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"].bool(),
        }

    def __call__(self, batch: dict[str, Any], device: str | torch.device | None = None) -> dict[str, Any]:
        out = dict(batch)
        if OBS_LANGUAGE_TOKENS not in out:
            task = out.get("task")
            if task is None:
                raise KeyError("MiniVLAProcessor needs either tokenized language keys or a 'task' string/list")
            if isinstance(task, str):
                task = [task]
            if self.config.add_newline_to_task:
                task = [item if item.endswith("\n") else f"{item}\n" for item in task]
            tokens = self._tokenize_text(task)
            out[OBS_LANGUAGE_TOKENS] = tokens["input_ids"]
            out[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].bool()

        if SUBTASK_LABEL in out and SUBTASK_TOKENS not in out:
            tokens = self._tokenize_text(out[SUBTASK_LABEL])
            out[SUBTASK_TOKENS] = tokens["input_ids"]
            out[SUBTASK_ATTENTION_MASK] = tokens["attention_mask"].bool()

        if device is None:
            device = self.config.device
        if device is not None:
            for key, value in list(out.items()):
                if torch.is_tensor(value):
                    out[key] = value.to(device)
        return out
