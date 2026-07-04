from __future__ import annotations

from typing import Any

import torch

from minivla.configuration_minivla import MiniVLAConfig
from minivla.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


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
            tokenizer = self._load_tokenizer()
            tokens = tokenizer(
                task,
                padding="max_length",
                truncation=True,
                max_length=self.config.tokenizer_max_length,
                return_tensors="pt",
            )
            out[OBS_LANGUAGE_TOKENS] = tokens["input_ids"]
            out[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].bool()

        if device is None:
            device = self.config.device
        if device is not None:
            for key, value in list(out.items()):
                if torch.is_tensor(value):
                    out[key] = value.to(device)
        return out

