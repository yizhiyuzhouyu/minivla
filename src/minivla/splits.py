from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_episode_split(path: str | Path | None, split_name: str | None = None) -> list[int] | None:
    """Load episode ids from a JSON split file.

    Supported formats:

    - `[0, 1, 2]`
    - `{"episodes": [0, 1, 2]}`
    - `{"val": [0, 1], "test": [2]}`
    - `{"splits": {"val": [0, 1], "test": [2]}}`
    """

    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    if isinstance(data, list):
        return [int(item) for item in data]
    if not isinstance(data, dict):
        raise ValueError(f"Split file must contain a list or object, got {type(data).__name__}")
    if split_name is not None:
        if split_name in data:
            return [int(item) for item in data[split_name]]
        splits = data.get("splits")
        if isinstance(splits, dict) and split_name in splits:
            return [int(item) for item in splits[split_name]]
        raise KeyError(f"Split {split_name!r} not found in {path}")
    if "episodes" in data:
        return [int(item) for item in data["episodes"]]
    if len(data) == 1:
        only_value = next(iter(data.values()))
        if isinstance(only_value, list):
            return [int(item) for item in only_value]
    raise ValueError("Split file object must contain 'episodes', a requested split, or a single list-valued key")
