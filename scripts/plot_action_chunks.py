from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_actions(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix == ".npy":
        import numpy as np

        data = np.load(path)
    else:
        raise ValueError(f"Unsupported action file suffix: {path.suffix}")
    if isinstance(data, dict):
        data = data.get("actions", data.get("action", data))
    return torch.as_tensor(data, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot action chunk dimensions over time.")
    parser.add_argument("--actions", required=True, help="Path to .pt, .npy, or .json action tensor.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--dims", type=int, nargs="*", default=None)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    actions = load_actions(Path(args.actions))
    if actions.ndim == 3:
        actions = actions[args.batch_index]
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape [T,D] or [B,T,D], got {tuple(actions.shape)}")

    dims = args.dims if args.dims is not None else list(range(actions.shape[-1]))
    steps = torch.arange(actions.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    for dim in dims:
        ax.plot(steps, actions[:, dim], label=f"dim{dim}")
    ax.set_xlabel("chunk step")
    ax.set_ylabel("action value")
    ax.set_title("MiniVLA action chunk")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
