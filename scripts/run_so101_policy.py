from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivla.postprocess import ActionPostProcessor, LatencyMonitor, PostProcessConfig


def load_json(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Observation JSON must contain an object")
    return data


def post_policy(url: str, payload: dict[str, Any], timeout: float) -> torch.Tensor:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "actions" in result:
        return torch.as_tensor(result["actions"], dtype=torch.float32)
    if "action" in result:
        return torch.as_tensor(result["action"], dtype=torch.float32)
    raise KeyError("Policy response must contain 'actions' or 'action'")


class SO101Adapter:
    """Explicit hardware boundary for SO-101 integration.

    Replace these methods with the local SO-101/LeRobot robot API calls. Keeping
    this adapter small makes it clear which part of the code touches hardware.
    """

    def read_observation(self) -> dict[str, Any]:
        raise NotImplementedError("Connect this method to the SO-101 camera/state API")

    def write_action(self, action: torch.Tensor) -> None:
        raise NotImplementedError("Connect this method to the SO-101 command API")

    def read_joints(self) -> torch.Tensor | None:
        return None


def should_stop(stop_file: str | None) -> bool:
    return stop_file is not None and Path(stop_file).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101-side MiniVLA policy client with safety filtering.")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8010/infer")
    parser.add_argument("--observation-json", default=None, help="Static observation payload for dry-run/debug.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override FM denoise steps sent to policy server.")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--server-select-action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop-file", default="/tmp/minivla_stop")

    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--action-min", type=float, default=-1.0)
    parser.add_argument("--action-max", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--joint-min", type=float, default=None)
    parser.add_argument("--joint-max", type=float, default=None)
    parser.add_argument("--action-mode", choices=["delta", "absolute"], default="delta")
    args = parser.parse_args()

    period = 1.0 / args.hz
    postprocessor = ActionPostProcessor(
        PostProcessConfig(
            action_min=args.action_min,
            action_max=args.action_max,
            max_delta=args.max_delta,
            joint_min=args.joint_min,
            joint_max=args.joint_max,
            action_mode=args.action_mode,
            ema_alpha=args.ema_alpha,
        )
    )
    latency = LatencyMonitor()
    adapter = SO101Adapter()
    static_observation = load_json(args.observation_json)

    for cycle in range(args.max_cycles):
        if should_stop(args.stop_file):
            print(f"stop_file_detected={args.stop_file}")
            break

        start = time.perf_counter()
        if static_observation:
            observation = dict(static_observation)
            current_joints = None
        elif args.dry_run:
            raise RuntimeError("--observation-json is required for dry-run without a hardware adapter")
        else:
            observation = adapter.read_observation()
            current_joints = adapter.read_joints()

        payload = dict(observation)
        if args.num_steps is not None:
            payload["num_steps"] = args.num_steps
        if args.server_select_action:
            payload["select_action"] = True
        actions = post_policy(args.policy_url, payload, timeout=args.request_timeout)
        action = postprocessor.select_action(actions)
        action, post_info = postprocessor(action, current_joints=current_joints)

        if args.dry_run:
            print(f"cycle={cycle} action={action.tolist()} postprocess={post_info.to_dict()}")
        else:
            adapter.write_action(action)

        elapsed = time.perf_counter() - start
        latency.record(elapsed)
        sleep_s = period - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

    summary = latency.summary()
    if summary["count"]:
        print(
            f"cycles={summary['count']} "
            f"mean_loop_ms={summary['mean_ms']:.3f} "
            f"mean_hz={summary['mean_hz']:.2f} "
            f"max_loop_ms={summary['max_ms']:.3f}"
        )


if __name__ == "__main__":
    main()
