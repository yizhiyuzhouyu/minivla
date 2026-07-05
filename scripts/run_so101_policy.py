from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch


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


class EMASmoother:
    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("EMA alpha must be in [0, 1]")
        self.alpha = alpha
        self.value: torch.Tensor | None = None

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        if self.value is None:
            self.value = action.clone()
        else:
            self.value = self.alpha * action + (1.0 - self.alpha) * self.value
        return self.value.clone()


class ActionSafetyFilter:
    def __init__(
        self,
        action_min: float,
        action_max: float,
        max_delta: float | None,
        joint_min: float | None,
        joint_max: float | None,
        action_mode: str,
    ) -> None:
        self.action_min = action_min
        self.action_max = action_max
        self.max_delta = max_delta
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.action_mode = action_mode
        self.prev_action: torch.Tensor | None = None

    def __call__(self, action: torch.Tensor, current_joints: torch.Tensor | None = None) -> torch.Tensor:
        action = action.clamp(self.action_min, self.action_max)
        if self.max_delta is not None and self.prev_action is not None:
            delta = (action - self.prev_action).clamp(-self.max_delta, self.max_delta)
            action = self.prev_action + delta
        if current_joints is not None and self.joint_min is not None and self.joint_max is not None:
            if self.action_mode == "delta":
                next_joints = current_joints + action[: current_joints.numel()]
            else:
                next_joints = action[: current_joints.numel()]
            next_joints = next_joints.clamp(self.joint_min, self.joint_max)
            if self.action_mode == "delta":
                action = action.clone()
                action[: current_joints.numel()] = next_joints - current_joints
            else:
                action = action.clone()
                action[: current_joints.numel()] = next_joints
        self.prev_action = action.clone()
        return action


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
    smoother = EMASmoother(args.ema_alpha)
    safety = ActionSafetyFilter(
        action_min=args.action_min,
        action_max=args.action_max,
        max_delta=args.max_delta,
        joint_min=args.joint_min,
        joint_max=args.joint_max,
        action_mode=args.action_mode,
    )
    adapter = SO101Adapter()
    static_observation = load_json(args.observation_json)

    latencies_ms: list[float] = []
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
        if actions.ndim == 3:
            action = actions[0, 0]
        elif actions.ndim == 2:
            if args.server_select_action:
                action = actions[0]
            else:
                action = actions[0]
        elif actions.ndim == 1:
            action = actions
        else:
            raise ValueError(f"Expected policy action tensor, got {tuple(actions.shape)}")

        action = smoother(action)
        action = safety(action, current_joints=current_joints)

        if args.dry_run:
            print(f"cycle={cycle} action={action.tolist()}")
        else:
            adapter.write_action(action)

        elapsed = time.perf_counter() - start
        latencies_ms.append(elapsed * 1000.0)
        sleep_s = period - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

    if latencies_ms:
        mean_ms = sum(latencies_ms) / len(latencies_ms)
        print(f"cycles={len(latencies_ms)} mean_loop_ms={mean_ms:.3f} mean_hz={1000.0 / mean_ms:.2f}")


if __name__ == "__main__":
    main()
