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


def post_policy(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "actions" not in result and "action" not in result:
        raise KeyError("Policy response must contain 'actions' or 'action'")
    return result


class SO101Adapter:
    def read_observation(self) -> dict[str, Any]:
        raise NotImplementedError("Connect this method to the SO-101 camera/state API")

    def write_action(self, action: torch.Tensor) -> None:
        raise NotImplementedError("Connect this method to the SO-101 command API")

    def read_joints(self) -> torch.Tensor | None:
        return None


def tensor_from_policy_result(result: dict[str, Any]) -> torch.Tensor:
    if "actions" in result:
        actions = torch.as_tensor(result["actions"], dtype=torch.float32)
    else:
        actions = torch.as_tensor(result["action"], dtype=torch.float32)
    if actions.ndim == 3:
        return actions[0, 0]
    if actions.ndim == 2:
        return actions[0]
    if actions.ndim == 1:
        return actions
    raise ValueError(f"Expected policy action tensor, got {tuple(actions.shape)}")


def scalar_or_list(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except TypeError:
            return str(value)
    return str(value)


def should_stop(stop_file: str | None) -> bool:
    return stop_file is not None and Path(stop_file).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Log MiniVLA rollout steps for post-SFT failure mining and RL data.")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8010/infer")
    parser.add_argument("--observation-json", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--server-select-action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop-file", default="/tmp/minivla_stop")
    parser.add_argument("--human-label", default=None, help="Optional label applied to every logged step.")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--trajectory-id", default=None)
    parser.add_argument("--operator", default=None)
    parser.add_argument("--save-observation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--video-key", default="video_path")
    parser.add_argument("--success", type=float, default=None)
    parser.add_argument("--stable-grasp", type=float, default=None)
    parser.add_argument("--collision-free", type=float, default=None)
    parser.add_argument("--smooth-action", type=float, default=None)
    parser.add_argument("--quality-score", type=float, default=None)

    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--action-min", type=float, default=-1.0)
    parser.add_argument("--action-max", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--joint-min", type=float, default=None)
    parser.add_argument("--joint-max", type=float, default=None)
    parser.add_argument("--action-mode", choices=["delta", "absolute"], default="delta")
    args = parser.parse_args()

    output = Path(args.output) if args.output is not None else Path("outputs/rollouts") / f"rollout_{int(time.time())}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    static_observation = load_json(args.observation_json)
    adapter = SO101Adapter()
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
    period = 1.0 / args.hz

    with output.open("a", encoding="utf-8") as handle:
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
            result = post_policy(args.policy_url, payload, timeout=args.request_timeout)
            raw_action = tensor_from_policy_result(result)
            post_action, post_info = postprocessor(raw_action, current_joints=current_joints)

            if not args.dry_run:
                adapter.write_action(post_action)

            elapsed = time.perf_counter() - start
            latency_ms = latency.record(elapsed)
            record = {
                "time": time.time(),
                "cycle": cycle,
                "episode_id": args.episode_id,
                "trajectory_id": args.trajectory_id or args.episode_id,
                "operator": args.operator,
                "human_label": args.human_label,
                "observation_keys": sorted(observation.keys()),
                "raw_action": raw_action.detach().cpu().tolist(),
                "postprocessed_action": post_action.detach().cpu().tolist(),
                "postprocess": post_info.to_dict(),
                "policy_diagnostics": {
                    key: scalar_or_list(value)
                    for key, value in result.items()
                    if key not in {"actions", "action"}
                },
                "latency_ms": latency_ms,
                "dry_run": args.dry_run,
            }
            if args.save_observation:
                record["observation"] = json_safe(observation)
            video_path = args.video_path or observation.get(args.video_key)
            if video_path is not None:
                record["video"] = {"path": str(video_path)}
            labels = {
                "success": args.success,
                "stable_grasp": args.stable_grasp,
                "collision_free": args.collision_free,
                "smooth_action": args.smooth_action,
                "quality_score": args.quality_score,
            }
            labels = {key: value for key, value in labels.items() if value is not None}
            if labels:
                record["labels"] = labels
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"cycle={cycle} logged={output} latency_ms={latency_ms:.3f}")

            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    print(f"summary={latency.summary()}")


if __name__ == "__main__":
    main()
