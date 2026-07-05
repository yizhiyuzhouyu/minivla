from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivla import MiniVLAPolicyRunner


def _tensor_to_json(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def _json_to_batch(payload: dict[str, Any]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "task" or isinstance(value, str):
            batch[key] = value
        elif isinstance(value, list):
            batch[key] = torch.tensor(value)
        else:
            batch[key] = value
    return batch


class PolicyHandler(BaseHTTPRequestHandler):
    runner: MiniVLAPolicyRunner

    def do_POST(self) -> None:
        if self.path != "/infer":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        num_steps = payload.pop("num_steps", None)
        select_action = bool(payload.pop("select_action", False))
        if select_action:
            result = self.runner.select_action(_json_to_batch(payload))
            response = {"action": _tensor_to_json(result["action"])}
        else:
            result = self.runner.infer(_json_to_batch(payload), num_steps=num_steps)
            response = {"actions": _tensor_to_json(result["actions"])}
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a MiniVLA checkpoint over a small HTTP API.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    PolicyHandler.runner = MiniVLAPolicyRunner.from_checkpoint(args.checkpoint, device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), PolicyHandler)
    print(f"serving MiniVLA policy on http://{args.host}:{args.port}/infer", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
