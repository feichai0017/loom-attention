#!/usr/bin/env python3
"""Tiny local QuillCache action-sink receiver.

It accepts the gateway's planned/committed action events and prints a compact
line for each request. This is intentionally stdlib-only so it can run on any
developer machine:

    python3 tools/action_sink_mock.py --port 9090
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    server_version = "QuillCacheActionSinkMock/0.1"

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return

        phase = event.get("phase")
        route = event.get("route", {})
        plan = event.get("plan", {})
        request = event.get("request", {})
        cache_actions = event.get("cache_actions", [])
        print(
            "event"
            f" phase={phase}"
            f" request_id={route.get('request_id') or request.get('id')}"
            f" mode={plan.get('mode')}"
            f" execution={plan.get('execution_worker_id')}"
            f" prefill={plan.get('prefill_worker_id')}"
            f" decode={plan.get('decode_worker_id')}"
            f" planner_actions={len(plan.get('actions', []))}"
            f" cache_actions={len(cache_actions)}",
            flush=True,
        )
        self._send_json(200, {"status": "ok"})

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
