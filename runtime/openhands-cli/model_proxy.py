"""Minimal zero-retry Chat Completions proxy for OpenHands case containers."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


CASE_ID = os.environ["METARIGOR_CASE_ID"]
UPSTREAM = os.environ["GATEWAY_BASE_URL"].rstrip("/") + "/chat/completions"
API_KEY = os.environ["GATEWAY_API_KEY"]
REQUEST_MODEL = os.environ["REQUEST_MODEL"]
EXPECTED_RESPONSE_MODEL = os.environ["EXPECTED_RESPONSE_MODEL"]
MAX_REQUESTS = int(os.environ.get("MAX_REQUESTS", "100"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "8"))
RECORD_PATH = Path("/records/model-calls.jsonl")
ALLOWED_PATH = f"/{CASE_ID}/v1/chat/completions"
_lock = threading.Lock()
_limiter = threading.BoundedSemaphore(MAX_CONCURRENCY)
_request_count = 0


def _append_record(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    with _lock:
        with RECORD_PATH.open("ab") as stream:
            stream.write(encoded)


class Handler(BaseHTTPRequestHandler):
    server_version = "MetaRigorOpenHandsProxy/1"

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _plain(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._plain(200, "ready")
        else:
            self._plain(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        global _request_count
        if self.path != ALLOWED_PATH:
            self._plain(404, "endpoint not allowed")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._plain(400, "invalid content length")
            return
        if length < 2 or length > 20_000_000:
            self._plain(413, "invalid request size")
            return
        body = self.rfile.read(length)
        try:
            request_payload = json.loads(body)
        except (TypeError, ValueError):
            self._plain(400, "invalid JSON")
            return
        requested_model = request_payload.get("model")
        if requested_model not in {REQUEST_MODEL, f"openai/{REQUEST_MODEL}"}:
            self._plain(400, "model binding mismatch")
            return
        with _lock:
            _request_count += 1
            request_index = _request_count
        if request_index > MAX_REQUESTS:
            self._plain(429, "case request limit reached")
            return

        started = time.perf_counter()
        status = 599
        response_body = b""
        error_type: str | None = None
        with _limiter:
            upstream_request = urllib.request.Request(
                UPSTREAM,
                data=body,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(upstream_request, timeout=600) as response:
                    status = response.status
                    response_body = response.read()
            except urllib.error.HTTPError as error:
                status = error.code
                response_body = error.read()
                error_type = "HTTPError"
            except Exception as error:  # provider transport boundary
                error_type = type(error).__name__
                response_body = json.dumps({"error": "upstream transport failed"}).encode()

        response_model: str | None = None
        usage: dict[str, Any] | None = None
        try:
            response_payload = json.loads(response_body)
            if isinstance(response_payload, dict):
                response_model = response_payload.get("model")
                usage = response_payload.get("usage")
        except (TypeError, ValueError):
            pass
        if 200 <= status < 300 and response_model != EXPECTED_RESPONSE_MODEL:
            status = 502
            response_body = json.dumps({"error": "response model binding mismatch"}).encode()
            error_type = "MODEL_MISMATCH"
        _append_record(
            {
                "case_id": CASE_ID,
                "request_index": request_index,
                "status": status,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "requested_model": requested_model,
                "response_model": response_model,
                "usage": usage,
                "request_sha256": hashlib.sha256(body).hexdigest(),
                "response_sha256": hashlib.sha256(response_body).hexdigest(),
                "error_type": error_type,
            }
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


if __name__ == "__main__":
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("READY", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
