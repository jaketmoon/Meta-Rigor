from __future__ import annotations

"""Local Responses transport compatibility lane for the Codex CLI V3 lightweight-format experiment."""


import asyncio
import json
from collections import Counter
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit
import httpx
from metarigor.local_run import canonical_json
PROXY_PREFIX = "/v1"


PROXY_ENDPOINT = "/v1/responses"


FORWARDED_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "content-type",
        "originator",
        "session-id",
        "thread-id",
        "user-agent",
        "x-client-request-id",
        "x-codex-beta-features",
        "x-codex-turn-metadata",
    }
)


class LoopbackResponsesProxy:
    """In-process HTTP forwarder for Codex Responses POST requests only; never logs request bodies."""

    def __init__(
        self,
        *,
        upstream_base_url: str,
        api_key: str,
        upstream_model_id: str | None = None,
    ) -> None:
        parsed = urlsplit(upstream_base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("proxy upstream must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("proxy upstream must not contain query or fragment")
        self._upstream_base_url = upstream_base_url.rstrip("/")
        self._upstream_prefix = parsed.path.rstrip("/")
        self._api_key = api_key
        self._upstream_model_id = upstream_model_id
        self._server: asyncio.AbstractServer | None = None
        self._client: httpx.AsyncClient | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._local_base_url: str | None = None
        self._request_count = 0
        self._response_count = 0
        self._request_body_bytes = 0
        self._response_body_bytes = 0
        self._status_counts: Counter[str] = Counter()
        self._proxy_error_count = 0
        self._request_model_rewrite_count = 0

    @property
    def local_base_url(self) -> str:
        if self._local_base_url is None:
            raise RuntimeError("loopback proxy has not started")
        return self._local_base_url

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("loopback proxy already started")
        self._client = httpx.AsyncClient(
            trust_env=False,
            http2=False,
            timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0),
        )
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        socket = self._server.sockets[0]
        port = int(socket.getsockname()[1])
        self._local_base_url = f"http://127.0.0.1:{port}{PROXY_PREFIX}"

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._handle(reader, writer))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header_bytes = await reader.readuntil(b"\r\n\r\n")
            request_line, *header_lines = header_bytes[:-4].split(b"\r\n")
            request_parts = request_line.decode("latin-1").split(" ", 2)
            if len(request_parts) != 3:
                await self._write_error(writer, HTTPStatus.BAD_REQUEST)
                return
            method, target, _http_version = request_parts
            request_headers = self._parse_headers(header_lines)
            path = urlsplit(target).path
            if method != "POST" or path != PROXY_ENDPOINT:
                await self._write_error(writer, HTTPStatus.NOT_FOUND)
                return
            body = self._rewrite_model_field(
                await self._read_request_body(reader, request_headers)
            )
            self._request_count += 1
            self._request_body_bytes += len(body)
            client = self._client
            if client is None:
                raise RuntimeError("loopback proxy client is not ready")
            query = urlsplit(target).query
            upstream_url = f"{self._upstream_base_url}/responses"
            if query:
                upstream_url += f"?{query}"
            headers = {
                name: value
                for name, value in request_headers.items()
                if name in FORWARDED_REQUEST_HEADERS
            }
            headers.update(
                {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": request_headers.get("content-type", "application/json"),
                    "Accept": request_headers.get("accept", "text/event-stream"),
                    "Accept-Encoding": "identity",
                }
            )
            async with client.stream(
                "POST", upstream_url, headers=headers, content=body
            ) as response:
                self._response_count += 1
                self._status_counts[str(response.status_code)] += 1
                await self._write_response_headers(writer, response)
                async for chunk in response.aiter_raw():
                    if not chunk:
                        continue
                    self._response_body_bytes += len(chunk)
                    writer.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    writer.write(chunk)
                    writer.write(b"\r\n")
                    await writer.drain()
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            self._proxy_error_count += 1
        except Exception:
            self._proxy_error_count += 1
            if not writer.is_closing():
                try:
                    await self._write_error(writer, HTTPStatus.BAD_GATEWAY)
                except (ConnectionError, RuntimeError):
                    pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    @staticmethod
    def _parse_headers(lines: list[bytes]) -> dict[str, str]:
        headers: dict[str, str] = {}
        for line in lines:
            if b":" not in line:
                continue
            name, value = line.split(b":", 1)
            headers[name.decode("latin-1").strip().casefold()] = value.decode("latin-1").strip()
        return headers

    @staticmethod
    async def _read_request_body(
        reader: asyncio.StreamReader, headers: dict[str, str]
    ) -> bytes:
        transfer_encoding = headers.get("transfer-encoding", "").casefold()
        if "chunked" in transfer_encoding:
            chunks: list[bytes] = []
            while True:
                size_line = await reader.readuntil(b"\r\n")
                size = int(size_line[:-2].split(b";", 1)[0], 16)
                if size == 0:
                    await reader.readuntil(b"\r\n")
                    break
                chunks.append(await reader.readexactly(size))
                await reader.readexactly(2)
            return b"".join(chunks)
        raw_length = headers.get("content-length")
        if raw_length is None:
            raise ValueError("Responses request has no content-length")
        length = int(raw_length)
        if length < 0 or length > 64 * 1024 * 1024:
            raise ValueError("Responses request body is outside proxy bound")
        return await reader.readexactly(length)

    def _rewrite_model_field(self, body: bytes) -> bytes:
        if not self._upstream_model_id:
            return body
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("Responses request body must be a JSON object")
        if payload.get("model") != self._upstream_model_id:
            payload["model"] = self._upstream_model_id
            self._request_model_rewrite_count += 1
        return canonical_json(payload)

    @staticmethod
    async def _write_response_headers(
        writer: asyncio.StreamWriter, response: httpx.Response
    ) -> None:
        try:
            reason = HTTPStatus(response.status_code).phrase
        except ValueError:
            reason = ""
        writer.write(f"HTTP/1.1 {response.status_code} {reason}\r\n".encode("latin-1"))
        for name in ("content-type", "cache-control", "x-request-id", "retry-after"):
            value = response.headers.get(name)
            if value is not None:
                writer.write(f"{name}: {value}\r\n".encode("latin-1"))
        writer.write(b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
        await writer.drain()

    @staticmethod
    async def _write_error(writer: asyncio.StreamWriter, status: HTTPStatus) -> None:
        body = json.dumps({"error": status.phrase}).encode("utf-8")
        writer.write(
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n".encode("latin-1")
        )
        writer.write(body)
        await writer.drain()

    async def close(self) -> None:
        server = self._server
        if server is not None:
            server.close()
            await server.wait_closed()
            self._server = None
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "transport": "IN_PROCESS_LOOPBACK_HTTP_FORWARDER",
            "provider_base_url": self.local_base_url,
            "upstream_base_url": self._upstream_base_url,
            "allowed_method": "POST",
            "allowed_path": PROXY_ENDPOINT,
            "credential_persistence": "NONE_IN_MEMORY_FORWARD_ONLY",
            "request_count": self._request_count,
            "response_count": self._response_count,
            "request_body_bytes": self._request_body_bytes,
            "response_body_bytes": self._response_body_bytes,
            "upstream_status_counts": dict(sorted(self._status_counts.items())),
            "proxy_error_count": self._proxy_error_count,
            "upstream_model_id": self._upstream_model_id,
            "request_model_rewrite_count": self._request_model_rewrite_count,
        }

