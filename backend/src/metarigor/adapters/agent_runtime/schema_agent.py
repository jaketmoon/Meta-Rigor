from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit
import httpx
from jsonschema import Draft202012Validator
from metarigor.schema_agent import SchemaAgentOutputError, SchemaAgentRequest, SchemaAgentResult, openai_chat_response_format, schema_agent_system_prompt, schema_agent_user_prompt, schema_validation_detail, strict_contract_schema
_AUTHORIZATION_PATTERN = re.compile(r"(?i)(authorization\s*:\s*)(?:bearer\s+)?[^\s,;]+")


_CREDENTIAL_PATTERN = re.compile(r"(?i)(api[_-]?key|token|bearer)(\s*[:=]?\s*)([^\s,;]+)")


_JSON_FENCE_PATTERN = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.IGNORECASE | re.DOTALL)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in pairs:
        if key in values:
            raise ValueError("duplicate JSON object key")
        values[key] = value
    return values


def _reject_non_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


class _SdkClient(Protocol):
    async def connect(self) -> None: ...

    async def query(self, prompt: str) -> None: ...

    def receive_response(self): ...

    async def disconnect(self) -> None: ...


def _default_client(options):
    from claude_agent_sdk import ClaudeSDKClient

    return ClaudeSDKClient(options)


def normalize_sdk_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("gateway base URL must be an absolute HTTP(S) origin")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path.removesuffix("/v1")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def normalize_openai_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("gateway base URL must be an absolute HTTP(S) URL")
    path = parsed.path.rstrip("/") + "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


class DirectSchemaAgent:
    """一次 direct chat-completions strict structured output；无 Agent loop。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        worker_model: str,
        reviewer_model: str,
        timeout_seconds: float = 120,
        max_output_tokens: int = 8_192,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key or not worker_model or not reviewer_model:
            raise ValueError("Direct Schema Agent requires credential and explicit models")
        if timeout_seconds <= 0 or timeout_seconds > 600:
            raise ValueError("Direct Schema Agent timeout must be between 0 and 600 seconds")
        if max_output_tokens <= 0:
            raise ValueError("Direct Schema Agent output budget must be positive")
        self._api_key = api_key
        self._base_url = normalize_openai_base_url(base_url)
        self._models = {"worker": worker_model, "reviewer": reviewer_model}
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
            http1=True,
            http2=False,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=0),
        )

    def binding(self, role: Literal["worker", "reviewer"]) -> dict[str, Any]:
        return {
            "runtime": "direct-openai-compatible-schema-v1",
            "model": self._models[role],
            "role": role,
            "thinking_mode": "disabled",
            "max_turns": 1,
            "timeout_seconds": self._timeout_seconds,
            "max_output_tokens": self._max_output_tokens,
            "structured_output_mode": "provider_json_schema",
            "provider_schema_enforcement": True,
            "local_schema_validation": "jsonschema-draft-2020-12",
            "automatic_retry_count": 0,
            "transport_retry_count": 0,
            "fallback_count": 0,
            "http_version": "HTTP/1.1",
            "keepalive_reuse": False,
            "base_url": self._base_url.rstrip("/"),
        }

    async def invoke(self, request: SchemaAgentRequest) -> SchemaAgentResult:
        Draft202012Validator.check_schema(request.output_schema)
        system_prompt = request.prompt
        if request.skill_text:
            system_prompt += "\n\nFollow this versioned Skill exactly:\n\n" + request.skill_text
        system_prompt += "\n\nReturn exactly one JSON object matching the supplied response format."
        schema_name = re.sub(r"[^a-zA-Z0-9_-]", "_", request.template_id)[:64] or "output"
        try:
            response = await self._client.post(
                "chat/completions",
                json={
                    "model": self._models[request.model_role],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                request.input_payload,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        },
                    ],
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": self._max_output_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": request.output_schema,
                        },
                    },
                },
            )
        except httpx.TimeoutException as error:
            raise ValueError(
                f"Direct Schema Agent exceeded {self._timeout_seconds:g}s timeout"
            ) from error
        except httpx.HTTPError as error:
            raise ValueError(
                f"Direct Schema Agent transport failed: {type(error).__name__}"
            ) from error
        if not response.is_success:
            raise SchemaAgentOutputError(
                f"Direct Schema Agent provider returned HTTP {response.status_code}",
                code="TERMINAL_ERROR",
                raw_model_output=self._redact(response.text[:4_000]),
            )
        try:
            provider_payload = response.json()
            raw_payload = provider_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise SchemaAgentOutputError(
                "Direct Schema Agent response did not match chat completions shape"
            ) from error
        if not isinstance(raw_payload, str):
            raise SchemaAgentOutputError("Direct Schema Agent returned no JSON text result")
        try:
            payload = ClaudeSchemaAgent._decode_json_text(raw_payload)
        except ValueError as error:
            raise SchemaAgentOutputError(
                "Direct Schema Agent returned invalid JSON text",
                raw_model_output=self._redact(raw_payload),
            ) from error
        if not isinstance(payload, dict):
            raise SchemaAgentOutputError(
                "Direct Schema Agent returned no structured output object",
                raw_model_output=self._redact(raw_payload),
            )
        errors = list(Draft202012Validator(request.output_schema).iter_errors(payload))
        if errors:
            raise SchemaAgentOutputError(
                schema_validation_detail("Direct Schema Agent output schema failed", errors[0]),
                raw_model_output=self._redact(raw_payload),
            )
        return SchemaAgentResult(payload=payload, raw_model_output=self._redact(raw_payload))

    async def aclose(self) -> None:
        await self._client.aclose()

    def _redact(self, value: str) -> str:
        value = value.replace(self._api_key, "[REDACTED]")
        value = _AUTHORIZATION_PATTERN.sub(r"\1[REDACTED]", value)
        return _CREDENTIAL_PATTERN.sub(r"\1\2[REDACTED]", value)


class ClaudeSchemaAgent:
    """一次请求、一次结构化结果；不恢复 session，也不持久化 invocation。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        worker_model: str,
        reviewer_model: str,
        timeout_seconds: float = 120,
        max_turns: int = 3,
        structured_output_mode: Literal["sdk_tool", "json_text"] = "sdk_tool",
        client_factory: Callable[[Any], _SdkClient] = _default_client,
    ) -> None:
        if not api_key or not worker_model or not reviewer_model:
            raise ValueError("Schema Agent requires credential and explicit models")
        if timeout_seconds <= 0 or timeout_seconds > 600:
            raise ValueError("Schema Agent timeout must be between 0 and 600 seconds")
        if max_turns < 2 or max_turns > 4:
            raise ValueError("Schema Agent max turns must be between 2 and 4")
        if structured_output_mode not in {"sdk_tool", "json_text"}:
            raise ValueError("Schema Agent structured output mode is unsupported")
        self._api_key = api_key
        self._base_url = base_url
        self._models = {"worker": worker_model, "reviewer": reviewer_model}
        self._timeout_seconds = timeout_seconds
        self._max_turns = max_turns
        self._structured_output_mode = structured_output_mode
        self._client_factory = client_factory

    def binding(self, role: Literal["worker", "reviewer"]) -> dict[str, Any]:
        return {
            "runtime": "claude-agent-sdk-schema-v4",
            "model": self._models[role],
            "role": role,
            "thinking_mode": "disabled",
            # output_format 由 SDK 内部的 structured-output tool 收束；额外 turn 不是
            # 重试或新 item，只允许同一次 schema-bound invocation 完成握手。
            "max_turns": self._max_turns if self._structured_output_mode == "sdk_tool" else 1,
            "timeout_seconds": self._timeout_seconds,
            "structured_output_mode": self._structured_output_mode,
            "input_token_counter": (
                "deepseek-v4-official-tokenizer-plus-sdk-envelope-v1"
                if self._models[role] == "deepseek-v4-flash"
                and self._structured_output_mode == "json_text"
                else None
            ),
            "input_token_envelope_tokens": (
                128
                if self._models[role] == "deepseek-v4-flash"
                and self._structured_output_mode == "json_text"
                else None
            ),
            "base_url": normalize_sdk_base_url(self._base_url) if self._base_url else None,
        }

    async def invoke(self, request: SchemaAgentRequest) -> SchemaAgentResult:
        Draft202012Validator.check_schema(request.output_schema)
        model = self._models[request.model_role]
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

        env = {"ANTHROPIC_API_KEY": self._api_key}
        if self._base_url:
            env["ANTHROPIC_BASE_URL"] = normalize_sdk_base_url(self._base_url)
        output_format = None
        max_turns = self._max_turns
        if self._structured_output_mode == "sdk_tool":
            output_format = {"type": "json_schema", "schema": request.output_schema}
        else:
            # 部分 Anthropic-compatible gateway 虽暴露 SDK structured-output tool，却不能稳定完成
            # 非 Claude 模型的 tool 握手；这里限制为单 item、单 turn，并由 adapter 使用同一份
            # 本地 schema contract 校验模型返回的普通 JSON。
            max_turns = 1
        system_prompt = schema_agent_system_prompt(
            prompt=request.prompt,
            skill_text=request.skill_text,
            output_schema=request.output_schema,
            structured_output_mode=self._structured_output_mode,
        )
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=model,
            thinking={"type": "disabled"},
            fallback_model=None,
            max_turns=max_turns,
            tools=[],
            allowed_tools=[],
            permission_mode="dontAsk",
            setting_sources=[],
            strict_mcp_config=True,
            mcp_servers={},
            output_format=output_format,
            env=env,
        )
        client = self._client_factory(options)
        terminal = None
        connected = False
        primary_error: BaseException | None = None
        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await client.connect()
                connected = True
                await client.query(schema_agent_user_prompt(request.input_payload))
                async for message in client.receive_response():
                    if isinstance(message, ResultMessage):
                        if terminal is not None:
                            raise ValueError("Schema Agent returned multiple terminal results")
                        terminal = message
        except BaseException as error:
            primary_error = error
            if isinstance(error, TimeoutError):
                raise ValueError(
                    f"Schema Agent exceeded {self._timeout_seconds:g}s timeout"
                ) from error
            raise
        finally:
            if connected:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(client.disconnect(), timeout=min(5, remaining))
                        connected = False
                    except BaseException:
                        if primary_error is None:
                            raise
        if terminal is None:
            raise ValueError("Schema Agent ended without a terminal result")
        if terminal.is_error:
            raw_result = getattr(terminal, "result", None)
            if raw_result is None:
                raw_result = getattr(terminal, "structured_output", None)
            raise SchemaAgentOutputError(
                self._terminal_error_detail(terminal),
                raw_model_output=self._redact_json_value(raw_result),
                code="TERMINAL_ERROR",
                metrics=self._terminal_metrics(terminal),
            )
        raw_model_output: Any | None
        if self._structured_output_mode == "json_text":
            raw_payload = getattr(terminal, "result", None)
            if not isinstance(raw_payload, str):
                raise SchemaAgentOutputError(
                    "Schema Agent returned no JSON text result",
                    raw_model_output=self._redact_json_value(raw_payload),
                    metrics=self._terminal_metrics(terminal),
                )
            try:
                payload = self._decode_json_text(raw_payload)
            except ValueError as error:
                raise SchemaAgentOutputError(
                    "Schema Agent returned invalid JSON text",
                    raw_model_output=self._redact_json_value(raw_payload),
                    metrics=self._terminal_metrics(terminal),
                ) from error
            raw_model_output = self._redact_json_value(raw_payload)
        else:
            payload = getattr(terminal, "structured_output", None)
            raw_model_output = self._redact_json_value(payload)
        if not isinstance(payload, dict):
            raise SchemaAgentOutputError(
                "Schema Agent returned no structured output object",
                raw_model_output=raw_model_output,
                metrics=self._terminal_metrics(terminal),
            )
        errors = list(Draft202012Validator(request.output_schema).iter_errors(payload))
        if errors:
            raise SchemaAgentOutputError(
                schema_validation_detail("Schema Agent output schema failed", errors[0]),
                raw_model_output=raw_model_output,
                metrics=self._terminal_metrics(terminal),
            )
        return SchemaAgentResult(
            payload=payload,
            raw_model_output=raw_model_output,
            metrics=self._terminal_metrics(terminal),
        )

    def _terminal_metrics(self, terminal: Any) -> dict[str, Any]:
        return {
            "duration_ms": terminal.duration_ms,
            "duration_api_ms": terminal.duration_api_ms,
            "num_turns": terminal.num_turns,
            "total_cost_usd": terminal.total_cost_usd,
            "usage": self._redact_json_value(terminal.usage),
            "model_usage": self._redact_json_value(
                {
                    key: asdict(value) if is_dataclass(value) else str(value)
                    for key, value in (terminal.model_usage or {}).items()
                }
            ),
        }

    def _terminal_error_detail(self, terminal: Any) -> str:
        """保留可诊断终态字段，同时避免把 credential 或完整模型结果写入 Issue。"""

        fields: dict[str, Any] = {}
        for name in (
            "subtype",
            "api_error_status",
            "stop_reason",
            "terminal_reason",
        ):
            value = getattr(terminal, name, None)
            if value is not None:
                fields[name] = self._redact(str(value))[:500]
        raw_errors = getattr(terminal, "errors", None)
        if isinstance(raw_errors, list):
            fields["errors"] = [self._redact(str(item))[:500] for item in raw_errors[:5]]
        detail = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"Schema Agent terminal error: {detail}"

    @staticmethod
    def _decode_json_text(raw: str) -> Any:
        """接受纯 JSON 或单一 Markdown fence，不接受夹带解释的自由文本。"""

        candidate = raw.strip()
        fenced = _JSON_FENCE_PATTERN.fullmatch(candidate)
        if fenced is not None:
            candidate = fenced.group(1).strip()
        try:
            return json.loads(
                candidate,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_non_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("Schema Agent returned invalid JSON text") from error

    def _redact_json_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {
                self._redact(str(key)): self._redact_json_value(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact_json_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._redact_json_value(item) for item in value]
        return value

    def _redact(self, value: str) -> str:
        if self._api_key:
            value = value.replace(self._api_key, "[REDACTED]")
        value = _AUTHORIZATION_PATTERN.sub(r"\1[REDACTED]", value)
        return _CREDENTIAL_PATTERN.sub(r"\1\2[REDACTED]", value)


class OpenAICompatibleJsonSchemaAgent:
    """通过 OpenAI-compatible Chat Completions 执行单次 JSON schema 任务。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        role: Literal["worker", "reviewer", "adjudicator"],
        timeout_seconds: float = 120,
        reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = None,
        structured_output_mode: Literal[
            "json_text",
            "provider_json_object",
            "provider_json_schema",
            "provider_json_schema_prompt",
            "text_wrapped",
        ] = "json_text",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        disable_thinking: bool = False,
        thinking: Literal["enabled", "disabled"] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key or not model:
            raise ValueError("OpenAI-compatible Schema Agent requires credential and model")
        if timeout_seconds <= 0 or timeout_seconds > 600:
            raise ValueError("Schema Agent timeout must be between 0 and 600 seconds")
        if structured_output_mode not in {
            "json_text",
            "provider_json_object",
            "provider_json_schema",
            "provider_json_schema_prompt",
            "text_wrapped",
        }:
            raise ValueError("OpenAI-compatible structured output mode is unsupported")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("OpenAI-compatible temperature must be between 0 and 2")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("OpenAI-compatible max output tokens must be positive")
        if thinking not in {None, "enabled", "disabled"}:
            raise ValueError("Unsupported explicit thinking mode")
        if disable_thinking and thinking is not None:
            raise ValueError("Use only one explicit thinking control")
        if (disable_thinking or thinking == "disabled") and reasoning_effort is not None:
            raise ValueError("Explicit thinking disable contradicts reasoning_effort")
        normalized = normalize_sdk_base_url(base_url)
        self._endpoint = normalized.rstrip("/") + "/v1/chat/completions"
        self._api_key = api_key
        self._base_url = normalized.rstrip("/") + "/v1"
        self._model = model
        self._role = role
        self._timeout_seconds = timeout_seconds
        self._reasoning_effort = reasoning_effort
        self._structured_output_mode = structured_output_mode
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._disable_thinking = disable_thinking
        self._thinking = thinking
        self._transport = transport

    def binding(self, role: Literal["worker", "reviewer", "adjudicator"]) -> dict[str, Any]:
        if role != self._role:
            raise ValueError(f"OpenAI-compatible Schema Agent is not bound to role {role}")
        if self._model == "deepseek-v4-flash" and self._structured_output_mode in {
            "json_text",
            "provider_json_object",
            "text_wrapped",
        }:
            token_counter = "deepseek-v4-official-tokenizer-plus-sdk-envelope-v1"
        else:
            token_counter = "openai-o200k-base-plus-chat-envelope-v1"
        binding = {
            "runtime": {
                "json_text": "openai-compatible-chat-json-v1",
                "provider_json_object": "openai-compatible-chat-json-object-v1",
                "provider_json_schema": "openai-compatible-chat-json-schema-v2",
                "provider_json_schema_prompt": ("openai-compatible-chat-json-schema-prompt-v3"),
                "text_wrapped": "openai-compatible-chat-text-wrapped-v1",
            }[self._structured_output_mode],
            "model": self._model,
            "role": role,
            "thinking_mode": "disabled" if self._reasoning_effort is None else "provider",
            "provider_reasoning_effort": self._reasoning_effort,
            "max_turns": 1,
            "timeout_seconds": self._timeout_seconds,
            "structured_output_mode": self._structured_output_mode,
            "input_token_counter": token_counter,
            "input_token_envelope_tokens": 256,
            "base_url": self._base_url,
        }
        if self._temperature is not None:
            binding["temperature"] = self._temperature
        if self._max_output_tokens is not None:
            binding["max_output_tokens"] = self._max_output_tokens
        if self._disable_thinking:
            binding["provider_thinking"] = {"type": "disabled"}
        if self._thinking is not None:
            binding["provider_thinking"] = {"type": self._thinking}
            binding["thinking_mode"] = "provider" if self._thinking == "enabled" else "disabled"
        return binding

    async def invoke(self, request: SchemaAgentRequest) -> SchemaAgentResult:
        if request.model_role != self._role:
            raise ValueError(
                f"OpenAI-compatible Schema Agent cannot execute role {request.model_role}"
            )
        Draft202012Validator.check_schema(request.output_schema)
        if self._structured_output_mode == "text_wrapped":
            properties = request.output_schema.get("properties")
            if (
                request.output_schema.get("type") != "object"
                or request.output_schema.get("additionalProperties") is not False
                or not isinstance(properties, dict)
                or set(properties) != {"assessment"}
                or properties["assessment"].get("type") != "string"
                or "assessment" not in request.output_schema.get("required", [])
            ):
                raise ValueError("text_wrapped mode requires one required assessment string field")
        system_prompt = schema_agent_system_prompt(
            prompt=request.prompt,
            skill_text=request.skill_text,
            output_schema=request.output_schema,
            structured_output_mode=self._structured_output_mode,
        )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": schema_agent_user_prompt(request.input_payload),
                },
            ],
        }
        if self._reasoning_effort is not None:
            body["reasoning_effort"] = self._reasoning_effort
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if self._max_output_tokens is not None:
            body["max_tokens"] = self._max_output_tokens
        if self._disable_thinking:
            body["thinking"] = {"type": "disabled"}
        if self._thinking is not None:
            body["thinking"] = {"type": self._thinking}
        if self._structured_output_mode in {
            "provider_json_schema",
            "provider_json_schema_prompt",
        }:
            body["response_format"] = openai_chat_response_format(
                template_id=request.template_id,
                output_schema=request.output_schema,
            )
        if self._structured_output_mode == "provider_json_object":
            body["response_format"] = {"type": "json_object"}
        started = asyncio.get_running_loop().time()
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout_seconds,
            ) as client:
                response = await client.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        except httpx.TimeoutException as error:
            raise ValueError(f"Schema Agent exceeded {self._timeout_seconds:g}s timeout") from error
        duration_ms = round((asyncio.get_running_loop().time() - started) * 1000)
        try:
            response_body = response.json()
        except ValueError as error:
            raise SchemaAgentOutputError(
                f"OpenAI-compatible gateway returned non-JSON HTTP {response.status_code}",
                code="TERMINAL_ERROR",
                metrics={"duration_ms": duration_ms, "duration_api_ms": duration_ms},
            ) from error
        response_metrics = self._response_metrics(response_body, duration_ms)
        if response.status_code < 200 or response.status_code >= 300:
            raise SchemaAgentOutputError(
                self._http_error_detail(response.status_code, response_body),
                code="TERMINAL_ERROR",
                metrics=response_metrics,
            )
        if not isinstance(response_body, dict):
            raise SchemaAgentOutputError(
                "OpenAI-compatible gateway returned no response object",
                metrics=response_metrics,
            )
        response_model = response_body.get("model")
        choices = response_body.get("choices")
        if not isinstance(response_model, str) or not isinstance(choices, list) or not choices:
            raise SchemaAgentOutputError(
                "OpenAI-compatible gateway response is missing model or choices",
                metrics=response_metrics,
            )
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        raw_payload = message.get("content") if isinstance(message, dict) else None
        if not isinstance(raw_payload, str):
            raise SchemaAgentOutputError(
                "OpenAI-compatible gateway returned no text result",
                raw_model_output=self._redact_json_value(raw_payload),
                metrics=response_metrics,
            )
        if self._structured_output_mode == "text_wrapped":
            payload = {"assessment": raw_payload.strip()}
        else:
            try:
                payload = ClaudeSchemaAgent._decode_json_text(raw_payload)
            except ValueError as error:
                raise SchemaAgentOutputError(
                    "OpenAI-compatible gateway returned invalid JSON text",
                    raw_model_output=self._redact_json_value(raw_payload),
                    metrics=response_metrics,
                ) from error
        if not isinstance(payload, dict):
            raise SchemaAgentOutputError(
                "OpenAI-compatible gateway returned no structured output object",
                raw_model_output=self._redact_json_value(raw_payload),
                metrics=response_metrics,
            )
        validation_schema = (
            strict_contract_schema(request.output_schema)
            if self._structured_output_mode
            in {"provider_json_schema", "provider_json_schema_prompt"}
            else request.output_schema
        )
        errors = list(Draft202012Validator(validation_schema).iter_errors(payload))
        if errors:
            raise SchemaAgentOutputError(
                schema_validation_detail("Schema Agent output schema failed", errors[0]),
                raw_model_output=self._redact_json_value(raw_payload),
                metrics=response_metrics,
            )
        return SchemaAgentResult(
            payload=payload,
            raw_model_output=self._redact_json_value(raw_payload),
            metrics=response_metrics,
        )

    def _response_metrics(self, response_body: Any, duration_ms: int) -> dict[str, Any]:
        usage = self._canonical_usage(
            response_body.get("usage") if isinstance(response_body, dict) else None
        )
        response_model = response_body.get("model") if isinstance(response_body, dict) else None
        metrics = {
            "duration_ms": duration_ms,
            "duration_api_ms": duration_ms,
            "num_turns": 1,
            "total_cost_usd": None,
            "usage": self._redact_json_value(usage),
            "model_usage": (
                {response_model: self._redact_json_value(usage)}
                if isinstance(response_model, str)
                else {}
            ),
        }
        choices = response_body.get("choices") if isinstance(response_body, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
            if isinstance(finish_reason, str):
                metrics["finish_reason"] = finish_reason
        return metrics

    @staticmethod
    def _canonical_usage(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        usage = dict(value)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if "input_tokens" not in usage and isinstance(prompt_tokens, int):
            usage["input_tokens"] = prompt_tokens
        if "output_tokens" not in usage and isinstance(completion_tokens, int):
            usage["output_tokens"] = completion_tokens
        return usage

    def _http_error_detail(self, status: int, body: Any) -> str:
        fields: dict[str, Any] = {"status": status}
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            for name in ("type", "code", "message"):
                value = error.get(name)
                if value is not None:
                    fields[name] = self._redact(str(value))[:500]
        detail = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"OpenAI-compatible terminal error: {detail}"

    def _redact_json_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {
                str(key).replace(self._api_key, "[REDACTED]"): self._redact_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact_json_value(item) for item in value]
        return value

    def _redact(self, value: str) -> str:
        value = value.replace(self._api_key, "[REDACTED]")
        value = _AUTHORIZATION_PATTERN.sub(r"\1[REDACTED]", value)
        return _CREDENTIAL_PATTERN.sub(r"\1\2[REDACTED]", value)


class RoleRoutedSchemaAgent:
    """按显式 worker/reviewer role 选择单一 runtime；不重试、不 fallback。"""

    def __init__(self, *, worker: Any, reviewer: Any, adjudicator: Any | None = None) -> None:
        self._runtimes = {"worker": worker, "reviewer": reviewer}
        if adjudicator is not None:
            self._runtimes["adjudicator"] = adjudicator

    def binding(self, role: Literal["worker", "reviewer", "adjudicator"]) -> dict[str, Any]:
        return self._runtimes[role].binding(role)

    async def invoke(self, request: SchemaAgentRequest) -> SchemaAgentResult:
        return await self._runtimes[request.model_role].invoke(request)


