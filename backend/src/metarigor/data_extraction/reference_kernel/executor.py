from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Literal
import httpx
from .contracts import KernelTask, RawTextResponse
from .prompts import SYSTEM_PROMPT
class GatewayRawTextError(RuntimeError):
    """Gateway failure containing neither response bodies nor credentials."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        http_status: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.http_status = http_status


class DeepSeekFlashExecutor:
    """Private DE OpenAI-compatible raw-text executor; one request per item."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        model_id: str,
        response_model: str | None = None,
        reasoning_effort: str | None = None,
        thinking_mode: Literal["provider_default", "disabled", "enabled"] | None = None,
        output_token_budget: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not model_id.strip() or (response_model is not None and not response_model.strip()):
            raise ValueError("configured model IDs must be nonempty")
        self._model_id = model_id
        self._response_model = response_model
        self._reasoning_effort = reasoning_effort
        if thinking_mode not in {None, "provider_default", "disabled", "enabled"}:
            raise ValueError("unsupported thinking mode")
        self._thinking_mode = thinking_mode or "provider_default"
        if output_token_budget is not None and output_token_budget < 1:
            raise ValueError("output token budget must be positive")
        self._output_contract = "legacy"
        self._output_token_budget = output_token_budget
        self._system_prompt = SYSTEM_PROMPT
        self._timeout_seconds = timeout_seconds
        self._usage: dict[KernelTask, dict[str, int | float | None]] = {
            task: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "reasoning_tokens": 0,
                "reasoning_response_count": 0,
            }
            for task in ("binary_outcomes", "continuous_outcomes")
        }
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def binding(self, task: KernelTask) -> Mapping[str, Any]:
        return {
            "executor": "openai_compatible_raw_text",
            "model_alias": self._model_id,
            "provider_model_requested": self._model_id,
            "provider_response_model_expected": self._response_model,
            "provider_reasoning_effort": self._reasoning_effort,
            "thinking_mode": self._thinking_mode,
            "response_format_type": "text",
            "output_contract": self._output_contract,
            "temperature": 1,
            "top_p": 1,
            "max_turns": 1,
            "max_output_tokens": self._output_token_budget
            or (50 if task == "binary_outcomes" else 70),
            "timeout_seconds": self._timeout_seconds,
            "automatic_retry_count": 0,
            "fallback_count": 0,
            "system_prompt_sha256": hashlib.sha256(self._system_prompt.encode("utf-8")).hexdigest(),
        }

    def _mark_usage_unknown(self, task: KernelTask) -> None:
        self._usage[task] = {
            "input_tokens": None,
            "output_tokens": None,
            "cost_usd": None,
            "reasoning_tokens": None,
            "reasoning_response_count": None,
        }

    async def generate(
        self,
        *,
        task: KernelTask,
        row_id: int,
        prompt: str,
        max_output_tokens: int,
    ) -> RawTextResponse:
        del row_id
        try:
            body = {
                "model": self._model_id,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 1,
                "top_p": 1,
                "max_tokens": max_output_tokens,
            }
            if self._reasoning_effort is not None:
                body["reasoning_effort"] = self._reasoning_effort
            if self._thinking_mode in {"disabled", "enabled"}:
                body["thinking"] = {"type": self._thinking_mode}
            response = await self._client.post("chat/completions", json=body)
        except httpx.TimeoutException as error:
            self._mark_usage_unknown(task)
            raise GatewayRawTextError("TIMEOUT", "gateway request timed out") from error
        except httpx.HTTPError as error:
            self._mark_usage_unknown(task)
            raise GatewayRawTextError(
                "TRANSPORT_ERROR", f"gateway transport failed: {type(error).__name__}"
            ) from error
        if not response.is_success:
            self._mark_usage_unknown(task)
            raise GatewayRawTextError(
                "HTTP_ERROR",
                f"gateway returned HTTP {response.status_code}",
                http_status=response.status_code,
            )
        try:
            payload = response.json()
        except (TypeError, ValueError) as error:
            self._mark_usage_unknown(task)
            raise GatewayRawTextError(
                "INVALID_RESPONSE",
                "gateway response did not match chat completions shape",
                http_status=response.status_code,
            ) from error
        usage = payload.get("usage")
        for metric, provider_key, convert in (
            ("input_tokens", "prompt_tokens", int),
            ("output_tokens", "completion_tokens", int),
            ("cost_usd", "cost", float),
        ):
            value = usage.get(provider_key) if isinstance(usage, dict) else None
            current = self._usage[task][metric]
            self._usage[task][metric] = (
                None if value is None or current is None else current + convert(value)
            )
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise GatewayRawTextError(
                "INVALID_RESPONSE",
                "gateway response did not match chat completions shape",
                http_status=response.status_code,
            ) from error
        # Keep usage metrics only; do not log reasoning_content or use it as extraction evidence.
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        current = self._usage[task]["reasoning_tokens"]
        self._usage[task]["reasoning_tokens"] = (
            current + reasoning_tokens
            if current is not None
            and isinstance(reasoning_tokens, int)
            and not isinstance(reasoning_tokens, bool)
            and reasoning_tokens >= 0
            else None
        )
        current = self._usage[task]["reasoning_response_count"]
        observed = choice["message"].get("reasoning_content")
        self._usage[task]["reasoning_response_count"] = (
            current + int(isinstance(observed, str) and bool(observed))
            if current is not None
            else None
        )
        if not isinstance(content, str):
            raise GatewayRawTextError(
                "INVALID_RESPONSE",
                "gateway response content was not text",
                http_status=response.status_code,
            )
        provider_model = payload.get("model")
        if not isinstance(provider_model, str) or not provider_model:
            raise GatewayRawTextError(
                "INVALID_RESPONSE",
                "gateway response did not identify the provider model",
                http_status=response.status_code,
            )
        if self._response_model is not None and provider_model != self._response_model:
            raise GatewayRawTextError(
                "MODEL_MISMATCH",
                "gateway response model differs from the requested Chat Completions binding",
                http_status=response.status_code,
            )
        return RawTextResponse(
            content=content,
            finish_reason=payload["choices"][0].get("finish_reason"),
            requested_model=self._model_id,
            provider_model=provider_model,
            http_status=response.status_code,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def usage_by_task(self) -> Mapping[KernelTask, Mapping[str, int | float | None]]:
        return {task: dict(usage) for task, usage in self._usage.items()}

