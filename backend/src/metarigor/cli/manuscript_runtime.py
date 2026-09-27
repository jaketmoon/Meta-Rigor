from __future__ import annotations

from typing import Literal
from metarigor.adapters.agent_runtime.schema_agent import OpenAICompatibleJsonSchemaAgent, RoleRoutedSchemaAgent
from metarigor.config import get_settings
MODEL = "deepseek-v4-flash"


def manuscript_runtime(
    *,
    include_adjudicator: bool = False,
    writer_model: str = MODEL,
    explicit_disable_thinking: bool = False,
    adjudicator_disable_thinking: bool = False,
    adjudicator_thinking: Literal["enabled", "disabled"] | None = None,
    adjudicator_reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = None,
    adjudicator_max_output_tokens: int | None = None,
    adjudicator_structured_output_mode: Literal[
        "json_text", "provider_json_object", "provider_json_schema", "text_wrapped"
    ] = "json_text",
) -> RoleRoutedSchemaAgent:
    settings = get_settings()
    if not settings.agent_gateway_base_url or settings.agent_gateway_api_key is None:
        raise RuntimeError("Manuscript requires configured gateway credentials")
    common = {
        "api_key": settings.agent_gateway_api_key.get_secret_value(),
        "base_url": settings.agent_gateway_base_url,
        "model": MODEL,
        "timeout_seconds": settings.agent_call_timeout_seconds,
        "temperature": 0,
        "max_output_tokens": 12_000,
    }
    return RoleRoutedSchemaAgent(
        worker=OpenAICompatibleJsonSchemaAgent(
            role="worker",
            structured_output_mode="json_text",
            disable_thinking=explicit_disable_thinking,
            **{**common, "model": writer_model},
        ),
        reviewer=OpenAICompatibleJsonSchemaAgent(
            role="reviewer",
            structured_output_mode="json_text",
            disable_thinking=explicit_disable_thinking,
            **{**common, "model": writer_model},
        ),
        adjudicator=(
            OpenAICompatibleJsonSchemaAgent(
                role="adjudicator",
                structured_output_mode=adjudicator_structured_output_mode,
                disable_thinking=adjudicator_disable_thinking,
                thinking=adjudicator_thinking,
                reasoning_effort=adjudicator_reasoning_effort,
                **{
                    **common,
                    "max_output_tokens": adjudicator_max_output_tokens
                    if adjudicator_max_output_tokens is not None
                    else common["max_output_tokens"],
                },
            )
            if include_adjudicator
            else None
        ),
    )


