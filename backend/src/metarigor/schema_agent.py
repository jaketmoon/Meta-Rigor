from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from jsonschema import Draft202012Validator, ValidationError
StructuredOutputMode = Literal[
    "sdk_tool",
    "json_text",
    "provider_json_schema",
    "provider_json_schema_prompt",
    "text_wrapped",
]


@dataclass(frozen=True, slots=True)
class SchemaAgentRequest:
    template_id: str
    model_role: Literal["worker", "reviewer", "adjudicator"]
    prompt: str
    input_payload: dict[str, Any]
    output_schema: dict[str, Any]
    skill_text: str | None = None


@dataclass(frozen=True, slots=True)
class SchemaAgentResult:
    payload: dict[str, Any]
    raw_model_output: Any | None = None
    metrics: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SchemaAgentInputSize:
    """最终 canonical 模型输入的模型 tokenizer 计数与 runtime 安全边界。"""

    canonical_payload_bytes: int
    canonical_payload_sha256: str
    tokenizer_content_tokens: int
    runtime_envelope_tokens: int
    input_tokens: int
    token_counter: str


_JSON_TEXT_OUTPUT_INSTRUCTION = (
    "\n\nReturn exactly one JSON object matching this JSON Schema. "
    "Emit every object property exactly once. Do not repeat a property to revise it. "
    "Do not use Markdown fences or include any text before or after the object."
    "\nJSON Schema:\n"
)


def schema_agent_system_prompt(
    *,
    prompt: str,
    skill_text: str | None,
    output_schema: dict[str, Any],
    structured_output_mode: StructuredOutputMode,
) -> str:
    """复用 runtime 的精确 system prompt 组装规则。"""

    rendered = prompt
    if skill_text:
        rendered += "\n\nFollow this versioned Skill exactly:\n\n" + skill_text
    if structured_output_mode in {"json_text", "provider_json_schema_prompt"}:
        prompt_schema = (
            strict_contract_schema(output_schema)
            if structured_output_mode == "provider_json_schema_prompt"
            else output_schema
        )
        rendered += _JSON_TEXT_OUTPUT_INSTRUCTION + json.dumps(
            prompt_schema,
            ensure_ascii=False,
            sort_keys=True,
        )
    return rendered


def openai_chat_response_format(
    *, template_id: str, output_schema: dict[str, Any]
) -> dict[str, Any]:
    """构造 OpenAI-compatible Chat Completions 的原生 Structured Outputs 参数。"""

    name = re.sub(r"[^A-Za-z0-9_-]", "_", template_id)[:64] or "structured_output"
    strict_schema = strict_contract_schema(output_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": strict_schema,
        },
    }


def strict_contract_schema(output_schema: dict[str, Any]) -> dict[str, Any]:
    """将任务 schema 投影为 provider strict wire contract。

    OpenAI-compatible strict Structured Outputs 要求每层 object 的全部 properties
    都列入 required。任务 schema 中带默认值的字段仍可由本地 Pydantic 省略，但模型
    wire output 必须显式返回，以免 gateway 在调用模型前拒绝整个 Stage。
    """

    strict_schema = deepcopy(output_schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            properties = node.get("properties")
            if node.get("type") == "object" and isinstance(properties, dict):
                additional_properties = node.get("additionalProperties")
                if additional_properties not in (None, False):
                    raise ValueError(
                        "provider_json_schema requires closed object schemas "
                        "with additionalProperties=false"
                    )
                node["additionalProperties"] = False
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(strict_schema)
    Draft202012Validator.check_schema(strict_schema)
    return strict_schema


def schema_agent_user_prompt(input_payload: dict[str, Any]) -> str:
    return json.dumps(
        input_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def count_schema_agent_input_size(
    *,
    template_id: str,
    prompt: str,
    skill_text: str | None,
    output_schema: dict[str, Any],
    input_payload: dict[str, Any],
    structured_output_mode: StructuredOutputMode,
    runtime: str,
    model: str,
    token_counter: str,
    runtime_envelope_tokens: int,
) -> SchemaAgentInputSize:
    """按模型 tokenizer 计算最终 system/user contract，并计入 runtime envelope。"""

    system_prompt = schema_agent_system_prompt(
        prompt=prompt,
        skill_text=skill_text,
        output_schema=output_schema,
        structured_output_mode=structured_output_mode,
    )
    user_prompt = schema_agent_user_prompt(input_payload)
    canonical_contract = {
        "model": model,
        "output_format": (
            {"type": "json_schema", "schema": output_schema}
            if structured_output_mode == "sdk_tool"
            else None
        ),
        "runtime": runtime,
        "runtime_envelope_tokens": runtime_envelope_tokens,
        "structured_output_mode": structured_output_mode,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    if structured_output_mode in {"provider_json_schema", "provider_json_schema_prompt"}:
        canonical_contract["response_format"] = openai_chat_response_format(
            template_id=template_id,
            output_schema=output_schema,
        )
    canonical_payload = json.dumps(
        canonical_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if (
        model == "deepseek-v4-flash"
        and token_counter == "deepseek-v4-official-tokenizer-plus-sdk-envelope-v1"
        and structured_output_mode in {"json_text", "text_wrapped"}
    ):
        from deepseek_tokenizer import ds_token

        rendered_prompt = (
            "<｜begin▁of▁sentence｜>"
            + system_prompt
            + "<｜User｜>"
            + user_prompt
            + "<｜Assistant｜></think>"
        )
        tokenizer_content_tokens = len(ds_token.encode(rendered_prompt))
    elif (
        model in {"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"}
        and token_counter == "openai-o200k-base-plus-chat-envelope-v1"
        and structured_output_mode
        in {
            "json_text",
            "provider_json_schema",
            "provider_json_schema_prompt",
            "text_wrapped",
        }
    ):
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        tokenizer_content_tokens = len(encoding.encode(system_prompt)) + len(
            encoding.encode(user_prompt)
        )
        if structured_output_mode in {"provider_json_schema", "provider_json_schema_prompt"}:
            response_format = canonical_contract["response_format"]
            tokenizer_content_tokens += len(
                encoding.encode(
                    json.dumps(
                        response_format,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            )
    else:
        raise ValueError("Schema Agent input token counter is not bound to this runtime/model/mode")
    return SchemaAgentInputSize(
        canonical_payload_bytes=len(canonical_payload),
        canonical_payload_sha256=hashlib.sha256(canonical_payload).hexdigest(),
        tokenizer_content_tokens=tokenizer_content_tokens,
        runtime_envelope_tokens=runtime_envelope_tokens,
        input_tokens=tokenizer_content_tokens + runtime_envelope_tokens,
        token_counter=token_counter,
    )


class SchemaAgentOutputError(ValueError):
    """模型结果不符合契约时保留可追溯的原始响应。"""

    def __init__(
        self,
        detail: str,
        *,
        raw_model_output: Any | None = None,
        code: Literal["INVALID_OUTPUT", "TERMINAL_ERROR"] = "INVALID_OUTPUT",
        metrics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.raw_model_output = raw_model_output
        self.code = code
        self.metrics = metrics


def schema_validation_detail(prefix: str, error: ValidationError) -> str:
    """只报告 schema 定位信息，避免把不可信实例值写入 Issue。"""

    validator = "unknown" if error.validator is None else str(error.validator)
    schema_path = "/".join(str(part) for part in error.absolute_schema_path) or "$"
    return f"{prefix}: validator={validator} schema_path={schema_path}"


class SchemaAgentPort(Protocol):
    async def invoke(self, request: SchemaAgentRequest) -> SchemaAgentResult: ...


class SchemaAgentRunner:
    """所有 Specialist 共用的单次 schema-bound 模型调用边界。"""

    def __init__(self, *, runtime: SchemaAgentPort) -> None:
        self._runtime = runtime

    def binding(self, role: Literal["worker", "reviewer", "adjudicator"]) -> dict[str, Any]:
        provider = getattr(self._runtime, "binding", None)
        if callable(provider):
            value = provider(role)
            if not isinstance(value, dict):
                raise TypeError("Schema Agent binding must be an object")
            return value
        runtime_type = type(self._runtime)
        return {"runtime": f"{runtime_type.__module__}.{runtime_type.__qualname__}", "role": role}

    def stage_payload(
        self,
        template,
        semantic_input: dict[str, Any],
        *,
        program_binding: dict[str, Any],
    ) -> dict[str, Any]:
        """构造既用于散列也用于实际调用的完整模型契约。"""

        return {
            "template_id": template.identifier,
            "prompt": template.prompt,
            "skill_text": template.skill_text,
            "skill_sha256": hashlib.sha256((template.skill_text or "").encode()).hexdigest(),
            "input_schema": template.input_schema,
            "output_schema": template.output_schema,
            "runtime": self.binding(template.model_role),
            "program_binding": program_binding,
            "semantic_input": semantic_input,
        }

    def input_size(self, template, semantic_input: dict[str, Any]) -> SchemaAgentInputSize:
        """估算 runtime 将实际发送的完整 canonical 模型输入。"""

        binding = self.binding(template.model_role)
        mode = binding.get("structured_output_mode")
        if mode not in {
            "sdk_tool",
            "json_text",
            "provider_json_schema",
            "provider_json_schema_prompt",
            "text_wrapped",
        }:
            raise ValueError("Schema Agent binding must declare structured_output_mode")
        runtime = binding.get("runtime")
        model = binding.get("model")
        token_counter = binding.get("input_token_counter")
        runtime_envelope_tokens = binding.get("input_token_envelope_tokens")
        if (
            not isinstance(runtime, str)
            or not isinstance(model, str)
            or not isinstance(token_counter, str)
            or not isinstance(runtime_envelope_tokens, int)
            or runtime_envelope_tokens < 0
        ):
            raise ValueError(
                "Schema Agent binding must declare runtime, model, and input token counter"
            )
        return count_schema_agent_input_size(
            template_id=template.identifier,
            prompt=template.prompt,
            skill_text=template.skill_text,
            output_schema=template.output_schema,
            input_payload=semantic_input,
            structured_output_mode=mode,
            runtime=runtime,
            model=model,
            token_counter=token_counter,
            runtime_envelope_tokens=runtime_envelope_tokens,
        )

    async def invoke_stage(self, template, stage_payload: dict[str, Any]) -> SchemaAgentResult:
        """只调用已经进入 canonical Stage input 的精确契约。"""

        semantic_input = stage_payload.get("semantic_input")
        program_binding = stage_payload.get("program_binding")
        if not isinstance(semantic_input, dict) or not isinstance(program_binding, dict):
            raise ValueError("Schema Agent stage payload is malformed")
        expected = self.stage_payload(
            template,
            semantic_input,
            program_binding=program_binding,
        )
        if stage_payload != expected:
            raise ValueError("Schema Agent stage payload does not match the active model contract")
        return await self.invoke(template, semantic_input)

    async def invoke(self, template, payload: dict) -> SchemaAgentResult:
        errors = list(Draft202012Validator(template.input_schema).iter_errors(payload))
        if errors:
            raise ValueError(schema_validation_detail("Agent input schema failed", errors[0]))
        return await self.invoke_schema(
            identifier=template.identifier,
            role=template.model_role,
            prompt=template.prompt,
            output_schema=template.output_schema,
            payload=payload,
            skill_text=template.skill_text,
        )

    async def invoke_schema(
        self,
        *,
        identifier: str,
        role: Literal["worker", "reviewer", "adjudicator"],
        prompt: str,
        output_schema: dict,
        payload: dict,
        skill_text: str | None = None,
    ) -> SchemaAgentResult:
        result = await self._runtime.invoke(
            SchemaAgentRequest(
                template_id=identifier,
                model_role=role,
                prompt=prompt,
                input_payload=payload,
                output_schema=output_schema,
                skill_text=skill_text,
            )
        )
        errors = list(Draft202012Validator(output_schema).iter_errors(result.payload))
        if errors:
            raise SchemaAgentOutputError(
                schema_validation_detail("Agent output schema failed", errors[0]),
                raw_model_output=result.raw_model_output,
                metrics=result.metrics,
            )
        return result


