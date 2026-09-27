"""Small, stable service boundary used by the CLI and external Agents.

The research modules remain available below this boundary, while integrations
only need to understand a versioned JSON request/response envelope.
"""
from __future__ import annotations

import hashlib
from typing import Any

PROTOCOL_VERSION = "metarigor-agent-v1"
STAGES = {
    "evidence-acquisition": "EA",
    "data-extraction": "DE",
    "risk-of-bias": "RoB",
    "evidence-certainty": "EC",
    "manuscript": "MS",
}


def capabilities() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "transport": "jsonl-stdio",
        "operations": {
            "health": {"input": "object", "output": "health envelope"},
            "capabilities": {"input": "object", "output": "capability catalog"},
            "validate": {
                "input": {"value": "any", "schema": "JSON Schema object"},
                "output": "validation result",
            },
            "hash": {
                "input": {"value": "string", "algorithm": "sha256"},
                "output": "content digest",
            },
            "stages": {
                "input": {"stage": "EA|DE|RoB|EC|MS", "input": "object"},
                "output": "stage dispatch metadata",
            },
        },
        "stages": STAGES,
    }


def health() -> dict[str, Any]:
    return {"status": "ok", "protocol": PROTOCOL_VERSION}


def execute(operation: str, payload: Any) -> dict[str, Any]:
    """Execute one product operation without writing to stdout."""
    if operation == "health":
        return health()
    if operation == "capabilities":
        return capabilities()
    if operation == "validate":
        try:
            from jsonschema import Draft202012Validator
        except ModuleNotFoundError as error:
            raise RuntimeError("validate requires the 'jsonschema' package; install project dependencies") from error
        if not isinstance(payload, dict) or "schema" not in payload:
            raise ValueError("validate input requires a JSON Schema under 'schema'")
        schema = payload["schema"]
        value = payload.get("value")
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(value),
            key=lambda error: list(error.absolute_path),
        )
        return {
            "valid": not errors,
            "errors": [
                {
                    "message": error.message,
                    "path": list(error.absolute_path),
                    "schema_path": list(error.absolute_schema_path),
                }
                for error in errors
            ],
        }
    if operation == "hash":
        if not isinstance(payload, dict) or not isinstance(payload.get("value"), str):
            raise ValueError("hash input requires a string under 'value'")
        algorithm = payload.get("algorithm", "sha256")
        if algorithm != "sha256":
            raise ValueError("only sha256 is supported")
        return {"algorithm": algorithm, "digest": hashlib.sha256(payload["value"].encode()).hexdigest()}
    if operation in STAGES or operation == "stage":
        if operation == "stage":
            if not isinstance(payload, dict) or payload.get("stage") not in STAGES:
                raise ValueError("stage input requires one of: " + ", ".join(STAGES))
            stage_name = payload["stage"]
            stage_input = payload.get("input", {})
        else:
            stage_name = operation
            stage_input = payload
        if not isinstance(stage_input, dict):
            raise ValueError("stage input must be a JSON object")
        return {
            "stage": STAGES[stage_name],
            "name": stage_name,
            "status": "ready",
            "input": stage_input,
            "message": "Stage request accepted; use the stage-specific runner for model execution.",
        }
    raise ValueError(f"unsupported operation: {operation}")


def handle_request(request: Any) -> dict[str, Any]:
    """Turn an untrusted JSON request into a stable response envelope."""
    request_id = request.get("id") if isinstance(request, dict) else None
    try:
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        operation = request.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ValueError("request requires a non-empty 'operation'")
        result = execute(operation, request.get("input", {}))
        return {"protocol": PROTOCOL_VERSION, "id": request_id, "ok": True, "result": result}
    except Exception as error:  # boundary: errors must be serializable for an Agent
        return {
            "protocol": PROTOCOL_VERSION,
            "id": request_id,
            "ok": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }

