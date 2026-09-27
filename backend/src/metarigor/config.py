from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from pydantic import BaseModel, SecretStr
class Settings(BaseModel):
    runs_root: Path = Path("output")
    agent_gateway_base_url: str | None = None
    agent_gateway_api_key: SecretStr | None = None
    evidence_fast_model: str = "deepseek-v4-flash"
    evidence_deep_model: str = "deepseek-v4-flash"
    agent_call_timeout_seconds: float = 300
    agent_call_max_turns: int = 3
    agent_structured_output_mode: str = "json_text"
    agent_context_window_tokens: int = 200000
    agent_reserved_output_tokens: int = 8192


@lru_cache
def get_settings():
    key = os.environ.get("METARIGOR_AGENT_GATEWAY_API_KEY")
    return Settings(agent_gateway_base_url=os.environ.get("METARIGOR_AGENT_GATEWAY_BASE_URL"),
                    agent_gateway_api_key=SecretStr(key) if key else None)


