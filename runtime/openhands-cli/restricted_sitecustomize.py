"""MetaRigor OpenHands CLI 实验限制层。

该模块由 Python 的 ``sitecustomize`` 机制在 OpenHands CLI 启动前加载。它只收窄上游
1.16.0 的运行能力：关闭 retry、condenser、skills、MCP 和 Task/subagent tool，并固定
单次 conversation 的最大迭代数。实验仍执行上游 OpenHands CLI 的 headless loop。
"""

from __future__ import annotations

import os
from typing import Any

from openhands.sdk import AgentContext
from openhands.sdk.llm import llm as sdk_llm_module
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool
from openhands_cli import setup
from openhands_cli.stores import agent_store


def _restricted_runtime_config(
    self: Any,
    agent: Any,
    session_id: str | None = None,
    *,
    critic_disabled: bool = False,
) -> Any:
    del self, session_id, critic_disabled
    llm = agent.llm.model_copy(
        update={
            "num_retries": 0,
            "temperature": 0,
            "max_output_tokens": int(os.environ.get("METARIGOR_MAX_OUTPUT_TOKENS", "8192")),
        }
    )
    context = AgentContext(
        skills=[],
        system_message_suffix=(
            "Your current working directory is /work. "
            "Use only /input and /work; network use other than the configured model endpoint "
            "is prohibited by the experiment runtime."
        ),
        load_user_skills=False,
        load_public_skills=False,
    )
    return agent.model_copy(
        update={
            "llm": llm,
            "tools": [
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
                Tool(name=TaskTrackerTool.name),
            ],
            "mcp_config": {},
            "agent_context": context,
            "condenser": None,
            "critic": None,
        }
    )


agent_store.AgentStore._apply_runtime_config = _restricted_runtime_config
setup.register_builtins_agents = lambda **_: None

# SDK 的外层 retry 和 LiteLLM/OpenAI transport 的 retry 是两层独立机制。前者由
# ``LLM.num_retries=0`` 关闭；这里显式给后者传 ``max_retries=0``，保证一次 agent step
# 最多对应一次真实 provider request。
_upstream_litellm_completion = sdk_llm_module.litellm_completion


def _single_attempt_litellm_completion(*args: Any, **kwargs: Any) -> Any:
    kwargs["max_retries"] = 0
    return _upstream_litellm_completion(*args, **kwargs)


sdk_llm_module.litellm_completion = _single_attempt_litellm_completion

_upstream_conversation = setup.Conversation


def _restricted_conversation(*args: Any, **kwargs: Any) -> Any:
    kwargs["max_iteration_per_run"] = int(
        os.environ.get("METARIGOR_MAX_ITERATIONS", "100")
    )
    return _upstream_conversation(*args, **kwargs)


setup.Conversation = _restricted_conversation
