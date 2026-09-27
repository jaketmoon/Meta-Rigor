from __future__ import annotations

import json
def locally_rejected_tool_call_count(stderr: bytes) -> int:
    """统计 Codex router 已拒绝、因而未执行的 tool call。"""

    count = 0
    for line in stderr.splitlines():
        if b"codex_core::tools::router" not in line:
            continue
        if b"unsupported call:" in line or (b"Fatal error: tool " in line and b" invoked" in line):
            count += 1
    return count


def successful_tool_call_count(stdout: bytes) -> int:
    """返回 JSONL 可直接证明已开始执行的 tool call 数。"""

    count = 0
    for raw_line in stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict) or event.get("type") != "item.started":
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") not in {
            None,
            "reasoning",
            "agent_message",
        }:
            count += 1
    return count


