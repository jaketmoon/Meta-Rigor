"""The installable MetaRigor command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from metarigor.application.service import capabilities, execute, handle_request


def _json(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None)


def _read_json(path: str | None) -> Any:
    raw = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON: {error}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metarigor", description="Agent-ready evidence review CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="check that the local service is available")
    sub.add_parser("capabilities", help="print the Agent integration contract")

    run = sub.add_parser("run", help="execute one operation from JSON")
    run.add_argument("operation", choices=(
        "health", "capabilities", "validate", "hash", "stage",
        "evidence-acquisition", "data-extraction", "risk-of-bias",
        "evidence-certainty", "manuscript",
    ))
    run.add_argument("--input", "-i", metavar="FILE", help="JSON input file; stdin when omitted")
    run.add_argument("--pretty", action="store_true")

    stage = sub.add_parser("stage", help="dispatch a request to one evidence-synthesis stage")
    stage.add_argument("name", choices=(
        "evidence-acquisition", "data-extraction", "risk-of-bias",
        "evidence-certainty", "manuscript",
    ))
    stage.add_argument("--input", "-i", metavar="FILE", help="stage input JSON; stdin when omitted")
    stage.add_argument("--pretty", action="store_true")

    agent = sub.add_parser("agent", help="serve newline-delimited JSON requests over stdin/stdout")
    agent.add_argument("--once", action="store_true", help="process one request and exit")
    agent.add_argument("--pretty", action="store_true", help="pretty-print each response (not recommended for pipes)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "health":
        print(_json(execute("health", {})))
        return 0
    if args.command == "capabilities":
        print(_json(capabilities(), pretty=True))
        return 0
    if args.command == "stage":
        try:
            payload = _read_json(args.input)
            result = execute(args.name, payload)
        except Exception as error:
            print(_json({"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}))
            return 1
        print(_json(result, pretty=args.pretty))
        return 0
    if args.command == "run":
        try:
            payload = _read_json(args.input) if args.operation not in {"health", "capabilities"} else {}
            result = execute(args.operation, payload)
        except Exception as error:
            print(_json({"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}))
            return 1
        print(_json(result, pretty=args.pretty))
        return 0
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            response = handle_request({"operation": "__invalid_json__", "input": {}})
            response["error"] = {"type": "JSONDecodeError", "message": str(error)}
            response["ok"] = False
        else:
            response = handle_request(request)
        print(_json(response, pretty=args.pretty), flush=True)
        if args.once:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

