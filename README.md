# MetaRigor

MetaRigor is an agent-ready evidence review service. It exposes a stable JSON interface through an installable CLI, so external agents can use it without knowing the internal Python modules.

## Installation

Use Python 3.12 or later. From the repository root on Linux or macOS (or WSL on Windows), run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
metarigor health
```

## CLI

View agent capabilities and the protocol version:

```bash
metarigor capabilities
```

Available basic operations:

- `health`: Check service health
- `capabilities`: Return the capability list
- `hash`: Compute the SHA-256 hash of a string
- `validate`: Validate input against a JSON Schema
- `stage`: Inspect metadata for evidence acquisition, data extraction, risk of bias, evidence certainty, or manuscript preparation

`metarigor run` reads JSON from stdin or from a file supplied with `--input`.

For example, compute a hash:

```bash
echo '{"value":"hello"}' | metarigor run hash
```

For example, validate JSON:

```bash
cat <<'JSON' | metarigor run validate
{
  "schema": {"type": "object", "required": ["case_id"]},
  "value": {"case_id": "demo"}
}
JSON
```

## Agent Integration

`metarigor agent` uses JSONL (one JSON object per line) over stdin/stdout, making it suitable for other agents, workflow orchestrators, and sandboxed processes.

Start the service:

```bash
metarigor agent
```

Send requests:

```json
{"id":"req-1","operation":"health","input":{}}
{"id":"req-2","operation":"hash","input":{"value":"hello"}}
```

Each response echoes the request ID. For example, the hash request returns:

```json
{
  "protocol": "metarigor-agent-v1",
  "id": "req-2",
  "ok": true,
  "result": {
    "algorithm": "sha256",
    "digest": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
  }
}
```

A request with an unsupported operation returns a structured error:

```json
{
  "protocol": "metarigor-agent-v1",
  "id": "req-3",
  "ok": false,
  "error": {
    "type": "ValueError",
    "message": "unsupported operation: unknown"
  }
}
```

For a single invocation, use:

```bash
echo '{"id":"one-shot","operation":"health","input":{}}' \
  | metarigor agent --once
```

## Project Structure

```text
backend/src/metarigor/
├── application/          # Product service boundary and operation dispatch
├── cli/                  # CLI entry points and JSONL agent adapter
├── adapters/             # External model and runtime adapters
├── data_extraction/      # Data extraction
├── evidence_acquisition/ # Evidence acquisition
├── evidence_certainty/   # Evidence certainty assessment
├── manuscript/           # Manuscript capabilities
└── risk_of_bias_v3/      # Risk-of-bias assessment
```

## Development Checks

```bash
PYTHONPATH=backend/src python3 -m compileall -q backend/src
PYTHONPATH=backend/src python3 -m metarigor health
```
