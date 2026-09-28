# MetaRigor

MetaRigor is an agent-ready evidence review service. It exposes a stable JSON interface through an installable CLI, so external agents can use it without knowing the internal Python modules.

## Installation

Requires Python 3.12 or later:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

To run JSON Schema validation, install the full dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

After installation, run:

```bash
metarigor --help
```

Alternatively, run directly without installing:

```bash
PYTHONPATH=backend/src python3 -m metarigor --help
```

## CLI

Check service health:

```bash
metarigor health
```

View agent capabilities and the protocol version:

```bash
metarigor capabilities
```

Available basic operations:

- `health`: Check service health
- `capabilities`: Return the capability list
- `hash`: Compute the SHA-256 hash of a string
- `validate`: Validate input against a JSON Schema

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

Each request returns a JSON object:

```json
{
  "protocol": "metarigor-agent-v1",
  "id": "req-2",
  "ok": true,
  "result": {
    "algorithm": "sha256",
    "digest": "..."
  }
}
```

Errors are returned as structured responses, not as log-formatted output:

```json
{
  "protocol": "metarigor-agent-v1",
  "id": "req-3",
  "ok": false,
  "error": {
    "type": "ValueError",
    "message": "..."
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

Research experiment scripts and frozen datasets remain in the repository, but product integrations should use the `metarigor` CLI and the `metarigor-agent-v1` protocol.

Project documentation, code comments, and active method prompts are in English. Original source documents, frozen case data, and reference answers retain their original language to preserve verbatim evidence and provenance hashes. Translating prompts changes model inputs; new runs are not byte-identical reproductions of runs using the original prompts.

## Development Checks

```bash
PYTHONPATH=backend/src python3 -m compileall -q backend/src
PYTHONPATH=backend/src python3 -m metarigor health
```
