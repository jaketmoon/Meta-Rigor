# MetaRigor

MetaRigor is an agent-ready evidence review service. It exposes a stable JSON interface through an installable CLI, so external agents can use it without knowing the internal Python modules.

## Installation

Use Python 3.12 or later. The commands below use a Linux or macOS shell. On Windows, use WSL and keep the checkout and virtual environment in the Linux filesystem, because some frozen input filenames contain colons.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[research]'
metarigor verify
```

`metarigor verify` checks frozen resource hashes, input schemas, and case inventories without a model gateway. A complete package prints `OK: 1443 input/resource hashes`. The same command works from a wheel installed with the `research` extra. For a source checkout, the equivalent is `python verify.py`.

## Reproduce the paper

The live experiments need a gateway account with access to the models used in the selected condition. Set the base URL to the gateway API root, normally ending in `/v1`. Enter the key locally and keep it out of the repository.

```bash
export METARIGOR_AGENT_GATEWAY_BASE_URL='https://your-gateway.example/v1'
read -r -s METARIGOR_AGENT_GATEWAY_API_KEY
export METARIGOR_AGENT_GATEWAY_API_KEY
```

To run one Data Extraction (DE) case, use a fresh output directory:

```bash
metarigor reproduce de --case song-2025 --output output/de-song
```

This sends the frozen case to `deepseek-v4-flash` through an OpenAI-compatible Chat Completions endpoint. The result is `output/de-song/song-2025.json`, with the model's candidate text, citation checks, and a status. Inspect the status before using the result. This run uses the gateway account's quota.

The runner also accepts the following stages. Each run needs a new `--output` directory. Use `metarigor reproduce --help` for the options.

| Stage | Conditions and inputs |
| --- | --- |
| `ea` (Evidence Acquisition) | `full`, `no-fast`, `no-deep`, `dual-deepseek`. Select `--phase initial` or `--phase fulltext`. |
| `de` (Data Extraction) | `full`, `no-task-contract`. |
| `rob` (Risk of Bias) | `full`, `no-task-contract`, `no-code-logic`. |
| `ec` (Evidence Certainty) | `full`, `no-task-contract`, `no-code-logic`. The full and no-code-logic conditions use the fixed 20-case set. |
| `ms` (Manuscript) | `full`, `no-task-contract`, `no-code-logic`. The latter two require `--source` pointing to a previous full MS output directory. |
| `judge` (Manuscript evaluation) | Requires `--source` pointing to the MS output directory. See `--judge-model`. |

The Chat Completions routes must support the structured output modes used by the selected stage. For EA, the FAST assessor uses the Claude Agent SDK and an Anthropic-compatible Messages route at the same gateway. Standard EA uses `gpt-5.6-luna` for DEEP and `gpt-5.6-terra` for adjudication. The `dual-deepseek` condition uses `deepseek-v4-flash` via the Claude Agent SDK.

### Run the CLI baselines

The Codex baseline requires Codex CLI `0.142.4` and a gateway that serves the Responses API for the selected model. Install the pinned version into the project's virtual environment directory so it can coexist with your regular Codex installation:

```bash
npm install --prefix .venv/codex-baseline @openai/codex@0.142.4
codex_binary=$(find .venv/codex-baseline/node_modules/@openai -type f -path '*/vendor/*/bin/codex')
"$codex_binary" --version
metarigor baseline de codex --case song-2025 \
  --codex-command "$codex_binary" \
  --output output/codex-de-song
```

For OpenHands, start Docker and build the image from the supplied runtime directory in the source checkout. The image uses OpenHands `1.16.0` and the gateway must serve Chat Completions.

```bash
docker build -t metarigor-openhands-cli:1.16.0-r3 runtime/openhands-cli
metarigor baseline de openhands --case song-2025 --output output/openhands-de-song
```

Both baselines default to `deepseek-v4-flash`. Set `--response-model` if the gateway returns a different identifier for the requested model. Each case has an `invocation-summary.json` with runtime and failure details. The checkout scripts `python reproduce.py` and `python baseline.py` accept the same arguments as their corresponding commands.

## CLI

Check that the local CLI responds:

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

The `stage` command and its JSON operations return stage metadata. Use `metarigor reproduce` to run experiments. Use `metarigor verify` to check the installed resources.

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

The CLI exposes the research runners as subcommands and the JSONL adapter through the `metarigor-agent-v1` protocol.

Project documentation, code comments, and active method prompts are in English. Original source documents, frozen case data, and reference answers retain their original language to preserve verbatim evidence and provenance hashes. Translating prompts changes model inputs; new runs are not byte-identical reproductions of runs using the original prompts.

## Development Checks

```bash
PYTHONPATH=backend/src python3 -m compileall -q backend/src
PYTHONPATH=backend/src python3 -m metarigor health
```
