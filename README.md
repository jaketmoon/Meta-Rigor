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

`metarigor verify` checks frozen resource hashes, input schemas, and case inventories without a model gateway. A complete package prints `OK: 1445 input/resource hashes`. The same command works from a wheel installed with the `research` extra. For a source checkout, the equivalent is `python verify.py`.

## Reproduce the paper

Fill in the API base URL and a model ID from your provider. Use the API base URL that precedes `/chat/completions`, including its version or routing prefix if needed. Enter the key at the prompt.

```bash
export METARIGOR_AGENT_GATEWAY_BASE_URL=''
export METARIGOR_MODEL=''
read -r -s METARIGOR_AGENT_GATEWAY_API_KEY
export METARIGOR_AGENT_GATEWAY_API_KEY
```

`METARIGOR_MODEL` supplies the model for both `reproduce` and `baseline`. Pass `--model` to select another model for one run. Each run needs a new `--output` directory.

| Stage | Conditions and inputs |
| --- | --- |
| `ea` (Evidence Acquisition) | `full`, `no-fast`, `no-deep`, `dual-deepseek`. Select `--phase initial` or `--phase fulltext`. |
| `de` (Data Extraction) | `full`, `no-task-contract`. |
| `rob` (Risk of Bias) | `full`, `no-task-contract`, `no-code-logic`. |
| `ec` (Evidence Certainty) | `full`, `no-task-contract`, `no-code-logic`. The full and no-code-logic conditions use the fixed 20-case set. |
| `ms` (Manuscript) | `full`, `no-task-contract`, `no-code-logic`. The latter two require `--source` pointing to a previous full MS output directory. |
| `judge` (Manuscript evaluation) | Requires `--source` pointing to the MS output directory. Use `--judge-model` to override the shared model and `--judge-format structured` to request structured scores. |

EA uses the paper's transports by default. FAST uses the Anthropic-compatible Messages API through the Claude Agent SDK, DEEP and adjudication use Chat Completions, and `dual-deepseek` uses Messages for all roles. Use `--ea-fast-model`, `--ea-deep-model`, and `--ea-adjudicator-model` if the roles need different model IDs. For providers that offer Chat Completions only, use `--ea-transport chat`. If the provider lacks native structured output, add `--structured-output prompt` for EA, RoB, EC, or a structured Judge. Outputs still undergo local schema validation.

```bash
metarigor reproduce ea --case turalde-mapili-2023 --ea-transport chat --structured-output prompt --output runs/ea
```

DE records both the requested and returned model IDs. Pass `--response-model` to require an exact return ID. For DE or MS, `--disable-thinking` sends the corresponding optional request parameter when your provider supports it. Use the model roles and transport settings reported in the paper for comparisons with its results.

EC's no-task-contract condition saves the first response, the calculator result, and the final request as separate stages under `cases/`. The deadline applies separately to each request. If final writing times out, the provisional report remains available and the result has no final `answer`.

Full MS uses optional value slots by default, as in the reference implementation. The verifier still checks Fact identifiers, placement, and required coverage. Directly written values remain subject to semantic review. Use `--ms-value-slots required` to run with the historical strict slot policy.

The EC stage layout and MS slot policy are adapted from the local MetaRigor reference checkout at `28863d22` (including the checkout's MS schema capacity adjustments). The relevant files are `backend/src/metarigor/evidence_certainty/benchmark_v2/task_contract_ablation.py` and `backend/src/metarigor/manuscript/publication_writing.py`.

### Run the CLI baselines

The Codex baseline requires Codex CLI `0.142.4` and a gateway that serves the Responses API for the selected model. Install the pinned version into the project's virtual environment directory so it can coexist with your regular Codex installation:

```bash
npm install --prefix .venv/codex-baseline @openai/codex@0.142.4
codex_binary=$(find .venv/codex-baseline/node_modules/@openai -type f -path '*/vendor/*/bin/codex')
"$codex_binary" --version
```

Pass the binary path with `--codex-command`.

For OpenHands, start Docker and build the image from the supplied runtime directory in the source checkout. The image uses OpenHands `1.16.0` and the gateway must serve Chat Completions.

```bash
docker build -t metarigor-openhands-cli:1.16.0-r3 runtime/openhands-cli
```

Both baselines use `METARIGOR_MODEL` or `--model`. Set `--response-model` if the gateway returns a different model ID. Use `metarigor baseline --help` for the options. Each case has an `invocation-summary.json` with runtime and failure details. The checkout scripts `python reproduce.py` and `python baseline.py` accept the same arguments as their corresponding commands.

### EC-ex80 extension

EC-ex80 combines the earlier EC-ex20 cohort and 60 additional review-level cases, with one binary outcome per review. Select it with `--ec-benchmark ec-ex80`. The original EC V2 cohort remains the default. EC-ex80 supports `full`, `no-task-contract`, `no-code-logic`, and both CLI baselines. Use `--case` to run a single case or omit it for all 80.

The [case catalog](inputs/ec-ex80/catalog.json) and [scoring labels](references/ec-ex80.json) ship with the code. The source documents have mixed reuse terms, so the prepared input bundle is supplied separately. Use `--ec-ex80-data PATH` to point to the prepared EC-ex80 folder, or place its `plan.json`, `prepared/`, and `inputs/` under `output/ec-ex80-data` in the working directory. The bundle comes from `var/benchmarks/ec-ex80-mr-gpt6-luna-subscription-20261111-r1` in the MetaRigor reference checkout. Validation pins the preparation plan, candidate inputs, and original source hashes. See the catalog for the original cohort and reference hashes.

```bash
metarigor verify --ec-ex80-data output/ec-ex80-data
metarigor reproduce ec --ec-benchmark ec-ex80 --case pmc-12960011-ec-ex3 --output runs/ec-ex80-smoke
python score.py ec-ex80 runs/ec-ex80-smoke/predictions.json
```

The live commands use the configured gateway. Full and `no-code-logic` export `predictions.json`. For `no-task-contract` or CLI outputs, transcribe the five explicit domain judgments and overall downgrade count into a JSON list with `case_id`, `domains` (an object keyed by domain), and `overall_downgrade_levels` (an integer or null). Keep unavailable judgments missing or null. Scoring always uses 400 domain judgments and 80 cases. MAE uses only explicit valid overall downgrade counts, and the reference labels come from sealed AI adjudication. An isolated case run verifies execution and still scores against the full denominator.

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
