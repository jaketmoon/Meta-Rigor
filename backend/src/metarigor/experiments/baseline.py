"""Run Codex or OpenHands against isolated case inputs."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, shlex, shutil, subprocess
from pathlib import Path
from .reproduce import MODEL_ENV, ROOT, check_gateway, configured_model
from metarigor.manuscript.ms_codex_natural_output import CodexManuscriptNaturalOutputExecutor
from metarigor.data_extraction.experiment_v5.openhands_cli import _execute_case_sync
from metarigor.data_extraction.experiment_v3.codex_cli_v3_proxy import LoopbackResponsesProxy

CODEX_VERSION = "0.142.4"
OPENHANDS_IMAGE = "metarigor-openhands-cli:1.16.0-r3"
OPENHANDS_DOCKERFILE = ROOT / "runtime/openhands-cli"


class BaselineSetupError(RuntimeError):
    pass


def codex_runtime(binary):
    command = shutil.which(binary)
    if command is None:
        raise BaselineSetupError(
            f"Codex CLI {CODEX_VERSION} is required. Install it or pass --codex-command."
        )
    try:
        version = subprocess.check_output([command, "--version"], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise BaselineSetupError(f"Cannot read the Codex CLI version: {error}") from error
    if version != f"codex-cli {CODEX_VERSION}":
        raise BaselineSetupError(f"The Codex runner requires version {CODEX_VERSION}. Found {version}.")
    return Path(command).resolve(), version


def check_openhands_image(image):
    if shutil.which("docker") is None:
        raise BaselineSetupError("The OpenHands baseline requires Docker on PATH.")
    try:
        server = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                capture_output=True, timeout=15)
        if server.returncode:
            raise BaselineSetupError("Cannot reach Docker. Start the Docker engine and check access.")
        inspected = subprocess.run(["docker", "image", "inspect", image],
                                   capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise BaselineSetupError(f"Cannot check the OpenHands Docker image: {error}") from error
    if inspected.returncode:
        build = shlex.join(["docker", "build", "-t", image, str(OPENHANDS_DOCKERFILE)])
        raise BaselineSetupError(f"OpenHands image {image!r} is unavailable. Build it with: {build}")


def ec_full_packages(root, selected_case):
    """Build both CLI inputs from Full's candidate/domain evidence, never gold."""
    from metarigor.evidence_certainty.benchmark_v1 import load_benchmark_catalog as load_v1
    from metarigor.evidence_certainty.benchmark_v2.dataset import (
        load_benchmark_catalog, load_review_level_sufficiency_contract,
        review_sufficiency_artifact_bytes,
    )
    from metarigor.evidence_certainty.benchmark_v2.live import _prepared_candidates, _domain_input
    from metarigor.evidence_certainty.models import GRADE_DOMAINS, canonical_json

    catalog=load_benchmark_catalog()
    contract=load_review_level_sufficiency_contract(catalog)
    v1={case.case_id:case for case in load_v1().cases}
    reviews={case.case_id:case for case in catalog.review_cases}
    dataset=ROOT/'backend/src/metarigor/evidence_certainty/benchmark_v2/dataset'
    for prepared in _prepared_candidates(catalog=catalog,review_contract=contract):
        if selected_case not in ('all',prepared.case_id):continue
        case_root=root/prepared.case_id
        case_root.mkdir(parents=True,exist_ok=False)
        payload=prepared.input.model_dump(mode='json')
        payload['full_domain_inputs']=[
            _domain_input(prepared,domain).model_dump(mode='json') for domain in GRADE_DOMAINS]
        artifacts=[]
        if prepared.case_id in v1:
            for document in v1[prepared.case_id].source_documents:
                artifacts.append((document.path,(ROOT/document.path).read_bytes(),document.sha256))
        else:
            review=reviews[prepared.case_id]
            source_manifest=json.loads((ROOT/review.source.manifest_path).read_bytes())
            for key in ('candidate','raw_source'):
                source=source_manifest[key]
                artifacts.append((source['path'],(dataset/source['path']).read_bytes(),source['sha256']))
            case_contract=contract.model_copy(update={'cases':tuple(
                case for case in contract.cases if case.case_id==prepared.case_id)})
            artifacts.extend(review_sufficiency_artifact_bytes(case_contract))
        bindings=[]
        for relative,content,expected in artifacts:
            if hashlib.sha256(content).hexdigest()!=expected:
                raise ValueError('EC Full source hash mismatch: '+relative)
            target=case_root/'sources'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
            bindings.append({'path':target.relative_to(case_root).as_posix(),'sha256':expected})
        content=canonical_json(payload)
        (case_root/'input.json').write_bytes(content)
        manifest={
            'schema_version':'ec-v2-cli-full-input-v1','case_id':prepared.case_id,
            'benchmark_id':prepared.input.benchmark_id,
            'input_contract':'MR_FULL_EC_SHARED_CANDIDATE_AND_DOMAIN_EVIDENCE',
            'input_json_path':'input.json','input_json_sha256':hashlib.sha256(content).hexdigest(),
            'accessible_files':['input-manifest.json','input.json']+[b['path'] for b in bindings],
            'source_artifacts':bindings,'gold_values_loaded':False,'reference_answers_loaded':False,
            'mr_outputs_loaded':False,'evaluator_loaded':False,
        }
        (case_root/'input-manifest.json').write_bytes(canonical_json(manifest))
    return root

async def run(a):
    packages=ROOT/'inputs/cli'/a.stage
    prompt=(packages/'prompt.txt').read_text()
    key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'];url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL']
    if a.system=='codex':
        command, version = codex_runtime(getattr(a, 'codex_command', 'codex'))
    else:
        check_openhands_image(a.image)
    try:
        a.output.mkdir(parents=True,exist_ok=False)
    except OSError as error:
        raise BaselineSetupError(f'--output must be a new writable directory: {error}') from error
    if a.ec_benchmark == 'ec-ex80':
        from metarigor.evidence_certainty.benchmark_v2.ec_ex80 import write_cli_packages

        packages = write_cli_packages(a.output / 'inputs', a.ec_ex80_data, a.case)
    elif a.stage=='ec':
        packages=ec_full_packages(a.output/'inputs',a.case)
        prompt=prompt.replace(
            'The complete permitted case input is the read-only input.json under INPUT_ROOT printed below.',
            'The permitted case input is input.json and the source_artifacts listed in input-manifest.json '
            'under INPUT_ROOT. These are shared with MR Full for both Codex CLI and OpenHands CLI. '
            'Read full_domain_inputs for all five domains: they contain the exact Full domain facts, '
            'including supplemental forest-plot transcriptions and text evidence. Use these alongside '
            'the candidate record and supplied source files; do not omit the supplemental evidence.')
    failures = []
    for case in sorted(p for p in packages.iterdir() if p.is_dir()):
        if a.case not in ('all',case.name):continue
        text=prompt.replace('CASE_ID',case.name)
        if a.system=='codex':
            proxy=LoopbackResponsesProxy(upstream_base_url=url,api_key=key,upstream_model_id=a.model)
            await proxy.start()
            try:
                executor=CodexManuscriptNaturalOutputExecutor(codex_command=command,codex_version=version,
                    codex_binary_sha256=hashlib.sha256(command.read_bytes()).hexdigest(),
                    base_url=proxy.local_base_url,api_key='metarigor-local-dummy',timeout_seconds=1800,
                    provider_model=a.model,response_model=a.response_model,
                    prompt_template=text,developer_instructions=(ROOT/'inputs/cli'/a.stage/'developer.txt').read_text())
                summary = await executor.execute(case_id=case.name,invocation_root=a.output/case.name,input_root=case)
            finally:await proxy.close()
        else:
            summary = await asyncio.to_thread(_execute_case_sync,case_id=case.name,invocation_root=a.output/case.name,
                input_root=case,api_key=key,base_url=url,image_id=a.image,timeout_seconds=1800,step_limit=100,
                prompt=text+'\nINPUT_ROOT: /input\nWorkspace: /work\n',request_model=a.model,response_model=a.response_model)
        if summary["harness_failures"]:
            failures.append(case.name)
    return failures

def main(argv=None, *, prog=None):
    p=argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument('stage',choices=['ea-initial','ea-fulltext','de','rob','ec','ms'])
    p.add_argument('system',choices=['codex','openhands'])
    p.add_argument('--case',default='all');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ec-benchmark', choices=['v2', 'ec-ex80'], default='v2',
                   help='EC case set (default v2)')
    p.add_argument('--ec-ex80-data', type=Path, help='Prepared EC-ex80 input bundle (default output/ec-ex80-data)')
    p.add_argument('--model', metavar='MODEL', help=f'Gateway model ID (defaults to {MODEL_ENV})')
    p.add_argument('--response-model', metavar='MODEL', help='Gateway response model ID if different from --model')
    p.add_argument('--image', default=OPENHANDS_IMAGE, help='Local OpenHands Docker image')
    p.add_argument('--codex-command', default='codex', help=f'Path to the Codex {CODEX_VERSION} executable')
    a=p.parse_args(argv);a.output=a.output.resolve()
    if a.stage != 'ec' and a.ec_benchmark != 'v2':
        p.error('--ec-benchmark applies only to EC')
    if a.ec_ex80_data is not None and a.ec_benchmark != 'ec-ex80':
        p.error('--ec-ex80-data requires --ec-benchmark ec-ex80')
    if a.ec_benchmark == 'ec-ex80':
        from metarigor.evidence_certainty.benchmark_v2.ec_ex80 import DEFAULT_DATA, case_ids, load_bundle

        a.ec_ex80_data = a.ec_ex80_data or DEFAULT_DATA
        try:
            load_bundle(a.ec_ex80_data)
        except (OSError, ValueError) as error:
            p.error(f'Cannot read prepared EC-ex80 inputs at {a.ec_ex80_data}: {error}')
        cases = set(case_ids())
    else:
        cases={path.name for path in (ROOT/'inputs/cli'/a.stage).iterdir() if path.is_dir()}
    if not cases or (a.case!='all' and a.case not in cases):
        p.error('Unknown or empty case selection. Available: '+', '.join(sorted(cases)))
    check_gateway(p)
    a.model = configured_model(a.model)
    if not a.model:
        p.error(f'Set {MODEL_ENV} or pass --model with a model ID from your gateway.')
    a.response_model = (a.response_model or '').strip() or a.model
    try:
        failures = asyncio.run(run(a))
    except BaselineSetupError as error:
        p.error(str(error))
    if failures:
        p.exit(1, f'Baseline failed for {", ".join(failures)}. See invocation-summary.json under {a.output}.\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
