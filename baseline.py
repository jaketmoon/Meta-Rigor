"""Run a native CLI baseline once against an isolated candidate-only input package."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, shutil, subprocess
from pathlib import Path
from reproduce import ROOT
from metarigor.manuscript.ms_codex_natural_output import CodexManuscriptNaturalOutputExecutor
from metarigor.data_extraction.experiment_v5.openhands_cli import _execute_case_sync
from metarigor.data_extraction.experiment_v3.codex_cli_v3_proxy import LoopbackResponsesProxy

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
    command=Path(shutil.which('codex') or 'codex').resolve()
    if a.system=='codex':
        version=subprocess.check_output([str(command),'--version'],text=True).strip()
        if '0.142.4' not in version:raise ValueError('Paper baseline requires Codex 0.142.4; found '+version)
    a.output.mkdir(parents=True,exist_ok=False)
    if a.stage=='ec':
        packages=ec_full_packages(a.output/'inputs',a.case)
        prompt=prompt.replace(
            'The complete permitted case input is the read-only input.json under INPUT_ROOT printed below.',
            'The permitted case input is input.json and the source_artifacts listed in input-manifest.json '
            'under INPUT_ROOT. These are shared with MR Full for both Codex CLI and OpenHands CLI. '
            'Read full_domain_inputs for all five domains: they contain the exact Full domain facts, '
            'including supplemental forest-plot transcriptions and text evidence. Use these alongside '
            'the candidate record and supplied source files; do not omit the supplemental evidence.')
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
                await executor.execute(case_id=case.name,invocation_root=a.output/case.name,input_root=case)
            finally:await proxy.close()
        else:
            await asyncio.to_thread(_execute_case_sync,case_id=case.name,invocation_root=a.output/case.name,
                input_root=case,api_key=key,base_url=url,image_id=a.image,timeout_seconds=1800,step_limit=100,
                prompt=text+'\nINPUT_ROOT: /input\nWorkspace: /work\n',request_model=a.model,response_model=a.response_model)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['ea-initial','ea-fulltext','de','rob','ec','ms'])
    p.add_argument('system',choices=['codex','openhands'])
    p.add_argument('--case',default='all');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='deepseek-v4-flash')
    p.add_argument('--response-model',default='deepseek-v4-flash',help='Actual response model identifier returned by your gateway')
    p.add_argument('--image',default='metarigor-openhands-cli:1.16.0-r3')
    a=p.parse_args();a.output=a.output.resolve()
    cases={path.name for path in (ROOT/'inputs/cli'/a.stage).iterdir() if path.is_dir()}
    if not cases or (a.case!='all' and a.case not in cases):
        p.error('Unknown or empty case selection. Available: '+', '.join(sorted(cases)))
    for name in ['METARIGOR_AGENT_GATEWAY_API_KEY','METARIGOR_AGENT_GATEWAY_BASE_URL']:
        if not os.environ.get(name):p.error('Set '+name)
    asyncio.run(run(a))
if __name__=='__main__':main()
