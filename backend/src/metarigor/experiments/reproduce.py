"""Run a stage with a model gateway and write results to a new directory."""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import gzip
import json
import os
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

from metarigor.research import RESEARCH_ROOT as ROOT

MODEL_ENV = "METARIGOR_MODEL"


def configured_model(value):
    if value is None:
        value = os.environ.get(MODEL_ENV, os.environ.get("METARIGOR_DEEPSEEK_MODEL", ""))
    return value.strip() or None


def check_gateway(parser):
    missing = [
        name for name in ("METARIGOR_AGENT_GATEWAY_BASE_URL", "METARIGOR_AGENT_GATEWAY_API_KEY")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        parser.error("Set " + " and ".join(missing) + " for model execution. See README.md.")
    try:
        base_url = os.environ["METARIGOR_AGENT_GATEWAY_BASE_URL"]
        url = urlsplit(base_url)
        valid = (
            url.scheme in {"http", "https"} and url.hostname and url.username is None
            and url.port != 0 and not url.query and not url.fragment
            and not any(character.isspace() for character in base_url)
        )
    except ValueError:
        valid = False
    if not valid:
        parser.error("METARIGOR_AGENT_GATEWAY_BASE_URL must be an absolute HTTP(S) API base URL.")

def read(path):
    path=Path(path)
    if path.suffix=='.gz':return json.loads(gzip.decompress(path.read_bytes()))
    if not path.exists() and path.with_suffix(path.suffix+'.gz').exists():
        return json.loads(gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes()))
    return json.loads(path.read_bytes())

def json_paths(directory):
    return sorted({p.with_suffix('') if p.suffix=='.gz' else p for p in directory.glob('*.json*')})

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)

def binding(path):
    from metarigor.manuscript.publication_models import BoundInput
    return BoundInput(input_path=str(path.resolve()), expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def runtime(*, model, role='worker', timeout_seconds=300, **kwargs):
    from metarigor.adapters.agent_runtime.schema_agent import OpenAICompatibleJsonSchemaAgent
    return OpenAICompatibleJsonSchemaAgent(api_key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'],
        base_url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL'], model=model, role=role,
        temperature=0, timeout_seconds=timeout_seconds, **kwargs)

async def de(args):
    from metarigor.data_extraction.reference_kernel.executor import DeepSeekFlashExecutor
    from metarigor.data_extraction.natural_optimization.mechanism import bind_citations, SELECTION_INSTRUCTION
    paths=[p/'source_citations_selection.json' for p in sorted((ROOT/'inputs/de').iterdir())]
    condition='source_citations_selection' if args.condition=='full' else 'source_citations'
    model = args.model
    executor=DeepSeekFlashExecutor(base_url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL'],
        api_key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'], timeout_seconds=1800,
        thinking_mode="disabled" if args.disable_thinking else "provider_default",
        model_id=model, response_model=args.response_model)
    limiter=asyncio.Semaphore(4)
    async def one(path):
        cid=path.parent.name
        if args.case not in ('all',cid):return
        inp=read(path)
        if condition=='source_citations':
            inp['prompt']=inp['prompt'].replace('\n'+SELECTION_INSTRUCTION,'',1)
            inp['condition']=condition
        async with limiter:
            try:
                r=await executor.generate(task='binary_outcomes',row_id=0,prompt=inp['prompt'],max_output_tokens=32768)
                bound=bind_citations(r.content,inp['citation_map'])
                for span in bound['spans']:
                    v=span['source_span'];text=(path.parent/v['view_path'].removeprefix('source-package/')).read_bytes()
                    if hashlib.sha256(text).hexdigest()!=v['view_sha256'] or text.decode()[v['start_char']:v['end_char']]!=v['quote']:
                        raise ValueError('source citation hash/offset mismatch')
                save(args.output/(cid+'.json'),{'case_id':cid,'condition':condition,'candidate':r.content,
                    'requested_model':r.requested_model,'response_model':r.provider_model,
                    'finish_reason':r.finish_reason,'citations':bound, 'status':'COMPLETED_WITH_ISSUES' if r.finish_reason not in (None,'stop') or bound['issues'] else 'COMPLETE'})
            except Exception as error:
                save(args.output/(cid+'.json'),{'case_id':cid,'requested_model':model,'status':'FAILED','error':str(error)})
                return cid
    try:return [cid for cid in await asyncio.gather(*(one(p) for p in paths)) if cid is not None]
    finally:await executor.aclose()

async def rob(args):
    from metarigor.rob_reproduction import run_case, MeasuredDirectAgent
    agent=MeasuredDirectAgent(api_key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'],
        base_url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL'],worker_model=args.model,
        reviewer_model=args.model,timeout_seconds=300, provider_schema=args.structured_output != 'prompt')
    arm={'full':'full','no-task-contract':'minus_decomposition','no-code-logic':'minus_programmatic_decision'}[args.condition]
    limiter=asyncio.Semaphore(4)
    try:
        rows=await asyncio.gather(*(run_case(read(p),arm,args.output,agent,limiter)
            for p in json_paths(ROOT/'inputs/rob') if args.case in ('all',p.stem)),return_exceptions=True)
        save(args.output/'execution.json',[str(r) if isinstance(r,Exception) else r for r in rows])
    finally:await agent.aclose()

async def ec(args):
    timeout = getattr(args, "ec_timeout", None) or 300
    mode = "json_text" if args.structured_output == "prompt" else "provider_json_schema_prompt"
    agent = runtime(model=args.model, timeout_seconds=timeout, structured_output_mode=mode)
    try:
        await run_ec(args, agent, timeout)
    finally:
        await agent.aclose()


async def run_ec(args, agent, timeout):
    if args.ec_benchmark == 'ec-ex80':
        from metarigor.evidence_certainty.benchmark_v2.ec_ex80 import run

        await run(data_root=args.ec_ex80_data, output=args.output, selected_case=args.case,
                  condition=args.condition, runtime=agent, timeout_seconds=timeout)
        return
    if args.condition!='no-task-contract':
        from metarigor.evidence_certainty.benchmark_v2.live import run_live_benchmark
        result=await run_live_benchmark(runtime=agent,runs_root=args.output,validation_id='ec-fresh-run',
            mechanism='domain_aggregate' if args.condition=='full' else 'direct')
        save(args.output/'execution.json',result.model_dump(mode='json'))
        return

    from metarigor.evidence_certainty.benchmark_v2.task_contract_ablation import run_case

    skill = (ROOT / "inputs/ec/methods.txt").read_text(encoding="utf-8")
    limiter = asyncio.Semaphore(4)

    async def one(path):
        result = await run_case(
            output=args.output, case_id=path.stem, request=read(path), skill=skill,
            runtime=agent, limiter=limiter, timeout_seconds=timeout,
        )
        save(args.output / f"{path.stem}.json", result)

    await asyncio.gather(*(one(p) for p in json_paths(ROOT / "inputs/ec") if args.case in ("all", p.stem)))

async def ms(args):
    from metarigor.manuscript.publication_pipeline import LocalManuscriptV2Pipeline
    from metarigor.manuscript.publication_models import ManuscriptRunRequestV2
    from metarigor.manuscript.journal_profiles import profile_path
    from metarigor.cli.manuscript_runtime import manuscript_runtime
    agent=manuscript_runtime(include_adjudicator=True,explicit_disable_thinking=args.disable_thinking,writer_model=args.model,
        adjudicator_structured_output_mode='text_wrapped')
    limiter=asyncio.Semaphore(4)
    rows=[]
    try:
        for p in sorted((ROOT/'inputs/ms').iterdir()):
            if args.case not in ('all',p.name):continue
            req=ManuscriptRunRequestV2(schema_version='manuscript-run-request.v2',run_id=str(uuid4()),
                fact_package=binding(p/'fact-package.json'),journal_profile=binding(profile_path('generic_prisma_2020_meta_analysis_v2')),
                profile_id='generic_prisma_2020_meta_analysis_v2',writer_mode='LLM',
                value_slots_required=getattr(args, "ms_value_slots", None) == "required",
                reference_package=binding(p/'source-fact-package.json'),input_curation=binding(p/'input-curation-audit.json'))
            pipe=LocalManuscriptV2Pipeline.create(root=args.output,request=req,agent_runtime=agent,model_call_limiter=limiter)
            result=await pipe.run();rows.append({'case_id':p.name,'run_root':str(result.run_root),'status':str(result.status)})
        save(args.output/'execution.json',rows)
    finally:await agent.aclose()

async def ms_ablation(args):
    from metarigor.cli.manuscript_proxy_comparison import initialize_ablation
    from metarigor.manuscript.ms_ablation import _generate_case, _worker_runtime
    from metarigor.manuscript.publication_judge import PublicationManuscriptJudge
    full={}
    for row in read(args.source/'execution.json'):
        if args.case not in ('all',row['case_id']):continue
        candidate=PublicationManuscriptJudge(candidate_root=Path(row['run_root']),
            validation_root=args.output/'unused',agent_runtime=None)._load_candidate()
        full[row['case_id']]=({**row,'run_id':candidate.run_id},candidate)
    condition={'no-task-contract':'NO_TASKCONTRACT_DECOMPOSITION',
        'no-code-logic':'NO_PROGRAM_MODEL_RESPONSIBILITY_SPLIT'}[args.condition]
    agent=_worker_runtime(disable_thinking=args.disable_thinking, model=args.model)
    try:
        target,manifest=initialize_ablation(args.output,condition,full,agent)
        rows=[]
        for cid in full:
            row=await _generate_case(root=target,manifest=manifest,case_id=cid,runtime=agent,
                model_limiter=asyncio.Semaphore(4),run_id=str(uuid4()))
            row['run_root']=str(target/'runs'/row['run_id']);rows.append(row)
        save(args.output/'execution.json',rows)
    finally:await agent.aclose()

async def judge(args):
    from metarigor.manuscript.publication_judge import PublicationManuscriptJudge
    from metarigor.adapters.agent_runtime.schema_agent import RoleRoutedSchemaAgent
    structured = args.judge_format == 'structured'
    mode = 'text_wrapped'
    if structured:
        mode = 'json_text' if args.structured_output == 'prompt' else 'provider_json_object'
    judge_agent=runtime(role='adjudicator',model=args.model,max_output_tokens=32768 if structured else 12000,
        structured_output_mode=mode)
    agent=RoleRoutedSchemaAgent(worker=judge_agent,reviewer=judge_agent,adjudicator=judge_agent)
    try:
        rows=[]
        for row in read(args.source/'execution.json'):
            if args.case not in ('all',row['case_id']):continue
            result=await PublicationManuscriptJudge(candidate_root=Path(row['run_root']),
                validation_root=args.output/row['case_id'],agent_runtime=agent,score_mode=True,
                structured_score=structured,array_score=structured).run()
            rows.append({'case_id':row['case_id'],'weighted_score':result.weighted_score,'status':result.status,
                'report_path':str(result.report_path)})
        save(args.output/'scores.json',rows)
    finally:await judge_agent.aclose()

async def ea(args):
    from metarigor.evidence_acquisition.benchmark_v3.ablation import ground_output,reconcile
    from metarigor.adapters.agent_runtime.schema_agent import ClaudeSchemaAgent
    from metarigor.schema_agent import SchemaAgentRequest
    from metarigor.local_run import RunFolder
    def messages_agent(model):
        return ClaudeSchemaAgent(
            api_key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'],
            base_url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL'],
            worker_model=model, reviewer_model=model,
            timeout_seconds=300, max_turns=3, structured_output_mode='json_text',
        )

    agents = {}
    model_roles = {'FAST': 'worker', 'DEEP': 'reviewer', 'ADJUDICATOR': 'adjudicator'}
    paper_transport = args.ea_transport != 'chat'
    for role, model in args.ea_models.items():
        if paper_transport and (role == 'FAST' or args.condition == 'dual-deepseek'):
            agents[role] = messages_agent(model)
            if role == 'ADJUDICATOR':
                model_roles[role] = 'worker'
            continue
        mode = 'provider_json_schema'
        if role == 'FAST' or args.structured_output == 'prompt' or args.condition == 'dual-deepseek':
            mode = 'json_text'
        agents[role] = runtime(
            model=model, role=model_roles[role], structured_output_mode=mode,
            reasoning_effort='high' if paper_transport and role == 'ADJUDICATOR' else None,
        )
    limiter=asyncio.Semaphore(8)
    roles = args.ea_roles
    async def invoke(task,role,folder,ground):
        model_role = model_roles[role]
        agent = agents[role]
        requested_model = agent.binding(model_role)['model']
        try:
            async with limiter:
                result=await agent.invoke(SchemaAgentRequest(template_id=task['template_id'],model_role=model_role,
                    prompt=task['prompt'],input_payload=task['semantic_input'],output_schema=task['output_schema'],skill_text=task['skill_text']))
            value=result.payload
            if ground:
                spans=ground_output(value,task,folder)
                value={**value,'supporting_refs':value['supporting_source_refs'],'grounding_status':'RESOLVED','source_spans':spans}
            return {**value, 'requested_model': requested_model}
        except Exception as error:return {'status':'FAILED','requested_model':requested_model,'error':str(error)}
    try:
        for directory in sorted((ROOT/'inputs/ea').iterdir()):
            if args.case not in ('all',directory.name):continue
            folder=RunFolder(directory)
            if args.phase=='initial':
                async def one(item,tasks):
                    values=dict(zip(roles,await asyncio.gather(*(invoke(tasks[r],r,folder,False) for r in roles))))
                    # Dual DeepSeek changes the model backend only; its screening
                    # decision must follow the Full condition exactly.
                    retained=all(v.get('decision')=='INCLUDE_FOR_FULL_TEXT' for v in values.values())
                    return {'record_id':item,'retained':retained,'assessments':values}
                rows=await asyncio.gather(*(one(k,v) for k,v in read(directory/'initial.json').items()))
            else:
                inp=read(directory/'fulltext.json')
                async def one(packet):
                    tasks=packet['tasks'];m=packet['manifest'];values={}
                    if not all(r in tasks for r in roles):decision='UNASSESSED'
                    else:
                        values=dict(zip(roles,await asyncio.gather(*(invoke(tasks[r],r,folder,True) for r in roles))))
                        if len(roles)==1:decision=values[roles[0]].get('decision','UNASSESSED')
                        else:
                            decision,basis=reconcile(values['FAST'],values['DEEP'],inp['exclusion_ids'],adjudicate=False)
                            if basis=='DISAGREEMENT_WITHOUT_ADJUDICATION':
                                task=copy.deepcopy(inp['adjudicator']);task['semantic_input']=copy.deepcopy(tasks['FAST']['semantic_input'])
                                for r in roles:
                                    value=values[r]
                                    task['semantic_input'][r.lower()+'_assessment']={k:value[k] for k in ['decision','criterion_ids','rationale','supporting_source_refs','missing_information']}
                                task['output_schema']['$defs']['EAV3PacketSourceRef']['properties']['document_id']['enum']=[d['document_id'] for d in m['documents']]
                                values['ADJUDICATOR']=await invoke(task,'ADJUDICATOR',folder,True)
                                decision,_=reconcile(values['FAST'],values['DEEP'],inp['exclusion_ids'],values['ADJUDICATOR'])
                    return {'packet_id':m['packet_id'],'study_id':m['study_id'],'report_ids':m['report_ids'],'decision':decision,'assessments':values}
                rows=await asyncio.gather(*(one(p) for p in inp['packets']))
            save(args.output/(directory.name+'.json'),rows)
    finally:
        for a in agents.values():await a.aclose()


def configure_models(parser, args):
    common_model = configured_model(args.model)
    if args.stage == 'ea':
        args.ea_roles = {
            'no-deep': ('FAST',), 'no-fast': ('DEEP',),
        }.get(args.condition, ('FAST', 'DEEP'))
        active_roles = args.ea_roles
        if len(active_roles) == 2 and args.phase == 'fulltext':
            active_roles += ('ADJUDICATOR',)
        args.ea_models = {}
        for role in active_roles:
            option = f'ea_{role.lower()}_model'
            model = (getattr(args, option) or '').strip() or common_model
            if not model:
                parser.error(f"Set {MODEL_ENV}, --model, or --{option.replace('_', '-')} for EA {role}.")
            args.ea_models[role] = model
        return

    judge_model = (args.judge_model or '').strip() if args.stage == 'judge' else ''
    args.model = judge_model or common_model
    if not args.model:
        parser.error(f"Set {MODEL_ENV} or pass --model with a model ID from your gateway.")


def main(argv=None, *, prog=None):
    p=argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument('stage',choices=['de','rob','ec','ms','ea','judge'])
    p.add_argument('--condition',default='full')
    p.add_argument('--source',type=Path,help='Fresh Full MS output, or the MS output to judge')
    p.add_argument('--model', '--deepseek-model', metavar='MODEL',
                   help=f'Shared gateway model ID (defaults to {MODEL_ENV})')
    p.add_argument('--judge-model', metavar='MODEL', help='Optional Judge model ID')
    p.add_argument('--judge-format', choices=['text', 'structured'], default='text',
                   help='Judge response format (default text)')
    p.add_argument('--ea-fast-model', metavar='MODEL', help='Optional EA FAST model ID')
    p.add_argument('--ea-deep-model', metavar='MODEL', help='Optional EA DEEP model ID')
    p.add_argument('--ea-adjudicator-model', metavar='MODEL', help='Optional EA full-text adjudicator model ID')
    p.add_argument('--ea-transport', choices=['paper', 'chat'], default='paper',
                   help='Use role-specific API transports or Chat Completions for all EA roles')
    p.add_argument('--structured-output', choices=['provider', 'prompt'], default='provider',
                   help='Request native structured output or prompt for locally validated JSON')
    p.add_argument('--response-model', metavar='MODEL', help='Optional exact DE response model ID')
    p.add_argument('--disable-thinking', action='store_true',
                   help='Send thinking.type=disabled for DE or MS if supported by the gateway')
    p.add_argument('--case',default='all')
    p.add_argument('--phase',choices=['initial','fulltext'],default='initial')
    p.add_argument('--ec-benchmark', choices=['v2', 'ec-ex80'], default='v2',
                   help='EC case set (default v2)')
    p.add_argument('--ec-ex80-data', type=Path, help='Prepared EC-ex80 input bundle (default output/ec-ex80-data)')
    p.add_argument("--ec-timeout", type=int, metavar="SECONDS",
                   help="Per-request EC timeout (1–600, default 300)")
    p.add_argument("--ms-value-slots", choices=["optional", "required"],
                   help="Full MS prose policy (default optional)")
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    allowed={'de':['full','no-task-contract'],'rob':['full','no-task-contract','no-code-logic'],
        'ec':['full','no-task-contract','no-code-logic'],'ms':['full','no-task-contract','no-code-logic'],'judge':['full'],
        'ea':['full','no-fast','no-deep','dual-deepseek']}
    if a.condition not in allowed[a.stage]:p.error(f'{a.stage} conditions: {allowed[a.stage]}')
    if a.response_model is not None and a.stage != 'de':
        p.error('--response-model applies only to DE')
    a.response_model = (a.response_model or '').strip() or None
    if a.disable_thinking and a.stage not in {'de', 'ms'}:
        p.error('--disable-thinking applies only to DE or MS')
    if a.ec_timeout is not None and (a.stage != "ec" or not 1 <= a.ec_timeout <= 600):
        p.error("--ec-timeout requires EC and a value between 1 and 600")
    if a.ms_value_slots is not None and (a.stage != "ms" or a.condition != "full"):
        p.error("--ms-value-slots applies only to the full MS condition")
    if a.stage != 'ec' and a.ec_benchmark != 'v2':
        p.error('--ec-benchmark applies only to EC')
    if a.ec_ex80_data is not None and a.ec_benchmark != 'ec-ex80':
        p.error('--ec-ex80-data requires --ec-benchmark ec-ex80')
    if a.stage=='ec' and a.ec_benchmark=='v2' and a.condition!='no-task-contract' and a.case!='all':
        p.error('EC V2 staged/direct uses the fixed 20-case set')
    from_previous_run = a.stage == 'judge' or (a.stage == 'ms' and a.condition != 'full')
    if from_previous_run:
        if not a.source:p.error('--source is required')
        a.source=a.source.resolve()
        try:
            cases={row['case_id'] for row in read(a.source/'execution.json')}
        except (OSError, ValueError, TypeError, KeyError):
            p.error(f'--source must contain an execution.json from a previous MS run: {a.source}')
    elif a.ec_benchmark == 'ec-ex80':
        from metarigor.evidence_certainty.benchmark_v2.ec_ex80 import DEFAULT_DATA, case_ids, load_bundle

        a.ec_ex80_data = a.ec_ex80_data or DEFAULT_DATA
        try:
            load_bundle(a.ec_ex80_data)
        except (OSError, ValueError) as error:
            p.error(f'Cannot read prepared EC-ex80 inputs at {a.ec_ex80_data}: {error}')
        cases = set(case_ids())
    elif a.stage in ('rob','ec'):
        cases={path.stem for path in json_paths(ROOT/'inputs'/a.stage)}
    else:
        cases={path.name for path in (ROOT/'inputs'/a.stage).iterdir() if path.is_dir()}
    if not cases or (a.case!='all' and a.case not in cases):
        p.error('Unknown or empty case selection. Available: '+', '.join(sorted(cases)))
    check_gateway(p)
    configure_models(p, a)
    a.output=a.output.resolve()
    try:
        a.output.mkdir(parents=True,exist_ok=False)
    except OSError as error:
        p.error(f'--output must be a new writable directory: {error}')
    action=ms_ablation if a.stage=='ms' and a.condition!='full' else globals()[a.stage]
    result = asyncio.run(action(a))
    if a.stage == 'de' and result:
        p.exit(1, f'DE failed for {", ".join(result)}. See the case JSON files under {a.output}.\n')

if __name__=='__main__':main()
