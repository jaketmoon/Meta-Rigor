"""Run the paper's frozen-input experiments; generated outputs always go to a new directory."""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import gzip
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'backend/src'))

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

def runtime(role='worker', model='deepseek-v4-flash', **kwargs):
    from metarigor.adapters.agent_runtime.schema_agent import OpenAICompatibleJsonSchemaAgent
    return OpenAICompatibleJsonSchemaAgent(api_key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'],
        base_url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL'], model=model, role=role,
        temperature=0, timeout_seconds=300, **kwargs)

async def de(args):
    from metarigor.data_extraction.reference_kernel.executor import DeepSeekFlashExecutor
    from metarigor.data_extraction.natural_optimization.mechanism import bind_citations, SELECTION_INSTRUCTION
    paths=[p/'source_citations_selection.json' for p in sorted((ROOT/'inputs/de').iterdir())]
    condition='source_citations_selection' if args.condition=='full' else 'source_citations'
    executor=DeepSeekFlashExecutor(base_url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL'],
        api_key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'], timeout_seconds=1800, thinking_mode="provider_default",
        model_id='deepseek-v4-flash')
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
                    'finish_reason':r.finish_reason,'citations':bound, 'status':'COMPLETED_WITH_ISSUES' if r.finish_reason not in (None,'stop') or bound['issues'] else 'COMPLETE'})
            except Exception as error:
                save(args.output/(cid+'.json'),{'case_id':cid,'status':'FAILED','error':str(error)})
    try:await asyncio.gather(*(one(p) for p in paths))
    finally:await executor.aclose()

async def rob(args):
    from metarigor.rob_reproduction import run_case, MeasuredDirectAgent
    agent=MeasuredDirectAgent(api_key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'],
        base_url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL'],worker_model='deepseek-v4-flash',
        reviewer_model='deepseek-v4-flash',timeout_seconds=300)
    arm={'full':'full','no-task-contract':'minus_decomposition','no-code-logic':'minus_programmatic_decision'}[args.condition]
    limiter=asyncio.Semaphore(4)
    try:
        rows=await asyncio.gather(*(run_case(read(p),arm,args.output,agent,limiter)
            for p in json_paths(ROOT/'inputs/rob') if args.case in ('all',p.stem)),return_exceptions=True)
        save(args.output/'execution.json',[str(r) if isinstance(r,Exception) else r for r in rows])
    finally:await agent.aclose()

async def ec(args):
    if args.condition!='no-task-contract':
        from metarigor.evidence_certainty.benchmark_v2.live import run_live_benchmark
        agent=runtime(structured_output_mode='provider_json_schema_prompt')
        try:
            result=await run_live_benchmark(runtime=agent,runs_root=args.output,validation_id='ec-fresh-run',
                mechanism='domain_aggregate' if args.condition=='full' else 'direct')
            save(args.output/'execution.json',result.model_dump(mode='json'))
        finally:await agent.aclose()
        return
    from dataclasses import asdict
    from metarigor.evidence_certainty.benchmark_v2.task_contract_ablation import PROMPT, FINAL_PROMPT, InitialEnvelope, FinalEnvelope
    from metarigor.evidence_certainty.rules import binary_imprecision_decision
    from metarigor.schema_agent import SchemaAgentRequest
    agent=runtime(structured_output_mode='provider_json_schema_prompt')
    skill=(ROOT/'inputs/ec/methods.txt').read_text()
    limiter=asyncio.Semaphore(4)
    async def one(p):
        inp=read(p);log=[]
        async def call(prompt,payload,model):
            async with limiter:
                r=await agent.invoke(SchemaAgentRequest(template_id='ec-task-contract-ablation',model_role='worker',
                    prompt=prompt,input_payload=payload,output_schema=model.model_json_schema(),skill_text=skill))
            log.append({'input':payload,'output':r.payload,'raw':r.raw_model_output});return model.model_validate(r.payload)
        try:
            first=await call(PROMPT,inp,InitialEnvelope);answer=first.answer
            if first.calculator is not None:
                calc={'result':asdict(binary_imprecision_decision(**first.calculator.model_dump()))}
                log.append({'calculator':calc})
                final=await call(FINAL_PROMPT,{**inp,'provisional_report':answer,
                    'calculator_arguments':first.calculator.model_dump(),'calculator_result':calc},FinalEnvelope)
                answer=final.answer
            save(args.output/(p.stem+'.json'),{'case_id':p.stem,'answer':answer,'trace':log})
        except Exception as error:save(args.output/(p.stem+'.json'),{'case_id':p.stem,'status':'FAILED','error':str(error),'trace':log})
    try:await asyncio.gather(*(one(p) for p in json_paths(ROOT/'inputs/ec') if args.case in ('all',p.stem)))
    finally:await agent.aclose()

async def ms(args):
    from metarigor.manuscript.publication_pipeline import LocalManuscriptV2Pipeline
    from metarigor.manuscript.publication_models import ManuscriptRunRequestV2
    from metarigor.manuscript.journal_profiles import profile_path
    from metarigor.cli.manuscript_runtime import manuscript_runtime
    agent=manuscript_runtime(include_adjudicator=True,explicit_disable_thinking=True,
        adjudicator_structured_output_mode='text_wrapped')
    limiter=asyncio.Semaphore(4)
    rows=[]
    try:
        for p in sorted((ROOT/'inputs/ms').iterdir()):
            if args.case not in ('all',p.name):continue
            req=ManuscriptRunRequestV2(schema_version='manuscript-run-request.v2',run_id=str(uuid4()),
                fact_package=binding(p/'fact-package.json'),journal_profile=binding(profile_path('generic_prisma_2020_meta_analysis_v2')),
                profile_id='generic_prisma_2020_meta_analysis_v2',writer_mode='LLM',
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
    agent=_worker_runtime(disable_thinking=True)
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
    model=args.judge_model
    judge_agent=runtime(role='adjudicator',model=model,max_output_tokens=32768 if model.startswith('gpt') else 12000,
        structured_output_mode='provider_json_object' if model.startswith('gpt') else 'text_wrapped')
    agent=RoleRoutedSchemaAgent(worker=judge_agent,reviewer=judge_agent,adjudicator=judge_agent)
    try:
        rows=[]
        for row in read(args.source/'execution.json'):
            if args.case not in ('all',row['case_id']):continue
            result=await PublicationManuscriptJudge(candidate_root=Path(row['run_root']),
                validation_root=args.output/row['case_id'],agent_runtime=agent,score_mode=True,
                structured_score=model.startswith('gpt'),array_score=model.startswith('gpt')).run()
            rows.append({'case_id':row['case_id'],'weighted_score':result.weighted_score,'status':result.status,
                'report_path':str(result.report_path)})
        save(args.output/'scores.json',rows)
    finally:await judge_agent.aclose()

async def ea(args):
    from metarigor.evidence_acquisition.benchmark_v3.ablation import ground_output,reconcile
    from metarigor.adapters.agent_runtime.schema_agent import ClaudeSchemaAgent
    from metarigor.schema_agent import SchemaAgentRequest
    from metarigor.local_run import RunFolder
    agents={
      'FAST':ClaudeSchemaAgent(api_key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'],base_url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL'],
          worker_model='deepseek-v4-flash',reviewer_model='deepseek-v4-flash',timeout_seconds=300,max_turns=3,structured_output_mode='json_text'),
      'DEEP':runtime(role='reviewer',model='gpt-5.6-luna',structured_output_mode='provider_json_schema'),
      'ADJUDICATOR':runtime(role='adjudicator',model='gpt-5.6-terra',reasoning_effort='high',structured_output_mode='provider_json_schema')}
    if args.condition=='dual-deepseek':
        for role in ['DEEP','ADJUDICATOR']:
            await agents[role].aclose()
            agents[role]=ClaudeSchemaAgent(api_key=os.environ['METARIGOR_AGENT_GATEWAY_API_KEY'],base_url=os.environ['METARIGOR_AGENT_GATEWAY_BASE_URL'],
                worker_model='deepseek-v4-flash',reviewer_model='deepseek-v4-flash',timeout_seconds=300,max_turns=3,structured_output_mode='json_text')
    limiter=asyncio.Semaphore(8)
    roles=('FAST',) if args.condition=='no-deep' else ('DEEP',) if args.condition=='no-fast' else ('FAST','DEEP')
    async def invoke(task,role,folder,ground):
        model_role={'FAST':'worker','DEEP':'reviewer','ADJUDICATOR':'adjudicator'}[role]
        if args.condition=='dual-deepseek' and role=='ADJUDICATOR':model_role='worker'
        try:
            async with limiter:
                result=await agents[role].invoke(SchemaAgentRequest(template_id=task['template_id'],model_role=model_role,
                    prompt=task['prompt'],input_payload=task['semantic_input'],output_schema=task['output_schema'],skill_text=task['skill_text']))
            value=result.payload
            if ground:
                spans=ground_output(value,task,folder)
                value={**value,'supporting_refs':value['supporting_source_refs'],'grounding_status':'RESOLVED','source_spans':spans}
            return value
        except Exception as error:return {'status':'FAILED','error':str(error)}
    try:
        for directory in sorted((ROOT/'inputs/ea').iterdir()):
            if args.case not in ('all',directory.name):continue
            folder=RunFolder(directory)
            if args.phase=='initial':
                async def one(item,tasks):
                    values=dict(zip(roles,await asyncio.gather(*(invoke(tasks[r],r,folder,False) for r in roles))))
                    # Dual DeepSeek changes the model backend only; its screening
                    # decision must follow the Full condition exactly.
                    retained=all(v.get('decision')=='INCLUDE' for v in values.values())
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['de','rob','ec','ms','ea','judge'])
    p.add_argument('--condition',default='full')
    p.add_argument('--source',type=Path,help='Fresh Full MS output, or the MS output to judge')
    p.add_argument('--judge-model',choices=['deepseek-v4-flash','gpt-5.6-luna'],default='deepseek-v4-flash')
    p.add_argument('--case',default='all')
    p.add_argument('--phase',choices=['initial','fulltext'],default='initial')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    allowed={'de':['full','no-task-contract'],'rob':['full','no-task-contract','no-code-logic'],
        'ec':['full','no-task-contract','no-code-logic'],'ms':['full','no-task-contract','no-code-logic'],'judge':['full'],
        'ea':['full','no-fast','no-deep','dual-deepseek']}
    if a.condition not in allowed[a.stage]:p.error(f'{a.stage} conditions: {allowed[a.stage]}')
    if a.stage=='ec' and a.condition!='no-task-contract' and a.case!='all':p.error('EC staged/direct uses the fixed 20-case set')
    if a.stage=='judge' or (a.stage=='ms' and a.condition!='full'):
        if not a.source:p.error('--source is required')
        a.source=a.source.resolve()
    if a.stage=='judge' or (a.stage=='ms' and a.condition!='full'):
        cases={row['case_id'] for row in read(a.source/'execution.json')}
    elif a.stage in ('rob','ec'):
        cases={path.stem for path in json_paths(ROOT/'inputs'/a.stage)}
    else:
        cases={path.name for path in (ROOT/'inputs'/a.stage).iterdir() if path.is_dir()}
    if not cases or (a.case!='all' and a.case not in cases):
        p.error('Unknown or empty case selection. Available: '+', '.join(sorted(cases)))
    for key in ['METARIGOR_AGENT_GATEWAY_API_KEY','METARIGOR_AGENT_GATEWAY_BASE_URL']:
        if not os.environ.get(key):p.error(f'Set {key}')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    os.chdir(ROOT)
    action=ms_ablation if a.stage=='ms' and a.condition!='full' else globals()[a.stage]
    asyncio.run(action(a))

if __name__=='__main__':main()
