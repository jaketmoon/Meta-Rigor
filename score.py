"""Score newly generated, identity-aligned predictions; never read archived system results."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from reproduce import ROOT, read

def ratio(n,d):return n/d if d else None

def unique(rows,key):
    out={}
    for r in rows:
        k=key(r)
        if k in out:raise ValueError('Duplicate evaluation identity: '+str(k))
        out[k]=r
    return out

def de(rows):
    key=lambda r:(r['case_id'],r['study_key'],r['target_id'])
    refs=[dict(r,case_id=p.stem) for p in (ROOT/'references/de').glob('*.json') for r in read(p)['study_target_cells']]
    pred=unique(rows,key);gold=unique(refs,key)
    if pred.keys()-gold.keys():raise ValueError('Prediction outside the fixed Study × Target grid')
    correct=complete=fields=field_correct=positive=valid_links=links=0
    maps={}
    for cid in {r['case_id'] for r in rows}:
        data=read(ROOT/'inputs/de'/cid/'source_citations_selection.json')
        maps[cid]={r['citation_id']:r for r in data['citation_map']}
    for k,g in gold.items():
        ref=g.get('values');is_positive=bool(ref) and g.get('accuracy_evaluation','SCOREABLE')=='SCOREABLE'
        if is_positive:positive+=1;fields+=len(ref)-1
        r=pred.get(k,{});values=r.get('values') or {};numeric={f:v for f,v in values.items() if f!='result_type' and v is not None}
        all_closed=bool(numeric)
        for f in numeric:
            citations=r.get('field_citations',{}).get(f,[]);closed=False
            for identifier in citations:
                c=maps[k[0]].get(identifier)
                if not c:continue
                source=(ROOT/'inputs/de'/k[0]/c['view_path'].removeprefix('source-package/')).read_bytes()
                if hashlib.sha256(source).hexdigest()==c['view_sha256'] and source.decode()[c['start_char']:c['end_char']]==c['quote']:
                    closed=True
            links+=1;valid_links+=closed;all_closed &= closed
        # DE always uses the paper's item-level claim-closure policy.
        if not all_closed:continue
        needed=(['intervention_events','intervention_group_size','comparator_events','comparator_group_size']
            if values.get('result_type')=='BINARY' else ['intervention_mean','intervention_standard_deviation','intervention_group_size','comparator_mean','comparator_standard_deviation','comparator_group_size'] if values.get('result_type')=='CONTINUOUS' else [])
        delivered=bool(needed) and all(f in numeric for f in needed)
        complete+=delivered
        if is_positive:
            matches=sum(values.get(f)==v for f,v in ref.items() if f!='result_type')
            field_correct+=matches
            correct+=delivered and matches==len(ref)-1 and values.get('result_type')==ref['result_type']
    assert len(gold)==405 and positive==195 and fields==828
    return {'units':405,'reference_results':positive,'correct_results':correct,'complete_candidates':complete,
        'recall':ratio(correct,positive),'precision':ratio(correct,complete),'f1':ratio(2*correct,positive+complete),
        'correct_fields':field_correct,'fields':fields,'field_accuracy':ratio(field_correct,fields),
        'unfiltered_claim_closure':ratio(valid_links,links),'evaluation_policy':'CLAIM_CLOSURE_ONLY'}

def rob(rows):
    from metarigor.risk_of_bias_v3.method import normalize_applicability,method_input_answers
    pred=unique(rows,lambda r:(r['case_id'],r['study_key'],r['target_id']))
    dc=oc=high=high_count=sqc=sqn=units=0
    for p in (ROOT/'references/rob').glob('*.json'):
        for g in read(p)['targets']:
            units+=1;r=pred.get((p.stem,g['study_key'],g['target_id']),{})
            labels=r.get('domain_labels',{})
            dc+=sum(labels.get(d['domain'])==d['judgment'] for d in g['domains'])
            oc+=r.get('overall')==g['overall_judgment']
            if g['overall_judgment']=='HIGH':high_count+=1;high+=r.get('overall')=='HIGH'
            for d in g['domains']:
                answers=normalize_applicability(d['domain'],method_input_answers(d['domain'],{q['question_id']:q['answer'] for q in d['questions']}))
                sqn+=len(answers);sqc+=sum(r.get('answers',{}).get(q)==v for q,v in answers.items())
    assert units==32 and high_count==13 and sqn==704
    return {'study_results':units,'domain_agreement':dc/160,'overall_agreement':oc/32,'high_risk_recall':high/13,'sq_agreement':sqc/sqn}

def ec(rows):
    from metarigor.evidence_certainty.benchmark_v2.dataset import load_combined_gold
    pred=unique(rows,lambda r:r['case_id']);dc=oc=case=0;errors=[]
    for g in load_combined_gold():
        r=pred.get(g.case_id,{})
        domains=r.get('domains',{})
        if isinstance(domains,list):domains={x['domain']:x.get('judgment') for x in domains}
        d=sum(domains.get(x.domain)==x.judgment for x in g.domains)
        dc+=d;overall=r.get('overall_downgrade_levels');match=overall==g.overall_downgrade_levels
        oc+=match;case+=d==5 and match
        if type(overall) is int and 0<=overall<=3:errors.append(abs(overall-g.overall_downgrade_levels))
    return {'domain_agreement':dc/100,'downgrade_agreement':oc/20,'case_exact':case/20,
        'downgrade_mae':ratio(sum(errors),len(errors)),'available_downgrades':len(errors)}

def ea(rows):
    reference=read(ROOT/'references/ea.json')
    names={'balbaa-2025':'balbaa','lin-2025':'lin','meng-2025':'meng','turalde-mapili-2023':'turalde','zhong-2022':'zhong'}
    truth={tuple(x) for x in reference['reference_primaries']}
    selected={(names.get(r['case_id'],r['case_id']),r['record_id']) for r in rows if r.get('retained') is True}
    tp=len(selected & truth);precision=ratio(tp,len(selected));recall=tp/25
    families={}
    for r in reference['reports']:
        if r['status']!='RESOLVED':continue
        for study in r['study_keys']:families.setdefault((r['case_id'],study),set()).add(r['record_id'])
    groups={}
    has_groups=any('group_ids' in r for r in rows)
    for r in rows:
        if r.get('retained') is not True:continue
        cid=names.get(r['case_id'],r['case_id'])
        for group in r.get('group_ids',[]):groups.setdefault((cid,group),set()).add(r['record_id'])
    complete=companions=companion_hit=0
    for (cid,study),members in families.items():
        primaries={rid for case,rid in truth if case==cid} & members
        if not primaries:continue
        got={rid for case,rid in selected if case==cid}
        complete+=any(members<=reports for (case,_),reports in groups.items() if case==cid)
        companions+=len(members-primaries);companion_hit+=len((members-primaries)&got)
    return {'reference_studies':25,'retained_reports':len(selected),'reference_hits':tp,
        'precision':precision,'recall':recall,'f1':ratio(2*tp,25+len(selected)),
        'f2':ratio(5*tp,100+len(selected)),'nnr':ratio(len(selected),tp),
        'family_recall':complete/25 if has_groups else None,'companion_recall':ratio(companion_hit,companions),
        'companion_count':companions}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['de','rob','ec','ea']);p.add_argument('predictions',type=Path)
    p.add_argument('--native-rob',action='store_true',help='Read candidate.json files beneath a fresh ROB output directory')
    a=p.parse_args()
    if a.native_rob:
        rows=[]
        for path in a.predictions.rglob('candidate.json'):
            c=read(path)
            for t in c['targets']:
                answers={q:v for item in t['items'].values() for q,v in c['items'][item]['answers'].items()}
                rows.append(dict(t,case_id=c['case_id'],answers=answers))
    else:rows=read(a.predictions)
    if not isinstance(rows,list):p.error('Predictions must be a JSON list of aligned records')
    result=de(rows) if a.stage=='de' else globals()[a.stage](rows)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
