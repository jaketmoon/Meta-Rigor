"""Check package integrity and validate bundled inputs offline."""
import argparse
import ast
import hashlib
from pathlib import Path

from metarigor.research import SOURCE_PACKAGE
from .reproduce import ROOT, read, json_paths


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument('--ec-ex80-data', type=Path, help='Also verify this prepared EC-ex80 input bundle')
    args = parser.parse_args(argv)
    manifest=read(ROOT/'inputs/checksums.json')
    for relative,expected in manifest.items():
        path=ROOT/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Missing or changed input: '+relative)
    for path in SOURCE_PACKAGE.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    de=list((ROOT/'inputs/de').iterdir());rob=json_paths(ROOT/'inputs/rob');ec=json_paths(ROOT/'inputs/ec')
    assert len(de)==10 and len(rob)==8 and len(ec)==20
    units=positive=fields=0
    for p in (ROOT/'references/de').glob('*.json'):
        for row in read(p)['study_target_cells']:
            units+=1
            if row.get('values') and row.get('accuracy_evaluation','SCOREABLE')=='SCOREABLE':
                positive+=1;fields+=len(row['values'])-1
    assert (units,positive,fields)==(405,195,828)
    from jsonschema import Draft202012Validator
    initial=packets=reports=0;report_ids=set()
    for d in (ROOT/'inputs/ea').iterdir():
        for roles in read(d/'initial.json').values():
            initial+=1
            for task in roles.values():Draft202012Validator(task['input_schema']).validate(task['semantic_input'])
        for packet in read(d/'fulltext.json')['packets']:
            packets+=1;report_ids.update((d.name,r) for r in packet['manifest']['report_ids'])
            for task in packet['tasks'].values():Draft202012Validator(task['input_schema']).validate(task['semantic_input'])
    assert (initial,packets,len(report_ids))==(822,252,302)
    common=sum(len((p/'records.jsonl').read_text().splitlines()) for p in (ROOT/'inputs/cli/ea-fulltext').iterdir() if p.is_dir())
    assert common==300
    from metarigor.evidence_certainty.benchmark_v2.live import _prepared_candidates
    for staged in (True,False):assert len(_prepared_candidates(include_review_sufficiency=staged))==20
    from metarigor.evidence_certainty.benchmark_v2 import ec_ex80

    ec_ex80.load_references()
    if args.ec_ex80_data is not None:
        source, plan = ec_ex80.load_bundle(args.ec_ex80_data)
        prepared = ec_ex80.prepare_cases(source, plan, 'all')
        print(f'OK: EC-ex80 candidate inputs and source bindings for {len(prepared)} cases.')
    print(f'OK: {len(manifest)} input/resource hashes; EA 822/300; DE 405/195/828; ROB 8 cases; EC 20; MS 10. No model calls.')
if __name__=='__main__':main()
