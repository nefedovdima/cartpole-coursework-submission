"""Compact normalized metrics -> canonical tables; no raw logs or simulation."""
import argparse
import csv
import io
import json
from pathlib import Path
from cartpole.experiments import robustness_analysis as a, robustness_reporting as r
ROOT=Path(__file__).resolve().parents[1]

def check(root=ROOT):
    root=Path(root)
    data=a.read_json(root/'results/robustness/normalized.json')
    r.validate_input(data)
    tables=r.build_tables(data['episodes'])
    r.regression_invariants(tables)
    target=root/'docs/assets/structured_robustness'
    for name, rows in tables.items():
        buf=io.StringIO(newline=''); writer=csv.DictWriter(buf,fieldnames=r.COLUMNS[name],lineterminator="\n");writer.writeheader();writer.writerows(rows)
        if buf.getvalue().encode() != (target/name).read_bytes():raise ValueError('canonical table differs: '+name)
    return dict(episodes=1960,pairs=980,successes=[441,796],resolved_exceedances=[585,0],on_hold_failures=184,
                scientific_sha256=data['scientific_sha256'],tables=len(tables),raw_recomputation='not_run')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--figures',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
    result=check()
    if args.figures:
        if args.output is None:p.error('--figures requires --output')
        result['export']=r.publish(a.read_json(ROOT/'results/robustness/normalized.json'),args.output)
    elif args.output is not None:p.error('--output requires --figures')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
