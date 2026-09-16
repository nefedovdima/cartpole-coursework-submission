"""Explicit evaluation of saved policies on the historical 20+3 protocol."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_mutually_exclusive_group(required=True);s.add_argument('--checkpoint',type=Path);s.add_argument('--classical',action='store_true')
    p.add_argument('--filter',choices=['off','on'],required=True)
    p.add_argument('--device',choices=['cpu','cuda:0','cuda:1'],default='cpu')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Without this flag only print the fixed evaluation plan.')
    a=p.parse_args()
    from cartpole.experiments.cloud_evaluation import validation_cases
    from cartpole.experiments.parallel_evaluation import evaluate_checkpoint
    cases=validation_cases()
    print(json.dumps(dict(episodes=len(cases),validation=20,named=3,horizon_seconds=10,filter=a.filter,device=a.device,workers=a.workers,execute=a.execute)))
    if a.execute:
        evaluate_checkpoint(a.checkpoint,cases,a.output,mode=a.filter,device=a.device,workers=a.workers,classical=a.classical)
if __name__=='__main__':main()
