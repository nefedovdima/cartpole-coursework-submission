"""Explicit fresh training plan/run, using the unchanged guarded scientific runner."""
import argparse
import json
import os
from pathlib import Path


def configuration(name, seed):
    from cartpole.experiments.cloud_config import load_config
    return load_config(name, seed, 'cpu', purpose='experiment')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init-session');q.add_argument('--session',type=Path,required=True)
    q.add_argument('--seconds',type=float,required=True);q.add_argument('--reserve-seconds',type=float,default=120.)
    for command in ('plan','run'):
        q=sub.add_parser(command);q.add_argument('--config',required=True);q.add_argument('--seed',type=int,required=True)
        if command=='run':
            q.add_argument('--session',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
            q.add_argument('--max-seconds',type=float,required=True)
    args=p.parse_args()
    if args.command=='init-session':
        from cartpole.experiments.time_only_session import TimeOnlySession
        result=TimeOnlySession.create(args.session,max_seconds=args.seconds,reserve_seconds=args.reserve_seconds).status()
    else:
        cfg=configuration(args.config,args.seed)
        if args.command=='plan':result=cfg
        else:
            if os.environ.get('CARTPOLE_BUDGET_SESSION')!=str(args.session.resolve()):
                raise ValueError('run through cloud_guard with the same session')
            if args.output.exists():raise FileExistsError('fresh run output required')
            from cartpole.experiments.time_only_session import TimeOnlySession
            from cartpole.experiments.cloud_runner import run_training
            session=TimeOnlySession(args.session)
            with session.lease():
                _,result=run_training(cfg,output=args.output,max_transitions=100000,
                                     max_seconds=args.max_seconds,session=session,evaluate_enabled=True)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
