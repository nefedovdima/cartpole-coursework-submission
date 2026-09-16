"""Full-horizon deterministic evaluation with persistent spawn workers.

Only complete episodes are reused. Interrupted episodes retain their attempt
logs and restart from the exact initial state, never an invented Drake history.
This evaluation-only loader does not weaken training/resume source checks.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
import json
import gzip
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import shutil
import time

import numpy as np

from cartpole.experiments.cloud_io import ROOT, atomic_json, atomic_bytes, file_lock, sha256, verify_manifest
from cartpole.rl.training_contract import canonical_hash

WORKER = {}


def inference_stack_config(config):
    """One audited registry-only migration for evaluation, never training resume.

    Main checkpoints predate the addition of development config names. All
    scientific sources and every other contract field must still match exactly.
    The saved config/model are not edited; check_stack sees the current registry.
    """
    from cartpole.rl.training_contract import identity
    saved = config['training_contract']
    current = identity(saved['id'])
    if saved == current:
        return config
    import copy
    migrated = copy.deepcopy(saved)
    name = 'cartpole/experiments/cloud_config.py'
    pair = ('34f0bebc98749f107bbc713b17aa580d6f92de38f3ba7787c36057e07f6a42f7',
            '0843c4000a7d33593c71d33b5204659e40cc8e8c4f55ee9ad3483562ae82754e')
    if (saved['sources'].get(name), current['sources'].get(name)) != pair:
        raise ValueError('unreviewed inference registry migration')
    migrated['sources'][name] = pair[1]
    if migrated != current:
        raise ValueError('inference migration includes other source/contract changes')
    return {**config, 'training_contract': current}


def save_episode(path, log):
    """Lossless full JSON, low compression level; no dropped state/command data."""
    raw=json.dumps(log,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    atomic_bytes(path,gzip.compress(raw,compresslevel=1,mtime=0))


def load_policy(checkpoint, device):
    """Own trusted checkpoint, fixed math/sources, no replay/optimizer updates."""
    from cartpole.experiments.cloud_learning import (algorithm_class, configure_device,
        implementation_hashes, policy_fingerprint, finite_model)
    from cartpole.experiments.inference_compatibility import observations_for_inference, compare_actions
    from cartpole.rl.training_contract import specification, identity
    from cartpole.experiments.cloud_checks import check_stack
    checkpoint=Path(checkpoint); verify_manifest(checkpoint)
    entries=json.loads((checkpoint/'manifest.json').read_text())
    if not {'model.zip','metadata.json','probes.json','replay.pkl','runtime.pt'} <= entries.keys():
        raise ValueError('incomplete checkpoint manifest')
    data=json.loads((checkpoint/'metadata.json').read_text());cfg=data['config']
    if not data['complete'] or data['implementation']!=implementation_hashes(cfg):
        raise ValueError('checkpoint/library identity differs')
    saved=cfg.get('training_contract')
    if saved is None:
        raise ValueError('legacy off checkpoint requires its frozen evaluation package; no implicit conversion')
    current=identity(saved['id'])
    if saved['specification']!=specification(saved['id']) or saved['sha256']!=current['sha256']:
        raise ValueError('checkpoint scientific contract differs')
    # Allow orchestration changes for inference, but not physics/reward/mapping.
    critical=[p for p in current['sources'] if not p.startswith('cartpole/experiments/')]
    critical += ['cartpole/experiments/lqr_hold.py','cartpole/experiments/gym_diagnostics.py','cartpole/experiments/safety_metrics.py']
    if any(saved['sources'].get(p)!=current['sources'][p] for p in critical):
        raise ValueError('checkpoint physical/reward/action/metric source differs')
    stack_config = inference_stack_config(cfg)
    check_stack(stack_config);requested=configure_device(device)
    model=algorithm_class(cfg).load(checkpoint/'model.zip',device=device,force_reset=True)
    if model.device!=requested or any(p.device!=requested for p in model.policy.parameters()):
        raise RuntimeError('evaluation device fallback')
    if (model.num_timesteps,model._n_updates,model.seed)!=(data['transitions'],data['updates'],data['seed']):
        raise ValueError('checkpoint counters differ')
    if model.training_contract!=saved:raise ValueError('saved model contract differs')
    finite_model(model)
    if policy_fingerprint(model)!=data['policy_sha256']:
        raise ValueError('loaded policy fingerprint differs from checkpoint')
    model.policy.set_training_mode(False)
    probes=json.loads((checkpoint/'probes.json').read_text())
    if (probes['training_seed'],probes['transitions'],probes['updates'],probes['device']) != (
            data['seed'],data['transitions'],data['updates'],data['device']):
        raise ValueError('probe checkpoint identity differs')
    observations=observations_for_inference(probes['observations'])
    actions=model.predict(observations,deterministic=True)[0]
    numerical=compare_actions(actions,probes['actions'],algorithm=cfg['algorithm'],count=len(observations),
                              saved_device=data['device'],loaded_device=str(requested))
    from cartpole.rl.training_contract import evaluation_action
    numerical['environment_actions']=compare_actions(evaluation_action(model,observations,filtered=True),
        probes['env_actions'],algorithm=cfg['algorithm'],count=len(observations),
        saved_device=data['device'],loaded_device=str(requested))
    numerical['registry_migration'] = stack_config is not cfg
    data={**data,'inference_compatibility':numerical}
    return model,data


def descriptor(checkpoint=None, *, classical=False):
    if classical:
        path=ROOT/'tests/fixtures/swing_up_nominal.npz'
        return dict(kind='classical',nominal_sha256=sha256(path),algorithm='classical',seed=None,
                    transitions=None,updates=None)
    p=Path(checkpoint).resolve();verify_manifest(p)
    d=json.loads((p/'metadata.json').read_text())
    return dict(kind='rl',checkpoint=str(p),checkpoint_manifest_sha256=sha256(p/'manifest.json'),
                model_sha256=sha256(p/'model.zip'),algorithm=d['config']['algorithm'],seed=d['seed'],
                transitions=d['transitions'],updates=d['updates'],
                training_contract_id=d['config']['training_contract']['id'])


def worker_init(source, mode, device, stop):
    import torch
    from cartpole.experiments.cloud_learning import configure_device
    from cartpole.rl.training_contract import make_training_env
    from cartpole.rl.methods import contract_id
    started=time.perf_counter();configure_device(device)
    if source['kind']=='rl':
        model,data=load_policy(source['checkpoint'],device)
        env=make_training_env(contract_id(source['algorithm'],mode))
        WORKER.update(model=model, data=data)
    else:
        from cartpole.control.swing_up import NominalTrajectory
        from cartpole.rl.env import CartPoleEnv
        WORKER['nominal']=NominalTrajectory.load(ROOT/'tests/fixtures/swing_up_nominal.npz')
        env=CartPoleEnv(reward_version='R1-v0',safety_filter=mode=='on')
    WORKER.update(env=env,source=source,stop=stop,device=device,
                  setup_seconds=time.perf_counter()-started,loads=1,episodes=0)


def valid_case_result(directory, *, specification_sha256=None):
    directory=Path(directory);p=directory/'completed.json'
    if not p.exists():return None
    pointer=json.loads(p.read_text());receipt=directory/pointer['receipt']
    if not receipt.resolve().is_relative_to(directory.resolve()) or sha256(receipt)!=pointer['sha256']:
        raise ValueError('episode receipt identity differs')
    row=json.loads(receipt.read_text());log=receipt.parent/row.get('log_file','episode.json')
    if specification_sha256 is not None and row.get('specification_sha256')!=specification_sha256:
        raise ValueError('episode model/mode/specification binding differs')
    if log.parent.resolve()!=receipt.parent.resolve():raise ValueError('unsafe episode log path')
    if sha256(log)!=row['log_sha256']:raise ValueError('episode log differs')
    return row


def run_case(task):
    from cartpole.experiments.gym_diagnostics import record_episode, ClassicalPolicy
    from cartpole.experiments.sac_pilot import episode_metrics
    from cartpole.experiments.cloud_evaluation import interval_margins
    from cartpole.experiments.safety_metrics import safety_metrics
    from cartpole.rl.training_contract import evaluation_action
    from cartpole.experiment_config import make_experiment_config
    import torch
    case,root=task['case'],Path(task['directory']);root.mkdir(parents=True,exist_ok=True)
    with file_lock(root/'.case.lock'):
        previous=valid_case_result(root,specification_sha256=task['specification_sha256'])
        if previous is not None:return previous
        attempts=len(list(root.glob('attempt_*')));out=root/f'attempt_{attempts:05d}';out.mkdir()
        if shutil.disk_usage(out).free<512*1024**2:raise OSError('evaluation episode save reserve exhausted')
        case_seed = case['seed'] if case['seed'] is not None else int(canonical_hash(case)[:8],16)
        random.seed(case_seed);np.random.seed(case_seed);torch.manual_seed(case_seed)
        policy=(ClassicalPolicy(make_experiment_config(),WORKER['nominal']) if WORKER['source']['kind']=='classical' else
                lambda obs,info:evaluation_action(WORKER['model'],obs,filtered=True))
        def checked(obs,info):
            if WORKER['stop'].is_set():raise RuntimeError('evaluation_interrupted')
            value=policy(obs,info)
            if not np.isfinite(value).all():raise FloatingPointError('nonfinite evaluation command')
            for attr in ('phase','raw_acceleration'):
                if hasattr(policy,attr):setattr(checked,attr,getattr(policy,attr))
            return value
        t=time.perf_counter();cpu=time.process_time()
        try:
            log=record_episode(WORKER['env'],checked,seed=case_seed,options={'initial_state':case['initial_state']},
                               on_failure=lambda partial:save_episode(out/'partial_episode.json.gz',partial),
                               metadata=dict(algorithm=WORKER['source']['algorithm'],training_seed=WORKER['source']['seed'],
                                             evaluation_case_id=case['id'],filter_mode=task['mode']))
        except Exception as exc:
            atomic_json(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),case=case))
            raise
        compute=time.perf_counter()-t;cpu_seconds=time.process_time()-cpu
        metrics=dict(id=case['id'],subset=case['subset'],**episode_metrics(log),**interval_margins(log),safety=safety_metrics(log))
        write=time.perf_counter();save_episode(out/'episode.json.gz',log);write_seconds=time.perf_counter()-write
        WORKER['episodes']+=1
        row=dict(index=task['index'],case=case,metrics=metrics,specification_sha256=task['specification_sha256'],
            log_file='episode.json.gz',log_sha256=sha256(out/'episode.json.gz'),
            log_bytes=(out/'episode.json.gz').stat().st_size,
            pid=os.getpid(),worker_model_loads=WORKER['loads'],worker_episodes=WORKER['episodes'],
            setup_seconds=WORKER['setup_seconds'],episode_seconds=compute,write_seconds=write_seconds,cpu_seconds=cpu_seconds,
            inference_compatibility=WORKER.get('data',{}).get('inference_compatibility'),
            rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if WORKER['device'].startswith('cuda') else None)
        atomic_json(out/'receipt.json',row)
        atomic_json(root/'completed.json',dict(receipt=str((out/'receipt.json').relative_to(root)),sha256=sha256(out/'receipt.json')))
        return row


def evaluate_checkpoint(checkpoint, cases, output, *, mode='on', device='cpu', workers=1,
                        guard=None, classical=False, diagnostic=False):
    from cartpole.experiments.cloud_evaluation import validation_cases, complete_aggregates
    from cartpole.experiments.sac_pilot import aggregate
    if type(workers)!=int or workers<1 or workers>64:raise ValueError('explicit workers must be 1..64')
    if mode not in ('on','off'):raise ValueError('filter mode must be on/off')
    ids=[c['id'] for c in cases]
    if not cases or len(set(ids))!=len(ids):raise ValueError('unique nonempty evaluation cases required')
    for c in cases:
        if not c['id'] or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in c['id']):
            raise ValueError('unsafe evaluation case ID')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=descriptor(checkpoint,classical=classical)
    from cartpole.rl.training_contract import identity
    from cartpole.rl.methods import contract_id
    ident=identity(contract_id('SAC' if classical else source['algorithm'],mode))
    # Source path is only a locator: restored identical checkpoints can move.
    source_identity={k:v for k,v in source.items() if k!='checkpoint'}
    spec=dict(schema=1,protocol='persistent-full-evaluation-v1',source=source_identity,
              cases=cases,cases_sha256=canonical_hash(cases),mode=mode,device=device,
              environment=ident,horizon=10.,diagnostic=diagnostic,
              log_format='lossless JSON gzip level1',
              evaluator_sources={name:sha256(ROOT/name) for name in (
                  'cartpole/experiments/parallel_evaluation.py','cartpole/experiments/inference_compatibility.py',
                  'cartpole/experiments/cloud_evaluation.py',
                  'cartpole/control/swing_up.py','cartpole/control/lqr.py')})
    with file_lock(out/'.evaluation.lock'):
        p=out/'specification.json'
        if p.exists() and json.loads(p.read_text())!=spec:raise ValueError('evaluation source/protocol/cases/device changed')
        if not p.exists():atomic_json(p,spec)
        specification_digest=sha256(p)
        if (out/'cases').exists() and any(d.name not in ids for d in (out/'cases').iterdir() if d.is_dir()):
            raise ValueError('unexpected episode ID in result directory')
        pending=[];rows={}
        for i,c in enumerate(cases):
            row=valid_case_result(out/'cases'/c['id'],specification_sha256=specification_digest)
            if row:
                if row['case']!=c or row['index']!=i:raise ValueError('episode identity/index differs')
                rows[i]=row
            else:pending.append(dict(index=i,case=c,directory=str((out/'cases'/c['id']).resolve()),mode=mode,
                                     specification_sha256=specification_digest))
        reused=len(rows);start=time.perf_counter();error=None
        ctx=mp.get_context('spawn');stop=ctx.Event();pool=None
        try:
            if pending:
                pool=ProcessPoolExecutor(max_workers=min(workers,len(pending)),mp_context=ctx,
                    initializer=worker_init,initargs=(source,mode,device,stop))
                futures={pool.submit(run_case,t):t['index'] for t in pending}
                while futures:
                    if guard:guard()
                    if shutil.disk_usage(out).free<1024**3:raise OSError('evaluation coordinator save reserve exhausted')
                    done,_=wait(futures,timeout=.1,return_when=FIRST_COMPLETED)
                    for f in done:
                        index=futures.pop(f);row=f.result()
                        if row['index']!=index or index in rows:raise ValueError('duplicate/misassigned episode')
                        rows[index]=row
                        atomic_json(out/'progress.json',dict(completed=len(rows),expected=len(cases),ids=[ids[i] for i in sorted(rows)]))
        except BaseException as exc:
            error=dict(type=type(exc).__name__,message=str(exc));stop.set();raise
        finally:
            if pool:pool.shutdown(wait=True,cancel_futures=True)
            # Workers commit receipts independently. Recover late completions
            # even if the coordinator was interrupted before receiving futures.
            for i,c in enumerate(cases):
                row=valid_case_result(out/'cases'/c['id'],specification_sha256=specification_digest)
                if row is not None:rows[i]=row
            ordered=[rows[i]['metrics'] for i in sorted(rows)]
            complete=len(rows)==len(cases) and error is None
            full=sorted(cases,key=lambda c:c['id'])==sorted(validation_cases(),key=lambda c:c['id'])
            report=dict(complete=complete,status='complete' if complete else 'partial',selection_eligible=complete and full and not diagnostic,
                diagnostic=diagnostic,scientific_result=not diagnostic,error=error,
                algorithm=source['algorithm'],training_seed=source['seed'],trained_transitions=source['transitions'],updates=source['updates'],
                training_contract_id=source.get('training_contract_id'),evaluation_contract_id=ident['id'],
                episodes=ordered,expected=len(cases),completed=len(rows),missing_ids=[ids[i] for i in range(len(cases)) if i not in rows],
                seconds=time.perf_counter()-start,reused_episodes=reused,new_episodes=len(rows)-reused,requested_workers=workers,
                control_transitions=sum(r['completion']['transition_count'] for r in ordered),
                validation=aggregate([r for r in ordered if r['subset']=='validation']),
                named=aggregate([r for r in ordered if r['subset']=='named']),all_cases=aggregate(ordered),
                subsets={name:aggregate([r for r in ordered if r['subset']==name]) for name in sorted({c['subset'] for c in cases})},
                specification_sha256=sha256(p),worker_records=[{k:v for k,v in rows[i].items() if k not in ('case','metrics')} for i in sorted(rows)])
            complete_aggregates(report,ordered)
            safety=[r['safety'] for r in ordered]
            accepted=sum(r['accepted_decisions'] for r in safety)
            interventions=sum(r['intervention_count'] for r in safety)
            report['safety_summary']=dict(episodes=len(safety),
                exceeds_024_episodes=sum(r['exceeds_024'] for r in safety),
                exceeds_024_resolved_episodes=sum(r['exceeds_024_resolved'] for r in safety),
                position_limit_events=sum(r['position_limit_events'] for r in safety),
                refusal_count=sum(r['refusal_count'] for r in safety),
                refusal_reasons={reason:sum(r['refusal_reasons'].get(reason,0) for r in safety)
                                 for reason in sorted({key for r in safety for key in r['refusal_reasons']})},
                accepted_decisions=accepted,interventions=interventions,
                intervention_fraction=interventions/accepted if accepted else None,
                intervention_abs_sum=sum(r['intervention_abs_sum'] for r in safety),
                intervention_abs_max=max((r['intervention_abs_max'] for r in safety),default=None),
                max_abs_position_interval=max((r['max_abs_position_interval'] for r in safety),default=None))
            atomic_json(out/'report.json',report)
            summary={k:v for k,v in report.items() if k not in ('episodes','worker_records','missing_ids')}
            summary['missing_count']=len(report['missing_ids'])
            atomic_json(out/'summary.json',summary)
        return report
