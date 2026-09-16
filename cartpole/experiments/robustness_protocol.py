"""Frozen development starts and main-policy bindings; no dynamics or final set."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import numpy as np
from cartpole.experiments.cloud_io import ROOT, sha256, verify_manifest
from cartpole.rl.training_contract import canonical_hash
from cartpole.safety import SafetyLimits, assess_cart_state

VERSION = 'structured-robustness-dev-v1'
CONFIG = ROOT/'configs/robustness'
GROUPS = ('nominal', 'position', 'velocity', 'angle', 'angular_velocity', 'boundary', 'joint')
FIELDS = ('cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity')
GENERATOR_SEED = 2026091601
RESERVE = 300


def read(path):
    return json.loads(Path(path).read_text())


def generate_states():
    limits = SafetyLimits(); L = limits.inner_position; A = limits.acceleration_limit
    natural = math.sqrt(9.8/.18)
    rng = np.random.Generator(np.random.PCG64(GENERATOR_SEED))
    groups = {g: [] for g in GROUPS}
    groups['nominal'] = rng.uniform([-.02, -.02, -.05, -.05], [.02, .02, .05, .05], (20, 4)).tolist()
    scales = dict(position=[.02,.04,.06,.08,.10,.12,.15,.18,.21,.23],
                  velocity=[k/10*.95*math.sqrt(2*A*L) for k in range(1,11)],
                  angle=[k*math.pi/11 for k in range(1,11)],
                  angular_velocity=[k/10*natural for k in range(1,11)])
    for group, column in zip(GROUPS[1:5], range(4)):
        for value in scales[group]:
            for sign in (-1, 1):
                state = [0.,0.,0.,0.]; state[column] = sign*value
                groups[group].append(state)
    for position in (.20, .23):
        for fraction in (.25,.5,.75,.9,.99):
            for sign in (-1,1):
                groups['boundary'].append([sign*position,sign*fraction*math.sqrt(2*A*(L-position)),0.,0.])
    for k in range(1,11):
        vector = (rng.uniform(-1,1,4)*np.array([.18,.6,math.pi,natural])*k/10).tolist()
        groups['joint'].extend([vector,[-v for v in vector]])
    cases = []
    for group, vectors in groups.items():
        for i, vector in enumerate(vectors):
            state = dict(zip(FIELDS,vector))
            admission = assess_cart_state(x=vector[0],v=vector[1])
            if not admission.admissible:raise ValueError('inadmissible generated start')
            cases.append(dict(id=f'{group}_{i:02d}',subset=group,seed=None,
                generator_seed=GENERATOR_SEED,generation_index=i,initial_state=state,
                state_sha256=canonical_hash(state),admission=json.loads(json.dumps(asdict(admission)))))
    return dict(version=VERSION,generator='NumPy PCG64',generator_seed=GENERATOR_SEED,
                fields=list(FIELDS),cases=cases,case_count=140,final=False)


def frozen():
    verify_manifest(CONFIG,'manifest.json')
    states, bindings = read(CONFIG/'states.json'), read(CONFIG/'bindings.json')
    if states != generate_states():raise ValueError('frozen state generation differs')
    models = bindings['models']
    expected = {f'{a}_seed{s}' for a in ('SAC','TQC') for s in range(3)}|{'classical'}
    if len(models)!=7 or {m['id'] for m in models}!=expected:raise ValueError('model roster differs')
    for name, digest in bindings['current_sources'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('scientific/classical source changed: '+name)
    return states,bindings


def model_paths(source, bindings):
    return {m['id']: Path(source)/m['checkpoint_relative'] for m in bindings['models'] if m['kind']=='rl'}


def verify_models(source, bindings):
    """Hash only six pinned checkpoints; never resolve a mutable best pointer."""
    paths = model_paths(source,bindings); checked=[]
    for model in bindings['models']:
        if model['kind']=='classical':continue
        point=paths[model['id']]
        if sha256(point/'manifest.json')!=model['checkpoint_manifest_sha256']:
            raise ValueError('pinned checkpoint manifest changed: '+str(point))
        entries=read(point/'manifest.json')
        if entries!=model['files']:raise ValueError('pinned payload roster changed')
        verify_manifest(point)
        data=read(point/'metadata.json')
        if (data['config']['algorithm'],data['seed'],data['transitions'],data['updates'],data['policy_sha256']) != (
            model['algorithm'],model['seed'],model['transitions'],model['updates'],model['policy_sha256']):
            raise ValueError('checkpoint metadata differs')
        if entries['model.zip']!=model['model_sha256'] or not data['complete']:raise ValueError('model binding differs')
        checked.append(dict(id=model['id'],checkpoint=str(point),files=len(entries)))
    return checked


def jobs(stage='main'):
    states, bindings = frozen(); result=[]
    for model in bindings['models']:
        for mode in ('off','on'):
            for group in GROUPS:
                if stage=='timing' and (model['id']!='SAC_seed0' or group!='nominal'):continue
                result.append(dict(id=f'{model["id"]}_{mode}_{group}',model_id=model['id'],
                    algorithm=model['algorithm'],seed=model['seed'],mode=mode,group=group,
                    case_ids=[c['id'] for c in states['cases'] if c['subset']==group]))
    if stage not in ('main','timing'):raise ValueError('unknown stage')
    return result


def classify(metrics):
    """Independent flags; a technical failure has no fabricated episode metrics."""
    if metrics is None:return dict(technical_error=True)
    safety=metrics['safety']; reasons=metrics['completion']['stop_reasons']
    return dict(technical_error=False,filter_refusal=safety['refusal_count']>0,
        constraint_event=any(r in ('position_limit','velocity_limit') for r in reasons),
        desired_exceedance=safety['exceeds_024_resolved'],
        working_position_exceedance=metrics['max_abs_position_interval']>.25+1e-12,
        working_velocity_exceedance=metrics['max_abs_velocity']>2.+1e-12,
        hard_position_exceedance=metrics['max_abs_position_interval']>.27+1e-12,
        hard_velocity_exceedance=metrics['max_abs_velocity']>2.5+1e-12,
        hard_acceleration_exceedance=metrics['max_abs_applied_acceleration']>5.+1e-12,
        hold_failure=not metrics['success_episode'],success=metrics['success_episode'])
