"""Fresh evaluation of an inference capsule; no execution without --execute."""
import argparse
import json
from pathlib import Path
from scripts.frozen_inference import verify_capsule, load_capsule


def plan(model_id, mode):
    from cartpole.experiments.cloud_evaluation import validation_cases
    if mode not in ('on', 'off'):
        raise ValueError('filter mode required')
    _, data, _ = verify_capsule(model_id)
    cases = validation_cases()
    return dict(model=model_id, algorithm=data['config']['algorithm'], seed=data['seed'],
                policy_sha256=data['policy_sha256'], filter=mode, episodes=len(cases),
                validation=20, named=3, horizon_seconds=10, workers=1,
                weights='original main-best', resume_supported=False)


def execute(model_id, mode, device, output):
    from cartpole.experiments.cloud_evaluation import evaluate, validation_cases
    from cartpole.experiments.cloud_io import atomic_json
    from cartpole.rl.methods import contract_id
    from cartpole.rl.training_contract import identity, make_training_env
    output = Path(output)
    if output.exists():
        raise FileExistsError('fresh output required; no implicit resume')
    spec = plan(model_id, mode)
    model, data, checks = load_capsule(model_id, device)
    cid = contract_id(spec['algorithm'], mode)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'plan.json', dict(spec, device=device, probe_checks=checks))
    return evaluate(model, validation_cases(), output, 'main_best_capsule',
        algorithm=spec['algorithm'], seed=spec['seed'],
        env_factory=lambda: make_training_env(cid), evaluation_contract=identity(cid),
        checkpoint_identity=dict(capsule_model_id=model_id, policy_sha256=data['policy_sha256']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--filter', choices=['on','off'], required=True)
    p.add_argument('--device', choices=['cpu','cuda:0','cuda:1'], default='cpu')
    p.add_argument('--output', type=Path)
    p.add_argument('--execute', action='store_true')
    args=p.parse_args()
    result=plan(args.model,args.filter)
    if args.execute:
        if args.output is None:p.error('--execute requires --output')
        result=execute(args.model,args.filter,args.device,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
