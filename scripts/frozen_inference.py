"""Read-only inference capsules bound to the original main-best checkpoints.

This is not the training/resume loader. Verify the original manifest and each
included payload before SB3 deserializes our trusted model. Replay/runtime are
intentionally absent; their original SHA remain in the historical manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOADS = ('model.zip', 'metadata.json', 'probes.json', 'source_manifest.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_capsule(model_id, root=ROOT):
    root = Path(root)
    bindings_path = root/'configs/robustness/bindings.json'
    bindings = json.loads(bindings_path.read_text())
    index = json.loads((root/'models/INDEX.json').read_text())
    if index['schema'] != 'inference-capsules-v1' or index['bindings_sha256'] != digest(bindings_path):
        raise ValueError('capsule index/bindings mismatch')
    expected_ids = {m['id'] for m in bindings['models'] if m['kind'] == 'rl'}
    ids = [m['model_id'] for m in index['models']]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids or model_id not in expected_ids:
        raise ValueError('unknown, duplicate or missing capsule')
    binding = next(m for m in bindings['models'] if m['id'] == model_id)
    entry = next(m for m in index['models'] if m['model_id'] == model_id)
    path = root/'models'/model_id
    if set(p.name for p in path.iterdir()) != set(PAYLOADS) or any((path/n).is_symlink() for n in PAYLOADS):
        raise ValueError('capsule must contain exactly four regular inference payloads')
    if set(entry['files']) != set(PAYLOADS):
        raise ValueError('capsule payload schema differs')
    if entry['original_manifest_sha256'] != binding['checkpoint_manifest_sha256']:
        raise ValueError('original manifest binding differs')
    for name in PAYLOADS:
        expected = binding['checkpoint_manifest_sha256'] if name == 'source_manifest.json' else binding['files'][name]
        if entry['files'][name] != expected or digest(path/name) != expected:
            raise ValueError('capsule SHA mismatch: '+name)
    original = json.loads((path/'source_manifest.json').read_text())
    if original != binding['files']:
        raise ValueError('original checkpoint manifest differs')
    data = json.loads((path/'metadata.json').read_text())
    probes = json.loads((path/'probes.json').read_text())
    if not data['complete'] or data['config']['algorithm'] != binding['algorithm']:
        raise ValueError('incomplete or wrong algorithm')
    for key in ('seed', 'transitions', 'updates', 'policy_sha256'):
        if data[key] != binding[key]:
            raise ValueError('checkpoint identity differs: '+key)
    if entry['checkpoint_relative'] != binding['checkpoint_relative'] or entry['training_resume_supported'] is not False:
        raise ValueError('capsule provenance/resume contract differs')
    return path, data, probes


def load_capsule(model_id, device='cpu', root=ROOT):
    path, data, probes = verify_capsule(model_id, root)
    from cartpole.experiments.cloud_learning import (algorithm_class, configure_device,
        implementation_hashes, policy_fingerprint, finite_model)
    from cartpole.experiments.cloud_checks import check_stack
    from cartpole.experiments.parallel_evaluation import inference_stack_config
    from cartpole.experiments.inference_compatibility import observations_for_inference, compare_actions
    from cartpole.rl.training_contract import evaluation_action
    cfg = data['config']
    if data['implementation'] != implementation_hashes(cfg):
        raise ValueError('checkpoint library implementation differs')
    # Exact scientific contract and all source pins; only the existing reviewed
    # cloud_config registry pair may migrate, and only in this inference path.
    check_stack(inference_stack_config(cfg))
    requested = configure_device(device)
    model = algorithm_class(cfg).load(path/'model.zip', device=device, force_reset=True)
    if model.device != requested or any(p.device != requested for p in model.policy.parameters()):
        raise ValueError('inference device fallback')
    if (model.num_timesteps, model._n_updates, model.seed) != (data['transitions'], data['updates'], data['seed']):
        raise ValueError('loaded counters differ')
    if model.training_contract != cfg['training_contract']:
        raise ValueError('loaded model contract differs')
    finite_model(model)
    if policy_fingerprint(model) != data['policy_sha256']:
        raise ValueError('loaded policy weights differ')
    if (probes['training_seed'], probes['transitions'], probes['updates'], probes['device']) != (data['seed'], data['transitions'], data['updates'], data['device']):
        raise ValueError('probe identity differs')
    model.policy.set_training_mode(False)
    obs = observations_for_inference(probes['observations'])
    kw = dict(algorithm=cfg['algorithm'], count=len(obs), saved_device=data['device'], loaded_device=str(requested))
    checks = dict(requests=compare_actions(model.predict(obs, deterministic=True)[0], probes['actions'], **kw),
                  environment_requests=compare_actions(evaluation_action(model, obs, filtered=True), probes['env_actions'], **kw))
    return model, data, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='all')
    parser.add_argument('--load-check', action='store_true', help='Load weights and compare saved probes; zero simulated transitions.')
    parser.add_argument('--device', choices=['cpu','cuda:0','cuda:1'], default='cpu')
    args = parser.parse_args()
    ids = [m['model_id'] for m in json.loads((ROOT/'models/INDEX.json').read_text())['models']]
    if args.model != 'all':
        ids = [args.model]
    results = []
    for model_id in ids:
        _, data, probes = verify_capsule(model_id)
        result = dict(model=model_id, transitions=data['transitions'], updates=data['updates'], payloads=4, sha='passed')
        if args.load_check:
            model, _, checks = load_capsule(model_id,args.device)
            result['inference'] = checks
            del model
        results.append(result)
    print(json.dumps(dict(results=results, simulated_transitions=0, training_updates=0, resume_supported=False), indent=2))

if __name__ == '__main__':
    main()
