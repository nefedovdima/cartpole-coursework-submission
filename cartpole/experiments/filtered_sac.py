"""S3a bounded SAC technical pilot. No provider calls or implicit session creation."""
import argparse
import json
import os
from pathlib import Path
import sys
import tarfile

from cartpole.experiments.cloud_io import ROOT, atomic_json, manifest, new_output, sha256, verify_manifest


def compatibility(expected_root, device):
    import cartpole
    from cartpole.experiments.cloud_checks import check_stack
    from cartpole.experiments.cloud_config import load_config
    from cartpole.experiments.cloud_learning import configure_device, implementation_hashes
    from cartpole.rl.training_contract import assert_environment, environment_factory
    if ROOT != Path(expected_root).resolve() or Path(cartpole.__file__).resolve() != ROOT/'cartpole/__init__.py':
        raise ValueError('wrong imported code root; clear PYTHONPATH and run from extracted package')
    count = verify_manifest(ROOT) if (ROOT/'manifest.json').exists() else None
    config = load_config('sac_filtered_pilot', 0, device)
    versions = check_stack(config); actual_device = configure_device(device)
    env = environment_factory(config)()
    try:
        assert_environment(env, config['training_contract'])
    finally:
        env.close()
    return dict(root=str(ROOT), executable=sys.executable, payload_files_verified=count,
                device=str(actual_device), versions=versions, training_contract=config['training_contract'],
                implementation=implementation_hashes(config), drake_steps=0, updates=0)


def pack_response(run, output):
    """Small review export; full validation logs/checkpoints remain in the run."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError('response output must be new and outside the run')
    paths = []
    for pattern in ('*.json', '*.jsonl', 'segments/*/*.json', 'evaluations/*.json',
                    'checkpoints/*/metadata.json', 'checkpoints/*/probes.json', 'checkpoints/*/manifest.json'):
        paths.extend(run.glob(pattern))
    paths = sorted(set(p for p in paths if p.is_file() and not p.is_symlink()))
    sizes = sum(p.stat().st_size for p in paths)
    if sizes > 64*1024**2:
        raise ValueError('small response cap exceeded (64 MiB uncompressed)')
    entries = {str(p.relative_to(run)): sha256(p) for p in paths}
    import io
    with tarfile.open(output, 'x:gz') as tar:
        for path in paths:
            tar.add(path, arcname='s3a-response/'+str(path.relative_to(run)), recursive=False)
        data = json.dumps(entries, sort_keys=True, indent=2).encode()
        info = tarfile.TarInfo('s3a-response/RESPONSE_MANIFEST.json'); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    digest = sha256(output)
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n')
    return dict(path=str(output), sha256=digest, bytes=output.stat().st_size, files=len(paths),
                omitted='model/replay/runtime/source_snapshot/full logs; preserve run separately')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('compat', 'preflight', 'pilot', 'resume'):
        q = sub.add_parser(name)
        q.add_argument('--expected-root', type=Path, required=True)
        q.add_argument('--device', required=True)
        if name != 'compat':
            q.add_argument('--output-dir', type=Path, required=name != 'resume')
            q.add_argument('--session', type=Path, required=name in ('pilot', 'resume'))
        if name in ('pilot', 'resume'):
            q.add_argument('--max-transitions', type=int, required=True)
            q.add_argument('--max-seconds', type=float, required=True)
        if name == 'resume':
            q.add_argument('--run-dir', type=Path, required=True)
    q = sub.add_parser('pack'); q.add_argument('--run-dir', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'pack':
        result = pack_response(args.run_dir, args.output)
    else:
        compat = compatibility(args.expected_root, args.device)
        if args.command == 'compat':
            result = compat
        else:
            from contextlib import nullcontext
            from cartpole.experiments.cloud_budget import Session
            from cartpole.experiments.cloud_config import load_config
            from cartpole.experiments.cloud_runner import run_training
            from cartpole.experiments.cloud_checkpoint import resolve_checkpoint, load_checkpoint
            from cartpole.experiments.cloud_checks import smoke_proof
            session = Session(args.session) if args.session else None
            if session and (os.environ.get('CARTPOLE_BUDGET_SESSION') != str(args.session.resolve())):
                raise ValueError('run the pilot/preflight through cloud_guard using this exact session')
            smoke = args.command == 'preflight'
            config = (json.loads((args.run_dir/'config.json').read_text()) if args.command == 'resume'
                      else load_config('sac_filtered_pilot', 0, args.device, purpose='smoke' if smoke else 'experiment'))
            if config['id'] != 'sac_filtered_pilot' or config['settings']['device'] != args.device:
                raise ValueError('resume requires this filtered pilot and the saved device')
            with session.lease() if session else nullcontext():
                if session:
                    status = session.enter_phase('setup')
                    if status['seconds_before_save_reserve'] <= 0:
                        raise ValueError('session budget exhausted')
                run, result = run_training(config, output=args.output_dir,
                    resume=args.run_dir if args.command == 'resume' else None,
                    max_transitions=12 if smoke else args.max_transitions,
                    max_seconds=60. if smoke else args.max_seconds, session=session,
                    evaluate_enabled=not smoke)
                atomic_json(run/'s3a_compatibility.json', compat)
                if smoke and (result['transitions'], result['gradient_updates'], result['status']) != (12, 4, 'completed'):
                    raise RuntimeError('preflight incomplete: requires 12 Drake transitions and 4 real updates')
                if smoke and result['status'] == 'completed':
                    result['smoke_proof'] = smoke_proof(run, args.device)
                    loaded, _, runtime = load_checkpoint(resolve_checkpoint(run), device=args.device, expected_config=config)
                    result['reload'] = dict(updates=loaded._n_updates, stored=loaded.replay_buffer.total_added,
                                            phase=runtime['run_state']['phase'])
                    atomic_json(run/'preflight.json', result)
                if session:
                    session.record(dict(event='s3a_'+args.command, run=str(run), stop_reason=result['stop_reason']))
                atomic_json(run/'manifest.json', manifest(run))
                result = dict(output=str(run), summary=result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
