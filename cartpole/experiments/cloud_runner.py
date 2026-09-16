"""Single-process, one-env SAC/TQC execution, with bounded resumable segments."""
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from cartpole.experiments.cloud_checkpoint import (
    load_checkpoint, prepare_resume_environment, prune_checkpoints, publish_pointer,
    resolve_checkpoint, save_checkpoint,
)
from cartpole.experiments.cloud_config import resolved_model, validate_config
from cartpole.experiments.cloud_benchmark import CollectionTiming
from cartpole.experiments.cloud_recovery import recover_journals, reconcile_losses, loss_totals
from cartpole.experiments.cloud_evaluation import evaluate, validation_cases
from cartpole.experiments.cloud_io import ROOT, append_json, atomic_json, manifest, new_output, sha256, snapshot
from cartpole.experiments.cloud_learning import create_model, finite_model, isolated_policy, policy_fingerprint
from cartpole.experiments.sac_pilot import selection_score
from cartpole.rl.pilot_config import make_pilot_env
from cartpole.rl.training_contract import environment_factory, assert_environment
from cartpole.rl.sac import assert_pilot_contract
from cartpole.rl.sac_diagnostics import ExperienceBlocks, conditional_actions, diagnostic_states, fixed_normal_samples


class RunLimit(Exception):
    pass


class StopSignals:
    def __init__(self):
        self.reason = None

    def receive(self, signum, frame=None):
        self.reason = signal.Signals(signum).name

    @contextmanager
    def installed(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in previous:
            signal.signal(s, self.receive)
        try:
            yield
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


def used_seconds(run, now=None):
    now = time.time() if now is None else now
    total = 0.
    for path in (Path(run)/'segments').glob('*/segment.json'):
        data = json.loads(path.read_text())
        # A crashed segment is charged through this resume, including downtime.
        total += data.get('duration_seconds', max(0., now-data['started_at_epoch']))
    return total


def seal_interrupted_segments(run):
    now = time.time()
    for path in (Path(run)/'segments').glob('*/segment.json'):
        data = json.loads(path.read_text())
        if 'duration_seconds' not in data:
            data.update(status='unclean_exit_detected_at_resume', duration_seconds=max(0., now-data['started_at_epoch']),
                        includes_unobserved_downtime=True, reconciled_at_epoch=now)
            atomic_json(path, data)


def assert_resume_sources(run):
    original = json.loads((Path(run)/'source_manifest.json').read_text())
    required = {name: digest for name, digest in original.items()
                if name.startswith(('cartpole/', 'configs/cloud/', 'tests/fixtures/'))}
    changed = [name for name, digest in required.items() if not (ROOT/name).is_file() or sha256(ROOT/name) != digest]
    if changed:
        raise ValueError(f'resume source/fixture/config mismatch: {changed}')


class TrainingCallback(BaseCallback):
    def __init__(self, runner):
        super().__init__(); self.runner = runner

    def _on_training_start(self):
        if self.model.replay_buffer.total_added != self.model.num_timesteps:
            raise AssertionError('initial replay/learner counters mismatch')
        append_json(self.runner.run/'history.jsonl', dict(event='learn_start', segment=self.runner.segment.name,
                    transitions=self.num_timesteps, updates=self.model._n_updates,
                    seeded_reset=not self.runner.resumed,
                    environment_seed=self.model.get_env().envs[0].unwrapped.np_random_seed))

    def _on_rollout_start(self):
        self.runner.guard(); self.collection_start = time.perf_counter()
        r = self.runner
        if self.model.num_timesteps >= self.model.learning_starts and not r.collection_timing.pending['updates']:
            r.block_started = self.collection_start
            r.block_excluded = r.eval_seconds+r.save_seconds

    def _on_step(self):
        self.runner.experience.observe(self.locals['infos'][0])
        if ('training_contract' in self.runner.config
                and 'safety_filter_refusal' in self.locals['infos'][0]['stop_reasons']):
            counts = self.runner.experience.counts
            counts['physical_terminations'] -= 1
            counts['filter_refusals'] = counts.get('filter_refusals', 0)+1
        # SB3 calls this BEFORE storing replay. Never return False or raise a
        # deadline here: the current physical transition must reach replay.
        return True

    def _on_rollout_end(self):
        r = self.runner
        seconds = time.perf_counter()-self.collection_start
        r.collection_seconds += seconds
        r.collection_timing.collect(warmup=self.num_timesteps <= self.model.learning_starts, seconds=seconds)
        r.phase = 'collected_pending_update' if self.num_timesteps > self.model.learning_starts else 'warmup'
        if self.num_timesteps % 1000 == 0:
            r.write_progress()
        r.experience.flush()
        if 'training_contract' in r.config:
            r.experience.rows[:] = r.experience.rows[-32:]
            r.experience.episodes[:] = r.experience.episodes[-16:]
        if self.num_timesteps == self.model.learning_starts:
            r.warmup_finished_at = r.cumulative_seconds()
        r.guard()
        if self.num_timesteps <= self.model.learning_starts:
            r.boundary_tasks()


class Runner:
    def __init__(self, config, run, *, max_transitions, max_seconds, session=None, resumed=False,
                 signals=None, evaluate_enabled=None, checkpoint_free_bytes=256*1024**2):
        validate_config(config)
        if (isinstance(max_transitions, bool) or not isinstance(max_transitions, int)
                or max_transitions < 1 or not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError('positive finite total transition/time limits required')
        self.config, self.run, self.session, self.resumed = config, Path(run), session, resumed
        if 'training_contract' in config and (max_transitions > config['protocol']['pilot_max_transitions']
                or max_seconds > config['protocol']['pilot_max_seconds']):
            raise ValueError('S3a technical pilot cap exceeded')
        self.signals = signals or StopSignals()
        self.evaluate_enabled = config['protocol'].get('automatic_evaluation', True) if evaluate_enabled is None else evaluate_enabled
        if config['protocol'].get('automatic_evaluation') is False and self.evaluate_enabled:
            raise ValueError('v2 mechanics excludes automatic evaluation; run a separate full evaluation')
        self.checkpoint_free_bytes = checkpoint_free_bytes
        self.next_disk_check = 0.
        previous_progress = None
        unclean = False
        accounting_delta = dict(discarded_known=0, uncertain=0)
        if resumed:
            assert_resume_sources(run)
            recover_journals(run)
            resume_point = resolve_checkpoint(run)
            resume_metadata = json.loads((resume_point/'metadata.json').read_text())
            if (self.run/'progress.json').exists():
                previous_progress = json.loads((self.run/'progress.json').read_text())
            old_segments = {p.parent.name: json.loads(p.read_text()) for p in (self.run/'segments').glob('*/segment.json')}
            unclean = any('duration_seconds' not in data for data in old_segments.values())
            self.recovery_accounting, accounting_delta = reconcile_losses(run, resume_metadata, previous_progress, old_segments)
            seal_interrupted_segments(run)
        else:
            self.recovery_accounting = dict(schema=1, segments={})
            atomic_json(self.run/'recovery_state.json', self.recovery_accounting)
        self.prior_seconds = used_seconds(run)
        self.started = time.perf_counter(); self.epoch = time.time()
        self.max_transitions, self.max_seconds = max_transitions, max_seconds
        reserve = config['save_reserve_seconds']
        self.stop_at = self.started + max(0., max_seconds-self.prior_seconds-reserve)
        if session:
            self.stop_at = min(self.stop_at, self.started+session.status()['seconds_before_save_reserve'])
        self.segment = self.run/'segments'/f'{len(list((self.run/"segments").glob("*"))):04d}'
        self.segment.mkdir(parents=True, exist_ok=False)
        self.segment_data = dict(started_at_epoch=self.epoch, status='running', resumed=resumed,
                                 prior_seconds=self.prior_seconds, total_transition_limit=max_transitions,
                                 cumulative_time_limit=max_seconds)
        atomic_json(self.segment/'segment.json', self.segment_data)
        self.phase = 'untrained'; self.collection_seconds = 0.
        self.collection_timing = CollectionTiming()
        self.block_started = None; self.block_excluded = 0.
        self.eval_seconds = self.save_seconds = 0.
        self.warmup_finished_at = None
        self.last_assessed = None; self.last_checkpoint = None
        self.last_report = None
        self.best_score = None
        self.experience = ExperienceBlocks(config['experience_block_size'])
        totals = loss_totals(self.recovery_accounting)
        self.discarded_known, self.unclean_allowance = totals['discarded_known'], totals['uncertain']
        self.cases = validation_cases()
        self.env_factory = environment_factory(config)
        self.env = self.env_factory()
        if 'training_contract' in config:
            assert_environment(self.env, config['training_contract'])
        else:
            self.env.reset(seed=config['settings']['seed'])
            assert_pilot_contract(self.env)
        if resumed:
            point = resume_point
            self.model, metadata, runtime = load_checkpoint(point, device=config['settings']['device'],
                                                            env=self.env, expected_config=config)
            self.phase = metadata['phase']; self.last_checkpoint = point
            self.experience = runtime['run_state']['experience']
            self.experience.flush(partial=True); self.experience.confirmed_in_episode = False
            reset = prepare_resume_environment(self.model, runtime)
            append_json(self.run/'history.jsonl', dict(event='resume', segment=self.segment.name,
                        source_generation=point.name, transitions=self.model.num_timesteps,
                        updates=self.model._n_updates, prior_seconds=self.prior_seconds,
                        discarded_collected_lower_bound=accounting_delta['discarded_known'],
                        unobserved_tail_budget_charge=accounting_delta['uncertain'],
                        unclean_exit_detected=unclean, **reset))
            if (self.run/'best_evaluation.json').exists():
                self.best_score = selection_score(json.loads((self.run/'best_evaluation.json').read_text()))
        else:
            self.model = create_model(config, self.env)
        self.model.set_logger(configure(folder=None, format_strings=[]))
        self.model.after_update = self.after_update
        self.initial_transitions = self.model.num_timesteps
        self.initial_updates = self.model._n_updates
        self.initial_train_host = self.previous_train_host = self.model.update_host_seconds
        self.initial_train_cuda = self.model.update_device_seconds
        self.telemetry_written = len(self.model.telemetry)
        n = self.model.num_timesteps
        self.next_evaluation = ((n+self.evaluation_interval-1)//self.evaluation_interval)*self.evaluation_interval
        if resumed:
            self.next_evaluation = min(self.next_evaluation, runtime['run_state'].get('next_evaluation', self.next_evaluation))
            self.restore_assessment(point)
        self.next_recovery = (self.model.num_timesteps//config['recovery_interval']+1)*config['recovery_interval']
        self.last_save_time = time.perf_counter()
        self.initialization_seconds = time.perf_counter()-self.started
        atomic_json(self.segment/'resolved_model.json', resolved_model(self.model, config))
        self.effective_target = max(0, max_transitions-self.discarded_known-self.unclean_allowance)
        self.write_progress()

    @property
    def evaluation_interval(self):
        return self.config['evaluation_interval']

    def restore_assessment(self, point):
        identity = (self.model.num_timesteps, self.model._n_updates)
        fingerprint = policy_fingerprint(self.model)
        for path in sorted((self.run/'evaluations').glob('*.json')):
            report = json.loads(path.read_text())
            if (report.get('complete') and report.get('selection_eligible')
                    and (report.get('trained_transitions'), report.get('updates')) == identity
                    and report.get('policy_sha256') == fingerprint):
                self.last_report, self.last_assessed = report, identity
                self.next_evaluation = (identity[0]//self.evaluation_interval+1)*self.evaluation_interval
                self.select_best(point, report)
                return

    def select_best(self, point, report):
        if report['selection_eligible']:
            # A recovery save may have a different ZIP/generation but the same
            # policy. Reuse is permitted only for that exact policy/counter pair.
            metadata = json.loads((point/'metadata.json').read_text())
            expected = (self.config['algorithm'], self.model.seed,
                        self.model.num_timesteps, self.model._n_updates, policy_fingerprint(self.model))
            actual = tuple(report[k] for k in ('algorithm', 'training_seed',
                           'trained_transitions', 'updates', 'policy_sha256'))
            saved = (metadata['config']['algorithm'], metadata['seed'],
                     metadata['transitions'], metadata['updates'], metadata['policy_sha256'])
            if actual != expected or saved != expected:
                raise ValueError('assessment policy/counters differ from selected checkpoint')
            if report['label'] != report['checkpoint_generation']:
                raise ValueError('assessment label/generation differs')
            if report['label'] == point.name and report['model_sha256'] != sha256(point/'model.zip'):
                raise ValueError('assessment model SHA differs from selected checkpoint')
            score = selection_score(report)
            if self.best_score is None or score >= self.best_score:
                self.best_score = score; publish_pointer(self.run, 'best', point)
                selected = dict(report, checkpoint_generation=point.name, label=point.name,
                                model_sha256=sha256(point/'model.zip'))
                if report['label'] != point.name:
                    selected['reused_from'] = report['label']
                atomic_json(self.run/'best_evaluation.json', selected)

    def write_progress(self):
        atomic_json(self.run/'progress.json', dict(segment=self.segment.name, transitions=self.model.num_timesteps,
                    updates=self.model._n_updates, timestamp=time.time(), maximum_unobserved_tail=1000))

    def cumulative_seconds(self):
        return self.prior_seconds+time.perf_counter()-self.started

    def guard(self):
        if self.signals.reason:
            raise RunLimit(self.signals.reason)
        if time.perf_counter() >= self.stop_at:
            raise RunLimit('wall_or_session_limit_save_reserve')
        if self.checkpoint_free_bytes and time.perf_counter() >= self.next_disk_check:
            import shutil
            self.next_disk_check = time.perf_counter()+1.
            # Keep space for one more atomic model/replay/runtime generation.
            rb = self.model.replay_buffer
            replay_bytes = sum(getattr(rb, name).nbytes for name in
                ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts', 'slot_sequence'))
            reserve = self.checkpoint_free_bytes+2*replay_bytes+64*1024**2
            if shutil.disk_usage(self.run).free < reserve:
                raise RunLimit('disk_limit_save_reserve')

    def flush_telemetry(self):
        for row in self.model.telemetry[self.telemetry_written:]:
            append_json(self.run/'training_updates.jsonl', dict(training_seed=self.model.seed,
                        cumulative_seconds=self.cumulative_seconds(), segment=self.segment.name, **row))
        self.telemetry_written = len(self.model.telemetry)

    def save(self):
        self.flush_telemetry()
        start = time.perf_counter()
        state = dict(phase=self.phase, cumulative_seconds=self.cumulative_seconds(), experience=self.experience,
                     next_evaluation=self.next_evaluation, recovery_accounting=loss_totals(self.recovery_accounting))
        point = save_checkpoint(self.model, self.run, self.config, state, self.cases,
                                min_free_bytes=self.checkpoint_free_bytes)
        self.save_seconds += time.perf_counter()-start
        self.last_checkpoint, self.last_save_time = point, time.perf_counter()
        self.write_progress()
        if self.last_report and self.last_assessed == (self.model.num_timesteps, self.model._n_updates):
            report = dict(self.last_report, label=point.name, checkpoint_generation=point.name,
                          model_sha256=sha256(point/'model.zip'), seconds=0.,
                          reused_from=self.last_report['label'])
            atomic_json(self.run/'evaluations'/(point.name+'.json'), report)
            atomic_json(self.run/'last_evaluation.json', report)
        else:
            atomic_json(self.run/'last_evaluation.json', dict(complete=False, selection_eligible=False,
                        checkpoint_generation=point.name, reason='these weights have no complete evaluation yet'))
        append_json(self.run/'history.jsonl', dict(event='checkpoint', generation=point.name,
                    transitions=self.model.num_timesteps, updates=self.model._n_updates,
                    phase=self.phase, cumulative_seconds=self.cumulative_seconds()))
        prune_checkpoints(self.run, self.config['keep_recovery'])
        return point

    def assess(self):
        self.guard()
        workers = int(os.environ.get('CARTPOLE_EVALUATION_WORKERS', '1'))
        if not 1 <= workers <= 64: raise ValueError('evaluation workers must be 1..64')
        fingerprint = policy_fingerprint(self.model)
        pin = self.run/'pinned_evaluation.json'
        saved_pin = json.loads(pin.read_text()) if pin.exists() else None
        counters = dict(transitions=self.model.num_timesteps, updates=self.model._n_updates)
        if (workers > 1 and saved_pin and saved_pin['policy_sha256'] == fingerprint
                and saved_pin.get('counters') == counters):
            point = self.run/'checkpoints'/saved_pin['generation']
            if not point.resolve().is_relative_to((self.run/'checkpoints').resolve()): raise ValueError('unsafe evaluation pin')
            if sha256(point/'manifest.json') != saved_pin['manifest_sha256']: raise ValueError('evaluation pin differs')
        else:
            point = self.save()
            if workers > 1:
                atomic_json(pin, dict(generation=point.name,policy_sha256=fingerprint,
                                      counters=counters,manifest_sha256=sha256(point/'manifest.json')))
        start = time.perf_counter()
        try:
            if workers > 1:
                from cartpole.experiments.parallel_evaluation import evaluate_checkpoint
                from cartpole.rl.methods import parse_contract
                _, mode = parse_contract(self.config['training_contract']['id'])
                report = evaluate_checkpoint(point,self.cases,self.run/'parallel_evaluations'/point.name,
                    mode=mode,device=str(self.model.device),workers=workers,guard=self.guard)
                report.update(policy_sha256=fingerprint, evaluation_workers=workers)
            else:
                report = evaluate(self.model, self.cases, self.run, point.name, algorithm=self.config['algorithm'],
                              seed=self.model.seed, guard=self.guard,
                              diagnostic=self.config['purpose'] != 'experiment' or 'training_contract' in self.config,
                              env_factory=self.env_factory,
                              checkpoint_identity=dict(checkpoint_generation=point.name, policy_sha256=policy_fingerprint(self.model),
                              logs_prefix=point.name, model_sha256=sha256(point/'model.zip'),
                              evaluation_started_cumulative=self.cumulative_seconds()))
        finally:
            self.eval_seconds += time.perf_counter()-start
        # The generic worker report has no Runner label. Bind both evaluator
        # branches before publishing or selecting; never invent it on reuse.
        report.update(label=point.name, checkpoint_generation=point.name, model_sha256=sha256(point/'model.zip'),
                      cumulative_seconds=self.cumulative_seconds(), scheduled_interval=self.config['evaluation_interval'],
                      logs_prefix=point.name)
        if report['complete']:
            report['diagnostic'] = self.config['purpose'] != 'experiment' or 'training_contract' in self.config
            report['scientific_result'] = not report['diagnostic']
        report['evaluation_origin'] = dict(label=point.name, checkpoint_generation=point.name,
            model_sha256=report['model_sha256'], policy_sha256=report['policy_sha256'])
        atomic_json(self.run/'evaluations'/(point.name+'.json'), report)
        if not report.get('complete') or not report.get('selection_eligible'):
            raise RunLimit('mandatory_evaluation_incomplete')
        self.last_assessed = (self.model.num_timesteps, self.model._n_updates)
        self.last_report = report
        self.select_best(point, report)
        if self.config['algorithm'] in ('SAC', 'TQC'):
            with isolated_policy(self.model):
                conditional = conditional_actions(self.model, diagnostic_states(), fixed_normal_samples())
            append_json(self.run/'conditional_actions.jsonl', dict(algorithm=self.config['algorithm'],
                        training_seed=self.model.seed, generation=point.name, **conditional))
        append_json(self.run/'history.jsonl', dict(event='evaluation', generation=point.name,
                    transitions=self.model.num_timesteps, updates=self.model._n_updates,
                    cumulative_seconds=self.cumulative_seconds(), success_rate=report['validation']['success_rate']))
        print(json.dumps(dict(event='evaluation', seed=self.model.seed, transitions=self.model.num_timesteps,
                              updates=self.model._n_updates, success=report['validation']['success_rate'])), flush=True)

    def boundary_tasks(self):
        self.flush_telemetry()
        n = self.model.num_timesteps
        if self.evaluate_enabled and n >= self.next_evaluation:
            scheduled = self.next_evaluation
            append_json(self.run/'history.jsonl', dict(event='evaluation_due', scheduled=scheduled, actual=n))
            self.assess()
            self.next_evaluation = (n//self.evaluation_interval+1)*self.evaluation_interval
        if n >= self.next_recovery or time.perf_counter()-self.last_save_time >= self.config['recovery_seconds']:
            self.next_recovery = (n//self.config['recovery_interval']+1)*self.config['recovery_interval']
            if self.last_assessed != (n, self.model._n_updates):
                self.save()

    def after_update(self, model):
        if self.collection_timing.update(model.update_host_seconds-self.previous_train_host):
            model.flush_cuda_timing()
            self.collection_timing.close_block(model.update_device_seconds-self.initial_train_cuda,
                                                wall_seconds=self.block_wall())
        self.previous_train_host = model.update_host_seconds
        self.phase = 'post_update'; self.guard(); self.boundary_tasks()

    def block_wall(self):
        return (max(0.,time.perf_counter()-self.block_started-self.eval_seconds-self.save_seconds+self.block_excluded)
                if self.block_started is not None else None)

    def execute(self):
        reason, error = 'transition_limit', None
        try:
            self.guard()
            if not self.resumed:
                if self.evaluate_enabled:
                    self.assess()
                    self.next_evaluation = self.evaluation_interval
                else:
                    self.save()
            if self.resumed and self.phase == 'collected_pending_update':
                # Finish exactly the update owed by the last stored transition.
                self.model.train(1, self.model.batch_size)
                append_json(self.run/'history.jsonl', dict(event='pending_update_completed',
                            transitions=self.model.num_timesteps, updates=self.model._n_updates))
            if self.resumed and self.evaluate_enabled:
                self.boundary_tasks()
                if self.model.num_timesteps >= self.effective_target and self.last_assessed != (self.model.num_timesteps, self.model._n_updates):
                    self.assess()
            if self.model.num_timesteps >= self.effective_target:
                raise RunLimit('total_transition_limit_already_reached')
            self.model.learn(total_timesteps=self.effective_target-self.model.num_timesteps,
                             callback=TrainingCallback(self), reset_num_timesteps=not self.resumed,
                             log_interval=None, progress_bar=False)
            if self.evaluate_enabled and self.last_assessed != (self.model.num_timesteps, self.model._n_updates):
                self.assess()
        except RunLimit as exc:
            reason = str(exc)
        except Exception as exc:
            reason = 'exception'; error = exc
            atomic_json(self.segment/'exception.json', dict(type=type(exc).__name__, message=str(exc),
                        transitions=self.model.num_timesteps, updates=self.model._n_updates,
                        physical_exception=self.env.last_exception,
                        traceback=traceback.format_exc(),
                        last_valid_checkpoint=self.last_checkpoint.name if self.last_checkpoint else None))
        finally:
            self.model.after_update = None
            self.model.flush_cuda_timing()
            self.collection_timing.close_block(self.model.update_device_seconds-self.initial_train_cuda,
                                               partial=True, wall_seconds=self.block_wall())
            self.experience.flush(partial=True)
            self.flush_telemetry()
            self.write_progress()
            if error is None:
                try:
                    self.save()
                except Exception as exc:
                    error, reason = exc, 'checkpoint_save_failed'
                    atomic_json(self.segment/'save_exception.json', dict(type=type(exc).__name__, message=str(exc)))
            atomic_json(self.segment/'experience_blocks.json', dict(rows=self.experience.rows, episodes=self.experience.episodes))
            elapsed = time.perf_counter()-self.started
            learning_complete = (self.model.num_timesteps >= self.max_transitions
                and self.model._n_updates == max(0, self.model.num_timesteps-self.model.learning_starts)
                and self.phase != 'collected_pending_update')
            evaluation_complete = (not self.evaluate_enabled or bool(self.last_report
                and self.last_report.get('complete') and self.last_report.get('selection_eligible')
                and self.last_assessed == (self.model.num_timesteps, self.model._n_updates)))
            complete = learning_complete and evaluation_complete
            self.segment_data.update(status='failed' if error else ('completed' if complete else 'stopped'),
                        learning_complete=learning_complete, mandatory_evaluation_complete=evaluation_complete,
                        stop_reason=reason, duration_seconds=elapsed, ended_at_epoch=time.time(),
                        start_transitions=self.initial_transitions, start_updates=self.initial_updates,
                        transitions=self.model.num_timesteps, updates=self.model._n_updates)
            atomic_json(self.segment/'segment.json', self.segment_data)
            try:
                replay_audit = self.model.replay_buffer.audit(self.model.num_timesteps)
            except Exception as audit_error:
                if error is None:
                    raise
                # Preserve the original integrator/learner exception and the
                # last published checkpoint when the failed step was unstored.
                replay_audit = dict(valid=False, error=str(audit_error),
                                    total_added=self.model.replay_buffer.total_added,
                                    learner_transitions=self.model.num_timesteps)
            summary = dict(schema=1, purpose=self.config['purpose'], algorithm=self.config['algorithm'],
                           training_seed=self.model.seed, status=self.segment_data['status'], stop_reason=reason,
                           transitions=self.model.num_timesteps, gradient_updates=self.model._n_updates,
                           current_lineage_transitions=self.model.num_timesteps,
                           lifetime_collected_lower_bound=self.model.num_timesteps+self.discarded_known,
                           unobserved_tail_budget_charge=self.unclean_allowance,
                           transition_budget_charged=self.model.num_timesteps+self.discarded_known+self.unclean_allowance,
                           replay=replay_audit,
                           cumulative_seconds=self.prior_seconds+elapsed, segment_seconds=elapsed,
                           full_cycle_transitions_per_second=(self.model.num_timesteps-self.initial_transitions)/elapsed,
                           full_cycle_updates_per_second=(self.model._n_updates-self.initial_updates)/elapsed,
                           timing=dict(initialization=self.initialization_seconds, collection=self.collection_seconds,
                                       training_host_cumulative=self.model.update_host_seconds,
                                       training_cuda_event_cumulative=self.model.update_device_seconds,
                                       training_host_segment=self.model.update_host_seconds-self.initial_train_host,
                                       training_cuda_segment=self.model.update_device_seconds-self.initial_train_cuda,
                                       collection_phases=self.collection_timing.record(),
                                       evaluation=self.eval_seconds, saving=self.save_seconds,
                                       warmup_finished_cumulative=self.warmup_finished_at),
                           peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                           cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.model.device) if self.model.device.type == 'cuda' else None,
                           cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.model.device) if self.model.device.type == 'cuda' else None,
                           best=json.loads((self.run/'best.json').read_text()) if (self.run/'best.json').exists() else None,
                           last=json.loads((self.run/'last.json').read_text()) if (self.run/'last.json').exists() else None,
                           validation_sha256=self.config['validation_sha256'],
                           session=self.session.status() if self.session else None,
                           final_test_set_used=False, bitwise_resume_guaranteed=False)
            diagnostics = getattr(self.model.replay_buffer, 'training_diagnostics', None)
            if diagnostics is not None:
                summary['training_diagnostics'] = diagnostics.record()
                summary['training_contract'] = self.config['training_contract']
                summary['seed_streams'] = self.model.training_seed_streams
                summary['training_resets'] = getattr(self.env.unwrapped, 'training_resets', None)
                summary['scientific_result'] = bool(self.config['protocol'].get('scientific_result', False) and complete)
                summary['counters'] = dict(collected=self.model.num_timesteps,
                    stored=self.model.replay_buffer.total_added, updates=self.model._n_updates,
                    physical_seconds=diagnostics.physical_seconds)
            atomic_json(self.segment/'summary.json', summary); atomic_json(self.run/'summary.json', summary)
            self.model.get_env().close()
        if error:
            raise error
        return summary


def run_training(config, *, output=None, max_transitions, max_seconds, session=None, resume=None,
                 signals=None, evaluate_enabled=None):
    if config['purpose'] == 'experiment' and session is None:
        raise ValueError('main training/resume requires the persistent budget session')
    from cartpole.experiments.cloud_checks import check_stack, hardware_report
    check_stack(config)
    run = Path(resume) if resume else new_output(output, config['id']+f'_seed{config["settings"]["seed"]}')
    if not resume:
        atomic_json(run/'config.json', config); snapshot(run)
        (run/'validation_states.json').write_bytes((ROOT/'tests/fixtures/validation_states.json').read_bytes())
    hardware = hardware_report(config['settings']['device'], run)
    if not resume:
        atomic_json(run/'hardware.json', hardware)
    from cartpole.experiments.cloud_io import file_lock
    with file_lock(run/'.run.lock'):
        try:
            runner = Runner(config, run, max_transitions=max_transitions, max_seconds=max_seconds,
                            session=session, resumed=bool(resume), signals=signals, evaluate_enabled=evaluate_enabled)
            atomic_json(runner.segment/'hardware.json', hardware)
            with runner.signals.installed():
                summary = runner.execute()
        finally:
            original_error = sys.exc_info()[1]
            try:
                atomic_json(run/'manifest.json', manifest(run))
            except Exception as manifest_error:
                if original_error is None:
                    raise
                original_error.add_note(f'Root manifest could not be published: {manifest_error}; checkpoint manifests remain available')
    return run, summary
