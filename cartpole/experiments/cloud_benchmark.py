"""Phase-separated, block-measured full-cycle planning; no learning changes."""
import math
import statistics


class CollectionTiming:
    block_size = 64

    def __init__(self):
        self.warmup = dict(transitions=0, seconds=0.)
        self.policy = dict(transitions=0, seconds=0.)
        self.blocks = []
        self.pending = dict(transitions=0, updates=0, collection=0., train_host=0., train_cuda=0.)
        self.previous_cuda = 0.

    def collect(self, *, warmup, seconds):
        phase = self.warmup if warmup else self.policy
        phase['transitions'] += 1; phase['seconds'] += seconds
        if not warmup:
            self.pending['transitions'] += 1; self.pending['collection'] += seconds

    def update(self, host_seconds):
        self.pending['updates'] += 1; self.pending['train_host'] += host_seconds
        return self.pending['updates'] >= self.block_size

    def close_block(self, cuda_seconds, *, partial=False, wall_seconds=None):
        if self.pending['updates']:
            self.pending['train_cuda'] = cuda_seconds-self.previous_cuda
            self.pending['pipeline_wall'] = wall_seconds
            self.pending['other_step_overhead'] = max(0., wall_seconds-self.pending['collection']
                -max(self.pending['train_host'],self.pending['train_cuda'])) if wall_seconds is not None else 0.
            self.blocks.append(dict(self.pending, partial=partial))
            self.pending = dict(transitions=0, updates=0, collection=0., train_host=0., train_cuda=0.)
        self.previous_cuda = cuda_seconds

    def record(self):
        return dict(warmup=self.warmup, policy=self.policy, blocks=self.blocks, block_size=self.block_size)


def measurement_quality(record):
    blocks = [b for b in record['blocks'] if not b['partial'] and b['updates'] == b['transitions'] == 64]
    rates = [(b.get('pipeline_wall') or b['collection']+max(b['train_host'], b['train_cuda']))/b['updates'] for b in blocks]
    mean = statistics.mean(rates) if rates else None
    cv = statistics.stdev(rates)/mean if len(rates) > 1 and mean > 0 else None
    enough = len(blocks) >= 4
    stable = enough and cv is not None and cv <= .25 and min(rates) > 0 and max(rates)/min(rates) <= 2.
    return dict(sufficient=enough, stable=stable, full_blocks=len(blocks), minimum_full_blocks=4,
                per_block_seconds_per_transition=rates, coefficient_of_variation=cv,
                projection_eligible=bool(stable),
                reason='usable_short_measurement' if stable else ('insufficient_post_warmup_blocks' if not enough else 'unstable_block_times'))


def project(total, *, starts, warmup_seconds_per_step, policy_seconds_per_step,
            train_seconds_per_update, initialization, constant_overhead, evaluation_seconds,
            evaluation_interval, save_seconds, recovery_interval, recovery_seconds, policy_overhead_per_step=0.):
    """Time-triggered saves also consume time: solve a conservative fixed point.

    Save counts are a union upper bound, not an assertion that simultaneous
    step/time/evaluation events create duplicate physical saves.
    """
    values = (warmup_seconds_per_step, policy_seconds_per_step, train_seconds_per_update,
              initialization, constant_overhead, evaluation_seconds, save_seconds, policy_overhead_per_step)
    if any(not math.isfinite(v) or v < 0 for v in values) or recovery_seconds <= save_seconds:
        raise ValueError('invalid timing or checkpoint cost exceeds its recovery period')
    updates = max(0, total-starts)
    eval_count = 1+math.ceil(total/evaluation_interval)
    components = dict(warmup_collection=min(total, starts)*warmup_seconds_per_step,
                      policy_collection=updates*policy_seconds_per_step,
                      training=updates*train_seconds_per_update,
                      policy_step_overhead=updates*policy_overhead_per_step,
                      initialization=initialization, constant_overhead=constant_overhead,
                      evaluation=eval_count*evaluation_seconds)
    base = sum(components.values()); estimate = base
    step_saves = total//recovery_interval
    for _ in range(1000):
        time_saves = math.ceil(estimate/recovery_seconds)
        save_count = step_saves+time_saves+eval_count+2
        updated = base+save_count*save_seconds
        if updated <= estimate:
            break
        estimate = updated
    else:
        raise ValueError('checkpoint timing projection did not converge')
    components['saving'] = save_count*save_seconds
    return dict(seconds=estimate, planning_seconds_with_25pct_headroom=estimate*1.25,
                components_seconds=components, updates=updates, evaluation_count=eval_count,
                recovery_step_upper_bound=step_saves, recovery_time_upper_bound=time_saves,
                save_count_upper_bound=save_count)


def forecast(summary, config, outer_seconds, eval_steps, saves):
    timing = summary['timing']; phases = timing['collection_phases']
    quality = measurement_quality(phases)
    full_eval = timing['evaluation']*23000/eval_steps if eval_steps else None
    projections = {}
    if quality['projection_eligible'] and summary['benchmark_complete'] and full_eval:
        w, p = phases['warmup'], phases['policy']
        train = max(timing['training_host_segment'], timing['training_cuda_segment'])
        step_overhead = sum(b.get('other_step_overhead',0.) for b in phases['blocks'])
        overhead = max(0., outer_seconds-sum(timing[k] for k in ('initialization','collection','evaluation','saving'))-train-step_overhead)
        for total in (100000, 300000):
            projections[str(total)] = project(total, starts=config['settings']['learning_starts'],
                warmup_seconds_per_step=w['seconds']/w['transitions'],
                policy_seconds_per_step=p['seconds']/p['transitions'],
                policy_overhead_per_step=step_overhead/summary['gradient_updates'],
                train_seconds_per_update=train/summary['gradient_updates'], initialization=timing['initialization'],
                constant_overhead=overhead, evaluation_seconds=full_eval,
                evaluation_interval=config['evaluation_interval'], save_seconds=timing['saving']/max(1,saves),
                recovery_interval=config['recovery_interval'], recovery_seconds=config['recovery_seconds'])
    return projections, quality, full_eval
