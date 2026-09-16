"""Versioned JSON logs: N transitions link N+1 independently indexed states.

Zero-duration transitions keep two state records with the same timestamp.
No dynamics or Drake imports are needed to read/replay a saved log.
"""
import json
import math
from dataclasses import asdict
from pathlib import Path

from cartpole.common import State
from cartpole.common.metrics import UprightThresholds, upright_angle_error, evaluate_hold


SCHEMA = 'cartpole.episode.v1'
STATE_FIELDS = ('cart_position', 'pole_angle', 'cart_velocity', 'pole_angular_velocity',
                'cart_acceleration')


def state_record(state, timestamp, index):
    return dict(index=index, time=float(timestamp), error=int(state.error),
                **{key: float(getattr(state, key)) for key in STATE_FIELDS})


def record_state(row):
    return State(**{key: row[key] for key in (*STATE_FIELDS, 'error')})


def new_log(metadata, state, timestamp=0.):
    return {'schema_version': SCHEMA, 'metadata': metadata,
            'states': [state_record(state, timestamp, 0)], 'transitions': []}


def append_transition(log, transition):
    index = len(log['transitions'])
    log['transitions'].append({
        'index': index, 'from_state': index, 'to_state': index+1,
        'time_start': log['states'][-1]['time'], 'time_end': transition.timestamp,
        'dt_requested': transition.dt_requested, 'dt_actual': transition.dt_actual,
        'u_requested': transition.u_requested, 'u_commanded': transition.u_commanded,
        'applied_acceleration': transition.applied_acceleration,
        'action_clipped': transition.action_clipped,
        'simulator_error': int(transition.simulator_error),
        'terminated': transition.terminated,
        'stop_reasons': [reason.value for reason in transition.stop_reasons],
        'boundary_events': [dict(reason=event.reason.value, side=event.side, time=event.time)
                            for event in transition.boundary_events],
    })
    if transition.safety_decision is not None:
        log['transitions'][-1]['safety_filter'] = asdict(transition.safety_decision)
    log['states'].append(state_record(transition.state, transition.timestamp, index+1))
    if log['metadata'].get('measure_cart_interval'):
        from cartpole.experiments.safety_metrics import interval_evidence
        log['transitions'][-1]['cart_interval'] = interval_evidence(
            log['states'][-2], log['states'][-1], log['transitions'][-1])


def finish_log(log, reason, *, exception=None, observed_state=None, observed_time=None):
    last = log['transitions'][-1] if log['transitions'] else None
    completion = dict(reason=reason, terminated=reason == 'terminal',
                      truncated=reason == 'horizon', time=log['states'][-1]['time'],
                      transition_count=len(log['transitions']),
                      stop_reasons=last['stop_reasons'] if last else [])
    if exception is not None:
        # An integrator may have progressed before raising. Keep that evidence
        # separate from successful transitions, and let the caller re-raise.
        completion['exception'] = {'type': type(exception).__name__, 'message': str(exception)}
        completion['observed_after_exception'] = state_record(observed_state, observed_time, None)
    log['completion'] = completion
    thresholds = UprightThresholds(**log['metadata']['upright_thresholds'])
    states, transitions = log['states'], log['transitions']
    log['metrics'] = {
        'duration': states[-1]['time'] - states[0]['time'],
        'max_abs_position': max(abs(row['cart_position']) for row in states),
        'max_abs_velocity': max(abs(row['cart_velocity']) for row in states),
        'max_abs_requested_acceleration': max((abs(row['u_requested']) for row in transitions), default=0.),
        'max_abs_commanded_acceleration': max((abs(row['u_commanded']) for row in transitions
                                               if row['u_commanded'] is not None), default=0.),
        'max_abs_applied_acceleration': max((abs(row['applied_acceleration']) for row in transitions
                                           if row['applied_acceleration'] is not None), default=0.),
        'final_angle_error': upright_angle_error(states[-1]['pole_angle']),
        'all_samples_in_upright_region': all(thresholds.contains(record_state(row)) for row in states),
        'clipped_transition_count': sum(row['action_clipped'] for row in transitions),
    }
    if 'required_hold_duration' in log['metadata']:
        log['evaluation'] = evaluate_hold(log, log['metadata']['required_hold_duration'])


def validate_log(log):
    """Reject inconsistent timing/indexing/action histories before playback."""
    def require(condition, message):
        if not condition:
            raise ValueError('invalid episode log: ' + message)

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def close(a, b):
        return math.isclose(a, b, rel_tol=0., abs_tol=1e-12)

    require(log.get('schema_version') == SCHEMA, 'unsupported schema')
    states, transitions = log['states'], log['transitions']
    require(len(states) == len(transitions)+1, 'expected N+1 states')
    for i, row in enumerate(states):
        require(row['index'] == i, 'state index')
        require(all(finite(row[key]) for key in ('time', *STATE_FIELDS)), 'non-finite state')
        require(row['time'] >= 0, 'negative timestamp')
    limit = log['metadata']['config']['max_acceleration']
    for i, tr in enumerate(transitions):
        if log['metadata'].get('experiment') == 'classical_swing_up':
            require('control_phase' in tr, 'missing classical control phase')
        if 'control_phase' in tr:
            require(tr['control_phase'] in ('swing_up', 'lqr'), 'unknown control phase')
        before, after = states[i:i+2]
        require((tr['index'], tr['from_state'], tr['to_state']) == (i, i, i+1), 'transition indices')
        require(all(finite(tr[key]) for key in ('time_start', 'time_end', 'dt_requested', 'dt_actual',
                                              'u_requested')), 'non-finite transition')
        require(close(tr['time_start'], before['time']) and close(tr['time_end'], after['time']), 'state/transition time')
        require(0 <= tr['dt_actual'] <= tr['dt_requested']+1e-12, 'invalid actual duration')
        require(close(after['time']-before['time'], tr['dt_actual']), 'elapsed time mismatch')
        require(tr['simulator_error'] == after['error'], 'native error mismatch')
        expected_command = max(-limit, min(tr['u_requested'], limit))
        safety = tr.get('safety_filter')
        if safety is None:
            require(tr['u_commanded'] == expected_command, 'limited command mismatch')
        else:
            require('safety_filter' in log['metadata'], 'missing filter configuration')
            require(safety['u_proposed'] == tr['u_requested'], 'filter proposed command mismatch')
            require(safety['u_filtered'] == tr['u_commanded'], 'filter command mismatch')
            if safety['feasible']:
                require(finite(tr['u_commanded']) and abs(tr['u_commanded']) <= limit, 'invalid filtered command')
                require(safety['intervened'] == (tr['u_requested'] != tr['u_commanded']), 'intervention flag')
            else:
                require(tr['u_commanded'] is None and tr['dt_actual'] == 0 and
                        tr['stop_reasons'] == ['safety_filter_refusal'] and bool(safety['reason']), 'invalid refusal')
        expected_clipped = tr['u_requested'] != expected_command if safety is None or safety['feasible'] else False
        require(tr['action_clipped'] == expected_clipped, 'clipping flag')
        if tr['dt_actual'] == 0:
            require(tr['applied_acceleration'] is None, 'zero step has no applied acceleration')
            require(all(before[key] == after[key] for key in STATE_FIELDS), 'zero step moved physical state')
        else:
            require(finite(tr['applied_acceleration']), 'missing applied acceleration')
            require(tr['applied_acceleration'] == after['cart_acceleration'], 'acceleration telemetry mismatch')
            if safety is not None:
                require(tr['applied_acceleration'] == tr['u_commanded'], 'filtered/applied command mismatch')
        require(tr['terminated'] == bool(tr['stop_reasons']), 'terminal flag mismatch')
        require(not tr['terminated'] or i == len(transitions)-1, 'post-terminal transition')
        for event in tr['boundary_events']:
            require(event['side'] in (-1, 1) and 0 <= event['time'] <= tr['dt_actual']+1e-12, 'boundary event time/side')
    end = log['completion']
    require(end['reason'] in ('horizon', 'terminal', 'exception'), 'completion reason')
    require(end['transition_count'] == len(transitions) and close(end['time'], states[-1]['time']), 'completion time/count')
    require(end['terminated'] == (end['reason'] == 'terminal') and end['truncated'] == (end['reason'] == 'horizon'), 'completion flags')
    if end['reason'] == 'horizon':
        require(not transitions or not transitions[-1]['terminated'], 'horizon masks terminal')
        require(close(end['time'], log['metadata']['horizon']), 'horizon not reached')
    elif end['reason'] == 'terminal':
        require(bool(transitions) and transitions[-1]['terminated'], 'terminal without event')
        require(end['stop_reasons'] == transitions[-1]['stop_reasons'], 'completion reasons')


def save_log(path, log):
    validate_log(log)
    payload = json.dumps(log, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def load_log(path):
    with Path(path).open(encoding='utf-8') as stream:
        log = json.load(stream)
    validate_log(log)
    return log
