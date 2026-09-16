"""Bounded request witnesses and O(1) aggregates, updated after replay.add."""
from collections import Counter, deque
import copy
import math

class Moments:
    def __init__(self):
        self.n = 0; self.mean = self.m2 = 0.

    def add(self, value):
        if value is None:
            return
        self.n += 1
        delta = value-self.mean
        self.mean += delta/self.n
        self.m2 += delta*(value-self.mean)

    def record(self):
        return dict(count=self.n, mean=self.mean if self.n else None,
                    variance=self.m2/self.n if self.n else None)


class TrainingDiagnostics:
    def __init__(self):
        self.counts = Counter(); self.reasons = Counter(); self.refusals = Counter()
        self.physical_seconds = self.correction_sum = self.correction_sq = 0.
        self.correction_max = self.correction_integral = self.correction_sq_integral = 0.
        self.maximum_x = self.hold_longest = self.return_sum = 0.
        self.physical_episode_seconds = 0.
        self.first = []; self.last = deque(maxlen=16); self.events = deque(maxlen=16)
        self.commands = {key: Moments() for key in ('requested', 'filtered', 'applied')}

    def observe(self, info, buffer_action, reward, sequence):
        self.counts['stored'] += 1
        dt = info['dt_actual']; self.physical_seconds += dt; self.return_sum += float(reward)
        self.physical_episode_seconds = info['time']
        decision = info.get('safety_filter')
        self.counts['working_clipping'] += int(abs(info['u_requested']) > 4.)
        if decision:
            self.counts['decisions'] += 1
            if decision['feasible']:
                self.counts['accepted'] += 1
                changed = decision['intervened']; d = decision['correction_abs']
                self.counts['interventions'] += int(changed)
                self.correction_sum += d; self.correction_sq += d*d
                self.correction_max = max(self.correction_max, d)
                self.correction_integral += d*dt; self.correction_sq_integral += d*d*dt
            else:
                self.refusals[decision['reason']] += 1; self.counts['refusals'] += 1
        evidence = info['cart_interval']
        self.maximum_x = max(self.maximum_x, evidence['max_abs_position_interval'])
        self.counts['intervals_exceeding_024'] += int(evidence['exceeds_024'])
        self.counts['intervals_exceeding_024_resolved'] += int(evidence['exceeds_024_resolved'])
        self.counts['position_limit_events'] += sum(e['reason'] == 'position_limit' for e in info['boundary_events'])
        self.counts['zero_duration'] += int(dt == 0)
        self.hold_longest = max(self.hold_longest, info['hold_metrics']['hold_longest'])
        end = info['completion_reason'] != 'running'
        if end:
            self.counts['episodes'] += 1
            self.counts['successes'] += int(info['hold_metrics']['success_episode'])
            self.counts['timeouts'] += int(info['completion_reason'] == 'horizon')
            self.counts['physical_failures'] += int(info['completion_reason'] == 'terminal'
                                                  and 'safety_filter_refusal' not in info['stop_reasons'])
            for reason in info['stop_reasons'] or [info['completion_reason']]:
                self.reasons[reason] += 1
        for key, value in zip(self.commands, (info['u_requested'], info['u_commanded'], info['applied_acceleration'])):
            self.commands[key].add(value)
        row = dict(sequence=sequence, buffer_action=float(buffer_action),
                   env_action=info['action_requested'], u_requested=info['u_requested'],
                   u_filtered=decision['u_filtered'] if decision else None,
                   u_commanded=info['u_commanded'], applied=info['applied_acceleration'],
                   dt_actual=dt, time=info['time'], reward=float(reward),
                   simulator_error=info['simulator_error'], stop_reasons=info['stop_reasons'],
                   completion_reason=info['completion_reason'], hold=info['hold_metrics'],
                   timeout=bool(info.get('TimeLimit.truncated', False)),
                   before=info['training_before_state'], after=info['state'], cart_interval=evidence)
        if 'request_index' in info:
            row.update(buffer_action=int(buffer_action), request_index=info['request_index'])
        if len(self.first) < 16:
            self.first.append(copy.deepcopy(row))
        self.last.append(copy.deepcopy(row))
        if end or decision and decision['intervened']:
            self.events.append(copy.deepcopy(row))

    def record(self):
        n = self.counts['accepted']; changes = self.counts['interventions']
        return dict(counts=dict(self.counts), physical_seconds=self.physical_seconds,
                    return_sum=self.return_sum, completion_reasons=dict(self.reasons),
                    refusal_reasons=dict(self.refusals), max_abs_position_interval=self.maximum_x,
                    max_contiguous_hold=self.hold_longest,
                    intervention_fraction=changes/n if n else None,
                    intervention_abs_mean=self.correction_sum/changes if changes else 0.,
                    correction_abs_mean_all_accepted=self.correction_sum/n if n else None,
                    intervention_abs_max=self.correction_max,
                    correction_rms=math.sqrt(self.correction_sq/n) if n else None,
                    correction_abs_integral=self.correction_integral,
                    correction_sq_integral=self.correction_sq_integral,
                    command_moments={k: v.record() for k, v in self.commands.items()},
                    first=self.first, last=list(self.last), recent_events=list(self.events),
                    definitions='success/hold sampled, no summed disjoint holds; interval x includes held-u vertex; '
                                'bounded witnesses 16 first +16 last +16 events; aggregates cover all stored transitions')
