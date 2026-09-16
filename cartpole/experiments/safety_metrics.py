"""Cart interval evidence, separate from the original sampled task success."""
from collections import Counter
import math


def interval_evidence(before, after, transition):
    x, v = before['cart_position'], before['cart_velocity']
    h, u = transition['dt_actual'], transition['applied_acceleration']
    values = [x, after['cart_position']]
    vertex = None
    dx = dv = 0.
    if h > 0 and u is not None:
        predicted = math.fsum((x, v*h, .5*u*h*h))
        values.append(predicted)
        dx = after['cart_position']-predicted
        dv = after['cart_velocity']-(v+u*h)
        if u != 0 and 0 < -v/u < h:
            vertex = -v/u
            values.append(math.fsum((x, v*vertex, .5*u*vertex*vertex)))
    maximum = max(map(abs, values))
    return dict(max_abs_position_interval=maximum, vertex_time=vertex,
                desired_excess=max(0., maximum-.24), exceeds_024=maximum > .24,
                exceeds_024_resolved=maximum > .24+1e-12,
                endpoint_position_residual=dx, endpoint_velocity_residual=dv)


def safety_metrics(log):
    intervals = [interval_evidence(a, b, tr) for a, b, tr in
                 zip(log['states'], log['states'][1:], log['transitions'])]
    decisions = [tr['safety_filter'] for tr in log['transitions'] if 'safety_filter' in tr]
    accepted = [d for d in decisions if d['feasible']]
    changes = [d['correction_abs'] for d in accepted if d['intervened']]
    refusals = Counter(d['reason'] for d in decisions if not d['feasible'])
    raw_max = max([abs(s['cart_position']) for s in log['states']] +
                  [r['max_abs_position_interval'] for r in intervals])
    return dict(filter_enabled='safety_filter' in log['metadata'],
                decisions=len(decisions), accepted_decisions=len(accepted),
                intervention_count=len(changes),
                intervention_fraction=len(changes)/len(accepted) if accepted else None,
                intervention_abs_sum=math.fsum(changes),
                intervention_abs_max=max(changes, default=0.),
                intervention_abs_mean=math.fsum(changes)/len(changes) if changes else 0.,
                refusal_count=sum(refusals.values()), refusal_reasons=dict(refusals),
                max_abs_position_interval=raw_max,
                desired_excess_max=max(0., raw_max-.24),
                exceeds_024=raw_max > .24, exceeds_024_resolved=raw_max > .24+1e-12,
                intervals_exceeding_024=sum(r['exceeds_024'] for r in intervals),
                position_limit_events=sum(e['reason']=='position_limit' for tr in log['transitions']
                                          for e in tr['boundary_events']),
                max_abs_endpoint_position_residual=max((abs(r['endpoint_position_residual']) for r in intervals), default=0.),
                max_abs_endpoint_velocity_residual=max((abs(r['endpoint_velocity_residual']) for r in intervals), default=0.),
                definitions={'position': 'max of actual Drake endpoints and analytical held-u quadratic including interior vertex',
                             'exceedance': 'strict >.24; separate resolved >.24+1e-12 m; neither alters execution',
                             'intervention_fraction': 'changed accepted filter decisions / all accepted filter decisions; refusals separate',
                             'hold': 'original consecutive recorded states; cart safety alone is not task success'})
