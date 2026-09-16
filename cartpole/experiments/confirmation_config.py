"""Fixed Stage1 winners; new training seeds only, with no parameter search."""
VERSION = 'development-confirmation-v1'
WINNERS = {'confirm_ddpg_on_warmup5000': 'ddpg_on_warmup5000',
           'confirm_sac_off_warmup5000': 'sac_off_warmup5000'}
NAMES = tuple(WINNERS)
SEEDS = (3, 4, 5)


def specification(name):
    from cartpole.experiments.development_config import specification as stage1
    from cartpole.experiments.cloud_io import ROOT, sha256
    if name not in WINNERS:
        raise ValueError('only audited warmup5000 winners are permitted')
    source = WINNERS[name]
    cfg = stage1(source)
    cfg['id'] = name
    cfg['development'].update(stage='confirmation', allowed_seeds=list(SEEDS),
        confirmation_authorized=True)
    cfg['confirmation'] = dict(version=VERSION, source_variant=source,
        registry_sha256=sha256(__file__),
        selection_evidence_sha256=sha256(ROOT/'configs/confirmation/selection_evidence.json'),
        criterion='all3 best validation success>=.8; at least2 last>=.8; all jobs complete',
        initialization='fresh; never load Stage1/baseline weights or replay')
    return cfg


def validation_metrics(report):
    """Recompute validation20 aggregates; named3 never enter a gate or score."""
    import math
    if not report['complete'] or not report['selection_eligible']:
        raise ValueError('incomplete/selection-ineligible report')
    episodes=report['episodes']
    rows=[r for r in episodes if r['subset']=='validation']
    named=[r for r in episodes if r['subset']=='named']
    if len(episodes)!=23 or len(rows)!=20 or len(named)!=3 or len({r['id'] for r in episodes})!=23:
        raise ValueError('expected20validation and3named, no missing/duplicate IDs')
    import json
    from cartpole.experiments.cloud_io import ROOT
    cases=json.loads((ROOT/'tests/fixtures/validation_states.json').read_text())['cases']
    for subset,selected in (('validation',rows),('named',named)):
        if {r['id'] for r in selected}!={c['id'] for c in cases if c['subset']==subset}:
            raise ValueError('case IDs assigned to the wrong subset')
    if any(not isinstance(r['success_episode'],bool) for r in rows):
        raise ValueError('invalid validation success flag')
    result=dict(episodes=20,success_rate=sum(r['success_episode'] for r in rows)/20,
                mean_hold_final=math.fsum(r['hold_final'] for r in rows)/20,
                mean_hold_longest=math.fsum(r['hold_longest'] for r in rows)/20)
    for k,v in result.items():
        saved=report['validation'][k]
        if not isinstance(saved,(int,float)) or not math.isfinite(saved) or not math.isfinite(v) or abs(saved-v)>1e-9:
            raise ValueError('validation aggregate differs from subset=validation: '+k)
    return result

def outcome(reports):
    """All failures/seeds retained; missing evidence cannot count as confirmation."""
    if set(reports) != set(SEEDS):
        raise ValueError('confirmation requires seeds3/4/5')
    import math
    best = []; last = []
    for seed in SEEDS:
        row = reports[seed]
        if not row['accepted_train'] or not row['accepted_external']:
            return dict(confirmed=False, reason='incomplete_or_failed_seed', seeds=list(SEEDS))
        for label, values in (('best', best), ('last', last)):
            report = row[label]
            if not report['complete'] or not report['selection_eligible'] or report['validation']['episodes'] != 20:
                raise ValueError('incomplete validation20')
            rate=validation_metrics(report)['success_rate']
            if not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0<=rate<=1:
                raise ValueError('invalid validation success rate')
            values.append(rate)
    return dict(confirmed=min(best)>=.8 and sum(x>=.8 for x in last)>=2,
                best_success_rates=best, last_success_rates=last, seeds=list(SEEDS),
                criterion='each best>=.8 and at least2 last>=.8',
                final_test_used=False, automatic_stage2=False)
