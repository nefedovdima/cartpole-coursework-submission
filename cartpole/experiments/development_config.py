"""Finite S12 development registry; never an override of the baseline matrix."""
import copy

VERSION = 'development-stage1-v1'
VARIANTS = {
    'ddpg_on_warmup5000': ('DDPG', 'on', 'learning_starts', 5000),
    'ddpg_on_lr1e4': ('DDPG', 'on', 'learning_rate', 1e-4),
    'sac_off_warmup5000': ('SAC', 'off', 'learning_starts', 5000),
    'sac_off_lr1e4': ('SAC', 'off', 'learning_rate', 1e-4),
}
NAMES = tuple(VARIANTS)


def specification(name):
    from cartpole.experiments.method_config import specification as baseline
    if name not in VARIANTS: raise ValueError('unapproved development variant')
    algorithm, mode, factor, value = VARIANTS[name]
    base_id = algorithm.lower()+'_training_v2_'+mode
    cfg = copy.deepcopy(baseline(base_id))
    previous = cfg['settings'][factor]
    cfg['id'] = name
    cfg['settings'][factor] = value
    cfg['development'] = dict(version=VERSION, baseline_id=base_id,
        change=dict(parameter=factor, baseline=previous, value=value),
        stage=1, allowed_seeds=[0,1,2], initialization='fresh',
        expected_updates=100000-cfg['settings']['learning_starts'],
        confirmation_seeds=[3,4,5], confirmation_authorized=False)
    from cartpole.experiments.cloud_io import sha256
    cfg['development']['registry_sha256'] = sha256(__file__)
    return cfg


def selection_key(seed_reports):
    """All three validation20 seeds, ranked independently for each method/mode.

    Missing/failed seeds are not silently dropped: they make a candidate
    ineligible. Equal keys require baseline retention (no winner by seed luck).
    """
    if set(seed_reports) != {0,1,2}: raise ValueError('all three seeds are required')
    rows = [seed_reports[s] for s in range(3)]
    if any(not r['complete'] or not r['selection_eligible'] for r in rows):
        raise ValueError('incomplete/failed seed; candidate cannot be selected')
    v = [r['validation'] for r in rows]
    success = [r['success_rate'] for r in v]
    return (min(success), sum(success)/3,
            min(r['mean_hold_final'] for r in v), sum(r['mean_hold_final'] for r in v)/3,
            sum(r['mean_hold_longest'] for r in v)/3)
