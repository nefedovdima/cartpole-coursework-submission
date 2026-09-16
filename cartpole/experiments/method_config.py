"""Explicit v2: mechanics has no automatic 23-episode assessment.

Main scenario remains a proposal, gated externally by an approved budget.
"""
import copy
from cartpole.rl.methods import ALGORITHMS, contract_id

NAMES = tuple(a.lower()+'_mechanics_v2' for a in ALGORITHMS) + tuple(
    a.lower()+'_training_v2_'+m for a in ALGORITHMS for m in ('on', 'off'))


def specification(name):
    if name not in NAMES:
        raise ValueError('unknown method configuration')
    from cartpole.experiments.cloud_config import specification as original
    from cartpole.rl.training_contract import identity
    algorithm = name.split('_')[0].upper()
    mechanics = '_mechanics_' in name
    mode = 'on' if mechanics else name.rsplit('_', 1)[1]
    cfg = copy.deepcopy(original('sac64_reference'))
    contract = identity(contract_id(algorithm, mode))
    cfg.update(id=name, algorithm=algorithm, preparation_version='local-ready-v2',
               training_contract=contract, environment=contract['specification'],
               evaluation_interval=10000, recovery_interval=256 if mechanics else 5000,
               recovery_seconds=60., telemetry_interval=128, experience_block_size=256,
               save_reserve_seconds=30.)
    s = cfg['settings']
    s.update(buffer_size=8192 if mechanics else 100000, learning_starts=128, batch_size=64)
    if algorithm == 'TQC':
        s['policy_kwargs']['n_quantiles'] = 25
        s['top_quantiles_to_drop_per_net'] = 2
    if algorithm in ('DDPG', 'DQN'):
        for key in ('ent_coef', 'target_entropy', 'use_sde', 'sde_sample_freq', 'use_sde_at_warmup'):
            s.pop(key, None)
        s['policy_kwargs'] = dict(net_arch=[64, 64], activation_fn='ReLU', optimizer_class='Adam',
                                  optimizer_kwargs={'eps': 1e-8}, normalize_images=True)
    if algorithm == 'DDPG':
        s.pop('stats_window_size')
        s.pop('target_update_interval')
        s['action_noise'] = 'independent_Gaussian_sigma_0.1'
        s['policy_kwargs'].update(n_critics=1, share_features_extractor=False)
    if algorithm == 'DQN':
        s.pop('action_noise')
        s.update(tau=1., target_update_interval=1000, exploration_fraction=.2,
                 exploration_initial_eps=1., exploration_final_eps=.05, max_grad_norm=10.)
    cfg['protocol'].update(version='local-ready-v2', profile_cuda=False,
        automatic_evaluation=not mechanics, evaluation=('none; mechanics only; no best selection' if mechanics else
            'complete fixed 20 validation + 3 named; deterministic; same training filter mode'),
        pilot_max_transitions=512 if mechanics else 100000,
        pilot_max_seconds=300. if mechanics else 7200., scientific_result=not mechanics,
        planned_transitions=512 if mechanics else 100000,
        seed_scheme=contract['specification']['seed_scheme'],
        main_budget_status='proposal requires benchmark and explicit expense approval' if not mechanics else 'mechanics',
        updates_per_transition='one after learning_starts; resume completes owed update once')
    return cfg
