"""Submission checks: no Drake rollout, checkpoint loading or learner updates."""
import copy
import json
from pathlib import Path
import unittest
from scripts.reproduce_results import compute, rows, selected, score, gate
from scripts.verify_submission import verify
ROOT=Path(__file__).resolve().parents[1]

class SubmissionTests(unittest.TestCase):
    def test_manifest_links_syntax_and_artifact_limits(self):
        self.assertGreater(verify()['import_hashes'],100)
    def test_reproduce_results(self):
        self.assertEqual(compute(),json.loads((ROOT/'results/tables/reproduced.json').read_text()))
    def test_named_cannot_enter_selection(self):
        values=selected(rows('main'),'evaluate_train_DDPG_on_seed')
        values[0]=dict(values[0],subset='named',episodes='3')
        with self.assertRaises(ValueError):score(values)
    def test_named_cannot_enter_confirmation(self):
        data=rows('confirmation');p='evaluate_train_confirm_ddpg_on_warmup5000_seed'
        best,last=selected(data,p),selected(data,p,'last')
        best[0]=dict(best[0],subset='named',episodes='3')
        with self.assertRaises(ValueError):gate(best,last)
    def test_exact_gate_not_one_lucky_seed(self):
        base=dict(subset='validation',episodes='20')
        best=[dict(base,seed=str(s),successes='16') for s in (3,4,5)]
        last=copy.deepcopy(best);last[2]['successes']='0'
        self.assertTrue(gate(best,last))
        best[0]['successes']='15';self.assertFalse(gate(best,last))
    def test_scientific_sources_unchanged(self):
        from cartpole.rl.training_contract import identity
        imported={r['path']:r['sha256'] for r in json.loads((ROOT/'provenance/imports.json').read_text())}
        for p,digest in identity()['sources'].items():self.assertEqual(digest,imported[p])
    def test_real_stack_identity_without_rollout(self):
        from cartpole.experiments.cloud_checks import check_stack
        from cartpole.experiments.cloud_config import load_config
        self.assertIn("torch",check_stack(load_config("sac_training_v2_on",0,"cpu")))
    def test_validation_and_named_roster(self):
        from cartpole.experiments.cloud_evaluation import validation_cases
        cases=validation_cases()
        self.assertEqual(sum(c['subset']=='validation' for c in cases),20)
        self.assertEqual(sum(c['subset']=='named' for c in cases),3)
    def test_prepared_robustness_is_frozen_and_not_final(self):
        from cartpole.experiments.robustness_protocol import generate_states
        generated=generate_states();saved=json.loads((ROOT/'configs/robustness/states.json').read_text())
        self.assertEqual(generated,saved);self.assertFalse(saved['final'])
        self.assertEqual(len(saved['cases']),140)
    def test_imports_do_not_require_hardware(self):
        from cartpole.control import BalanceLQRControl
        from cartpole.simulator.pydrake.simulator import CartPoleSimulator
        from cartpole.rl.env import CartPoleEnv
        from cartpole.experiments.parallel_evaluation import load_policy
        self.assertTrue(all(callable(v) for v in (BalanceLQRControl,CartPoleSimulator,CartPoleEnv,load_policy)))
if __name__=='__main__':unittest.main()
