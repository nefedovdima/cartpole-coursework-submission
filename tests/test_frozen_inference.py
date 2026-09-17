"""Capsule identities and explicit command routing; no model/Drake execution."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from scripts import frozen_inference as f
from scripts import evaluate_capsule as e

class InferenceCapsules(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        shutil.copytree(f.ROOT/'configs/robustness',self.root/'configs/robustness')
        (self.root/'models').mkdir();shutil.copyfile(f.ROOT/'models/INDEX.json',self.root/'models/INDEX.json')
        shutil.copytree(f.ROOT/'models/SAC_seed0',self.root/'models/SAC_seed0')
    def test_all_six_real_capsule_identities(self):
        for a in ('SAC','TQC'):
            for s in range(3):f.verify_capsule(f'{a}_seed{s}')
    def test_corrupt_model_rejected_before_load(self):
        p=self.root/'models/SAC_seed0/model.zip';p.write_bytes(p.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'SHA'):f.verify_capsule('SAC_seed0',self.root)
    def test_missing_payload_rejected(self):
        (self.root/'models/SAC_seed0/probes.json').unlink()
        with self.assertRaises(ValueError):f.verify_capsule('SAC_seed0',self.root)
    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):f.verify_capsule('SAC_seed99',self.root)
    def test_index_cannot_rebind_weights(self):
        p=self.root/'models/INDEX.json';obj=json.loads(p.read_text());obj['models'][0]['files']['model.zip']='0'*64;p.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):f.verify_capsule('SAC_seed0',self.root)
    def test_unexpected_runtime_rejected(self):
        (self.root/'models/SAC_seed0/runtime.pt').touch()
        with self.assertRaises(ValueError):f.verify_capsule('SAC_seed0',self.root)
    def test_plan_never_loads_or_simulates(self):
        with patch.object(e,'load_capsule',side_effect=AssertionError('must not load')):
            p=e.plan('SAC_seed0','on')
            self.assertEqual((p['episodes'],p['validation'],p['named']),(23,20,3))
    def test_existing_output_rejected_without_loading(self):
        with patch.object(e,'load_capsule',side_effect=AssertionError('must not load')):
            with self.assertRaises(FileExistsError):e.execute('SAC_seed0','on','cpu',self.root)
    def test_execution_adapter_mock_only(self):
        with patch.object(e,'load_capsule',return_value=(object(),{'policy_sha256':'test'},{})),patch('cartpole.experiments.cloud_evaluation.evaluate',return_value={'mock':True}) as run:
            self.assertEqual(e.execute('SAC_seed0','off','cpu',self.root/'new'),{'mock':True})
            self.assertEqual(run.call_args.kwargs['evaluation_contract']['id'],'cartpole-request-v2-SAC-off')
            self.assertEqual(len(run.call_args.args[1]),23)
if __name__=='__main__':unittest.main()
