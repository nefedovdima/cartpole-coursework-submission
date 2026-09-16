import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from cartpole.experiments.inference_compatibility import compare_actions, observations_for_inference
from cartpole.experiments.parallel_evaluation import load_policy
from cartpole.experiments.cloud_config import load_config
from cartpole.experiments.cloud_io import atomic_json, manifest
from cartpole.experiments.cloud_learning import policy_fingerprint


class InferenceCompatibilityTests(unittest.TestCase):
    def compare(self,a,b,algorithm='SAC',saved='cuda:0',loaded='cpu'):
        return compare_actions(a,b,algorithm=algorithm,count=len(a),saved_device=saved,loaded_device=loaded)

    def test_reported_gpu_to_cpu_rounding_is_accepted(self):
        actual=np.full((23,1),.5,dtype=np.float32);saved=actual.copy()
        actual[15,0]=0.006495475769042969;saved[15,0]=0.0064955949783325195
        with self.assertRaises(AssertionError):np.testing.assert_allclose(actual,saved,rtol=1e-6,atol=1e-7)
        for algorithm in ('SAC','TQC','DDPG'):
            r=self.compare(actual,saved,algorithm)
            self.assertEqual(r['max_abs_difference'],2**-23)
            self.assertEqual(r['atol'],2e-6);self.assertEqual(r['rtol'],0)
            self.compare(saved,actual,algorithm,saved='cpu',loaded='cuda:1')

    def test_material_difference_rejected_also_near_saturation(self):
        for a,b in ((0.,1e-4),(.9,.900003),(0.,2.000001e-6)):
            with self.assertRaisesRegex(ValueError,'inference incompatible'):self.compare([[a]],[[b]])

    def test_same_device_requires_exact_values(self):
        self.compare([[.1]],[[.1]],saved='cpu')
        with self.assertRaises(ValueError):self.compare([[.1]],[[.10000001]],saved='cpu')

    def test_dqn_requires_exact_integer_decision(self):
        r=self.compare([0,4,8],[0,4,8],'DQN');self.assertTrue(r['exact_required'])
        for a,b in (([4],[5]),([4.0],[4.0]),([4],[4.00000001]),([9],[9]),([-1],[-1]),([True],[True])):
            with self.assertRaises(ValueError):self.compare(a,b,'DQN')

    def test_nonfinite_actions_rejected_on_both_sides(self):
        for v in (np.nan,np.inf,-np.inf):
            for a,b in ((v,0.),(0.,v),(v,v)):
                with self.assertRaises(ValueError):self.compare([[a]],[[b]])

    def test_shape_type_and_domain_are_not_broadcast_or_clipped(self):
        for a,b in (([[0.]], [0.]),([[0.]],[[0.],[0.]]),([['0']],[[0.]]),([[1.000001]],[[1.000001]])):
            with self.assertRaises(ValueError):self.compare(a,b)

    def test_observations_validated_before_model(self):
        self.assertEqual(observations_for_inference([[0.]*5]).dtype,np.float32)
        for x in ([0.]*5,[],[[0.]*4],[['0']*5],[[np.nan]*5],[[np.inf]*5],[[1e100]*5]):
            with self.assertRaises(ValueError):observations_for_inference(x)


class LoaderIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.config=load_config('sac_mechanics_v2',0,'cpu')
        self.policy=torch.nn.Linear(5,1);self.policy.set_training_mode=lambda _:None
        self.policy.scale_action=lambda x:x;self.policy.unscale_action=lambda x:x
        from gymnasium.spaces import Box
        self.model=SimpleNamespace(device=torch.device('cpu'),policy=self.policy,num_timesteps=512,_n_updates=384,
            seed=0,training_contract=self.config['training_contract'],action_space=Box(-1,1,(1,),np.float32),
            predict=lambda x,deterministic:(np.zeros((len(x),1),np.float32),None))
        self.data=dict(complete=True,config=self.config,implementation={'test':'identity'},transitions=512,
            updates=384,seed=0,device='cuda:0',policy_sha256=policy_fingerprint(self.model))
        self.probes=dict(observations=[[0.]*5]*23,actions=[[0.]]*23,env_actions=[[0.]]*23,
                         training_seed=0,transitions=512,updates=384,device='cuda:0')
        for n in ('model.zip','replay.pkl','runtime.pt'):(self.root/n).write_bytes(b'fixture')
        self.save()
        self.loader=MagicMock();self.loader.load.return_value=self.model
        for target,value in (('cloud_learning.algorithm_class',self.loader),('cloud_learning.implementation_hashes',{'test':'identity'}),
                             ('cloud_learning.configure_device',torch.device('cpu')),('cloud_checks.check_stack',{})):
            p=patch('cartpole.experiments.'+target,return_value=value);p.start();self.addCleanup(p.stop)

    def save(self):
        atomic_json(self.root/'metadata.json',self.data);atomic_json(self.root/'probes.json',self.probes)
        atomic_json(self.root/'manifest.json',manifest(self.root))

    def test_identity_and_numeric_receipts_are_separate(self):
        model,data=load_policy(self.root,'cpu')
        self.assertEqual(data['policy_sha256'],policy_fingerprint(model))
        self.assertEqual(data['inference_compatibility']['comparisons'],23)
        self.assertTrue(data['inference_compatibility']['environment_actions']['finite'])

    def test_corrupt_artifact_rejected_before_deserialization(self):
        for name in ('model.zip','metadata.json','probes.json','replay.pkl','runtime.pt'):
            original=(self.root/name).read_bytes();(self.root/name).write_bytes(original+b'broken')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):load_policy(self.root,'cpu')
            (self.root/name).write_bytes(original)
        self.loader.load.assert_not_called()

    def test_missing_manifest_member_cannot_skip_identity(self):
        entries=manifest(self.root);entries.pop('model.zip');atomic_json(self.root/'manifest.json',entries)
        with self.assertRaisesRegex(ValueError,'incomplete checkpoint manifest'):load_policy(self.root,'cpu')
        self.loader.load.assert_not_called()

    def test_changed_loaded_weights_rejected_even_with_identical_actions(self):
        with torch.no_grad():self.policy.weight.add_(.1)
        with self.assertRaisesRegex(ValueError,'fingerprint'):load_policy(self.root,'cpu')

    def test_nonfinite_weights_rejected(self):
        with torch.no_grad():self.policy.weight[0,0]=torch.nan
        with self.assertRaises(FloatingPointError):load_policy(self.root,'cpu')

    def test_probe_counters_rejected_even_with_rehashed_artifact(self):
        self.probes['transitions']=511;self.save()
        with self.assertRaisesRegex(ValueError,'probe checkpoint identity'):load_policy(self.root,'cpu')

    def test_mapping_mismatch_rejected_separately(self):
        self.probes['env_actions'][0]=[.01];self.save()
        with self.assertRaisesRegex(ValueError,'inference incompatible'):load_policy(self.root,'cpu')


if __name__=='__main__':unittest.main()
