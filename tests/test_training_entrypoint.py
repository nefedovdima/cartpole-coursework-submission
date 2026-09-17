"""Fresh-training wrapper contract only; never execute a learner."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import train

class TrainingEntrypoint(unittest.TestCase):
    def test_main_scientific_settings(self):
        cfg=train.configuration('sac_training_v2_on',2)
        self.assertEqual(cfg['settings']['seed'],2)
        self.assertEqual(cfg['settings']['buffer_size'],100000)
        self.assertEqual(cfg['settings']['learning_starts'],128)
        self.assertEqual(cfg['settings']['batch_size'],64)
    def test_warmup_is_separate_config(self):
        cfg=train.configuration('ddpg_on_warmup5000',0)
        self.assertEqual(cfg['settings']['learning_starts'],5000)
        self.assertEqual(cfg['settings']['buffer_size'],100000)
    def test_run_refuses_without_guard(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True),patch('sys.argv',['train','run','--config','sac_training_v2_on','--seed','0','--session',d+'/session.json','--output',d+'/run','--max-seconds','60']):
            with self.assertRaisesRegex(ValueError,'cloud_guard'):train.main()
            self.assertEqual(list(Path(d).iterdir()),[])
if __name__=='__main__':unittest.main()
