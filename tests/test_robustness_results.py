"""Submission normalized-data integration; synthetic failure tests live in analysis/reporting."""
import hashlib
import json
import unittest
from scripts import reproduce_robustness as r

class RobustnessResults(unittest.TestCase):
    def test_six_canonical_tables_reproduced_bytewise(self):
        result=r.check()
        self.assertEqual((result['episodes'],result['pairs'],result['tables']),(1960,980,6))
        self.assertEqual(result['raw_recomputation'],'not_run')
    def test_historical_media_duplicates_identical(self):
        manifest=json.loads((r.ROOT/'media/VIDEO_MANIFEST.json').read_text())
        self.assertEqual(len(manifest['videos']),6)
        for p in (r.ROOT/'media').rglob('*'):
            if p.is_file():self.assertEqual(p.read_bytes(),(r.ROOT/'docs/report/media'/p.relative_to(r.ROOT/'media')).read_bytes())
    def test_reporting_payloads(self):
        base=r.ROOT/'docs/assets/structured_robustness'
        manifest=json.loads((base/'MANIFEST.json').read_text())
        self.assertEqual(len(manifest['files_sha256']),18)
        for name,digest in manifest['files_sha256'].items():
            self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(),digest)
if __name__=='__main__':unittest.main()
