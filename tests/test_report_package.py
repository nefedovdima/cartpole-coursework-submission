"""PDF edition and publication-link integrity, without TeX or scientific runtime."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('report_inputs',ROOT/'docs/report/scripts/source_inputs.py')
inputs=importlib.util.module_from_spec(spec);spec.loader.exec_module(inputs)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

class ReportPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'docs/report';self.root.mkdir(parents=True)
        (self.root/'sections').mkdir()
        for rel,text in {'report.pdf':'new PDF bytes','report.tex':'source','references.tex':'references',
                         'appendix_links.tex':(ROOT/'docs/report/appendix_links.tex').read_text(),
                         'sections/intro.tex':r'\begin{center}\small\url{\SubmissionRepositoryURL}\end{center}','data.csv':'1,2\n'}.items():
            (self.root/rel).write_text(text)
        files={'data.csv':dict(sha256=sha(self.root/'data.csv'),bytes=4)};derived={}
        for n in range(7):
            rel=f'document{n}.md';(self.root/rel).write_text('portable document')
            files[rel]=dict(sha256='1'*64,bytes=10)
            derived[rel]=dict(original_sha256='1'*64,current_sha256=sha(self.root/rel))
        (self.root/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(files=files)))
        self.port=dict(schema='report-portability-v2',original_source_manifest_sha256=sha(self.root/'SOURCE_MANIFEST.json'),
                       derived_documents=derived,historical_pdf=dict(source_commit='e53d39699dacd0490d875df8d2ec7642503f3081',
                           path='docs/report/report.pdf',sha256='0a4d6360d34f60b415a1e0b45e493296b9068e31f9b2b380ebdf1c7217356f04'),
                       current_pdf=dict(path='report.pdf',sha256=sha(self.root/'report.pdf'),bytes=(self.root/'report.pdf').stat().st_size,
                           source_files_sha256={p:sha(self.root/p) for p in ('report.tex','references.tex','appendix_links.tex','sections/intro.tex')}))
        self.save();self.mock=patch.object(inputs,'ROOT',self.root);self.mock.start();self.addCleanup(self.mock.stop)
    def save(self):
        (self.root/'PORTABILITY.json').write_text(json.dumps(self.port))
    def test_current_pdf_and_historical_identity_are_separate(self):
        self.assertEqual(inputs.validate_sources(),dict(original=1,portable_descriptions=7))
    def test_corrupt_pdf_rejected(self):
        (self.root/'report.pdf').write_text('modified PDF')
        with self.assertRaisesRegex(AssertionError,'current PDF SHA'):inputs.validate_sources()
    def test_source_edit_requires_new_edition_binding(self):
        (self.root/'sections/intro.tex').write_text('edited source')
        with self.assertRaises(AssertionError):inputs.validate_sources()
    def test_historical_pdf_identity_cannot_be_relabelled(self):
        self.port['historical_pdf']['sha256']=self.port['current_pdf']['sha256'];self.save()
        with self.assertRaises(AssertionError):inputs.validate_sources()
    def test_scientific_data_check_remains_active(self):
        (self.root/'data.csv').write_text('3,4\n')
        with self.assertRaises(AssertionError):inputs.validate_sources()
    def test_derivative_document_check_remains_active(self):
        (self.root/'document0.md').write_text('changed')
        with self.assertRaises(AssertionError):inputs.validate_sources()
    def test_single_repository_entry_point(self):
        self.assertEqual(inputs.validate_links()['mode'],'repository-index')
    def test_unknown_repository_rejected(self):
        p=self.root/'appendix_links.tex';p.write_text(p.read_text().replace('nefedovdima','another-owner'))
        with self.assertRaises(AssertionError):inputs.validate_links()
    def test_local_launch_rejected(self):
        p=self.root/'sections/intro.tex';p.write_text(r'\href{run:data.csv}{data}'+p.read_text())
        with self.assertRaisesRegex(AssertionError,'file link'):inputs.validate_links()
    def test_duplicate_repository_link_rejected(self):
        p=self.root/'sections/intro.tex';p.write_text(p.read_text()*2)
        with self.assertRaisesRegex(AssertionError,'one repository'):inputs.validate_links()
    def test_material_macros_cannot_restore_links(self):
        p=self.root/'appendix_links.tex';p.write_text(p.read_text().replace(r'\newcommand{\eref}[2]{#2}',r'\newcommand{\eref}[2]{\url{#1}}'))
        with self.assertRaisesRegex(AssertionError,'plain material'):inputs.validate_links()
    def test_repository_link_must_be_at_end(self):
        p=self.root/'sections/intro.tex';p.write_text(p.read_text()+' trailing text')
        with self.assertRaisesRegex(AssertionError,'end the report'):inputs.validate_links()
    def test_real_package(self):
        with patch.object(inputs,'ROOT',ROOT/'docs/report'):
            self.assertEqual(inputs.validate_sources()['original'],29)
            self.assertEqual(inputs.validate_links()['mode'],'repository-index')
if __name__=='__main__':unittest.main()
