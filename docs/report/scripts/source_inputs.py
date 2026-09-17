"""Validate preserved sources and explicitly pinned portable document derivatives."""
from pathlib import Path
import hashlib
import json
import re
ROOT=Path(__file__).resolve().parents[1]
def validate_sources():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    portable=json.loads((ROOT/'PORTABILITY.json').read_text())
    assert sha(ROOT/'SOURCE_MANIFEST.json')==portable['original_source_manifest_sha256']
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    derived=portable['derived_documents']
    assert len(derived)==7 and all(p.endswith('.md') for p in derived)
    for rel,spec in manifest['files'].items():
        path=ROOT/rel
        if rel in derived:
            assert derived[rel]['original_sha256']==spec['sha256']
            assert sha(path)==derived[rel]['current_sha256'],rel
        else:
            assert path.stat().st_size==spec['bytes'] and sha(path)==spec['sha256'],rel
    assert portable['schema']=='report-portability-v2'
    assert portable['historical_pdf']==dict(
        source_commit='e53d39699dacd0490d875df8d2ec7642503f3081',
        path='docs/report/report.pdf',
        sha256='0a4d6360d34f60b415a1e0b45e493296b9068e31f9b2b380ebdf1c7217356f04')
    current=portable['current_pdf']
    assert current['path']=='report.pdf'
    assert sha(ROOT/current['path'])==current['sha256'], 'current PDF SHA'
    assert (ROOT/current['path']).stat().st_size==current['bytes'], 'current PDF size'
    expected={'report.tex','references.tex','appendix_links.tex'} | {
        str(p.relative_to(ROOT)) for p in (ROOT/'sections').glob('*.tex')}
    assert set(current['source_files_sha256'])==expected, 'report source roster'
    for rel,digest in current['source_files_sha256'].items():
        assert sha(ROOT/rel)==digest, rel
    return dict(original=len(manifest['files'])-len(derived),portable_descriptions=len(derived))


def validate_links():
    settings=(ROOT/'appendix_links.tex').read_text()
    expected='https://github.com/nefedovdima/cartpole-coursework-submission'
    matches=re.findall(r'\\newcommand\{\\SubmissionRepositoryURL\}\{([^{}]*)\}',settings)
    assert matches==[expected], 'exact submission repository URL required'
    for name in ('eref','vref','applink'):
        assert re.findall(r'\\newcommand\{\\'+name+r'\}\[2\]\{([^{}]*)\}',settings)==['#2'], 'plain material label: '+name
    text='\n'.join(p.read_text() for p in sorted((ROOT/'sections').glob('*.tex')))
    authored=settings+'\n'+(ROOT/'report.tex').read_text()+'\n'+text
    assert not re.search(r'run:|file:|\\href|\\hyperlinkfile',authored,re.I), 'local or direct file link forbidden'
    assert text.count(r'\url{\SubmissionRepositoryURL}')==1, 'one repository entry point required'
    assert expected not in text, 'use the single URL setting'
    assert text.rstrip().endswith(r'\begin{center}\small\url{\SubmissionRepositoryURL}\end{center}'), 'repository block must end the report'
    for rel in re.findall(r'\\applink\{([^}]+)\}',text):
        path=(ROOT/'../..'/rel).resolve()
        assert path.is_relative_to(ROOT.parent.parent.resolve()) and path.is_file(),rel
    return dict(mode='repository-index', url=expected)
