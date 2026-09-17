"""Validate preserved sources and explicitly pinned portable document derivatives."""
from pathlib import Path
import hashlib
import json
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
    assert sha(ROOT/'report.pdf')==portable['unchanged_pdf_sha256']
    return dict(original=len(manifest['files'])-len(derived),portable_descriptions=len(derived))
