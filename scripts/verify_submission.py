"""Read-only verification of the portable, SHA-listed submission payload."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root=Path(root).resolve();manifest=root/'provenance/SHA256SUMS'
    records={}
    for line in manifest.read_text().splitlines():
        digest,rel=line.split('  ',1)
        if rel in records or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('duplicate or invalid checksum')
        if Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('unsafe manifest path')
        records[rel]=digest
    links=[];total=0
    allowed_zips={f'models/{a}_seed{s}/model.zip' for a in ('SAC','TQC') for s in range(3)}
    for rel,digest in records.items():
        p=root/rel
        if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('SHA mismatch: '+rel)
        if any(v in Path(rel).parts for v in ('.git','verification','raw','__pycache__','replay.pkl','runtime.pt')) or p.name=='AGENTS.md':raise ValueError('excluded artifact: '+rel)
        size=p.stat().st_size;total+=size
        if size>=50*1024**2:raise ValueError('oversized artifact: '+rel)
        if p.suffix.lower() in ('.pkl','.pt','.gz') or (p.suffix=='.zip' and rel not in allowed_zips):raise ValueError('unapproved binary: '+rel)
        if p.suffix=='.py':ast.parse(p.read_text(),filename=rel)
        if p.suffix in ('.py','.md','.json','.csv','.tex','.txt','.svg','.toml','.cff','.sh'):
            text=p.read_text()
            if re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+|[A-Z]:\\Users\\|/workspace/[A-Za-z0-9_.-]+',text):raise ValueError('private absolute path: '+rel)
            if re.search(r'-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE).*KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{32,}',text):raise ValueError('possible secret: '+rel)
        if p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
                target=target.strip('<>').split('#',1)[0]
                if not target or re.match(r'[a-zA-Z]+:',target):continue
                q=(p.parent/target).resolve()
                if not q.is_relative_to(root) or not q.exists():raise ValueError('broken relative link: '+rel+' -> '+target)
                if q.is_file() and str(q.relative_to(root)) not in records and q!=manifest:raise ValueError('link outside listed payload: '+rel)
                links.append((rel,target))
    from scripts.frozen_inference import verify_capsule
    for name in sorted(allowed_zips):verify_capsule(Path(name).parent.name,root)
    video=json.loads((root/'media/VIDEO_MANIFEST.json').read_text())
    for v in video['videos']:
        for name,digest in [(v['file'],v['sha256']),(v['poster'],v['poster_sha256'])]:
            p=root/'media'/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=digest or p.read_bytes()!=(root/'docs/report/media'/name).read_bytes():raise ValueError('media identity differs')
    for rel,digest in video['code_sha256'].items():
        if records.get(rel)!=digest:raise ValueError('renderer source identity differs')
    return dict(import_hashes=len(records),payload_bytes=total,relative_markdown_links=len(links),syntax='OK',models=6,videos=6,large_files=0)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(verify(),indent=2))
if __name__=='__main__':main()
