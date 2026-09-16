"""Static portable-artifact verification. No training or simulated transitions."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def verify(root=ROOT):
    root=Path(root);count=0
    for row in json.loads((root/'provenance/imports.json').read_text()):
        p=root/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('import SHA mismatch: '+row['path'])
        count+=1
    links=[]
    for path in root.rglob('*.md'):
        if any(x.startswith('.') for x in path.relative_to(root).parts):continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            target=target.strip('<>').split('#',1)[0]
            if not target or re.match(r'[a-zA-Z]+:',target):continue
            q=(path.parent/target).resolve()
            if not q.is_relative_to(root.resolve()) or not q.exists():raise ValueError(f'broken relative link in {path.name}: {target}')
            links.append((str(path.relative_to(root)),target))
    for area in ('cartpole','scripts','tests'):
        for path in (root/area).rglob('*.py'):ast.parse(path.read_text(),filename=str(path))
    for path in root.rglob('*'):
        if not path.is_file() or any(x.startswith('.') for x in path.relative_to(root).parts):continue
        if path.stat().st_size>50_000_000:raise ValueError('oversized Git artifact: '+path.name)
        if path.suffix.lower() in ('.pkl','.pt','.zip','.gz'):raise ValueError('unapproved recovery/model artifact: '+path.name)
        if path.suffix in ('.py','.md','.json','.csv','.tex','.txt','.svg','.toml','.cff','.sh'):
            content=path.read_text()
            if re.search(r'/(?:home|Users)/[^\s"<>]+|[A-Z]:\\Users\\',content):raise ValueError('personal absolute path: '+str(path))
            if re.search(r'-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE).*KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{32,}',content):raise ValueError('possible secret: '+str(path))
    return dict(import_hashes=count,relative_markdown_links=len(links),syntax='OK',large_files=0)

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(verify(),indent=2))
if __name__=='__main__':main()
