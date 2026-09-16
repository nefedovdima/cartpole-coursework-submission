"""Local files, provenance and atomic publication; no account or network calls."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        with os.fdopen(handle, 'w') as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_json(path, data):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def atomic_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path); sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def new_output(output=None, prefix='cloud'):
    path = Path(output) if output else ROOT/'results'/datetime.now(timezone.utc).strftime(prefix+'_%Y%m%dT%H%M%S_%fZ')
    path.mkdir(parents=True, exist_ok=False)
    return path


def manifest(directory, *, exclude=('manifest.json',)):
    directory = Path(directory)
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and not p.is_symlink() and str(p.relative_to(directory)) not in exclude}


def verify_manifest(directory, filename='manifest.json'):
    directory = Path(directory)
    entries = json.loads((directory/filename).read_text())
    for name, digest in entries.items():
        path = directory/name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('unsafe manifest path')
        if sha256(path) != digest:
            raise ValueError(f'checksum mismatch: {name}')
    return len(entries)


def source_files(root=ROOT):
    """Explicit project allowlist: no .git, environments, results or credentials."""
    paths = []
    for pattern in ('cartpole/**/*.py', 'tests/*.py', 'tests/fixtures/*.npz',
                    'tests/fixtures/*.json', 'tests/fixtures/*.md', 'configs/cloud/*.json', 'configs/development/*.json', 'configs/confirmation/*.json',
                    'scripts/cloud/*.py', 'scripts/cloud/*.sh', 'docs/*.md',
                    'review-e6d/cartpole-e6d-audit-20260914/*.md',
                    'review-e6d/cartpole-e6d-audit-20260914/*.json',
                    'review-e6d/cartpole-e6d-audit-20260914/*.py',
                    'review-e6d/cartpole-e6d-audit-20260914/*.txt',
                    'review-e6d/cartpole-e6d-audit-20260914/SHA256SUMS',
                    'requirements*.txt', 'pyproject.toml', 'readme.md'):
        paths.extend(root.glob(pattern))
    result = sorted(set(paths))
    if any(p.is_symlink() for p in result):
        raise ValueError('source bundle does not accept symlinks')
    return result


def snapshot(output, root=ROOT):
    output = Path(output)
    files = source_files(root)
    hashes = {}
    for path in files:
        relative = path.relative_to(root)
        dest = output/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest); hashes[str(relative)] = sha256(dest)
    atomic_json(output/'source_manifest.json', hashes)
    if (root/'.git').exists():
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
        # Only allowlisted source paths can enter the patch.
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--',
                    *[str(p.relative_to(root)) for p in files]], cwd=root)
        (output/'source.diff').write_bytes(diff)
        git = dict(sha=head, dirty=bool(status), status_porcelain=status)
    else:
        origin = root/'bundle_provenance.json'
        git = json.loads(origin.read_text())['git'] if origin.exists() else dict(sha=None, dirty=None)
    atomic_json(output/'provenance.json', dict(git=git, source_manifest_sha256=sha256(output/'source_manifest.json'),
                python=platform.python_version(), executable=sys.executable))
    (output/'environment.freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    return git


def bundle(output):
    """A portable current-code archive, including reviewed uncommitted files."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cartpole-bundle-') as tmp:
        root = Path(tmp)/'cartpole-coursework'; root.mkdir()
        for src in source_files():
            dest = root/src.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
        atomic_json(root/'bundle_provenance.json', dict(git=dict(sha=head, dirty=bool(status), status_porcelain=status),
                    purpose='E6d portable sources; includes uncommitted code; no results or secrets'))
        atomic_json(root/'manifest.json', manifest(root))
        verify_manifest(root)
        # Stable member metadata; archive hash still identifies the exact content.
        def normalize(info):
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
            return info
        with tarfile.open(output, 'x:gz') as archive:
            archive.add(root, arcname='cartpole-coursework', filter=normalize)
    output.with_suffix(output.suffix+'.sha256').write_text(sha256(output)+'  '+output.name+'\n')
    return dict(path=str(output), sha256=sha256(output), bytes=output.stat().st_size)


@contextmanager
def file_lock(path):
    import fcntl
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another process already owns this run/session') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def child_process(command, stream, *, timeout=None):
    """Forward termination to the owned child and allow its atomic save to finish."""
    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    requested = []
    def forward(signum, frame):
        requested.append(signum)
        if child.poll() is None:
            os.killpg(child.pid, signum)
    for s in previous:
        signal.signal(s, forward)
    try:
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise
        if requested:
            raise InterruptedError(f'queue/benchmark stopped by {signal.Signals(requested[-1]).name}; child finished saving')
        if code:
            raise subprocess.CalledProcessError(code, command)
        return code
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
