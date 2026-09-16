"""Pre-download host policy and cgroup-aware memory, using only stdlib."""
from pathlib import Path
import platform
import shutil
import subprocess


def gpu_policy(rows):
    result = []
    for row in rows.splitlines():
        driver, name, capability = [part.strip() for part in row.split(',')]
        if tuple(map(int, driver.split('.'))) < (560, 35, 5):
            raise RuntimeError('cu126 requires NVIDIA driver >=560.35.05')
        if capability not in ('8.6', '8.9'):
            raise RuntimeError(f'{name}: compute capability {capability} outside first-run cu126 policy (8.6/8.9); driver alone is insufficient')
        result.append(dict(driver=driver, name=name, compute_capability=capability))
    if not result:
        raise RuntimeError('no visible NVIDIA GPU')
    return result


def check_platform():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or platform.python_version() != '3.12.3':
        raise ValueError('reviewed host requires Linux x86_64 and CPython exactly 3.12.3')
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 35):
        raise ValueError('glibc >=2.35 required')


def check_gpu():
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('NVIDIA GPU/driver unavailable; no CUDA installation')
    return gpu_policy(subprocess.check_output(['nvidia-smi',
                      '--query-gpu=driver_version,name,compute_cap', '--format=csv,noheader'], text=True))


def memory_report(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    raw = {line.split(':')[0]: int(line.split()[1])*1024 for line in (proc/'meminfo').read_text().splitlines()}
    limits = []; memberships = (proc/'self/cgroup').read_text().splitlines()
    for membership in memberships:
        _, controllers, relative = membership.split(':', 2)
        if '..' in Path(relative).parts:
            raise ValueError('unresolvable cgroup namespace path; inspect memory quota before running')
        unified = controllers == ''
        if not unified and 'memory' not in controllers.split(','):
            continue
        base = cgroup if unified else cgroup/'memory'
        names = ('memory.max','memory.current') if unified else ('memory.limit_in_bytes','memory.usage_in_bytes')
        node = base/relative.lstrip('/')
        # Include ancestor quotas: sibling usage can exhaust a parent limit.
        while node.is_relative_to(base):
            limit_file, usage_file = (node/n for n in names)
            if limit_file.is_file() and usage_file.is_file():
                value = limit_file.read_text().strip()
                limit = None if value == 'max' else int(value)
                if limit is not None and limit >= 2**60:
                    limit = None
                usage = int(usage_file.read_text())
                limits.append(dict(path=str(node), version=2 if unified else 1, limit_bytes=limit,
                                   current_bytes=usage, available_bytes=max(0, limit-usage) if limit is not None else None))
            if node == base:
                break
            node = node.parent
    finite = [entry for entry in limits if entry['limit_bytes'] is not None]
    return dict(host_total_bytes=raw['MemTotal'], host_available_bytes=raw['MemAvailable'],
                cgroup_membership=memberships, cgroup_limits=limits,
                effective_total_bytes=min([raw['MemTotal']]+[r['limit_bytes'] for r in finite]),
                effective_available_bytes=min([raw['MemAvailable']]+[r['available_bytes'] for r in finite]),
                quota_observed=bool(limits))
