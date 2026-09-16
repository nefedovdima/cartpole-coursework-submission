"""Bound an entire owned Linux process tree by the existing rental session.

Stdlib-only: can wrap installation before Torch/Drake exist. No provider API.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from cartpole.experiments.cloud_budget import Session, number
from cartpole.experiments.cloud_io import atomic_json, file_lock, new_output


def processes():
    rows = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            rows[int(path.parent.name)] = (int(fields[1]), int(fields[19]), fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return rows


def descendants(root, rows):
    owned = set(); frontier = {root}
    while frontier:
        frontier = {pid for pid, (parent, _, _) in rows.items() if parent in frontier and pid not in owned}
        owned.update(frontier)
    return {pid: rows[pid][1] for pid in owned}


def owned_processes(root):
    """Read only this process subtree, including children of non-main threads.

    A subreaper adopts detached descendants. Re-reading its task children also
    finds those orphans without scanning every process on a large shared host.
    """
    rows={};frontier=[root]
    while frontier:
        pid=frontier.pop()
        if pid in rows:continue
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            rows[pid]=(int(fields[1]),int(fields[19]),fields[0])
            for task in Path(f'/proc/{pid}/task').glob('*/children'):
                try:frontier.extend(int(p) for p in task.read_text().split())
                except (OSError,ValueError):pass
        except (OSError,ValueError,IndexError):pass
    return rows


def emergency_cleanup(child, grace):
    deadline=time.monotonic()+grace
    while True:
        rows=owned_processes(os.getpid());owned=descendants(os.getpid(),rows)
        live=[pid for pid in owned if rows[pid][2]!='Z']
        for pid in live:
            try:os.kill(pid,signal.SIGTERM if time.monotonic()<deadline else signal.SIGKILL)
            except ProcessLookupError:pass
        child.poll()
        for pid in owned:
            if pid != child.pid:
                try:os.waitpid(pid,os.WNOHANG)
                except ChildProcessError:pass
        if not live:return
        time.sleep(.05)


def supervise(command, session, *, phase, max_seconds, output=None, grace_seconds=30., job_id=None):
    number(max_seconds, 'max_seconds', positive=True); number(grace_seconds, 'grace_seconds', positive=True)
    out = new_output(output, 'cloud_guard')
    # This lease spans the whole operation. Child cloud commands may acquire
    # the shorter Session.lease themselves; using that same lock would deadlock.
    parallel = hasattr(session, 'operation')
    if parallel and job_id is None:
        raise ValueError('parallel session requires an explicit unique job ID')
    operation = session.operation(job_id, phase) if parallel else file_lock(session.path.with_suffix('.operations.lease'))
    with operation:
        with session.lease():
            status = session.enter_phase(phase)
        allowed = min(max_seconds, status['seconds_before_save_reserve'])
        grace = min(grace_seconds, session.data['reserve_seconds'])
        if allowed <= 0:
            result = dict(status='not_started', reason=('session_time_exhausted' if session.data.get('accounting_mode')=='time-only' else 'session_budget_exhausted'),
                          accounting_mode=session.data.get('accounting_mode','monetary'),returncode=None)
            atomic_json(out/'operation.json', result)
            return out, result
        libc = ctypes.CDLL(None, use_errno=True)
        previous_subreaper=ctypes.c_int()
        if libc.prctl(37, ctypes.byref(previous_subreaper), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(),'cannot read Linux subreaper state')
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), 'Linux subreaper required to own orphaned descendants')
        started = time.monotonic(); deadline = started+allowed
        reason = 'completed'; stopped = []; tracked = {}; child = None
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for sig in old_handlers:
            signal.signal(sig, lambda signum, frame: stopped.append(signum))
        report = dict(status='running', accounting_mode=session.data.get('accounting_mode','monetary'), command=command, phase=phase, session=str(session.path.resolve()),
                      allowed_seconds=allowed, grace_seconds=grace, started_at_epoch=time.time())
        env = dict(os.environ, CARTPOLE_BUDGET_SESSION=str(session.path.resolve()))
        if parallel:
            env.update(CARTPOLE_SESSION_JOB=job_id, CARTPOLE_JOB_PHASE=phase)
        try:
            atomic_json(out/'operation.json', report)
            with (out/'console.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                termination_at = None; exited_at = None
                # A spawn resource tracker can finish just after its parent.
                # This small passive drain is inside the same deadline, not an
                # extension and not acceptance of a surviving descendant.
                exit_drain = min(.5, grace)
                while True:
                    rows = owned_processes(os.getpid()); tracked.update(descendants(os.getpid(), rows))
                    live = {p for p, birth in tracked.items() if p in rows and rows[p][1] == birth and rows[p][2] != 'Z'}
                    code = child.poll()
                    now = time.monotonic()
                    if code is not None:
                        if exited_at is None: exited_at = now
                        # poll() can reap a child that was live in the previous
                        # /proc snapshot. Never classify that stale PID as a leak.
                        rows = owned_processes(os.getpid())
                        tracked.update(descendants(os.getpid(), rows))
                        live = {p for p, birth in tracked.items() if p != child.pid
                                and p in rows and rows[p][1] == birth and rows[p][2] != 'Z'}
                    budget_expired = session.status()['seconds_before_save_reserve'] <= 0
                    leaked = code is not None and live and now >= exited_at + exit_drain
                    if termination_at is None and (stopped or now >= deadline or budget_expired or leaked):
                        reason = signal.Signals(stopped[-1]).name if stopped else ('timeout' if now >= deadline or budget_expired else 'descendant_cleanup')
                        termination_at = now
                    if termination_at is not None:
                        sig = signal.SIGKILL if now >= termination_at+grace else signal.SIGTERM
                        for pid in live:
                            try:
                                # Recheck birth before signalling: a stale scan
                                # must not target a reused PID outside our tree.
                                fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                                if int(fields[19]) != tracked[pid] or fields[0] == 'Z': continue
                                os.kill(pid, sig)
                            except (ProcessLookupError, FileNotFoundError):
                                pass
                    for pid in tracked:
                        if pid != child.pid:
                            try:
                                os.waitpid(pid, os.WNOHANG)
                            except ChildProcessError:
                                pass
                    if code is not None and not live:
                        break
                    time.sleep(.05)
                if reason == 'completed' and code != 0:
                    reason='command_failed'
                report.update(status='completed' if reason == 'completed' and code == 0 else 'stopped_or_failed',
                              reason=reason, returncode=code, seconds=time.monotonic()-started,
                              natural_exit_drain_limit_seconds=exit_drain,
                              tree_pids=list(tracked), all_owned_processes_stopped=True,
                              session_after=session.status())
        except BaseException as exc:
            if child is not None:
                emergency_cleanup(child,grace)
            report.update(status='failed',reason='wrapper_exception',returncode=child.poll() if child else None,
                          exception=dict(type=type(exc).__name__,message=str(exc)),seconds=time.monotonic()-started)
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            libc.prctl(36,previous_subreaper.value,0,0,0)
            atomic_json(out/'operation.json', report)
    return out, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--phase', choices=('setup','main','confirmation'), required=True)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--grace-seconds', type=float, default=30.)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--job-id', help='required for concurrent monetary/schema-2 or time-only/schema-3 sessions')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args(); command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        p.error('command required after --')
    from cartpole.experiments.parallel_budget import open_session
    out, report = supervise(command, open_session(args.session), phase=args.phase, max_seconds=args.max_seconds,
                            grace_seconds=args.grace_seconds, output=args.output_dir, job_id=args.job_id)
    print(json.dumps(dict(output=str(out), result=report), allow_nan=False))
    raise SystemExit(0 if report['status'] == 'completed' else 124 if report['reason'] == 'timeout' else 1)


if __name__ == '__main__':
    main()
