"""Explicit schema-2 rental budget: short transactions, union phase clocks.

Each instance owns a separate session. Rental cost is elapsed calendar time,
never the sum of concurrent job durations. Schema-1 sessions remain unchanged.
"""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from cartpole.experiments.cloud_budget import Session, number, price_plan
from cartpole.experiments.cloud_io import atomic_json, file_lock

PHASES = ('setup', 'main', 'confirmation')


@contextmanager
def transaction(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def plan(offer, budget, caps):
    # Reuse tariff validation, but the new explicitly supplied budget replaces
    # the historical $12.50 cap; no default or inferred spending authorization.
    base = price_plan(offer)
    budget = number(budget, 'budget_usd', positive=True)
    if set(caps) != set(PHASES):
        raise ValueError('explicit setup/main/confirmation dollar caps required')
    caps = {k:number(v, k+'_cap', positive=True) for k,v in caps.items()}
    available = min(budget, number(offer['remaining_balance_usd'], 'remaining_balance_usd'))
    reserve = max(1., base['estimated_traffic_usd']+base['other_known_charges_usd'])
    if sum(caps.values())+reserve > available+1e-12:
        raise ValueError('phase allocations plus reserve exceed explicit session budget/balance')
    return dict(total_budget_usd=budget, available_usd=available, phase_caps_usd=caps,
                protected_reserve_usd=reserve, effective_hourly_usd=base['effective_hourly_usd'],
                available_active_seconds=(available-reserve)*3600/base['effective_hourly_usd'],
                accounting='rental wall time once; phase active unions may overlap and are not added to cost')


class ParallelSession:
    required_schema=2
    def __init__(self, path, *, clock=time.time, phase=None, job=None):
        self.path, self.clock = Path(path), clock
        self.phase = phase or os.environ.get('CARTPOLE_JOB_PHASE', 'setup')
        self.job = job or os.environ.get('CARTPOLE_SESSION_JOB')
        self._reload()
        if self.data['schema'] != 2 or self.data['price_plan'] != plan(
                self.data['offer'], self.data['budget_usd'], self.data['phase_caps_usd']):
            raise ValueError('parallel session budget identity changed')
        self._frozen_identity = self.identity()

    def identity(self):
        return {k:self.data[k] for k in ('session_id','offer','budget_usd','phase_caps_usd',
            'rental_started_at','hard_deadline_epoch','reserve_seconds')}

    @classmethod
    def create(cls, path, offer, rental_start, *, budget_usd, phase_caps_usd,
               max_seconds, reserve_seconds=120., clock=time.time):
        path = Path(path)
        with transaction(path.with_suffix('.create.lock')):
            if path.exists():
                raise FileExistsError('session exists; never reset rental start/deadline/budget')
            p = plan(offer, budget_usd, phase_caps_usd)
            start = number(rental_start, 'rental_start', positive=True)
            if start > clock()+1: raise ValueError('future rental start')
            duration = min(number(max_seconds, 'max_seconds', positive=True), p['available_active_seconds'])
            if offer.get('maximum_rental_duration_seconds') is not None:
                duration = min(duration, number(offer['maximum_rental_duration_seconds'], 'offer duration', positive=True))
            reserve = number(reserve_seconds, 'reserve_seconds', positive=True)
            if start+duration-clock() <= reserve: raise ValueError('no compute time after reserve')
            atomic_json(path, dict(schema=2, session_id=uuid.uuid4().hex, offer=offer,
                budget_usd=budget_usd, phase_caps_usd=phase_caps_usd, price_plan=p,
                rental_started_at=start, hard_deadline_epoch=start+duration, reserve_seconds=reserve,
                accounted_at=clock(), phase_seconds={k:0. for k in PHASES}, active={}, history=[]))
        return cls(path, clock=clock)

    def _reload(self):
        self.data = json.loads(self.path.read_text())
        if self.data.get('schema')!=self.required_schema:raise ValueError('session mode/schema changed')
        if hasattr(self,'_frozen_identity') and self.identity()!=self._frozen_identity:
            raise ValueError('running session deadline/budget changed')

    def _charge(self):
        now = self.clock()
        elapsed = max(0., now-self.data['accounted_at'])
        for phase in {r['phase'] for r in self.data['active'].values()}:
            self.data['phase_seconds'][phase] += elapsed
        self.data['accounted_at'] = now

    def enter_phase(self, phase):
        if phase not in PHASES: raise ValueError('unknown phase')
        self.phase = phase
        return self.status()

    @contextmanager
    def lease(self):
        # The guard owns a per-job OS lock throughout execution. Never hold the
        # shared JSON transaction lock around learner/evaluation work.
        self._reload()
        if not self.job or self.job not in self.data['active']:
            raise RuntimeError('parallel worker must be registered by cloud_guard')
        yield

    @contextmanager
    def operation(self, job, phase):
        if not job or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in job):
            raise ValueError('safe nonempty job ID required')
        locks = self.path.parent/(self.path.name+'.jobs'); locks.mkdir(exist_ok=True)
        with file_lock(locks/(job+'.lock')):
            with transaction(self.path.with_suffix('.state.lock')):
                self._reload(); self._charge()
                # A stale registration retains union charges until recovered;
                # holding this OS lock proves its former guard no longer owns it.
                self.data['active'][job] = dict(phase=phase, pid=os.getpid(), started_at=self.clock())
                self.data['history'].append(dict(event='job_start', job=job, phase=phase, time=self.clock()))
                atomic_json(self.path, self.data)
            self.job, self.phase = job, phase
            try: yield
            finally:
                with transaction(self.path.with_suffix('.state.lock')):
                    self._reload(); self._charge(); self.data['active'].pop(job, None)
                    self.data['history'].append(dict(event='job_end', job=job, time=self.clock()))
                    atomic_json(self.path, self.data)

    def status(self):
        self._reload()
        d, now = self.data, self.clock()
        used = dict(d['phase_seconds'])
        for phase in {r['phase'] for r in d['active'].values()}:
            used[phase] += max(0., now-d['accounted_at'])
        rate = d['price_plan']['effective_hourly_usd']
        phase_remaining = {p:max(0., d['phase_caps_usd'][p]*3600/rate-used[p]) for p in PHASES}
        remaining = min(d['hard_deadline_epoch']-now-d['reserve_seconds'],
                        phase_remaining[self.phase]-d['reserve_seconds'])
        return dict(phase=self.phase, session_id=d['session_id'], active_jobs=d['active'],
            elapsed_rental_seconds=max(0., now-d['rental_started_at']), phase_used_seconds=used,
            phase_remaining_seconds=phase_remaining, seconds_before_save_reserve=max(0., remaining),
            hard_deadline_epoch=d['hard_deadline_epoch'],
            stop_new_tasks_at_epoch=d['hard_deadline_epoch']-d['reserve_seconds'],
            estimated_accrued_usd=max(0., now-d['rental_started_at'])*rate/3600,
            price_plan=d['price_plan'])

    def record(self, event):
        with transaction(self.path.with_suffix('.state.lock')):
            self._reload(); self._charge()
            self.data['history'].append(dict(time=self.clock(), **event)); atomic_json(self.path, self.data)


def open_session(path):
    schema=json.loads(Path(path).read_text())['schema']
    if schema==3:
        from cartpole.experiments.time_only_session import TimeOnlySession
        return TimeOnlySession(path)
    if schema==2:return ParallelSession(path)
    if schema==1:return Session(path)
    raise ValueError('unknown session schema')


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init');q.add_argument('--offer',type=Path,required=True);q.add_argument('--session',type=Path,required=True)
    q.add_argument('--budget-usd',type=float,required=True);q.add_argument('--phase-caps',type=Path,required=True)
    q.add_argument('--rental-started-at',required=True);q.add_argument('--max-seconds',type=float,required=True)
    q.add_argument('--reserve-seconds',type=float,default=120.)
    q=sub.add_parser('init-time-only');q.add_argument('--session',type=Path,required=True)
    q.add_argument('--max-seconds',type=float,required=True);q.add_argument('--reserve-seconds',type=float,default=120.)
    q=sub.add_parser('status');q.add_argument('--session',type=Path,required=True)
    a=p.parse_args()
    if a.command=='init':
        stamp=datetime.fromisoformat(a.rental_started_at.replace('Z','+00:00'))
        if stamp.tzinfo is None:raise ValueError('timestamp timezone required')
        s=ParallelSession.create(a.session,json.loads(a.offer.read_text()),stamp.timestamp(),budget_usd=a.budget_usd,
            phase_caps_usd=json.loads(a.phase_caps.read_text()),max_seconds=a.max_seconds,reserve_seconds=a.reserve_seconds)
    elif a.command=='init-time-only':
        from cartpole.experiments.time_only_session import TimeOnlySession
        s=TimeOnlySession.create(a.session,max_seconds=a.max_seconds,reserve_seconds=a.reserve_seconds)
    else:s=open_session(a.session)
    print(json.dumps(s.status(),indent=2))

if __name__=='__main__':main()
