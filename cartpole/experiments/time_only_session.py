"""Explicit execution-time session. No tariff, balance, rental cost or money cap."""
import json
import time
import uuid
from pathlib import Path

from cartpole.experiments.cloud_budget import number
from cartpole.experiments.cloud_io import atomic_json
from cartpole.experiments.parallel_budget import ParallelSession, PHASES, transaction


class TimeOnlySession(ParallelSession):
    required_schema=3
    def __init__(self, path, *, clock=time.time, phase=None, job=None):
        import os
        self.path,self.clock=Path(path),clock
        self.phase=phase or os.environ.get('CARTPOLE_JOB_PHASE','setup')
        self.job=job or os.environ.get('CARTPOLE_SESSION_JOB')
        self._reload()
        d=self.data
        if d.get('schema')!=3 or d.get('accounting_mode')!='time-only':
            raise ValueError('explicit time-only schema-3 session required')
        if any(k in d for k in ('offer','price_plan','budget_usd','phase_caps_usd','rental_started_at')):
            raise ValueError('time-only session cannot contain monetary/rental fields')
        for k in ('started_at','hard_deadline_epoch','reserve_seconds','max_seconds'):
            number(d[k],k,positive=True)
        if d['hard_deadline_epoch']!=d['started_at']+d['max_seconds'] or d['reserve_seconds']>=d['max_seconds']:
            raise ValueError('time-only deadline identity invalid')
        if set(d['phase_seconds'])!=set(PHASES):raise ValueError('invalid phase clock')
        self._frozen_identity=self.identity()

    def _reload(self):
        super()._reload()
        if self.data.get('accounting_mode')!='time-only' or any(k in self.data for k in
                ('offer','price_plan','budget_usd','phase_caps_usd','rental_started_at')):
            raise ValueError('time-only session cannot mix accounting modes')

    def identity(self):
        return {k:self.data[k] for k in ('schema','accounting_mode','session_id','started_at',
            'max_seconds','hard_deadline_epoch','reserve_seconds')}

    @classmethod
    def create(cls,path,*,max_seconds,reserve_seconds=120.,started_at=None,clock=time.time):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        duration=number(max_seconds,'max_seconds',positive=True)
        reserve=number(reserve_seconds,'reserve_seconds',positive=True)
        start=clock() if started_at is None else number(started_at,'started_at',positive=True)
        if start>clock()+1 or start+duration-clock()<=reserve:
            raise ValueError('no compute time after reserve or future start')
        with transaction(path.with_suffix('.create.lock')):
            if path.exists():raise FileExistsError('session exists; never reset mode/start/deadline')
            atomic_json(path,dict(schema=3,accounting_mode='time-only',session_id=uuid.uuid4().hex,
                started_at=start,max_seconds=duration,hard_deadline_epoch=start+duration,reserve_seconds=reserve,
                accounted_at=clock(),phase_seconds={k:0. for k in PHASES},active={},history=[]))
        return cls(path,clock=clock)

    def status(self):
        self._reload();d=self.data;now=self.clock();used=dict(d['phase_seconds'])
        for phase in {r['phase'] for r in d['active'].values()}:
            used[phase]+=max(0.,now-d['accounted_at'])
        return dict(accounting_mode='time-only',phase=self.phase,session_id=d['session_id'],active_jobs=d['active'],
            elapsed_session_seconds=max(0.,now-d['started_at']),phase_used_seconds=used,
            seconds_before_save_reserve=max(0.,d['hard_deadline_epoch']-now-d['reserve_seconds']),
            hard_deadline_epoch=d['hard_deadline_epoch'],stop_new_tasks_at_epoch=d['hard_deadline_epoch']-d['reserve_seconds'],
            monetary_accounting=False,note='execution-time limit only; provider billing is not tracked')
