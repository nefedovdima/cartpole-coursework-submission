"""Persistent $12.50 planning ledger. A timer cannot cap a provider's bill."""
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from cartpole.experiments.cloud_io import atomic_json, file_lock


PHASE_CAPS = {'setup': 1.50, 'main': 8.00, 'confirmation': 2.00}


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a known finite number; unknown is not zero')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return float(value)


def price_plan(offer):
    """All traffic rates use decimal GB, explicitly converted from offer units."""
    rate = number(offer['rental_hourly_usd'], 'rental_hourly_usd', positive=True)
    disk = number(offer['disk_gb'], 'disk_gb', positive=True)
    included = offer['rental_includes_storage']
    if type(included) is not bool:
        raise ValueError('rental_includes_storage must explicitly be true/false')
    storage = (0. if included else disk*number(offer['storage_gb_hourly_usd'], 'storage_gb_hourly_usd'))
    traffic = sum(number(offer[d+'_gb'], d+'_gb')*number(offer[d+'_usd_per_gb'], d+'_usd_per_gb')
                  for d in ('ingress', 'egress'))
    other = number(offer['other_known_charges_usd'], 'other_known_charges_usd')
    balance = number(offer['remaining_balance_usd'], 'remaining_balance_usd')
    spent = number(offer['spent_before_session_usd'], 'spent_before_session_usd')
    available = min(balance, max(0., 12.50-spent))
    # New sessions may declare a tighter explicit spend cap. Historical offers
    # keep their exact price-plan representation; this never increases a limit.
    session_cap = offer.get('session_budget_usd')
    if session_cap is not None:
        available = min(available, number(session_cap, 'session_budget_usd', positive=True))
    reserve = max(1., traffic+other)
    hourly = rate+storage
    spendable = min(sum(PHASE_CAPS.values()), max(0., available-reserve))
    return dict(total_budget_usd=12.50, available_usd=available,
                phase_caps_usd=PHASE_CAPS, protected_reserve_usd=reserve,
                estimated_traffic_usd=traffic, other_known_charges_usd=other,
                rental_hourly_usd=rate, storage_additional_hourly_usd=storage,
                effective_hourly_usd=hourly, spendable_usd=spendable,
                available_active_seconds=3600*spendable/hourly,
                warning='Estimates only. Python exit does not stop rental; stopped storage remains billable.')


class Session:
    """One ledger and one lease across setup, seeds, queues and resume.

    Absolute rental start includes installation and idle time. Phase accounting
    continues between commands, and never resets at the next seed.
    """
    def __init__(self, path, *, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.data = json.loads(self.path.read_text())
        if self.data['schema'] != 1:
            raise ValueError('unknown budget session schema')
        if self.data['price_plan'] != price_plan(self.data['offer']):
            raise ValueError('session price configuration changed')

    @classmethod
    def create(cls, path, offer, rental_start, *, max_seconds, reserve_seconds=120., clock=time.time):
        path = Path(path)
        if path.exists():
            raise FileExistsError('session already exists; reuse it, do not reset the budget')
        plan = price_plan(offer)
        start = number(rental_start, 'rental_start', positive=True)
        now = clock()
        if start > now+1:
            raise ValueError('rental start cannot be in the future')
        duration = min(number(max_seconds, 'max_seconds', positive=True), plan['available_active_seconds'])
        if offer.get('maximum_rental_duration_seconds') is not None:
            duration = min(duration, number(offer['maximum_rental_duration_seconds'],
                                            'maximum_rental_duration_seconds', positive=True))
        reserve = number(reserve_seconds, 'reserve_seconds', positive=True)
        if duration <= reserve:
            raise ValueError('no compute time remains after the saving reserve')
        data = dict(schema=1, offer=offer, price_plan=plan, rental_started_at=start,
                    hard_deadline_epoch=start+duration, reserve_seconds=reserve,
                    phase='setup', phase_started_at=start,
                    phase_seconds={key: 0. for key in PHASE_CAPS}, history=[])
        atomic_json(path, data)
        return cls(path, clock=clock)

    def lease(self):
        return file_lock(self.path.with_suffix(self.path.suffix+'.lease'))

    def _reload(self):
        self.data = json.loads(self.path.read_text())

    def enter_phase(self, phase):
        if phase not in PHASE_CAPS:
            raise ValueError('unknown budget phase')
        self._reload()
        old, now = self.data['phase'], self.clock()
        if old != phase:
            self.data['phase_seconds'][old] += max(0., now-self.data['phase_started_at'])
            self.data.update(phase=phase, phase_started_at=now)
            self.data['history'].append(dict(event='phase', previous=old, phase=phase, time=now))
            atomic_json(self.path, self.data)
        return self.status()

    def status(self):
        self._reload()
        d, now = self.data, self.clock()
        used = copy.deepcopy(d['phase_seconds'])
        used[d['phase']] += max(0., now-d['phase_started_at'])
        hourly = d['price_plan']['effective_hourly_usd']
        phase_remaining = {p: max(0., PHASE_CAPS[p]*3600/hourly-used[p]) for p in PHASE_CAPS}
        overall = max(0., d['hard_deadline_epoch']-now-d['reserve_seconds'])
        remaining = min(overall, max(0., phase_remaining[d['phase']]-d['reserve_seconds']))
        return dict(phase=d['phase'], elapsed_rental_seconds=max(0., now-d['rental_started_at']),
                    phase_used_seconds=used, phase_remaining_seconds=phase_remaining,
                    seconds_before_save_reserve=remaining, hard_deadline_epoch=d['hard_deadline_epoch'],
                    stop_new_tasks_at_epoch=d['hard_deadline_epoch']-d['reserve_seconds'],
                    estimated_accrued_usd=max(0., now-d['rental_started_at'])*hourly/3600,
                    price_plan=d['price_plan'])

    def can_start(self, requested_seconds):
        return self.status()['seconds_before_save_reserve'] >= number(requested_seconds, 'requested_seconds', positive=True)

    def record(self, event):
        self._reload()
        self.data['history'].append(dict(time=self.clock(), **event))
        atomic_json(self.path, self.data)


def projected_volume(plan, seconds_per_100k, seconds_per_300k):
    return {phase: dict(available_seconds=min(plan['available_active_seconds'], dollars*3600/plan['effective_hourly_usd']),
                        complete_100k_runs=int(min(plan['available_active_seconds'], dollars*3600/plan['effective_hourly_usd'])/seconds_per_100k),
                        complete_300k_runs=int(min(plan['available_active_seconds'], dollars*3600/plan['effective_hourly_usd'])/seconds_per_300k))
            for phase, dollars in PHASE_CAPS.items()}


def main():
    """This stdlib-only command also works BEFORE installing Torch/Drake."""
    import argparse
    p=argparse.ArgumentParser(description=__doc__); subs=p.add_subparsers(dest='command',required=True)
    a=subs.add_parser('plan'); a.add_argument('--offer',type=Path,required=True)
    a.add_argument('--benchmark',type=Path); a.add_argument('--output',type=Path)
    a=subs.add_parser('init'); a.add_argument('--offer',type=Path,required=True)
    a.add_argument('--session',type=Path,required=True)
    a.add_argument('--rental-started-at',required=True,help='actual ISO-8601 timestamp, including timezone')
    a.add_argument('--max-seconds',type=float,required=True); a.add_argument('--reserve-seconds',type=float,default=120.)
    a=subs.add_parser('status'); a.add_argument('--session',type=Path,required=True)
    args=p.parse_args()
    if args.command == 'status':
        result=Session(args.session).status()
    else:
        offer=json.loads(args.offer.read_text())
        if args.command == 'init':
            stamp=datetime.fromisoformat(args.rental_started_at.replace('Z','+00:00'))
            if stamp.tzinfo is None:
                raise ValueError('rental start must include its timezone')
            result=Session.create(args.session,offer,stamp.timestamp(),max_seconds=args.max_seconds,
                                  reserve_seconds=args.reserve_seconds).status()
        else:
            result=price_plan(offer)
            if args.benchmark:
                data=json.loads(args.benchmark.read_text()); rows=data.get('runs',[data])
                result['alternatives_not_additive']=[]
                for row in rows:
                    forecasts=row['forecasts']
                    if not all(str(n) in forecasts for n in (100000,300000)):
                        continue
                    seconds=[forecasts[str(n)]['planning_seconds_with_25pct_headroom'] for n in (100000,300000)]
                    result['alternatives_not_additive'].append(dict(config=row['config'],device=row['device'],
                        predicted_100k_usd=seconds[0]*result['effective_hourly_usd']/3600,
                        predicted_300k_usd=seconds[1]*result['effective_hourly_usd']/3600,
                        volume=projected_volume(result,*seconds)))
            if args.output:
                if args.output.exists():
                    raise FileExistsError(args.output)
                atomic_json(args.output,result)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    main()
