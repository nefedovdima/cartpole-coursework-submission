"""Evidence-bound videos of saved states; no scientific runtime is imported.

Data validation uses the stdlib analyzer. Pillow and ffmpeg are used only when
rendering. Unwrapped log angles are preserved, not reduced modulo 2*pi.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from cartpole.experiments import robustness_analysis as a

VERSION = 'saved-robustness-video-v2'
FPS = 30
WIDTH, HEIGHT = 1920, 1080
HOLD_FRAMES = 60
GROUPS = ('nominal', 'position', 'velocity', 'angle', 'angular_velocity', 'boundary', 'joint')
GROUP_TITLES = ('Номинальный старт', 'Смещение тележки', 'Начальная скорость тележки',
                'Начальный угол маятника', 'Начальная угловая скорость',
                'Старт у границы', 'Совместные возмущения')
MODELS = ('SAC_seed0', 'SAC_seed1', 'SAC_seed2', 'TQC_seed0', 'TQC_seed1', 'TQC_seed2', 'classical')
SCREEN_MODELS = ('SAC_seed0', 'SAC_seed1', 'SAC_seed2', 'classical', 'TQC_seed0', 'TQC_seed1', 'TQC_seed2')
MODEL_COLORS = dict(zip(MODELS, ('#297e9b', '#4170a0', '#53669a', '#418374', '#5c8571', '#6c7960', '#94743d')))
MODEL_COLORS['DDPG_seed5'] = '#986a87'
STATUS_COLORS = dict(hold='#19364b', good='#347660', warning='#975e43')
NAMES = ('01_nominal_controllers', '02_ddpg_best_vs_last', '03_filter_safety_rescue',
         '04_success_vs_safety', '05_robustness_groups', '06_final_controllers')
TITLES = ('Подъём и удержание', 'DDPG: лучшие и последние веса',
          'Вмешательство фильтра', 'Удержание и безопасность',
          'Разные начальные условия', 'Семь контроллеров: общий старт')
PURPOSES = (*TITLES[:4], 'Примеры SAC для семи групп начальных условий',
            'Семь контроллеров на одном общем начальном состоянии')
REVIEW_SHA = '0c3b3685612f35da89332f8b36b165e96fd76d0056f12bf647a34e5d57f019e3'
RECOVERY_SHA = 'b2390e4f695ce45b2549423a2a591bf65e0ba0c5351771182e042bbb09496f18'
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
BOLD_FONT = FONT.with_name('DejaVuSans-Bold.ttf')
BUILD_COMMAND = ('.venv-cloud-check/bin/python -B -m cartpole.experiments.robustness_video '
    '--review artifacts/robustness-20260916/robustness-review-1789558151385076965.tar.gz '
    '--recovery artifacts/robustness-20260916/robustness-recovery-1789558163887525846.tar.gz '
    '--historical-media ../cartpole-coursework-submission/media '
    '--historical-episodes ../cartpole-intermediate-report-20260916/sources/episodes '
    '--confirmation-analysis verification/confirmation_audit_1789515957912724921/analysis '
    '--output reports/research-report-20260917/media --verify-repeat')
require = a.require


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def identity(row):
    return row['model_id'], row['case_id'], row['filter']


def select(rows):
    """Only metadata/metrics select clips; never inspect appearance or rerun a policy."""
    require(rows and all(r['stage'] == 'main' for r in rows), 'main only; timing forbidden')
    index = a.unique(rows, identity, 'episode identity')
    pairs = {}
    for r in rows:
        require(r['filter'] in ('off', 'on'), 'unknown filter')
        require(r['case_id'].rsplit('_', 1)[0] == r['group'], 'case/group mismatch')
        pairs.setdefault((r['model_id'], r['case_id']), {})[r['filter']] = r
    for pair in pairs.values():
        require(set(pair) == {'off', 'on'}, 'missing paired episode')
        for k in ('state_sha256', 'model_sha256', 'training_seed', 'group'):
            a.exact(pair['off'][k], pair['on'][k], 'pair '+k)
    rescue, trade = [], []
    for key, pair in sorted(pairs.items()):
        off, on = (pair[m]['metrics'] for m in ('off', 'on'))
        unsafe = off['safety']['exceeds_024_resolved']
        if (unsafe or 'position_limit' in off['completion']['stop_reasons']) and not on['safety']['exceeds_024_resolved'] and on['success_episode']:
            rescue.append(key)
        if unsafe and off['success_episode'] and not on['success_episode'] and not on['safety']['exceeds_024_resolved']:
            trade.append(key)
    require(rescue and len(trade) == 7, 'expected rescue candidates and exactly seven trade-off pairs')
    def get(mid, cid, mode='on'):
        require((mid, cid, mode) in index, 'missing required episode: '+str((mid,cid,mode)))
        return index[mid, cid, mode]
    # Incomplete evidence fails closed. We do not silently skip nominal_00.
    sets = [[get(m, 'nominal_00') for m in ('classical','SAC_seed0','TQC_seed0')], [],
            [get(*rescue[0], m) for m in ('off','on')],
            [get(*trade[0], m) for m in ('off','on')],
            [get('SAC_seed0', g+'_00') for g in GROUPS],
            [get(m, 'joint_00') for m in MODELS]]
    return sets, dict(rescue_candidates=len(rescue), rescue_pair=list(rescue[0]),
                      tradeoff_candidates=len(trade), tradeoff_pairs=[list(k) for k in trade])


def selected_payloads(path, expected):
    """Safe streaming, bounded selected gzip members; never extract tar members."""
    found = {}; seen = set(); total = 0
    with tarfile.open(path, 'r|gz') as archive:
        for info in archive:
            name = a.safe_name(info.name.rstrip('/'))
            require(name not in seen, 'duplicate archive path: '+name); seen.add(name)
            require(info.isdir() or info.isfile(), 'non-regular archive member')
            if info.isdir():
                continue
            total += info.size
            require(info.size <= a.MAX_MEMBER and total <= a.MAX_TOTAL, 'archive size limit')
            if name in expected:
                with archive.extractfile(info) as stream:
                    payload = stream.read(a.MAX_MEMBER+1)
                require(len(payload) == info.size, 'truncated member')
                require(hashlib.sha256(payload).hexdigest() == expected[name], 'selected log SHA: '+name)
                found[name] = a.decode_raw_log(payload)[0]
    require(set(found) == set(expected), 'missing selected logs: '+str(sorted(set(expected)-set(found))))
    return found


def unwrap_angles(values, *, wrapped=False):
    """Logs use unwrapped radians. Explicit wrapped inputs use shortest increments.

    Never guess that a full revolution in an unwrapped log is a branch cut.
    Exactly pi increments in wrapped input are ambiguous and rejected.
    """
    require(values and all(math.isfinite(x) for x in values), 'non-finite/empty angles')
    if not wrapped:
        return list(values)
    result = [values[0]]
    for before, after in zip(values, values[1:]):
        delta = math.remainder(after-before, 2*math.pi)
        require(abs(abs(delta)-math.pi) > 1e-12, 'ambiguous wrapped angle increment')
        result.append(result[-1]+delta)
    return result


@dataclass
class Episode:
    row: dict
    log: dict
    evidence: dict
    label: str

    def __post_init__(self):
        self.states = self.log['states']; self.transitions = self.log['transitions']
        self.times = [s['time'] for s in self.states]
        self.angles = unwrap_angles([s['pole_angle'] for s in self.states])
        self.end = self.times[-1]

    def sample(self, timestamp):
        require(math.isfinite(timestamp) and timestamp >= 0, 'invalid display time')
        t = min(timestamp, self.end)
        i = bisect_right(self.times, t)-1
        require(i >= 0, 'display before first state')
        s = self.states[i]
        if i == len(self.states)-1:
            return dict(x=s['cart_position'], theta=self.angles[i], time=t, intervention=False, ended=True)
        f = (t-self.times[i])/(self.times[i+1]-self.times[i])
        nxt = self.states[i+1]
        sf = self.transitions[i].get('safety_filter')
        changed = bool(sf and sf['feasible'] and sf['intervened'])
        return dict(x=s['cart_position']+f*(nxt['cart_position']-s['cart_position']),
                    theta=self.angles[i]+f*(self.angles[i+1]-self.angles[i]),
                    time=t, intervention=changed, ended=False)

    def outcome(self):
        """Presentation only; never change the saved success/safety metrics."""
        return self.final_status()[0]['text']

    def final_components(self):
        """Last recorded state against recorded hold thresholds, not interval causality."""
        s = self.states[-1]; th = self.log['metadata']['upright_thresholds']
        delta = s['pole_angle']-math.pi
        values = dict(angle=abs(math.atan2(math.sin(delta), math.cos(delta))),
                      angular_velocity=abs(s['pole_angular_velocity']),
                      position=abs(s['cart_position']), velocity=abs(s['cart_velocity']))
        require(set(th)==set(values) and all(math.isfinite(x) and x>0 for x in th.values()),
                'invalid saved hold thresholds')
        return {k:dict(value=x, threshold=th[k], exceeds=x>th[k]) for k,x in values.items()}

    def final_status(self, *, explain_components=False):
        m = self.row['metrics']
        unsafe = m['safety']['exceeds_024_resolved']
        reasons = m['completion']['stop_reasons']
        first = 'Удержание выполнено' if m['success_episode'] else 'Удержание не выполнено'
        role = 'good' if m['success_episode'] and not unsafe else 'hold'
        if reasons or m['completion']['terminated']:
            role = 'warning'
            if reasons == ['position_limit']:
                limit = self.log['metadata']['config']['max_position']
                require(math.isfinite(limit) and limit>0, 'invalid saved position limit')
                first = 'Остановка: предел '+format(limit, '.12g').replace('.', ',')+' м'
            elif reasons == ['velocity_limit']:
                first = 'Остановка: предел скорости'
            elif any('filter' in r for r in reasons):
                first = 'Остановка: отказ фильтра'
            else:
                first = 'Остановка эпизода'
        lines = [dict(text=first, role=role), dict(
            text='Граница нарушена' if unsafe else 'Без превышения границы',
            role='warning' if unsafe else 'good')]
        if explain_components and not m['success_episode'] and not m['completion']['terminated']:
            failed = [k for k,c in self.final_components().items() if c['exceeds']]
            labels = dict(velocity='Скорость тележки выше порога', position='Положение тележки вне порога',
                          angle='Угол маятника вне порога', angular_velocity='Угловая скорость выше порога')
            if failed:
                note = labels[failed[0]] if len(failed)==1 else 'Несколько условий удержания не выполнены'
                lines.append(dict(text=note, role='warning'))
        return lines

    def manifest(self):
        r = self.row; c = self.log['completion']
        events = [dict(start=t['time_start'], end=t['time_end']) for t in self.transitions
                  if t.get('safety_filter') and t['safety_filter']['feasible'] and t['safety_filter']['intervened']]
        return dict(controller=self.log['metadata']['algorithm'], model_id=r['model_id'], seed=r['training_seed'],
            filter=r['filter'], group=r['group'], case_id=r['case_id'],
            initial_state={k:self.states[0][k] for k in a.FIELDS}, model_sha256=r['model_sha256'],
            outcome=self.outcome(), success=r['metrics']['success_episode'], completion=c,
            final_status=self.final_status(explain_components='checkpoint_label' in self.evidence),
            final_components=self.final_components(),
            hold_final=r['metrics']['hold_final'], hold_longest=r['metrics']['hold_longest'],
            exceeds_024_resolved=r['metrics']['safety']['exceeds_024_resolved'],
            saved_states=len(self.states), saved_transitions=len(self.transitions),
            intervention_intervals=events, evidence=self.evidence, log_sha256=r['log_sha256'])


def validate_log(log, row, initial, spec=None):
    """Bind saved frames to checked receipts; no reward/dynamics/filter recomputation."""
    meta = log['metadata']; states = log['states']; transitions = log['transitions']
    algo = 'classical' if row['model_id']=='classical' else row['model_id'].split('_')[0]
    a.exact([meta['algorithm'],meta['training_seed'],meta['evaluation_case_id'],meta['filter_mode']],
            [algo,row['training_seed'],row['case_id'],row['filter']], 'log identity metadata')
    require(meta['reset_mode']=='explicit', 'non-explicit reset')
    require(states and len(states)==len(transitions)+1, 'state/transition count')
    for source in (meta['initial_state'], states[0]):
        a.exact({k:source[k] for k in a.FIELDS}, initial, 'initial state')
    require(states[0]['time']==0, 'episode must start at zero')
    a.exact(log['completion'],row['metrics']['completion'], 'saved completion')
    for key in ('success_episode','hold_final','hold_longest'):
        a.exact(log['evaluation'][key],row['metrics'][key], 'saved '+key)
    require(log['completion']['time']==states[-1]['time'], 'end time')
    require(log['completion']['transition_count']==len(transitions), 'completion count')
    count = 0
    for i,s in enumerate(states):
        require(s['index']==i, 'state index')
        require(all(isinstance(s[k],(int,float)) and math.isfinite(s[k]) for k in (*a.FIELDS,'time')), 'non-finite state')
    for i,t in enumerate(transitions):
        a.exact([t['index'],t['from_state'],t['to_state']], [i,i,i+1], 'transition indices')
        a.exact([t['time_start'],t['time_end']], [states[i]['time'],states[i+1]['time']], 'transition time')
        require(t['dt_actual']>=0 and abs(t['time_end']-t['time_start']-t['dt_actual'])<1e-12, 'dt mismatch')
        require(t['dt_actual']>0 or i==len(transitions)-1, 'zero time before final event')
        sf=t.get('safety_filter')
        require(bool(sf)==(row['filter']=='on'), 'transition filter metadata')
        if sf:
            require(type(sf['intervened']) is bool and type(sf['feasible']) is bool, 'filter flag type')
            count += int(sf['feasible'] and sf['intervened'])
    require(count==row['metrics']['safety']['intervention_count'], 'intervention count')
    if spec:
        env=spec['environment']['specification']
        for key,value in dict(config=env['physical_config'],integrator=env['integrator'],reward=env['reward'],
                              control_interval=.01,horizon=10.,required_hold_duration=2.,
                              upright_thresholds=dict(angle=math.pi/18,angular_velocity=.5,position=.2,velocity=.2)).items():
            a.exact(meta[key],value,'metadata science '+key)
        if algo=='classical':
            require(meta['environment']=='CourseworkCartPole-v0' and 'training_contract' not in meta, 'classical actual API')
        else:
            a.exact(meta['training_contract'],spec['environment'],'training contract')
        if row['filter']=='on':
            a.exact(meta['safety_filter'],env['filter']['limits'],'safety metadata')
    require(meta['config']['pole_length']==.18, 'unexpected pole geometry')


def common_start(episodes, *, same_model=False):
    require(episodes, 'empty comparison')
    first = episodes[0]
    for ep in episodes[1:]:
        a.exact(ep.row['case_id'],first.row['case_id'],'common case')
        a.exact({k:ep.states[0][k] for k in a.FIELDS}, {k:first.states[0][k] for k in a.FIELDS},'common initial state')
        if same_model:
            a.exact(ep.row['model_sha256'], first.row['model_sha256'], 'paired model')


def historical(media, episodes, audit):
    """Check all four old MP4s; DDPG binds to external audit rows, NOT names."""
    entries=a.read_json(media/'manifest.json'); require(len(entries)==5,'legacy manifest roster')
    output=[]; verified=[]
    with (audit/'episodes.csv').open() as f:
        audited=list(csv.DictReader(f))
    checkpoints=a.read_json(audit/'selected_checkpoints.json')
    for e in entries:
        for key in ('video','archived_log_name'):
            require(Path(e[key]).name==e[key], 'unsafe historical filename')
        require(a.file_sha(media/e['video'])==e['video_sha256'], 'legacy video SHA')
        payload=(episodes/e['archived_log_name']).read_bytes()
        require(hashlib.sha256(payload).hexdigest()==e['archived_log_sha256'], 'legacy log SHA')
        log=a.decode_raw_log(payload)[0]; receipt_path=episodes/(e['id']+'_receipt.json')
        receipt=a.read_json(receipt_path)
        require(receipt['log_sha256']==e['archived_log_sha256'], 'legacy receipt log SHA')
        algorithm=log['metadata']['algorithm']
        row=dict(stage=e['phase'], model_id=('classical' if algorithm=='classical' else algorithm+'_seed'+str(e['seed'])),
            training_seed=e['seed'],case_id=e['case'],group=receipt['case']['subset'],filter='on',
            model_sha256=e.get('model_sha256'), metrics=receipt['metrics'],log_sha256=e['archived_log_sha256'])
        validate_log(log,row,e['initial_state'])
        for k in ('success','hold_final','hold_longest','duration','frames_source','interventions'):
            actual=dict(success=receipt['metrics']['success_episode'],hold_final=receipt['metrics']['hold_final'],
                hold_longest=receipt['metrics']['hold_longest'],duration=log['completion']['time'],
                frames_source=len(log['states']),interventions=receipt['metrics']['safety']['intervention_count'])[k]
            a.exact(e[k],actual,'legacy manifest '+k)
        info=dict(id=e['id'],phase=e['phase'],job=e['job'],mode=e['mode'],seed=e['seed'],
            case_id=e['case'],initial_state=e['initial_state'],model_sha256=e.get('model_sha256'),
            checkpoint=e['checkpoint'],transitions=e['transitions'],log_sha256=e['archived_log_sha256'],
            old_video=e['video'],old_video_sha256=e['video_sha256'],receipt_sha256=a.file_sha(receipt_path))
        if algorithm=='DDPG':
            require(e['phase']=='confirmation' and e['seed']==5 and e['case']=='validation_1000', 'DDPG origin')
            match=[r for r in audited if r['category']=='external' and r['run_id']==e['job']
                   and r['checkpoint_label']==e['checkpoint'] and r['evaluation_filter']=='on' and r['case_id']==e['case']]
            require(len(match)==1,'ambiguous/missing DDPG audit'); ar=match[0]
            for key in ('model_sha256','log_sha256','receipt_sha256'):
                require(ar[key]==info[key],'DDPG audit '+key)
            cp=[r for r in checkpoints if r['run_id']=='train_confirm_ddpg_on_warmup5000_seed5' and r['checkpoint_label']==e['checkpoint']]
            require(len(cp)==1 and cp[0]['model_sha256']==e['model_sha256'] and cp[0]['transitions']==e['transitions'], 'DDPG checkpoint audit')
            info['checkpoint_manifest_sha256']=cp[0]['checkpoint_manifest_sha256']
            evidence=dict(log='historical_episodes/'+e['archived_log_name'],receipt='historical_episodes/'+receipt_path.name,
                receipt_sha256=info['receipt_sha256'],audit='confirmation_analysis/episodes.csv',audit_log_source=ar['source'],
                model_sha256=e['model_sha256'],checkpoint_manifest_sha256=cp[0]['checkpoint_manifest_sha256'],
                checkpoint_label=e['checkpoint'],training_transitions=e['transitions'])
            output.append(Episode(row,log,evidence,'DDPG · '+('лучшие веса' if e['checkpoint']=='best' else 'последние веса')))
        verified.append(info)
    require(len(output)==2 and {e.evidence['checkpoint_label'] for e in output}=={'best','last'}, 'DDPG two versions')
    output.sort(key=lambda e:e.evidence['checkpoint_label'])
    require(output[0].row['model_sha256']!=output[1].row['model_sha256'], 'DDPG distinct weights expected')
    common_start(output)
    return output,verified


def prepare(args):
    protocol=a.Protocol()
    review=a.load_export(args.review,REVIEW_SHA); normalized=a.import_compact(review,protocol)
    sets,selection=select(normalized['episodes'])
    print('Evidence: checking recovery hashes/receipts (no raw metric audit)',flush=True)
    recovery=a.load_export(args.recovery,RECOVERY_SHA)
    a.compare_exports(review,recovery); a.recovery_bindings(recovery,protocol)
    require(a.import_compact(recovery,protocol)['scientific_sha256']==normalized['scientific_sha256'],'science binding')
    rows={identity(r):r for group in sets for r in group}; paths={}; expected={}
    for key,r in rows.items():
        path=f"queue/jobs/{r['job_id']}/evaluation/cases/{r['case_id']}/attempt_00000/episode.json.gz"
        expected[path]=r['log_sha256']; paths[key]=path
    payloads=selected_payloads(args.recovery,expected); bound={}
    for key,r in rows.items():
        ep=f"queue/jobs/{r['job_id']}/evaluation"; path=paths[key]
        spec=recovery.documents[ep+'/specification.json']
        validate_log(payloads[path],r,protocol.cases[r['case_id']]['initial_state'],spec)
        label='Классика' if r['model_id']=='classical' else r['model_id'].replace('_seed',' · запуск ')
        bound[key]=Episode(r,payloads[path],dict(archive='main_recovery',log=path,
            receipt=path.replace('episode.json.gz','receipt.json'),
            receipt_sha256=recovery.hashes[path.replace('episode.json.gz','receipt.json')],
            specification=ep+'/specification.json',specification_sha256=recovery.hashes[ep+'/specification.json']),label)
    videos=[[bound[identity(r)] for r in group] for group in sets]
    videos[1],old=historical(args.historical_media,args.historical_episodes,args.confirmation_analysis)
    for i in (0,1,2,3,5): common_start(videos[i],same_model=i in (2,3))
    return videos,dict(scientific_sha256=normalized['scientific_sha256'],selection=selection,
        main_review=review.evidence,main_recovery=recovery.evidence,historical_videos=old,
        selected_unique_main_logs=len(rows),verification=dict(archive_integrity='passed',compact_consistency='passed',
        selected_log_binding='passed',full_raw_metric_recomputation='not_run',models_loaded=False,rollouts=0),
        classical_caveat='Specification has SAC-like environment contract; actual saved classical logs use CourseworkCartPole-v0 without training contract. Rendering uses actual saved coordinates.')


def segments(episodes, sequential=False):
    offset=0; result=[]
    groups=[[e] for e in episodes] if sequential else [episodes]
    for i,group in enumerate(groups):
        if sequential:
            result.append(dict(start=offset,frames=FPS,kind='title',title=GROUP_TITLES[i],episodes=[])); offset+=FPS
        motion=math.ceil(max(e.end for e in group)*FPS)
        result.append(dict(start=offset,frames=motion+HOLD_FRAMES,kind='motion',title=GROUP_TITLES[i] if sequential else None,
                           episodes=group,motion_frames=motion,display_extent=comparison_extent(group)))
        offset+=motion+HOLD_FRAMES
    return result


def display_time(segment, local_frame):
    return min(local_frame/FPS,max(e.end for e in segment['episodes']))


def screen_episodes(episodes):
    """Presentation permutation only; frozen selection roster remains untouched."""
    if len(episodes)!=7:
        return list(episodes)
    by_model = a.unique(episodes, lambda e:e.row['model_id'], 'screen model')
    require(set(by_model)==set(SCREEN_MODELS), 'screen roster')
    return [by_model[m] for m in SCREEN_MODELS]


def model_color(ep):
    require(ep.row['model_id'] in MODEL_COLORS, 'unknown model color')
    return MODEL_COLORS[ep.row['model_id']]


def comparison_extent(episodes):
    """Conservative enclosure of saved/display-interpolated geometry, not dynamics.

    Bound x and sin(theta) independently within each pair of recorded samples,
    including internal sine extrema. One common isotropic scale per comparison.
    """
    extent=.34  # includes desired-boundary labels with panel padding
    for ep in episodes:
        length=ep.log['metadata']['config']['pole_length']
        for left,right in zip(ep.states,ep.states[1:]):
            lo,hi=sorted((left['pole_angle'],right['pole_angle']))
            sins=[math.sin(lo),math.sin(hi)]
            if hi-lo>=2*math.pi:
                sins.extend((-1.,1.))
            else:
                for k in range(math.ceil((lo-math.pi/2)/math.pi),math.floor((hi-math.pi/2)/math.pi)+1):
                    sins.append(1. if k%2==0 else -1.)
            xlo,xhi=sorted((left['cart_position'],right['cart_position']))
            extent=max(extent,abs(xlo+length*min(sins))+.02,abs(xhi+length*max(sins))+.02,
                       abs(xlo)+.055,abs(xhi)+.055)
        if len(ep.states)==1:
            extent=max(extent,abs(ep.states[0]['cart_position'])+length+.02)
    return extent


class Painter:
    def __init__(self):
        from PIL import Image, ImageDraw, ImageFont
        self.Image=Image; self.ImageDraw=ImageDraw; self.ImageFont=ImageFont
        self.fonts={}; self.ink='#19364b'; self.gray='#647687'; self.bg='#f6f8fa'

    def font(self,size,bold=False):
        key=(size,bold)
        if key not in self.fonts:self.fonts[key]=self.ImageFont.truetype(str(BOLD_FONT if bold else FONT),size)
        return self.fonts[key]

    def text(self,draw,xy,text,size=30,fill=None,bold=False,anchor=None):
        draw.text(xy,text,font=self.font(size,bold),fill=fill or self.ink,anchor=anchor)

    def panel(self,draw,box,ep,t,extent):
        x0,y0,x1,y1=box; w=x1-x0; h=y1-y0; small=w<500
        draw.rounded_rectangle(box,radius=18,fill='white',outline='#dce4eb',width=2)
        size=24 if small else 32
        self.text(draw,(x0+24,y0+20),ep.label,size,bold=True)
        self.text(draw,(x0+24,y0+62),'Фильтр '+('включён' if ep.row['filter']=='on' else 'выключен'),size-3,fill=self.gray)
        s=ep.sample(t)
        self.text(draw,(x1-24,y0+105),f"t = {s['time']:.3f} с",size-2,anchor='ra')
        # Same physical scale in all panels of a comparison. y screen points down.
        scale=min((w-48)/(2*extent),(h-260)/.52); cx=(x0+x1)/2; cy=y0+150+(h-270)/2
        rail_y=cy+max(10,scale*.034)
        draw.line((cx-(extent-.01)*scale,rail_y,cx+(extent-.01)*scale,rail_y),fill='#aebac4',width=3)
        for side in (-1,1):
            bx=cx+side*.24*scale
            for yy in range(int(cy-.21*scale),int(cy+.22*scale),12):
                draw.line((bx,yy,bx,min(yy+5,cy+.22*scale)),fill='#b87c49',width=2)
            label='x = −0,24 м' if side<0 else 'x = +0,24 м'
            self.text(draw,(bx,cy+.22*scale+9),label,19 if small else 25,fill='#86613f',anchor='mt')
        px=cx+s['x']*scale; py=cy
        bw=max(26,scale*.07); bh=max(14,scale*.03)
        color=model_color(ep)
        draw.rounded_rectangle((px-bw/2,py,px+bw/2,py+bh),radius=5,fill=color)
        for dx in (-bw*.30,bw*.30):
            rr=max(3,scale*.007);draw.ellipse((px+dx-rr,py+bh-rr,px+dx+rr,py+bh+rr),fill=self.ink)
        # theta=0 down, theta=pi up. Pole length is .18 m in both coordinates.
        bx=px+.18*math.sin(s['theta'])*scale; by=py+.18*math.cos(s['theta'])*scale
        draw.line((px,py,bx,by),fill=self.ink,width=5 if small else 7)
        radius=7 if small else 11
        draw.ellipse((bx-radius,by-radius,bx+radius,by+radius),fill=color)
        draw.ellipse((px-5,py-5,px+5,py+5),fill='white',outline=self.ink,width=2)
        if s['intervention']:
            self.text(draw,((x0+x1)/2,y1-86),'Вмешательство фильтра',20 if small else 28,fill='#b36525',anchor='mt')
        if s['ended']:
            lines=ep.final_status(explain_components='checkpoint_label' in ep.evidence)
            for j,line in enumerate(lines):
                self.text(draw,((x0+x1)/2,y1-34*(len(lines)-j)-18),line['text'],
                          21 if small else 28,fill=STATUS_COLORS[line['role']],anchor='mt')

    def frame(self,index,segment,local):
        im=self.Image.new('RGB',(WIDTH,HEIGHT),self.bg);d=self.ImageDraw.Draw(im)
        if segment['kind']=='title':
            self.text(d,(WIDTH/2,HEIGHT/2),segment['title'],58,bold=True,anchor='mm');return im
        self.text(d,(60,40),segment['title'] or TITLES[index],42,bold=True)
        eps=screen_episodes(segment['episodes']); n=len(eps); t=display_time(segment,local)
        extent=segment['display_extent']
        if n==7: cols,rows=4,2
        else:cols,rows=n,1
        gap=22; left=40;top=120; pw=(WIDTH-2*left-gap*(cols-1))/cols; ph=(HEIGHT-top-40-gap*(rows-1))/rows
        for i,ep in enumerate(eps):
            col=i%cols;row=i//cols;xx=left+col*(pw+gap);yy=top+row*(ph+gap)
            self.panel(d,(xx,yy,xx+pw,yy+ph),ep,t,extent)
        return im


def encode(path,index,episodes,painter):
    segs=segments(episodes,index==4);total=sum(s['frames'] for s in segs)
    command=['ffmpeg','-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}',
             '-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','18',
             '-threads','4','-pix_fmt','yuv420p','-movflags','+faststart','-map_metadata','-1',str(path)]
    h=hashlib.sha256()
    with tempfile.TemporaryFile() as errors:
        p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors)
        try:
            for s in segs:
                for j in range(s['frames']):
                    data=painter.frame(index,s,j).tobytes();h.update(data);p.stdin.write(data)
            p.stdin.close();require(p.wait()==0,'ffmpeg encoding failed')
        except BaseException:
            p.stdin.close();p.terminate();p.wait();errors.seek(0)
            print(errors.read().decode(errors='replace'),file=sys.stderr);raise
    return dict(frames=total,duration=total/FPS,rgb_frames_sha256=h.hexdigest())


def probe_and_decode(path,expected):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)]))
    require(len(probe['streams'])==1,'unexpected audio/extra stream');s=probe['streams'][0]
    require(s['codec_name']=='h264' and s['pix_fmt']=='yuv420p' and (s['width'],s['height'])==(WIDTH,HEIGHT),'video format')
    require(s['avg_frame_rate']=='30/1' and s['r_frame_rate']=='30/1','video fps')
    require(int(s['nb_read_frames'])==expected['frames'],'frame count')
    require(abs(float(probe['format']['duration'])-expected['duration'])<.001,'video duration')
    require(path.stat().st_size<50*1024**2,'video >=50 MiB')
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    # Parse MP4 top-level boxes, not a coincidental byte sequence in compressed data.
    boxes=[]
    with path.open('rb') as f:
        while f.tell()<path.stat().st_size:
            head=f.read(8);require(len(head)==8,'truncated MP4 box')
            size=int.from_bytes(head[:4],'big');kind=head[4:].decode('ascii')
            if size==1:size=int.from_bytes(f.read(8),'big');header=16
            else:header=8
            require(size>=header,'invalid MP4 box');boxes.append(kind);f.seek(size-header,1)
    require(boxes.index('moov')<boxes.index('mdat'),'missing faststart')
    return dict(codec=s['codec_name'],pixel_format=s['pix_fmt'],width=s['width'],height=s['height'],fps=30,
                frames=int(s['nb_read_frames']),duration=float(probe['format']['duration']),
                bytes=path.stat().st_size,sha256=a.file_sha(path),full_decode='passed',faststart=True)


def extract_frame(video,frame,path):
    subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(video),'-vf',f'select=eq(n\\,{frame})',
                    '-frames:v','1','-update','1',str(path)],check=True,stdout=subprocess.DEVNULL)


def qa_frames(path,index,episodes,total,directory):
    """Actual encoded frames, including every segment endpoint and sampled intervention."""
    frames={0:'start',total//2:'middle',total-1:'final'}
    for si,s in enumerate(segments(episodes,index==4)):
        if s['kind']!='motion':continue
        frames[s['start']]=f'segment_{si}_start'
        frames[s['start']+s['frames']-1]=f'segment_{si}_final'
        for ei,ep in enumerate(s['episodes']):
            if ep.log['completion']['terminated']:
                at=s['start']+math.ceil(ep.end*FPS)
                frames[at]=f'terminal_{si}_{ei}'
                frames[max(s['start'],at-1)]=f'preterminal_{si}_{ei}'
            # Show a real sampled decision. Do not lengthen/dilate short events.
            for j in range(s['motion_frames']):
                if ep.sample(j/FPS)['intervention']:
                    frames[s['start']+j]=f'intervention_{si}_{ei}';break
            for j in range(s['motion_frames']):
                if abs(ep.sample(j/FPS)['x'])>.24+1e-12:
                    frames[s['start']+j]=f'exceedance_{si}_{ei}';break
    directory.mkdir()
    result=[]
    for frame,label in sorted(frames.items()):
        dest=directory/f'{frame:05d}_{label}.png';extract_frame(path,frame,dest)
        result.append(dict(frame=frame,pts=frame/FPS,label=label,file=dest.name))
    return result


def readme_text():
    return '''# Шесть видеороликов по сохранённым траекториям

Это визуализация симуляционных данных, не новое испытание и не реальная установка.
Ни политика, ни фильтр, ни динамика не исполнялись. Timing40 не использован.

## Выбор

1. classical, SAC seed0, TQC seed0: on, общий nominal_00.
2. Исторический confirmation DDPG warmup5000 seed5: best 90000 и last 100000,
   validation_1000, on. Это **разные веса** и другой этап. SHA старых четырёх MP4,
   пяти журналов и их receipts проверены; DDPG дополнительно связан по SHA с
   external-строками принятого confirmation-аудита и selected_checkpoints.json.
3. Первая по (model_id,case_id) из 402 пар: SAC_seed0 / angle_01.
   off: resolved >0.24 либо position_limit; on: без resolved и успешен.
4. Первая из семи пар: SAC_seed0 / angle_16. off успешен, но имеет resolved
   >0.24; on без resolved, но без успеха. Отсутствие удержания не названо аварией.
5. Примеры SAC_seed0 для семи групп начальных условий: on, индекс00,
   в заданном порядке. Все семь выбранных примеров успешны.
6. Семь контроллеров на одном общем начальном состоянии joint_00, on.
   Верхний ряд: SAC0, SAC1, SAC2, классика; нижний: TQC0, TQC1, TQC2 и пустая
   ячейка. Все семь выбранных примеров успешны. Выбор эпизодов не менялся.

Лексикографический порядок стандартный Python (Unicode), не порядок успеха.
Неполные связи вызывают ошибку, автоматического эстетического fallback нет.
Клипы описывают выбранные примеры; они не заменяют агрегаты 1960 эпизодов.
№5/6 не доказывают 100% успеха серии или глобальную устойчивость. Не все
старты являются OOD: position_00 лежит на границе замкнутого training support.
Успех — исходный критерий полного горизонта и последних двух секунд удержания;
он отделён от превышения желаемой границы. Событие position_limit — ±0.25 м.
Пунктирные линии в видео — желаемые ±0.24 м, а не физическое событие.
Итоговые строки отделяют удержание от resolved-превышения (>0.24+1e-12 м),
показываются только после завершения; исходный success_episode не изменён.
В №4 удержание off выполнено при нарушенной границе; on не удерживает, но
не превышает её. Надпись о position_limit берёт предел из saved config.max_position.
Цвет закреплён за model_id во всём комплекте, а не за номером панели или режимом.

Происхождение прежних MP4 (они сохранены, кадры новых роликов рисуются заново
из связанных журналов):

| Старый MP4 | Источник | Сценарий / веса |
|---|---|---|
| classical_success | исходная main, evaluate_classical | bottom, on, фиксированная классика |
| sac_on_success | исходная main, SAC-on seed1 | bottom, best 50000, on |
| tqc_on_success | исходная main, TQC-on seed0 | bottom, best 80000, on |
| ddpg_failure_or_degradation | confirmation, DDPG warmup5000 seed5 | validation_1000, best 90000 / last 100000, on |

DDPG: final hold 7.9 с у best и 0 с у last; longest hold 7.9 и 1.16 с.
В последнем отсчёте last скорость тележки 0.243579 м/с при пороге 0.20 м/с,
хотя маятник близок к вертикали. Успех нельзя определять только на глаз по углу.
Пояснение под последними весами проверяет все четыре компоненты последнего
отсчёта против metadata.upright_thresholds: здесь превышена только скорость.
Это не утверждение о единственной причине неуспеха на всём интервале либо
о других DDPG-эпизодах. Данные и критерии не пересчитывались ради подписи.
Метаданные старых replay в results/ относятся к прежним E6-сериям (например,
SAC E6c seed1, step90000, bottom), не к этим frozen robustness checkpoints.
Их траектории не подставляются вместо требуемых main-сценариев.

## Время и геометрия

Точные x, theta, v, omega и time прочитаны из states. Переходы связываются по
индексам и time_start/time_end/dt_actual. theta=0 вниз, pi вверх; длина 0.18 м.
Геометрия: (x+0.18 sin(theta), -0.18 cos(theta)); одинаковый масштаб осей.
Масштаб общий внутри сравнения; границы экранной геометрии рассчитаны по
сохранённым x/theta и синусным экстремумам отображаемой интерполяции. Поэтому
№1 использует большую установку без обрезки траектории или растяжения осей.
Полные обороты исходного неограниченного угла сохраняются. Только экранные x
и theta линейно интерполируются между соседними отсчётами; в данных ничего
не изменяется. Wrapped-ввод требует явного режима, ровно pi неоднозначно.
Дискретные события и индикаторы не интерполируются: вмешательство действует
на [time_start,time_end). По завершении панель застывает на точном конечном
состоянии, её часы останавливаются; другие панели продолжаются синхронно.
Точное конечное время (включая укороченный переход) сохранено в manifest;
экранное время округлено до миллисекунды. При CFR30 событие впервые видно на
следующем кадре, задержка меньше 1/30 с; короткое вмешательство может полностью
попасть между кадрами. Интервалы вмешательств в manifest сохраняются точно,
они не растягиваются ради заметности. Нулевой переход не создаёт движения;
при равном времени берётся последний записанный отсчёт. Финальная выдержка
2 с. В обзоре групп — отдельная секундная заставка перед каждым фрагментом.
Видео само по себе не является проверкой экстремумов между отсчётами;
resolved-метрики взяты из проверенных свидетельств, без нового raw-аудита.

## Воспроизведение

Из корня checkout (Pillow, ffmpeg/ffprobe и DejaVu Sans уже должны быть доступны):

```bash
.venv-cloud-check/bin/python -B -m cartpole.experiments.robustness_video \\
  --review artifacts/robustness-20260916/robustness-review-1789558151385076965.tar.gz \\
  --recovery artifacts/robustness-20260916/robustness-recovery-1789558163887525846.tar.gz \\
  --historical-media ../cartpole-coursework-submission/media \\
  --historical-episodes ../cartpole-intermediate-report-20260916/sources/episodes \\
  --confirmation-analysis verification/confirmation_audit_1789515957912724921/analysis \\
  --output reports/research-report-20260917/media --verify-repeat
```

Существующий output отвергается. Для повторной сборки задайте новый output;
явный --overwrite сохраняет прежний каталог под отдельным backup-именем.
--check-only выполняет чтение/сверку без рендера и без создания каталогов.
Входные архивы проверяются по закреплённому внешнему SHA и всем внутренним
manifest SHA; читаются потоково. Декодируются только выбранные журналы.
Модели/pickle/replay не загружаются. Временный каталог создаётся вне исходных
данных и удаляется после завершения; output публикуется переименованием.
Кодировщик: libx264 veryfast, CRF18, четыре потока. Одновременно запускать два
рендера с одним output нельзя. Публикация media и отдельного QA-каталога не
является общей файловой транзакцией; существующий путь проверяется повторно
перед переименованием, символические ссылки отвергаются.

В VIDEO_MANIFEST пути evidence относительны к логическим корням main_recovery,
historical_episodes и confirmation_analysis из команды выше. Это ссылки на
локальные исходные свидетельства, не обещание наличия полных логов в пакете.
Точная версия рендера — SHA модуля, базовый Git HEAD и SHA зависимых исходников.
Историческая оговорка classical: specification содержит SAC-подобный контракт,
но реальные логи используют CourseworkCartPole-v0 без training contract;
визуализация использует фактически записанные координаты.

QA_REPORT.md описывает фактически выполненные технические и визуальные проверки.
'''


def build(args):
    require(not args.output.is_symlink(),'output must not be a symlink')
    require(not args.output.exists() or args.overwrite or args.check_only,'output already exists (choose a new path)')
    inputs=[args.review,args.recovery,args.historical_media/'manifest.json',
            args.confirmation_analysis/'episodes.csv',args.confirmation_analysis/'selected_checkpoints.json']
    inputs += sorted(args.historical_episodes.glob('*.json*'))+sorted(args.historical_media.glob('*.mp4'))
    before={str(p):a.file_sha(p) for p in inputs}
    videos,evidence=prepare(args)
    if args.check_only:
        require(before=={str(p):a.file_sha(p) for p in inputs},'input changed')
        print(json.dumps(dict(status='passed',**evidence),ensure_ascii=False,indent=2));return
    require(FONT.is_file() and BOLD_FONT.is_file(),'DejaVu fonts unavailable')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # All products staged separately. An encoding/check failure never publishes a partial set.
    with tempfile.TemporaryDirectory(prefix='.video-build-',dir=args.output.parent) as scratch:
        scratch=Path(scratch);stage=scratch/'media';stage.mkdir();(stage/'posters').mkdir()
        qa=scratch/'qa_frames';qa.mkdir();painter=Painter();records=[];qa_records={}
        for i,(name,episodes) in enumerate(zip(NAMES,videos)):
            print('Render '+name,flush=True);path=stage/(name+'.mp4')
            rendered=encode(path,i,episodes,painter);checked=probe_and_decode(path,rendered)
            if args.verify_repeat:
                print('Verify repeated render '+name,flush=True)
                repeat=scratch/(name+'.mp4');again=encode(repeat,i,episodes,painter)
                require(again==rendered and a.file_sha(repeat)==checked['sha256'],'repeat render differs')
                repeat.unlink()
            qa_records[name]=qa_frames(path,i,episodes,rendered['frames'],qa/name)
            extract_frame(path,rendered['frames']-1,stage/'posters'/(name+'.png'))
            shown=screen_episodes(episodes) if i!=4 else episodes
            records.append(dict(file=name+'.mp4',purpose=PURPOSES[i],**checked,
                rgb_frames_sha256=rendered['rgb_frames_sha256'],repeat_bit_identical=bool(args.verify_repeat),
                episodes=[ep.manifest() for ep in episodes],poster='posters/'+name+'.png',
                screen_order=[dict(panel=k, row=k//4 if len(shown)==7 and i!=4 else 0,
                    column=k%4 if len(shown)==7 and i!=4 else (0 if i==4 else k),
                    fragment=k if i==4 else None, model_id=ep.row['model_id'],filter=ep.row['filter'],
                    case_id=ep.row['case_id'],model_sha256=ep.row['model_sha256'],
                    log_sha256=ep.row['log_sha256'],color=model_color(ep)) for k,ep in enumerate(shown)],
                renderer_version=VERSION,renderer_sha256=a.file_sha(Path(__file__)),
                build_command=BUILD_COMMAND,
                timeline=[{k:s[k] for k in ('start','frames','kind','title')}
                          for s in segments(episodes,i==4)],
                poster_sha256=a.file_sha(stage/'posters'/(name+'.png'))))
            print('Verified '+name,flush=True)
        require(before=={str(p):a.file_sha(p) for p in inputs},'input changed during build')
        code={'cartpole/experiments/robustness_video.py':a.file_sha(Path(__file__)),
              'cartpole/experiments/robustness_analysis.py':a.file_sha(Path(a.__file__))}
        from PIL import __version__ as pillow_version
        manifest=dict(schema=VERSION,source_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.ROOT,text=True).strip(),
            code_sha256=code,font_sha256={f.name:a.file_sha(f) for f in (FONT,BOLD_FONT)},
            environment=dict(python=sys.version.split()[0],pillow=pillow_version,
                ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]),
            build_command=BUILD_COMMAND,
            encoding=dict(codec='libx264',preset='veryfast',crf=18,threads=4,pixel_format='yuv420p',audio=False,faststart=True),
            input_aliases=dict(main_review=args.review.name,main_recovery=args.recovery.name,
                historical_media='historical_media',historical_episodes='historical_episodes',confirmation_analysis='confirmation_analysis'),
            evidence=evidence,auxiliary_sha256={p.name:before[str(p)] for p in inputs if p not in (args.review,args.recovery)},
            input_unchanged=True,videos=records,qa_extracted_frames=qa_records)
        write_json(stage/'VIDEO_MANIFEST.json',manifest);(stage/'README.md').write_text(readme_text())
        (stage/'QA_REPORT.md').write_text('# Техническая проверка\n\nВсе 6 MP4: H.264, yuv420p, 1920×1080, 30 fps, без звука;\n'
            'ffprobe count_frames и длительность совпали с расписанием; полное декодирование ffmpeg -xerror прошло;\n'
            'moov перед mdat (faststart); каждый файл меньше 50 МиБ.\n\n'
            +('Повторно отрендерены все шесть роликов: SHA MP4 и потоков RGB совпали побайтово.\n' if args.verify_repeat else 'Повторный рендер не запускался.\n')
            +'\nИсходные архивы и исторические входы сохранили SHA. Модели не загружались, rollout нет.\n'
            'Полный новый пересчёт raw-метрик не выполнялся. Проверены compact/receipt bindings и выбранные логи.\n'
            '\nКадры начала, середины, каждого финала, границы и вмешательства извлечены из MP4.\n'
            '**Визуальная проверка: ожидает отдельного просмотра извлечённых кадров.**\n')
        publish_output(stage,qa,args.output,args.overwrite)
    print(json.dumps(dict(output=str(args.output),videos=len(records),input_unchanged=True),ensure_ascii=False),flush=True)


def publish_output(stage, qa, output, overwrite=False):
    """Single writer per destination. Recheck after the long render, before rename.

    This is not a concurrent-writer transaction across media and QA directories.
    A destination appearing during rendering is rejected without --overwrite.
    Explicit replacement keeps both old directories as backups.
    """
    require(not output.is_symlink(),'output must not be a symlink')
    qa_dest=output.parent/(output.name+'-qa-frames')
    require(not qa_dest.is_symlink(),'QA must not be a symlink')
    require(not output.exists() or overwrite,'output appeared during rendering')
    require(not qa_dest.exists() or (overwrite and output.exists()),'QA directory already exists')
    if output.exists():
        require(output.is_dir() and (output/'VIDEO_MANIFEST.json').is_file(),'not a video output directory')
        backup=output.with_name(output.name+'.backup-'+a.file_sha(output/'VIDEO_MANIFEST.json')[:12])
        qa_backup=qa_dest.with_name(backup.name+'-qa-frames')
        require(not backup.exists() and not backup.is_symlink() and not qa_backup.exists()
                and not qa_backup.is_symlink(),'backup exists')
        output.rename(backup)
        if qa_dest.exists():qa_dest.rename(qa_backup)
    qa.rename(qa_dest);stage.rename(output)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--historical-media',type=Path,required=True);p.add_argument('--historical-episodes',type=Path,required=True)
    p.add_argument('--confirmation-analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-repeat',action='store_true');p.add_argument('--check-only',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args(argv)
    try:build(args)
    except (ValueError,KeyError,OSError,subprocess.SubprocessError) as exc:p.exit(1,'video build failed: '+str(exc)+'\n')


if __name__=='__main__':
    main()
