"""Synthetic saved records; no local experiment archives or scientific runtime."""
import argparse
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from cartpole.experiments import robustness_video as v
from tests.test_robustness_reporting import pair, change_safety


def selection_fixture():
    rows=[]
    for model in v.MODELS:
        for group in v.GROUPS:
            rows+=pair(model,group)
    # Seven tradeoffs, deliberately reverse numerical insertion order.
    for i in reversed(range(1,8)):
        ps=pair('SAC_seed0','angle',i,(True,False));change_safety(ps[0],max_abs_position_interval=.245)
        rows+=ps
    ps=pair('SAC_seed0','angle',8,(False,True));change_safety(ps[0],max_abs_position_interval=.25);rows+=ps
    ps=pair('TQC_seed0','angle',9,(False,True));change_safety(ps[0],max_abs_position_interval=.25);rows+=ps
    return rows


def episode(mode='on',end=.1):
    state={k:0. for k in v.a.FIELDS}
    states=[dict(index=i,time=t,**dict(state,cart_position=x,pole_angle=th))
            for i,(t,x,th) in enumerate(((0.,0.,0.),(.05,.01,math.pi/2),(end,.03,math.pi)))]
    completion=dict(time=end,transition_count=2,reason='horizon',stop_reasons=[],terminated=False,truncated=True)
    metrics=dict(success_episode=False,hold_final=0.,hold_longest=0.,completion=completion,
                 safety=dict(intervention_count=int(mode=='on'),exceeds_024_resolved=False))
    row=dict(stage='main',model_id='SAC_seed0',training_seed=0,filter=mode,group='nominal',case_id='nominal_00',
             model_sha256='a'*64,log_sha256='b'*64,metrics=metrics)
    trans=[]
    for i in range(2):
        t=dict(index=i,from_state=i,to_state=i+1,time_start=states[i]['time'],time_end=states[i+1]['time'],dt_actual=states[i+1]['time']-states[i]['time'])
        if mode=='on': t['safety_filter']=dict(feasible=True,intervened=i==0)
        trans.append(t)
    log=dict(metadata=dict(algorithm='SAC',training_seed=0,evaluation_case_id='nominal_00',filter_mode=mode,
             initial_state=state,reset_mode='explicit',config=dict(pole_length=.18,max_position=.25),
             upright_thresholds=dict(angle=math.pi/18,angular_velocity=.5,position=.2,velocity=.2)),states=states,transitions=trans,
             completion=completion,evaluation=dict(success_episode=False,hold_final=0.,hold_longest=0.))
    return v.Episode(row,log,{},'SAC · запуск 0')


class SelectionTests(unittest.TestCase):
    def test_deterministic_selection_and_exact_roster(self):
        rows=selection_fixture();expected,info=v.select(rows)
        random.Random(67).shuffle(rows);actual,again=v.select(rows)
        self.assertEqual(expected,actual);self.assertEqual(info,again)
        self.assertEqual(info['rescue_pair'],['SAC_seed0','angle_08'])
        self.assertEqual(info['tradeoff_pairs'][0],['SAC_seed0','angle_01'])
        self.assertEqual([len(x) for x in actual],[3,0,2,2,7,7])
        self.assertEqual([r['model_id'] for r in actual[5]],list(v.MODELS))
        self.assertEqual({r['case_id'] for r in actual[5]},{'joint_00'})
        self.assertEqual([r['case_id'] for r in actual[4]],[g+'_00' for g in v.GROUPS])

    def test_off_on_pair_order(self):
        groups,_=v.select(selection_fixture())
        for i in (2,3):
            self.assertEqual([r['filter'] for r in groups[i]],['off','on'])
            self.assertEqual(groups[i][0]['case_id'],groups[i][1]['case_id'])
            self.assertEqual(groups[i][0]['model_id'],groups[i][1]['model_id'])

    def test_duplicates_rejected(self):
        rows=selection_fixture()
        with self.assertRaisesRegex(ValueError,'duplicate'):v.select(rows+[rows[0]])

    def test_missing_pair_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing paired'):v.select(selection_fixture()[1:])

    def test_missing_nominal_not_replaced(self):
        rows=[r for r in selection_fixture() if not (r['model_id']=='classical' and r['case_id']=='nominal_00')]
        with self.assertRaisesRegex(ValueError,'missing required'):v.select(rows)

    def test_pair_state_and_model_binding(self):
        for field in ('state_sha256','model_sha256','training_seed'):
            rows=selection_fixture();rows[0][field]='tampered'
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'pair'):v.select(rows)

    def test_timing_rejected(self):
        rows=selection_fixture();rows[0]['stage']='timing'
        with self.assertRaisesRegex(ValueError,'timing'):v.select(rows)

    def test_wrong_tradeoff_population(self):
        rows=[r for r in selection_fixture() if r['case_id']!='angle_07']
        with self.assertRaisesRegex(ValueError,'seven'):v.select(rows)


class TimelineTests(unittest.TestCase):
    def test_angle_unwrap_across_branch(self):
        xs=v.unwrap_angles([math.pi-.1,-math.pi+.1,-math.pi+.3],wrapped=True)
        self.assertAlmostEqual(xs[1],math.pi+.1);self.assertAlmostEqual(xs[2],math.pi+.3)

    def test_unwrapped_multiple_turns_preserved(self):
        values=[6*math.pi,8*math.pi,9*math.pi]
        self.assertEqual(v.unwrap_angles(values),values)

    def test_ambiguous_or_nonfinite_angles_rejected(self):
        with self.assertRaises(ValueError):v.unwrap_angles([0.,math.pi],wrapped=True)
        for val in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):v.unwrap_angles([0.,val])

    def test_interpolate_display_only(self):
        ep=episode();saved=copy.deepcopy(ep.log);s=ep.sample(.025)
        self.assertEqual(s['x'],.005);self.assertAlmostEqual(s['theta'],math.pi/4)
        self.assertEqual(saved,ep.log)

    def test_intervention_half_open_and_terminal(self):
        ep=episode()
        self.assertTrue(ep.sample(0.)['intervention']);self.assertTrue(ep.sample(.049)['intervention'])
        self.assertFalse(ep.sample(.05)['intervention']);self.assertFalse(ep.sample(1.)['intervention'])
        self.assertFalse(episode('off').sample(.01)['intervention'])

    def test_shared_clock_and_shortened_stop(self):
        early=episode(end=.081);later=episode(end=.1)
        self.assertEqual(early.sample(.09)['time'],.081)
        self.assertEqual(early.sample(.09)['x'],.03)
        self.assertEqual(later.sample(.09)['time'],.09)
        s=v.segments([early,later])[0]
        self.assertEqual(s['frames'],3+60);self.assertEqual(v.display_time(s,3),.1)

    def test_zero_duration_terminal_no_motion(self):
        ep=episode();ep.states[-1]['time']=.05;ep.states[-1]['cart_position']=.01
        ep= v.Episode(ep.row,ep.log,{},ep.label)
        s=ep.sample(.05);self.assertTrue(s['ended']);self.assertEqual(s['x'],.01)

    def test_sequential_titles_and_full_endpoints(self):
        seg=v.segments([episode() for _ in range(7)],True)
        self.assertEqual(len(seg),14);self.assertEqual(sum(s['frames'] for s in seg),7*(30+3+60))
        self.assertEqual([s['title'] for s in seg[::2]],list(v.GROUP_TITLES))

    def test_common_case_and_initial_state(self):
        left,right=episode('off'),episode('on');v.common_start([left,right],same_model=True)
        right.row['case_id']='nominal_01'
        with self.assertRaisesRegex(ValueError,'common case'):v.common_start([left,right])
        right.row['case_id']='nominal_00';right.states[0]['cart_velocity']=.0001
        with self.assertRaisesRegex(ValueError,'initial state'):v.common_start([left,right])

    def test_safety_not_conflated_with_success(self):
        ep=episode();ep.row['metrics']['safety']['exceeds_024_resolved']=True
        self.assertEqual(ep.outcome(),'Удержание не выполнено')
        ep.row['metrics']['success_episode']=True;self.assertEqual(ep.outcome(),'Удержание выполнено')
        self.assertEqual(ep.final_status()[1]['text'],'Граница нарушена')
        self.assertNotEqual(ep.final_status()[0]['role'],'good')
        ep.log['completion']['stop_reasons']=['position_limit'];self.assertEqual(ep.outcome(),'Остановка: предел 0,25 м')


class PresentationTests(unittest.TestCase):
    def draw(self,eps,frame=3,index=0):
        painter=v.Painter();calls=[];original=painter.text
        def text(draw,xy,message,size=30,fill=None,bold=False,anchor=None):
            bbox=draw.textbbox(xy,message,font=painter.font(size,bold),anchor=anchor)
            calls.append(dict(text=message,color=fill or painter.ink,bbox=bbox))
            return original(draw,xy,message,size,fill,bold,anchor)
        with patch.object(painter,'text',side_effect=text):
            im=painter.frame(index,v.segments(eps)[0],frame)
        self.assertEqual(im.size,(1920,1080))
        for c in calls:
            self.assertTrue(0<=c['bbox'][0]<c['bbox'][2]<=1920,c)
            self.assertTrue(0<=c['bbox'][1]<c['bbox'][3]<=1080,c)
        return im,calls

    def test_four_hold_safety_combinations_drawn_without_mutation(self):
        for success in (False,True):
            for unsafe in (False,True):
                ep=episode();ep.row['metrics']['success_episode']=success
                ep.row['metrics']['safety']['exceeds_024_resolved']=unsafe
                before=copy.deepcopy(ep.row)
                with self.subTest(success=success,unsafe=unsafe):
                    _,calls=self.draw([ep]);texts=[c['text'] for c in calls]
                    self.assertIn('Удержание выполнено' if success else 'Удержание не выполнено',texts)
                    self.assertIn('Граница нарушена' if unsafe else 'Без превышения границы',texts)
                    self.assertNotIn('Успех',texts)
                    if unsafe:self.assertFalse(any(c['text']=='Удержание выполнено' and c['color']==v.STATUS_COLORS['good'] for c in calls))
                    self.assertEqual(before,ep.row)

    def test_tradeoff_final_pair_drawing(self):
        off,on=episode('off'),episode('on')
        off.row['metrics']['success_episode']=True;off.row['metrics']['safety']['exceeds_024_resolved']=True
        _,calls=self.draw([off,on],index=3)
        left=[c['text'] for c in calls if c['bbox'][2]<960]
        right=[c['text'] for c in calls if c['bbox'][0]>960]
        self.assertIn('Удержание выполнено',left);self.assertIn('Граница нарушена',left)
        self.assertIn('Удержание не выполнено',right);self.assertIn('Без превышения границы',right)

    def test_no_early_verdict_even_with_future_violation(self):
        ep=episode();ep.row['metrics']['success_episode']=True
        ep.row['metrics']['safety']['exceeds_024_resolved']=True
        for frame in (0,1,2):
            _,calls=self.draw([ep],frame)
            self.assertFalse(any('Удержание' in c['text'] or 'Граница нарушена' in c['text'] or 'Без превышения' in c['text'] for c in calls))

    def test_position_stop_drawing_and_freeze(self):
        early,later=episode('off',.081),episode('on',.2)
        early.log['completion'].update(terminated=True,truncated=False,stop_reasons=['position_limit'])
        early.row['metrics']['safety']['exceeds_024_resolved']=True
        early.states[-1]['cart_position']=.25
        _,calls=self.draw([early,later],3,2);texts=[c['text'] for c in calls]
        self.assertIn('Остановка: предел 0,25 м',texts)
        self.assertIn('t = 0.081 с',texts);self.assertIn('t = 0.100 с',texts)
        self.assertEqual(texts.count('x = +0,24 м'),2)
        self.assertEqual(early.sample(.19)['x'],.25)
        self.assertEqual(early.sample(.19)['theta'],early.states[-1]['pole_angle'])

    def test_stop_threshold_from_saved_config(self):
        ep=episode();ep.log['completion']['stop_reasons']=['position_limit']
        ep.log['metadata']['config']['max_position']=.26
        self.assertEqual(ep.outcome(),'Остановка: предел 0,26 м')
        del ep.log['metadata']['config']['max_position']
        with self.assertRaises(KeyError):ep.outcome()

    def test_other_stop_causes_never_position_label(self):
        for reason in ('velocity_limit','safety_filter_refusal','simulator_error','unknown'):
            ep=episode();ep.log['completion'].update(terminated=True,stop_reasons=[reason])
            with self.subTest(reason=reason):
                self.assertNotIn('0,25',ep.outcome());self.assertIn('Остановка',ep.outcome())

    def test_desired_exceedance_does_not_invent_stop(self):
        ep=episode();ep.row['metrics']['safety']['exceeds_024_resolved']=True
        self.assertFalse(any('Остановка' in c['text'] for c in ep.final_status()))

    def test_saved_final_velocity_explanation_drawn(self):
        ep=episode();ep.evidence['checkpoint_label']='last';ep.states[-1]['cart_velocity']=.24357904491649754
        _,calls=self.draw([ep],index=1)
        self.assertIn('Скорость тележки выше порога',[c['text'] for c in calls])
        self.assertEqual([k for k,c in ep.final_components().items() if c['exceeds']],['velocity'])
        ep.states[-1]['cart_velocity']=.2
        self.assertEqual(len(ep.final_status(explain_components=True)),2)

    def test_multiple_final_failures_not_mislabelled_as_velocity_only(self):
        ep=episode();ep.states[-1].update(cart_velocity=.24,pole_angle=0.)
        self.assertEqual(ep.final_status(explain_components=True)[-1]['text'],'Несколько условий удержания не выполнены')
        del ep.log['metadata']['upright_thresholds']
        with self.assertRaises(KeyError):ep.final_status(explain_components=True)

    def test_periodic_final_angle_uses_saved_threshold(self):
        ep=episode();ep.states[-1]['pole_angle']=-3*math.pi
        self.assertFalse(ep.final_components()['angle']['exceeds'])
        ep.states[-1]['pole_angle']+=.18
        self.assertTrue(ep.final_components()['angle']['exceeds'])

    def test_stable_model_colors_and_screen_roster(self):
        eps=[]
        for mid in v.MODELS:
            ep=episode();ep.row['model_id']=mid;ep.label=mid;eps.append(ep)
        before=copy.deepcopy([e.manifest() for e in eps]);shown=v.screen_episodes(list(reversed(eps)))
        self.assertEqual([e.row['model_id'] for e in shown],list(v.SCREEN_MODELS))
        self.assertEqual([e.manifest() for e in eps],before)
        for ep in eps:
            color=v.model_color(ep);ep.row['filter']='off';self.assertEqual(v.model_color(ep),color)
        _,calls=self.draw(eps,index=5)
        labels=[c for c in calls if c['text'] in v.MODELS]
        self.assertEqual([c['text'] for c in labels],list(v.SCREEN_MODELS))
        self.assertTrue(all(c['bbox'][1]<600 for c in labels[:4]))
        self.assertTrue(all(c['bbox'][1]>600 for c in labels[4:]))
        for c,i in zip(labels,range(7)):
            self.assertLess(c['bbox'][2],40+(i%4+1)*465.5)

    def test_interpolated_geometry_enclosed_and_common_scale(self):
        eps=[episode(),episode()];eps[0].states[1].update(cart_position=.2,pole_angle=0.)
        eps[0].states[-1].update(cart_position=.2,pole_angle=math.pi)
        extent=v.comparison_extent(eps)
        self.assertGreaterEqual(extent,.2+.18+.02)
        for ep in eps:
            for k in range(101):
                s=ep.sample(ep.end*k/100)
                self.assertLessEqual(abs(s['x']+.18*math.sin(s['theta'])),extent-.019999999)
        self.assertEqual(v.segments(eps)[0]['display_extent'],extent)

    def test_screen_reorder_preserves_common_start_and_selection(self):
        selected,_=v.select(selection_fixture());eps=[]
        for row in selected[5]:
            ep=episode();ep.row['model_id']=row['model_id'];ep.row['case_id']='joint_00';eps.append(ep)
        v.common_start(v.screen_episodes(eps))
        self.assertEqual({id(e) for e in eps},{id(e) for e in v.screen_episodes(eps)})
        self.assertEqual([r['model_id'] for r in selected[5]],list(v.MODELS))


class IntegrityTests(unittest.TestCase):
    def test_log_binds_valid_record(self):
        ep=episode();v.validate_log(ep.log,ep.row,ep.log['metadata']['initial_state'])

    def test_identity_corruption_rejected(self):
        for key,value in [('training_seed',1),('algorithm','TQC'),('evaluation_case_id','joint_00'),('filter_mode','off')]:
            ep=episode();ep.log['metadata'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'identity'):v.validate_log(ep.log,ep.row,ep.log['metadata']['initial_state'])

    def test_initial_state_corruption_rejected(self):
        ep=episode();ep.states[0]['cart_position']=.001
        with self.assertRaisesRegex(ValueError,'initial'):v.validate_log(ep.log,ep.row,ep.log['metadata']['initial_state'])

    def test_nonfinite_and_time_corruption(self):
        for corrupt in ('nan','time','flag'):
            ep=episode()
            if corrupt=='nan':ep.states[1]['cart_position']=float('nan')
            elif corrupt=='time':ep.transitions[0]['time_end']=.03
            else:ep.transitions[0]['safety_filter']['intervened']=False
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):v.validate_log(ep.log,ep.row,ep.log['metadata']['initial_state'])

    def test_tar_sha_missing_duplicate_path_and_no_mutation(self):
        data=gzip.compress(json.dumps({'states':[]}).encode(),mtime=0);sha=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.tar.gz'
            def archive(names):
                with tarfile.open(path,'w:gz') as tf:
                    for name in names:
                        info=tarfile.TarInfo(name);info.size=len(data);tf.addfile(info,io.BytesIO(data))
            archive(['episode.json.gz']);before=path.read_bytes()
            self.assertEqual(v.selected_payloads(path,{'episode.json.gz':sha}),{'episode.json.gz':{'states':[]}})
            self.assertEqual(path.read_bytes(),before);self.assertEqual(list(Path(tmp).iterdir()),[path])
            with self.assertRaisesRegex(ValueError,'SHA'):v.selected_payloads(path,{'episode.json.gz':'0'*64})
            with self.assertRaisesRegex(ValueError,'missing'):v.selected_payloads(path,{'missing.json.gz':sha})
            archive(['episode.json.gz']*2)
            with self.assertRaisesRegex(ValueError,'duplicate'):v.selected_payloads(path,{'episode.json.gz':sha})
            archive(['../episode.json.gz'])
            with self.assertRaises(ValueError):v.selected_payloads(path,{})

    def test_output_exists_fails_before_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output=Path(tmp),overwrite=False,check_only=False)
            with self.assertRaisesRegex(ValueError,'output already'):v.build(args)

    def test_stdlib_import_has_no_scientific_runtime(self):
        code="from cartpole.experiments import robustness_video; import sys; assert not any(x in sys.modules for x in ['torch','pydrake','numpy','cartpole.safety','cartpole.rl.env','PIL'])"
        subprocess.run([sys.executable,'-B','-S','-c',code],check=True,cwd=v.a.ROOT)

    def test_destination_appearing_during_render_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'stage';qa=root/'qa';out=root/'out'
            for p in (stage,qa,out):p.mkdir()
            (out/'user-file').write_text('keep')
            with self.assertRaisesRegex(ValueError,'appeared'):v.publish_output(stage,qa,out)
            self.assertEqual((out/'user-file').read_text(),'keep');self.assertTrue(stage.exists())

    def test_symlink_output_is_rejected_even_with_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.symlink_to(root/'absent')
            args=argparse.Namespace(output=out,overwrite=True,check_only=False)
            with self.assertRaisesRegex(ValueError,'symlink'):v.build(args)
            self.assertTrue(out.is_symlink())

    def test_explicit_overwrite_preserves_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'stage';qa=root/'qa';out=root/'out';oldqa=root/'out-qa-frames'
            for p in (stage,qa,out,oldqa):p.mkdir()
            (stage/'new').write_text('new');(out/'VIDEO_MANIFEST.json').write_text('{}')
            (oldqa/'old-frame').write_text('old');v.publish_output(stage,qa,out,True)
            self.assertEqual((out/'new').read_text(),'new')
            self.assertEqual(len(list(root.glob('out.backup-*/VIDEO_MANIFEST.json'))),1)
            self.assertEqual(len(list(root.glob('out.backup-*-qa-frames/old-frame'))),1)


if __name__=='__main__':unittest.main()
