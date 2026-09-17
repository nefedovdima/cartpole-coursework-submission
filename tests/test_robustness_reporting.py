"""Synthetic data only; no local archives, model loading, network or simulation."""
import copy
import csv
import json
import importlib.util
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cartpole.experiments import robustness_analysis as a
from cartpole.experiments import robustness_reporting as r
from tests.test_robustness_analysis import dataset, metrics, seal


def pair(model='SAC_seed0', group='nominal', index=0, outcome=(True, True)):
    result = []
    for mode, success in zip(r.MODES, outcome):
        m = metrics(mode)
        if not success:
            m.update(success_episode=False, hold_final=0.)
        result.append(dict(stage='main', model_id=model, group=group, filter=mode,
            case_id=f'{group}_{index:02}', job_id=f'{model}_{mode}_{group}',
            training_seed=None if model == 'classical' else int(model[-1]),
            state_sha256='a'*64, model_sha256='b'*64, log_sha256='c'*64,
            metrics=m, flags=a.flags(m)))
    return result


def change_safety(row, **values):
    s = row['metrics']['safety']; s.update(values)
    if 'max_abs_position_interval' in values:
        maximum = values['max_abs_position_interval']
        s.update(desired_excess_max=max(0., maximum-.24), exceeds_024=maximum > .24,
                 exceeds_024_resolved=maximum > .24+1e-12,
                 intervals_exceeding_024=int(maximum > .24))
    row['flags'] = a.flags(row['metrics'])


def small_rows():
    # All four possible pair outcomes; all three controller families.
    rows = pair(outcome=(True, True))
    rows += pair('TQC_seed1', index=1, outcome=(False, True))
    rows += pair('classical', index=2, outcome=(True, False))
    rows += pair('SAC_seed2', index=3, outcome=(False, False))
    change_safety(rows[4], max_abs_position_interval=.245)
    change_safety(rows[1], intervention_count=2, intervention_fraction=.002,
        intervention_abs_sum=6., intervention_abs_max=4., intervention_abs_mean=3.)
    change_safety(rows[3], intervention_count=1, intervention_fraction=.001,
        intervention_abs_sum=9., intervention_abs_max=9., intervention_abs_mean=9.)
    return rows


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.rows = small_rows()
        self.tables = r.build_tables(self.rows)

    def test_counts_denominators_and_safety_conjunction(self):
        off, on = self.tables['summary_overall.csv']
        self.assertEqual((off['episodes'], off['successes'], off['resolved_exceedances']), (4,2,1))
        self.assertEqual((off['success_rate'], off['safe_success_rate']), (.5,.25))
        self.assertEqual((on['episodes'], on['horizons'], on['horizon_hold_failures']), (4,4,2))

    def test_off_interventions_are_null_including_counts(self):
        for table in ('summary_overall.csv','summary_by_controller.csv','summary_by_group.csv','summary_controller_group.csv'):
            for row in self.tables[table]:
                if row['filter'] == 'off':
                    self.assertTrue(all(row[k] is None for k in ('filter_decisions','interventions','intervention_rate')))

    def test_weighted_decisions_and_corrections(self):
        i = self.tables['intervention_summary.csv'][0]
        self.assertEqual((i['accepted_decisions'], i['interventions'], i['intervention_rate']), (4000,3,3/4000))
        self.assertEqual((i['correction_abs_sum_m_s2'], i['correction_abs_mean_m_s2'], i['correction_abs_max_m_s2']), (15.,5.,9.))
        with self.assertRaisesRegex(ValueError,'only defined for on'):
            r.intervention_summary(self.rows)

    def test_four_paired_outcomes_and_off_only_exceedance(self):
        p = self.tables['paired_outcomes.csv'][0]
        self.assertEqual([p[k] for k in ('pair_count','both_success','on_only_success','off_only_success','neither_success')], [4,1,1,1,1])
        self.assertEqual(p['paired_success_gain'],0)
        self.assertEqual(p['off_only_success_with_resolved_exceedance'],1)
        self.assertEqual(p['off_success_with_resolved_exceedance'],1)

    def test_strict_and_resolved_remain_distinct(self):
        change_safety(self.rows[0], max_abs_position_interval=.24+1e-13)
        result = r.build_tables(self.rows)['summary_overall.csv'][0]
        self.assertEqual((result['strict_exceedances'], result['resolved_exceedances']), (2,1))
        self.assertNotEqual(result['legacy_max_abs_position_interval_m'], result['safety_max_abs_position_interval_m'])

    def test_event_and_horizon_separate(self):
        row = self.rows[0]; m = row['metrics']
        m.update(success_episode=False, hold_final=0., hold_longest=0., first_hold_start=None, first_hold_confirmation=None, duration=1.)
        m['completion'].update(reason='terminal', terminated=True, truncated=False, time=1., stop_reasons=['position_limit','velocity_limit'])
        change_safety(row, position_limit_events=1)
        result = r.build_tables(self.rows)['summary_overall.csv'][0]
        self.assertEqual((result['horizons'],result['horizon_hold_failures'],result['position_limit'],result['velocity_limit']), (3,2,1,1))

    def test_zero_denominators(self):
        self.assertIsNone(r.ratio(0,0))
        self.assertIsNone(r.summary([], 'on')['success_rate'])
        self.assertIsNone(r.intervention_summary([])['intervention_rate'])
        self.assertIsNone(r.intervention_summary([])['correction_abs_mean_m_s2'])
        self.assertIsNone(r.paired_summary([])['paired_success_gain'])

    def test_duplicate(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            r.build_tables(self.rows+[self.rows[0]])

    def test_incomplete_pair(self):
        with self.assertRaisesRegex(ValueError,'incomplete pair'):
            r.build_tables(self.rows[:-1])

    def test_main_timing_never_mixed(self):
        self.rows[0]['stage']='timing'
        with self.assertRaisesRegex(ValueError,'main/timing'):
            r.build_tables(self.rows)

    def test_pair_bindings(self):
        for key, bad in (('state_sha256','z'*64),('model_sha256','z'*64),('training_seed',99),('group','angle')):
            with self.subTest(key=key):
                rows=copy.deepcopy(self.rows); rows[1][key]=bad
                with self.assertRaisesRegex(ValueError,'pair binding'):
                    r.build_tables(rows)

    def test_unknown_controller_group_mode(self):
        for key in ('model_id','group','filter'):
            rows=copy.deepcopy(self.rows); rows[0][key]='unknown'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'roster identity'):
                r.build_tables(rows)

    def test_nonfinite_rejected(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            rows=copy.deepcopy(self.rows); rows[0]['metrics']['reward']=value
            with self.assertRaises(ValueError): r.build_tables(rows)
            with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
                r.write_json(Path(d)/'bad.json', {'bad':value})

    def test_stable_csv_json_order_and_null_encoding(self):
        shuffled=copy.deepcopy(self.rows); random.Random(42).shuffle(shuffled)
        other=r.build_tables(shuffled)
        self.assertEqual(a.canonical(self.tables), a.canonical(other))
        with tempfile.TemporaryDirectory() as d:
            for name, columns in r.COLUMNS.items():
                first, second = Path(d)/name, Path(d)/('second-'+name)
                r.write_csv(first, columns, self.tables[name]); r.write_csv(second, columns, other[name])
                self.assertEqual(first.read_bytes(),second.read_bytes())
            with (Path(d)/'summary_overall.csv').open() as f:
                off=next(csv.DictReader(f))
            self.assertEqual(off['interventions'],'')


class InputIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=a.Protocol()
        cls.data=dataset(cls.protocol,stage='main')

    def test_complete_exact_frozen_roster(self):
        # The synthetic SHA is explicitly substituted ONLY in this unit test.
        with patch.object(r,'SCIENTIFIC_SHA', self.data['scientific_sha256']):
            result=r.validate_input(self.data)
        self.assertEqual((result['episodes'],result['jobs'],len(result['pairs'])),(1960,98,980))
        counts=CounterForTest(self.data['episodes'])
        self.assertEqual(set(counts.values()),{20})
        self.assertEqual(len(counts),98)

    def test_valid_but_unapproved_scientific_sha(self):
        with self.assertRaisesRegex(ValueError,'unexpected scientific SHA'):
            r.validate_input(self.data)

    def test_timing_input(self):
        with self.assertRaisesRegex(ValueError,'main only'):
            r.validate_input(dataset(self.protocol))

    def test_frozen_corruptions_after_reseal(self):
        for key,value in (('model_id','SAC_seed8'),('state_sha256','f'*64),('model_sha256','f'*64),
                          ('training_seed',8),('group','angle'),('job_id','invented')):
            d=copy.deepcopy(self.data); d['episodes'][0][key]=value; seal(d)
            with self.subTest(key=key), self.assertRaises(ValueError):r.validate_input(d)

    def test_duplicate_missing_pair_and_entire_cell(self):
        for mutate in (lambda rows:rows.append(rows[0]),lambda rows:rows.pop(),
                       lambda rows:rows.__delitem__(slice(0,20))):
            d=copy.deepcopy(self.data); mutate(d['episodes']); seal(d)
            with self.assertRaises(ValueError):r.validate_input(d)

    def test_provenance_tamper_and_scientific_tamper(self):
        for field in ('scientific_sha256','envelope_sha256'):
            d=copy.deepcopy(self.data); d[field]='f'*64
            with self.assertRaises(ValueError):r.validate_input(d)
        d=copy.deepcopy(self.data); d['provenance']['review']['manifest_files']=99
        with self.assertRaisesRegex(ValueError,'envelope SHA'):r.validate_input(d)

    def test_export_clock_provenance_not_in_scientific_tables(self):
        d=copy.deepcopy(self.data)
        d['provenance']['review']['export_manifest_sha256']='1'*64
        a.seal_envelope(d)
        with patch.object(r,'SCIENTIFIC_SHA',d['scientific_sha256']):
            r.validate_input(d)
        self.assertEqual(a.canonical(r.build_tables(self.data['episodes'])),a.canonical(r.build_tables(d['episodes'])))
        self.assertNotEqual(d['envelope_sha256'],self.data['envelope_sha256'])


def CounterForTest(rows):
    from collections import Counter
    return Counter((r['model_id'],r['group'],r['filter']) for r in rows)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name); self.out=self.base/'report'
        self.data=dict(episodes=small_rows(),scientific_sha256=r.SCIENTIFIC_SHA,
                      envelope_sha256='e'*64, provenance={'synthetic':True})

    def publish_synthetic(self, renderer=None):
        # Publication mechanics isolated from production input/golden checks.
        def dummy(tables, target): (target/'synthetic.svg').write_text('<svg/>')
        with patch.object(r,'validate_input'),patch.object(r,'regression_invariants'),patch.object(r,'render_figures',renderer or dummy):
            return r.publish(self.data,self.out)

    def test_manifest_hashes_and_no_private_paths(self):
        before=copy.deepcopy(self.data)
        result=self.publish_synthetic()
        manifest=a.read_json(self.out/'MANIFEST.json')
        self.assertEqual(result['manifest_sha256'],a.file_sha(self.out/'MANIFEST.json'))
        self.assertEqual(set(manifest['files_sha256']),{p.name for p in self.out.iterdir()}-{'MANIFEST.json'})
        for name,digest in manifest['files_sha256'].items():self.assertEqual(digest,a.file_sha(self.out/name))
        summary=a.read_json(self.out/'analysis_summary.json')
        self.assertEqual(summary['verification']['compact_consistency'],'not_run')
        self.assertEqual(summary['verification']['raw_metric_recomputation'],'not_run')
        for name in r.COLUMNS:
            self.assertNotIn(str(a.ROOT), (self.out/name).read_text())
        self.assertEqual(self.data,before)
        self.assertEqual(list(self.base.iterdir()),[self.out])

    def test_existing_output_untouched(self):
        self.out.mkdir(); (self.out/'sentinel').write_text('keep')
        with self.assertRaises(FileExistsError):r.publish(self.data,self.out)
        self.assertEqual((self.out/'sentinel').read_text(),'keep')

    def test_existing_dangling_symlink_refused(self):
        self.out.symlink_to(self.base/'missing')
        with self.assertRaises(FileExistsError):r.publish(self.data,self.out)
        self.assertTrue(self.out.is_symlink())

    def test_render_failure_leaves_no_partial_directory(self):
        def fail(tables,target):
            (target/'partial.png').write_bytes(b'partial')
            raise RuntimeError('synthetic failure')
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):self.publish_synthetic(fail)
        self.assertEqual(list(self.base.iterdir()),[])

    def test_destination_appearing_during_render_not_overwritten(self):
        def race(tables,target):
            self.out.mkdir(); (self.out/'owner').write_text('other')
        with self.assertRaises(FileExistsError):self.publish_synthetic(race)
        self.assertEqual(list(self.base.iterdir()),[self.out])
        self.assertEqual((self.out/'owner').read_text(),'other')

    def test_stdlib_core_import_no_ml_or_plotting(self):
        code="import sys; from cartpole.experiments import robustness_reporting; assert not any(m in sys.modules for m in ('torch','numpy','pydrake','matplotlib'))"
        subprocess.run([sys.executable,'-S','-B','-c',code],cwd=a.ROOT,check=True,capture_output=True)

    def test_render_all_figures_closed_and_deterministic(self):
        if importlib.util.find_spec('matplotlib') is None:
            self.skipTest('Matplotlib only needed for rendering')
        rows=[row for model in r.MODELS for group in r.GROUPS for row in pair(model,group)]
        tables=r.build_tables(rows)
        first=self.base/'first'; second=self.base/'second'; first.mkdir(); second.mkdir()
        r.render_figures(tables,first)
        import matplotlib.pyplot as plt
        self.assertEqual(plt.get_fignums(),[])
        r.render_figures(tables,second)
        self.assertEqual(plt.get_fignums(),[])
        expected={name+ext for name in r.FIGURES for ext in ('.png','.svg')}
        self.assertEqual({p.name for p in first.iterdir()},expected)
        for name in expected:self.assertEqual((first/name).read_bytes(),(second/name).read_bytes())

    def test_heatmap_controller_labels_fit_canvas(self):
        if importlib.util.find_spec('matplotlib') is None:
            self.skipTest('Matplotlib only needed for rendering')
        from matplotlib.figure import Figure
        original = Figure.savefig
        checked = []
        def verify(figure, filename, *args, **kwargs):
            if Path(filename).stem in ('success_delta_heatmap', 'intervention_rate_heatmap'):
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                for label in figure.axes[0].get_yticklabels():
                    box = label.get_window_extent(renderer)
                    self.assertGreaterEqual(box.x0, 0)
                    self.assertLessEqual(box.x1, figure.bbox.width)
                    self.assertGreaterEqual(box.y0, 0)
                    self.assertLessEqual(box.y1, figure.bbox.height)
                checked.append(Path(filename).name)
            return original(figure, filename, *args, **kwargs)
        rows=[row for model in r.MODELS for group in r.GROUPS for row in pair(model,group)]
        with patch.object(Figure, 'savefig', verify):
            r.render_figures(r.build_tables(rows), self.base)
        self.assertEqual(len(checked), 4)


if __name__ == '__main__':
    unittest.main()
