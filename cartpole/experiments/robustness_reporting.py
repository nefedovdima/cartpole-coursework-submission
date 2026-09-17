"""Publication tables/figures for the pinned structured robustness main experiment.

Pure stdlib calculations; Matplotlib is imported only for rendering. No archive,
model or dynamics is loaded. Input verification reuses the normalized validator.
CSV empty cells mean null (not zero); rates are fractions, corrections are m/s².
A new output directory is published by same-filesystem rename after all files
are ready. Single-writer CLI: concurrent publishers to the same path are not
supported; do not use a shared destination. No existing destination is replaced.
The summary hashes every table/figure; MANIFEST.json also hashes the summary.
The manifest's own hash is returned on stdout (no circular self-hash).
"""
import argparse
from collections import Counter, defaultdict
import csv
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from cartpole.experiments import robustness_analysis as audit

SCHEMA = 'structured-robustness-reporting-v1'
SCIENTIFIC_SHA = 'c1cac3775426748520db67957e988fb7cf3706148d1cc515eca26e88118fb63b'
MODELS, GROUPS = audit.MODELS, audit.GROUPS
MODES = ('off', 'on')
BASE_COLUMNS = (
    'filter', 'episodes', 'successes', 'success_rate', 'horizons', 'horizon_rate',
    'horizon_hold_failures', 'position_limit', 'velocity_limit', 'strict_exceedances',
    'resolved_exceedances', 'resolved_exceedance_rate',
    'successes_without_resolved_exceedance', 'safe_success_rate',
    'filter_decisions', 'interventions', 'intervention_rate', 'refusals',
    'legacy_max_abs_position_interval_m', 'safety_max_abs_position_interval_m')
PAIR_COLUMNS = ('scope', 'id', 'pair_count', 'both_success', 'on_only_success',
    'off_only_success', 'neither_success', 'paired_success_gain',
    'off_success_with_resolved_exceedance', 'off_only_success_with_resolved_exceedance')
INTERVENTION_COLUMNS = ('scope', 'id', 'episodes', 'accepted_decisions', 'interventions',
    'intervention_rate', 'refusals', 'correction_abs_sum_m_s2',
    'correction_abs_mean_m_s2', 'correction_abs_max_m_s2')
COLUMNS = {
    'summary_overall.csv': BASE_COLUMNS,
    'summary_by_controller.csv': ('controller',) + BASE_COLUMNS,
    'summary_by_group.csv': ('group',) + BASE_COLUMNS,
    'summary_controller_group.csv': ('controller', 'group') + BASE_COLUMNS,
    'paired_outcomes.csv': PAIR_COLUMNS,
    'intervention_summary.csv': INTERVENTION_COLUMNS,
}
FIGURES = ('overall_success_safety', 'success_by_group', 'success_delta_heatmap',
           'intervention_rate_heatmap', 'paired_outcomes')
GROUP_LABELS = ('Nominal', 'Позиция', 'Скорость', 'Угол', 'Угловая\nскорость',
                'Граница', 'Совместные')
COLORS = {'off': '#697586', 'on': '#14886c'}
DEFINITIONS = {
    'episode_rates': 'All success/safety/horizon rates divide by episodes in the row; no pooled timing.',
    'success': 'Full 10 s horizon and final consecutive sampled hold >= 2 s (tolerance 1e-12 s).',
    'hold': '|wrap(theta-pi)|<=10 degrees, |omega|<=0.5 rad/s, |x|<=0.20 m, |v|<=0.20 m/s; sampled states.',
    'horizon': "metrics.completion.reason == 'horizon'; distinct from success.",
    'horizon_hold_failures': 'Horizon reached but success_episode is false.',
    'position_limit': 'Episodes with position_limit in completion.stop_reasons (working boundary +/-0.25 m).',
    'velocity_limit': 'Episodes with velocity_limit in completion.stop_reasons (+/-2 m/s).',
    'strict_exceedances': 'Episodes with safety.exceeds_024: interval maximum >0.24 m.',
    'resolved_exceedances': 'Episodes with safety.exceeds_024_resolved: interval maximum >0.24+1e-12 m.',
    'safe_success': 'success_episode AND NOT exceeds_024_resolved; a descriptive conjunction, not a safety guarantee.',
    'filter_decisions': 'Accepted filter decisions, denominator for interventions; null for off.',
    'intervention_rate': 'sum(intervention_count)/sum(accepted_decisions); null for off or zero denominator.',
    'correction_mean': 'sum(intervention_abs_sum)/sum(intervention_count); null if no interventions, m/s². Absolute filtered minus raw requested command, including amplitude limiting.',
    'refusals': 'Sum of refused decisions, separate from accepted decisions and interventions.',
    'paired_success_gain': '(on_only_success - off_only_success)/pair_count for identical model and state.',
    'legacy_max_abs_position_interval_m': 'Maximum metrics.max_abs_position_interval; actual endpoints plus interior quadratic vertex.',
    'safety_max_abs_position_interval_m': 'Maximum metrics.safety.max_abs_position_interval; actual endpoints, predicted endpoint and interior vertex using fsum.',
    'csv_null': 'Empty field; JSON null. All rates are fractions in [0,1], gain in [-1,1].',
}
LIMITS = [
    'Fixed development design: descriptive paired results, not an independent final test or global stability claim.',
    'Equal group weighting and 3 SAC + 3 TQC + 1 classical controllers are design weights, not population frequencies.',
    'Seeds share initial states; mirrored and structured states are not independent random observations.',
    'Nominal uses 20 new lower-box draws; it tests the training reset law, not all admissible states.',
    'All starts were selected within filter admission K. Off/on comparison is conditional on this selection.',
    'Evaluation off/on changes the deployed action/admission/refusal path, not training. These are the same frozen models.',
    'Classical specification has a known environment provenance mismatch; common physics/hold apply, controller interfaces differ.',
    'Classical requests can exceed +/-4 m/s², while RL requests are bounded. Correction magnitudes include this amplitude limiting and do not isolate additional position/velocity protection.',
    'Compact normalized metrics are reaggregated here; the previously accepted raw audit is not rerun or certified by this tool.',
    'Historical documentation still describes a prepared pilot; this candidate does not edit or supersede historical files.',
]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summary(rows, mode):
    audit.require(mode in MODES and all(r['filter'] == mode for r in rows), 'summary mode')
    m = [r['metrics'] for r in rows]
    safe = [v['safety'] for v in m]
    n = len(m)
    success = sum(v['success_episode'] for v in m)
    horizons = sum(v['completion']['reason'] == 'horizon' for v in m)
    resolved = sum(v['exceeds_024_resolved'] for v in safe)
    safe_success = sum(v['success_episode'] and not v['safety']['exceeds_024_resolved'] for v in m)
    accepted = sum(v['accepted_decisions'] for v in safe) if mode == 'on' else None
    changes = sum(v['intervention_count'] for v in safe) if mode == 'on' else None
    return dict(filter=mode, episodes=n, successes=success, success_rate=ratio(success, n),
        horizons=horizons, horizon_rate=ratio(horizons, n),
        horizon_hold_failures=sum(v['completion']['reason'] == 'horizon' and not v['success_episode'] for v in m),
        position_limit=sum('position_limit' in v['completion']['stop_reasons'] for v in m),
        velocity_limit=sum('velocity_limit' in v['completion']['stop_reasons'] for v in m),
        strict_exceedances=sum(v['exceeds_024'] for v in safe), resolved_exceedances=resolved,
        resolved_exceedance_rate=ratio(resolved, n), successes_without_resolved_exceedance=safe_success,
        safe_success_rate=ratio(safe_success, n), filter_decisions=accepted, interventions=changes,
        intervention_rate=ratio(changes, accepted), refusals=sum(v['refusal_count'] for v in safe),
        legacy_max_abs_position_interval_m=max((v['max_abs_position_interval'] for v in m), default=None),
        safety_max_abs_position_interval_m=max((v['max_abs_position_interval'] for v in safe), default=None))


def index_pairs(rows):
    """Reusable small-fixture core; production requires the complete frozen roster."""
    pairs = defaultdict(dict)
    for row in rows:
        audit.require(row['stage'] == 'main', 'main/timing cannot be mixed')
        audit.require(row['model_id'] in MODELS and row['group'] in GROUPS and row['filter'] in MODES, 'unknown roster identity')
        pair = pairs[row['model_id'], row['case_id']]
        audit.require(row['filter'] not in pair, 'duplicate episode')
        audit.validate_metrics(row['metrics'], row['filter'])
        audit.exact(row['flags'], audit.flags(row['metrics']), 'flags')
        pair[row['filter']] = row
    for modes in pairs.values():
        audit.require(set(modes) == set(MODES), 'incomplete pair')
        for key in ('state_sha256', 'model_sha256', 'training_seed', 'group'):
            audit.exact(modes['off'][key], modes['on'][key], 'pair binding: ' + key)
    return [pairs[k] for k in sorted(pairs)]


def paired_summary(pairs):
    counts = Counter()
    for p in pairs:
        off, on = (p[m]['metrics'] for m in MODES)
        key = {(True, True): 'both_success', (False, True): 'on_only_success',
               (True, False): 'off_only_success', (False, False): 'neither_success'}[
                   off['success_episode'], on['success_episode']]
        counts[key] += 1
        if off['success_episode'] and off['safety']['exceeds_024_resolved']:
            counts['off_success_with_resolved_exceedance'] += 1
            counts['off_only_success_with_resolved_exceedance'] += not on['success_episode']
    result = {k: counts[k] for k in PAIR_COLUMNS[3:] if k != 'paired_success_gain'}
    return dict(pair_count=len(pairs), **result,
        paired_success_gain=ratio(counts['on_only_success'] - counts['off_only_success'], len(pairs)))


def intervention_summary(rows):
    audit.require(all(r['filter'] == 'on' for r in rows), 'interventions only defined for on')
    safe = [r['metrics']['safety'] for r in rows]
    accepted = sum(v['accepted_decisions'] for v in safe)
    changes = sum(v['intervention_count'] for v in safe)
    total = math.fsum(v['intervention_abs_sum'] for v in safe)
    return dict(episodes=len(rows), accepted_decisions=accepted, interventions=changes,
        intervention_rate=ratio(changes, accepted), refusals=sum(v['refusal_count'] for v in safe),
        correction_abs_sum_m_s2=total, correction_abs_mean_m_s2=ratio(total, changes),
        correction_abs_max_m_s2=max((v['intervention_abs_max'] for v in safe), default=None))


def build_tables(rows):
    """Calculate, never copy baseline numbers. Stable even if input rows reorder."""
    rows = sorted(rows, key=lambda r: (r['model_id'], r['case_id'], r['filter']))
    pairs = index_pairs(rows)
    by_controller = [m for m in MODELS if any(r['model_id'] == m for r in rows)]
    by_group = [g for g in GROUPS if any(r['group'] == g for r in rows)]
    tables = {name: [] for name in COLUMNS}
    tables['summary_overall.csv'] = [summary([r for r in rows if r['filter'] == mode], mode) for mode in MODES]
    for col, values, name, field in (
        ('controller', by_controller, 'summary_by_controller.csv', 'model_id'),
        ('group', by_group, 'summary_by_group.csv', 'group')):
        for value in values:
            for mode in MODES:
                tables[name].append({col: value, **summary([r for r in rows if r[field] == value and r['filter'] == mode], mode)})
    for model in by_controller:
        for group in by_group:
            for mode in MODES:
                selected = [r for r in rows if (r['model_id'], r['group'], r['filter']) == (model, group, mode)]
                tables['summary_controller_group.csv'].append(dict(controller=model, group=group, **summary(selected, mode)))
    selections = [('overall', 'all', pairs)]
    selections += [('controller', m, [p for p in pairs if p['on']['model_id'] == m]) for m in by_controller]
    selections += [('group', g, [p for p in pairs if p['on']['group'] == g]) for g in by_group]
    for scope, name, selected in selections:
        tables['paired_outcomes.csv'].append(dict(scope=scope, id=name, **paired_summary(selected)))
        tables['intervention_summary.csv'].append(dict(scope=scope, id=name,
            **intervention_summary([p['on'] for p in selected])))
    return tables


def validate_input(dataset):
    audit.require(dataset['stage'] == 'main', 'reporting accepts main only')
    analysis = audit.analyze(dataset, audit.Protocol())
    audit.require(dataset['scientific_sha256'] == SCIENTIFIC_SHA, 'unexpected scientific SHA')
    counts = Counter((r['model_id'], r['group'], r['filter']) for r in dataset['episodes'])
    audit.require(counts == Counter({(m, g, f): 20 for m in MODELS for g in GROUPS for f in MODES}), 'reporting roster')
    audit.require((analysis['episodes'], analysis['jobs'], len(analysis['pairs'])) == (1960, 98, 980), 'main counts')
    return analysis


def regression_invariants(tables):
    # ONLY assertions for SCIENTIFIC_SHA, never inputs to calculation.
    expected = [dict(episodes=980, successes=441, horizons=446, horizon_hold_failures=5,
        position_limit=534, velocity_limit=0, resolved_exceedances=585, successes_without_resolved_exceedance=394),
        dict(episodes=980, successes=796, horizons=980, horizon_hold_failures=184,
        position_limit=0, velocity_limit=0, resolved_exceedances=0, successes_without_resolved_exceedance=796,
        filter_decisions=980000, interventions=115355, refusals=0)]
    for row, reference in zip(tables['summary_overall.csv'], expected):
        audit.exact({k: row[k] for k in reference}, reference, 'overall regression')
    p = tables['paired_outcomes.csv'][0]
    reference = dict(both_success=434, on_only_success=362, off_only_success=7, neither_success=177,
                     off_only_success_with_resolved_exceedance=7)
    audit.exact({k: p[k] for k in reference}, reference, 'paired regression')


def write_json(path, value):
    path.write_bytes(audit.canonical(value) + b'\n')


def write_csv(path, columns, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            audit.require(set(row) == set(columns), 'CSV schema')
            audit.canonical(row)  # rejects NaN/Infinity before writing
            writer.writerow(row)


def render_figures(tables, output):
    """Lazy rendering; scope font cache to a disposable directory in the output."""
    with tempfile.TemporaryDirectory(prefix='.mpl-', dir=output) as cache:
        old = os.environ.get('MPLCONFIGDIR')
        os.environ['MPLCONFIGDIR'] = cache
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                                 'svg.hashsalt': SCHEMA, 'axes.spines.top': False,
                                 'axes.spines.right': False}):
                _render(tables, output, plt)
        finally:
            if old is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = old


def _render(tables, output, plt):
    def save(fig, name):
        try:
            fig.savefig(output / (name + '.png'), dpi=160, metadata={'Software': SCHEMA})
            fig.savefig(output / (name + '.svg'), metadata={'Date': None, 'Creator': SCHEMA})
        finally:
            plt.close(fig)

    # Close our own figures also if drawing fails before save; do not close callers'.
    existing = set(plt.get_fignums())
    try:
        fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
        overall = tables['summary_overall.csv']
        for row, offset in zip(overall, (-.19, .19)):
            vals = [row[k] * 100 for k in ('success_rate', 'safe_success_rate', 'resolved_exceedance_rate')]
            bars = ax.bar([i+offset for i in range(3)], vals, .38, color=COLORS[row['filter']], label='filter ' + row['filter'])
            ax.bar_label(bars, fmt='%.1f%%', padding=3)
        ax.set(xticks=range(3), xticklabels=['Успех задачи', 'Успех без resolved\nпревышения |x|', 'Resolved превышение\n|x| > 0.24 + 10⁻¹² м'],
               ylabel='Доля эпизодов, % (n = 980 на режим)', ylim=(0, 103), title='Успех и безопасность — отдельные показатели')
        ax.legend(); ax.grid(axis='y', alpha=.2)
        save(fig, FIGURES[0])

        fig, ax = plt.subplots(figsize=(11, 5.3), layout='constrained')
        ax.axvspan(-.5, .5, color='#eaf2fa', zorder=0)
        ax.axvline(.5, color='#8799ad', linestyle='--', linewidth=1)
        for mode, offset in zip(MODES, (-.19, .19)):
            lookup = {r['group']: r for r in tables['summary_by_group.csv'] if r['filter'] == mode}
            bars = ax.bar([i+offset for i in range(7)], [lookup[g]['success_rate']*100 for g in GROUPS], .38, color=COLORS[mode], label='filter '+mode)
            ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=3)
        ax.set(xticks=range(7), xticklabels=GROUP_LABELS, ylabel='Успех, % (n = 140 на группу и режим)', ylim=(0, 113), title='Nominal и шесть групп возмущённых стартов')
        ax.legend(loc='upper right', ncol=2); ax.grid(axis='y', alpha=.2)
        save(fig, FIGURES[1])

        lookup = {(r['controller'], r['group'], r['filter']): r for r in tables['summary_controller_group.csv']}
        for name, intervention in ((FIGURES[2], False), (FIGURES[3], True)):
            data = [[100 * (lookup[m,g,'on']['intervention_rate'] if intervention else
                      lookup[m,g,'on']['success_rate'] - lookup[m,g,'off']['success_rate']) for g in GROUPS] for m in MODELS]
            # Explicit margins avoid backend-dependent constrained-layout clipping
            # of the long controller names next to a colorbar (PNG and SVG).
            fig, ax = plt.subplots(figsize=(11.5, 6.5))
            fig.subplots_adjust(left=.16, right=.88, bottom=.16, top=.91)
            im = ax.imshow(data, vmin=0 if intervention else -100, vmax=100,
                           cmap='YlGnBu' if intervention else 'BrBG', aspect='auto')
            for i, line in enumerate(data):
                for j, value in enumerate(line):
                    rgba = im.cmap(im.norm(value))
                    luminance = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    ax.text(j, i, f'{value:.1f}' if intervention else f'{value:+.0f}', ha='center', va='center', color='black' if luminance > .55 else 'white')
            ax.set(xticks=range(7), xticklabels=GROUP_LABELS, yticks=range(7), yticklabels=MODELS,
                title='Вмешательства filter on / принятые решения, %' if intervention else 'Парное изменение успеха: on − off, п.п. (n = 20 пар/ячейку)')
            fig.colorbar(im, ax=ax, label='% принятых решений' if intervention else 'Процентные пункты')
            save(fig, name)

        fig, ax = plt.subplots(figsize=(11, 5.4), layout='constrained')
        records = {r['id']: r for r in tables['paired_outcomes.csv'] if r['scope'] == 'group'}
        bottom = [0]*7
        for key, label, color in (
            ('both_success', 'Оба успешны', '#4b7eae'), ('on_only_success', 'Успех только on', COLORS['on']),
            ('off_only_success', 'Успех только off', COLORS['off']), ('neither_success', 'Оба неуспешны', '#d7dce3')):
            values = [records[g][key] for g in GROUPS]
            ax.bar(range(7), values, bottom=bottom, label=label, color=color)
            bottom = [a+b for a,b in zip(bottom, values)]
        ax.set(xticks=range(7), xticklabels=GROUP_LABELS, ylim=(0, 140),
               ylabel='Число пар (n = 140 на группу)', title='Парные исходы одного контроллера на одном старте')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2)
        save(fig, FIGURES[4])
    finally:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)


def publish(dataset, output):
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError('output already exists: ' + str(output))
    validate_input(dataset)
    tables = build_tables(dataset['episodes'])
    regression_invariants(tables)
    temporary = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    try:
        for name, columns in COLUMNS.items():
            write_csv(temporary / name, columns, tables[name])
        render_figures(tables, temporary)
        payloads = {p.name: audit.file_sha(p) for p in sorted(temporary.iterdir())}
        summary_data = dict(schema=SCHEMA, scientific_sha256=dataset['scientific_sha256'],
            counts=dict(episodes=len(dataset['episodes']), pairs=tables['paired_outcomes.csv'][0]['pair_count'],
                jobs=len(tables['summary_controller_group.csv']), controllers=len(MODELS), groups=len(GROUPS)),
            definitions=DEFINITIONS, overall=tables['summary_overall.csv'], paired=tables['paired_outcomes.csv'][0],
            invariants=dict(frozen_identity='passed', pinned_scientific_sha='passed', regression_totals='passed',
                complete_pairs=True, episodes_per_cell=20, timing_included=False),
            verification=dict(normalized_consistency='passed', envelope_integrity='passed',
                compact_consistency='not_run', source_provenance='not_revalidated', raw_metric_recomputation='not_run'),
            tables=list(COLUMNS), figures=[n+ext for n in FIGURES for ext in ('.png', '.svg')],
            files_sha256=payloads, interpretation_limits=LIMITS,
            hash_scope='Summary hashes tables/figures; MANIFEST.json hashes all payloads including this summary. Manifest SHA is on stdout.')
        write_json(temporary / 'analysis_summary.json', summary_data)
        payloads = {**payloads, 'analysis_summary.json': audit.file_sha(temporary / 'analysis_summary.json')}
        manifest = dict(schema=SCHEMA, scientific_sha256=dataset['scientific_sha256'],
            normalized_envelope_sha256=dataset['envelope_sha256'], source_provenance=dataset['provenance'],
            code_sha256={str(p.relative_to(audit.ROOT)): audit.file_sha(p) for p in
                         (Path(__file__).resolve(), Path(audit.__file__).resolve())}, files_sha256=payloads)
        write_json(temporary / 'MANIFEST.json', manifest)
        result = dict(schema=SCHEMA, scientific_sha256=dataset['scientific_sha256'],
                      counts=summary_data['counts'], files=len(payloads)+1,
                      manifest_sha256=audit.file_sha(temporary / 'MANIFEST.json'))
        if os.path.lexists(output):
            raise FileExistsError('output appeared during generation: ' + str(output))
        temporary.rename(output)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normalized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = publish(audit.read_json(args.normalized), args.output)
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, 'reporting failed: ' + str(exc) + '\n')
    print(audit.canonical(result).decode())


if __name__ == '__main__':
    main()
