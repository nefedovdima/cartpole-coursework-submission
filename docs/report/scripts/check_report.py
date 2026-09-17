"""Read-only checks of the report's compact evidence. No scientific runtime imports."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
checks = []

def passed(name):
    checks.append(name)
    print('PASS:', name)

def rows(path):
    with (ROOT / 'data' / path).open() as stream:
        return list(csv.DictReader(stream))

def obj(path):
    return json.loads((ROOT / path).read_text())

def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)

def main():
    from source_inputs import validate_sources
    passed('source integrity: '+str(validate_sources()))
    expected_runs = {'main': 15, 'stage1': 12, 'confirmation': 6}
    for stage, count in expected_runs.items():
        rr = rows(stage + '/runs.csv')
        assert len(rr) == count == len({r['run_id'] for r in rr})
        for r in rr:
            assert int(r['transitions']) == int(r['buffer_size']) == 100000
            expected = 95000 if 'warmup5000' in r.get('variant', '') else 99872
            assert int(r['updates']) == expected
            assert r['status'] == 'completed'
    passed('33 audited training counters; warmup updates distinguished')
    main_rows = rows('main/evaluation_summary.csv')
    methods = [('SAC', 'on'), ('TQC', 'on'), ('DQN', 'on'), ('DDPG', 'on'), ('SAC', 'off')]
    expected = [[20,20,20], [20,20,20], [20,14,20], [20,0,0], [0,0,0]]
    table = []
    for (alg, mode), best in zip(methods, expected):
        for seed in range(3):
            values = []
            for label, ef in [('best','on'), ('best','off'), ('last','on'), ('last','off')]:
                rr = [r for r in main_rows if r['category'] == 'external' and r['algorithm'] == alg
                      and r['training_filter'] == mode and r['seed'] == str(seed)
                      and r['checkpoint_label'] == label and r['evaluation_filter'] == ef
                      and r['subset'] == 'validation']
                assert len(rr) == 1 and int(rr[0]['episodes']) == 20
                values.append(rr[0]['successes'])
                if label == 'best' and ef == mode:
                    assert int(rr[0]['successes']) == best[seed]
            table.append(f'{alg}-{mode} & {seed} & ' + ' & '.join(values) + r' \\')
    assert (ROOT/'figures/main_table.tex').read_text() == '\n'.join(table)+'\n'+r'\bottomrule'+'\n'
    passed('main validation20 successes and generated 15-row table')
    detail = rows('stage1/detailed_comparison.csv')
    selection = obj('data/stage1/selection_audit.json')
    for group in selection.values():
        assert group['named_excluded']
        for variant, key in group['keys'].items():
            rr = [r for r in detail if r['category'] == 'external' and r['variant'] == variant
                  and r['subset'] == 'validation' and r['checkpoint_label'] == 'best'
                  and r['evaluation_filter'] == r['training_filter']]
            assert len(rr) == 3 and {r['seed'] for r in rr} == {'0','1','2'}
            assert all(int(r['episodes']) == 20 for r in rr)
            p = [int(r['successes']) / 20 for r in rr]
            hf = [float(r['mean_hold_final']) for r in rr]
            hl = [float(r['mean_hold_longest']) for r in rr]
            actual = [min(p),sum(p)/3,min(hf),sum(hf)/3,sum(hl)/3]
            for a,b in zip(actual,key): close(a,b)
        win = group['selected_for_confirmation']
        assert win.endswith('warmup5000')
        assert tuple(group['keys'][win]) > tuple(group['keys'][group['baseline']])
    passed('all six exact selection tuples recomputed, validation only')
    seeds = rows('confirmation/seeds_0_5.csv')
    for variant, gate in obj('data/confirmation/confirmation_gate.json').items():
        rr = [r for r in seeds if r['variant'] == variant and int(r['seed']) >= 3]
        assert len(rr) == 3 and {r['seed'] for r in rr} == {'3','4','5'}
        accepted = all(int(r['best_successes']) >= 16 for r in rr) and sum(int(r['last_successes']) >= 16 for r in rr) >= 2
        assert not accepted and gate['confirmed'] == accepted
        assert gate['subset'] == 'validation' and gate['states_per_seed'] == 20 and gate['named_excluded']
    hyp = next(r for r in rows('stage1/hypothesis_assessment.csv') if r['variant'] == 'sac_off_warmup5000')
    assert int(hyp['predeclared_diagnostic_seeds_passing']) < 2
    assert hyp['predeclared_hypothesis_supported'] == 'False'
    passed('confirmation gates and unsupported SAC early-coverage mechanism kept separate')
    for seed, expected in [(3, (273.6816135092021,706.0465536492355,.8895,.3325)), (5,(752.568819705203,665.452665557242,7.8915,.0115))]:
        r=next(r for r in seeds if r['algorithm']=='DDPG' and int(r['seed'])==seed)
        for key,value in zip(['best_return','last_return','best_hold_final','last_hold_final'],expected):close(r[key],value)
    components=rows('confirmation/hold_component_summary.csv')
    for seed,v_pass in [(3,.4788557213930348),(5,.5980099502487561)]:
        r=next(r for r in components if r['algorithm']=='DDPG' and int(r['seed'])==seed and r['label']=='last' and r['window']=='last2')
        assert int(r['episodes'])==int(r['present_episodes'])==20
        for key in ['angle_pass_fraction','omega_pass_fraction','x_pass_fraction']:close(r[key],1)
        close(r['v_pass_fraction'],v_pass)
    trace=rows('confirmation/ddpg_seed5_trajectory.csv')
    assert Counter(r['label'] for r in trace)==Counter(best=1001,last=1001)
    for label in ['best','last']:
        rr=[r for r in trace if r['label']==label]
        assert float(rr[0]['time_s'])==0 and float(rr[-1]['time_s'])==10
        assert all(float(a['time_s'])<float(b['time_s']) for a,b in zip(rr,rr[1:]))
    passed('DDPG examples: reward/hold numbers, velocity criterion, saved time grid')

    cells = rows('robustness/summary_controller_group.csv')
    models = [f'{a}_seed{s}' for a in ('SAC','TQC') for s in range(3)] + ['classical']
    groups = ['nominal','position','velocity','angle','angular_velocity','boundary','joint']
    assert len(cells) == 98
    assert {(r['controller'],r['group'],r['filter']) for r in cells} == {(m,g,f) for m in models for g in groups for f in ('off','on')}
    assert all(int(r['episodes']) == 20 for r in cells)
    totals = {r['filter']: r for r in rows('robustness/summary_overall.csv')}
    keys = ['successes','horizons','position_limit','velocity_limit','strict_exceedances','resolved_exceedances','successes_without_resolved_exceedance','refusals']
    for mode in ('off','on'):
        rr = [r for r in cells if r['filter'] == mode]
        for key in keys: assert sum(int(r[key]) for r in rr) == int(totals[mode][key]), key
    assert [int(totals[f]['successes']) for f in ('off','on')] == [441,796]
    assert [int(totals[f]['successes_without_resolved_exceedance']) for f in ('off','on')] == [394,796]
    assert [int(totals[f]['resolved_exceedances']) for f in ('off','on')] == [585,0]
    assert [int(totals[f]['horizons']) for f in ('off','on')] == [446,980]
    assert [int(totals[f]['position_limit']) for f in ('off','on')] == [534,0]
    assert all(int(totals[f]['refusals']) == 0 for f in ('off','on'))
    assert int(totals['on']['horizons'])-int(totals['on']['successes']) == 184
    assert totals['off']['intervention_rate'] == ''
    group_counts = [sum(int(r['successes']) for r in cells if r['group']==g and r['filter']=='on') for g in groups]
    assert group_counts == [139,122,109,102,111,97,116]
    for family, expected in [('SAC',(279,392)),('TQC',(101,319)),('classical',(61,85))]:
        assert tuple(sum(int(r['successes']) for r in cells if r['controller'].startswith(family) and r['filter']==f) for f in ('off','on')) == expected
    for family, count in [('SAC',60),('TQC',59),('classical',20)]:
        assert sum(int(r['successes']) for r in cells if r['controller'].startswith(family)
                   and r['filter']=='on' and r['group']=='nominal') == count
    paired = rows('robustness/paired_outcomes.csv')[0]
    assert [int(paired[k]) for k in ['pair_count','both_success','on_only_success','off_only_success','neither_success','off_only_success_with_resolved_exceedance']] == [980,434,362,7,177,7]
    interventions = rows('robustness/intervention_summary.csv')
    per = [r for r in interventions if r['scope'] == 'controller']
    assert sum(int(r['interventions']) for r in per) == 115355
    assert sum(int(r['accepted_decisions']) for r in per) == 980000
    close(interventions[0]['intervention_rate'],115355/980000)
    for family, numerator, denominator in [('SAC',4209,420000),('TQC',57179,420000),('classical',53967,140000)]:
        rr=[r for r in per if r['id'].startswith(family)]
        assert sum(int(r['interventions']) for r in rr)==numerator
        assert sum(int(r['accepted_decisions']) for r in rr)==denominator
    passed('98 cells, 1960 episodes, 980 pairs; rates, group/family counts, weighted interventions')
    states = obj('data/configs/states.json')['cases']
    assert len(states) == 140 and len({r['state_sha256'] for r in states}) == 140
    assert Counter(r['subset'] for r in states) == Counter({g:20 for g in groups})
    support=Counter();joint=Counter()
    for r in states:
        s=r['initial_state'];x,v,t,w=[s[k] for k in ['cart_position','cart_velocity','pole_angle','pole_angular_velocity']]
        assert abs(x)<=.24-1e-6 and abs(v)<=2-1e-6
        assert x+max(v,0)**2/8<=.24-1e-6 and -x+max(-v,0)**2/8<=.24-1e-6
        outside=sum(abs(a)>b for a,b in zip([x,v,t,w],[.02,.02,.05,.05]))
        if not outside:support[r['subset']]+=1
        if r['subset']=='joint':joint[outside]+=1
    assert support == Counter(nominal=20,position=2)
    assert joint == Counter({2:2,3:10,4:8})
    passed('140 saved starts: K admission and joint training-support classification')
    source_files=[ROOT/'report.tex',ROOT/'references.tex',*sorted((ROOT/'sections').glob('*.tex'))]
    text='\n'.join(p.read_text() for p in source_files)
    cited={x.strip() for group in re.findall(r'\\cite\{([^}]+)\}',text) for x in group.split(',')}
    bib=set(re.findall(r'\\bibitem\{([^}]+)\}',text)); assert cited==bib
    assert len(bib) == 13 and not any(key.startswith('project_') for key in bib)
    labels=set(re.findall(r'\\label\{([^}]+)\}',text)) | set(re.findall(r'\\fig[^\n]*\{(fig:[^}]+)\}',text))
    refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text));assert refs<=labels, refs-labels
    assert {x for x in labels if x.startswith(('eq:','fig:','tab:'))}<=refs
    for link in re.findall(r'\\eref\{([^}]+)\}',text):assert (ROOT/'data'/link).is_file(),link
    for stem in re.findall(r'\\fig\{([^}]+)\}',text): assert (ROOT/'figures'/f'{stem}.pdf').is_file(),stem
    for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
        if '#' not in name: assert (ROOT/name).is_file(),name
    video_stems = set(re.findall(r'\\vref\{([^}]+)\}',text))
    video_manifest = obj('media/VIDEO_MANIFEST.json')
    assert len(video_manifest['videos']) == len(video_stems) == 6
    assert video_stems == {Path(v['file']).stem for v in video_manifest['videos']}
    for v in video_manifest['videos']:
        payload = (ROOT/'media'/v['file']).read_bytes()
        assert len(payload) == v['bytes'] and hashlib.sha256(payload).hexdigest() == v['sha256']
        assert hashlib.sha256((ROOT/'media'/v['poster']).read_bytes()).hexdigest() == v['poster_sha256']
    for name, digest in video_manifest['documentation_sha256'].items():
        assert hashlib.sha256((ROOT/'media'/name).read_bytes()).hexdigest() == digest
    checkout = ROOT.parents[1]
    for name,digest in video_manifest['code_sha256'].items():
        if (checkout/name).is_file():
            assert hashlib.sha256((checkout/name).read_bytes()).hexdigest() == digest
        else:
            print('INFO: renderer source outside standalone report; historical binding retained:',name)
    assert video_manifest['evidence']['scientific_sha256'] == 'c1cac3775426748520db67957e988fb7cf3706148d1cc515eca26e88118fb63b'
    passed('six MP4, six posters, accepted media documentation SHA and relative video links')
    front=(ROOT/'sections/00_front.tex').read_text().split(r'\unnumbered{Введение}')[0]
    assert len(front)<2000
    assert 'БПМИ-233' in text
    assert not re.search(r'/home/|/workspace/',text)
    assert not any(ord(c)<32 and c not in '\n\t' for c in text)
    if (ROOT/'report.aux').exists():
        aux=(ROOT/'report.aux').read_text()
        for label,letter in [('app:main','А'),('app:selection','Б'),('app:repro','В'),('app:videos','Г')]:
            assert '\\newlabel{'+label+'}{{'+letter+'}' in aux,(label,letter)
    passed(f'{len(bib)} bibliography entries all cited; labels, electronic links, abstract, group')
    for p in source_files+list((ROOT/'scripts').glob('*')):
        for i,line in enumerate(p.read_text().splitlines(),1):assert line==line.rstrip(),(p,i)
    passed('authored source whitespace')
    print(json.dumps({'checks':len(checks),'new_training':0,'new_rollouts':0,'raw_reaudit':False},ensure_ascii=False))

if __name__=='__main__':main()
