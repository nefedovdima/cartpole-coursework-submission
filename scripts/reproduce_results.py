"""Recompute displayed statistics from compact audited tables, without simulation."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]

def rows(phase):
    with (ROOT/'results/tables'/phase/'evaluation_summary.csv').open(newline='') as f:return list(csv.DictReader(f))

def selected(data, prefix, label='best', mode=None):
    result=[r for r in data if r['category']=='external' and r['run_id'].startswith(prefix)
            and r['checkpoint_label']==label and r['subset']=='validation'
            and r['evaluation_filter']==(mode or r['training_filter'])]
    result.sort(key=lambda r:int(r['seed']))
    if len(result)!=3 or len({r['seed'] for r in result})!=3 or any(int(r['episodes'])!=20 for r in result):
        raise ValueError('selection needs three seeds, each with exactly 20 validation states')
    return result

def score(data):
    if len(data)!=3 or any(r['subset']!='validation' or int(r['episodes'])!=20 for r in data):
        raise ValueError('validation-only selection')
    successes=[int(r['successes'])/20 for r in data]
    final=[float(r['mean_hold_final']) for r in data]
    return [min(successes),mean(successes),min(final),mean(final),mean(float(r['mean_hold_longest']) for r in data)]

def gate(best,last):
    if any(r['subset']!='validation' or int(r['episodes'])!=20 for r in best+last):
        raise ValueError('validation-only confirmation')
    if len(best)!=3 or len(last)!=3 or [r['seed'] for r in best]!=[r['seed'] for r in last]:raise ValueError('unpaired seeds')
    return all(int(r['successes'])>=16 for r in best) and sum(int(r['successes'])>=16 for r in last)>=2

def compute():
    data={p:rows(p) for p in ('main','stage1','confirmation')}
    summary={'main':{},'stage1':{},'confirmation':{},'selection':{}}
    for method in ('SAC_on','TQC_on','DDPG_on','DQN_on','SAC_off'):
        prefix='evaluate_train_'+method+'_seed'
        summary['main'][method]={label:[int(r['successes']) for r in selected(data['main'],prefix,label,mode='on')]
                                 for label in ('best','last')}
    saved=json.loads((ROOT/'results/tables/stage1/selection_audit.json').read_text())
    for method in ('DDPG_on','SAC_off'):
        keys={method.lower()+'_baseline':score(selected(data['main'],'evaluate_train_'+method+'_seed'))}
        for factor in ('warmup5000','lr1e4'):
            name=method.lower()+'_'+factor
            selected_rows=selected(data['stage1'],'evaluate_train_'+name+'_seed')
            keys[name]=score(selected_rows)
            summary['stage1'][name]=dict(best=[int(r['successes']) for r in selected_rows],evaluation_mode='training mode')
        for name,key in keys.items():
            if not all(math.isclose(a,b,rel_tol=0,abs_tol=1e-12) for a,b in zip(key,saved[method]['keys'][name])):
                raise ValueError('archived selection score differs: '+name)
        baseline=method.lower()+'_baseline'
        winner=max(keys,key=lambda k:tuple(keys[k]))
        if tuple(keys[winner])<=tuple(keys[baseline]):winner=baseline
        if winner!=saved[method]['selected_for_confirmation']:raise ValueError('archived winner differs')
        summary['selection'][method]=dict(keys=keys,winner=winner,subset='validation',named_excluded=True)
        name=method.lower()+'_warmup5000';prefix='evaluate_train_confirm_'+name+'_seed'
        best=selected(data['confirmation'],prefix);last=selected(data['confirmation'],prefix,'last')
        summary['confirmation'][name]=dict(best=[int(r['successes']) for r in best],last=[int(r['successes']) for r in last],confirmed=gate(best,last),subset='validation',evaluation_mode='training mode')
    expected={'SAC_on':[20,20,20],'TQC_on':[20,20,20],'DDPG_on':[20,0,0],'DQN_on':[20,14,20],'SAC_off':[0,0,0]}
    for method,values in expected.items():
        if summary['main'][method]['best']!=values:raise ValueError('main result differs: '+method)
    archived=json.loads((ROOT/'results/tables/confirmation/confirmation_gate.json').read_text())
    for name,value in summary['confirmation'].items():
        if value['confirmed']!=archived[name]['confirmed']:raise ValueError('gate result differs')
        if value['best']!=[r['best_successes'] for r in archived[name]['rows']]:raise ValueError('best counts differ')
        if value['last']!=[r['last_successes'] for r in archived[name]['rows']]:raise ValueError('last counts differ')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    result=compute()
    if a.check:
        if result!=json.loads((ROOT/'results/tables/reproduced.json').read_text()):raise ValueError('displayed results stale')
        print('All main counts, validation-only selection scores and confirmation gates agree.')
    elif a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    else:print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
