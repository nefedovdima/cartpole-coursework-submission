"""Figures from pinned compact CSV and saved state samples. No project imports."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'build' / 'mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'svg.hashsalt': 'cartpole-report-20260917', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white'})
COLORS = ['#176C91', '#B35820', '#397B4A']
STYLES = ['-', '--', ':']
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

def read(name):
    with (ROOT / 'data' / name).open() as f:
        return list(csv.DictReader(f))

def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None})
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    from source_inputs import validate_sources
    validate_sources()

    # Accurate schematic: theta=0 downward, bob=(x+l sin theta,-l cos theta).
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    ax.set(xlim=(-.33, .37), ylim=(-.25, .115), aspect='equal'); ax.axis('off')
    ax.plot([-.3,.3],[-.018,-.018],color='.35',lw=2)
    for x in [-.24,.24]:
        ax.plot([x,x],[-.05,.075],ls='--',color='#B35820',lw=1)
        ax.text(x,.09,f'{x:+.2f} м'.replace('.',','),ha='center',fontsize=9)
    ax.add_patch(Rectangle((-.042,-.018),.084,.036,facecolor='#E4EBF0',edgecolor='#176C91'))
    for x in [-.027,.027]:ax.add_patch(Circle((x,-.03),.009,color='.25'))
    angle=.75; bx=.18*math.sin(angle); by=-.18*math.cos(angle)
    ax.plot([0,bx],[0,by],color='#176C91',lw=3)
    ax.add_patch(Circle((bx,by),.011,color='#176C91'))
    ax.add_patch(Circle((0,0),.005,color='black'))
    ax.plot([0,0],[0,-.2],ls=':',color='.5')
    ax.add_patch(Arc((0,0),.13,.13,theta1=-90,theta2=math.degrees(angle)-90,color='.25'))
    ax.text(.023,-.089,r'$\theta$',fontsize=12)
    ax.text(.10,-.054,'l = 0,18 м',rotation=-47,ha='center')
    ax.annotate('',xy=(.18,.038),xytext=(.035,.038),arrowprops={'arrowstyle':'->','color':'#176C91'})
    ax.text(.11,.058,r'$u=\ddot{x}$',ha='center',fontsize=12)
    ax.annotate('',xy=(-.28,-.15),xytext=(-.28,-.07),arrowprops={'arrowstyle':'->'})
    ax.text(-.267,-.115,'g')
    ax.text(.02,-.224,'Угол 0 — вниз; π — вверх. Границы относятся к шарниру.',ha='center',fontsize=8)
    save(fig,'model')

    fig,ax=plt.subplots(figsize=(6.8,2.35));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    boxes=[(.1,1.3,2.1,'Политика /\nклассический\nконтроллер'),(3.05,1.3,2.5,'Фильтр on\nили limiter off'),(6.65,1.3,3.1,'CartPoleEpisode\nи исходный Drake')]
    for x,y,w,t in boxes:
        ax.add_patch(Rectangle((x,y),w,.85,fc='#F1F5F7',ec='#176C91',lw=1.3));ax.text(x+w/2,y+.425,t,ha='center',va='center')
    for x1,x2,t in [(2.2,3.05,'запрос'),(5.55,6.65,'команда')]:
        ax.annotate('',xy=(x2,1.73),xytext=(x1,1.73),arrowprops={'arrowstyle':'->'});ax.text((x1+x2)/2,2.32,t,ha='center',fontsize=8)
    ax.annotate('',xy=(1.15,1.3),xytext=(8.2,1.3),arrowprops={'arrowstyle':'->','connectionstyle':'bar,fraction=-.12'})
    ax.text(4.7,.22,'Состояние → наблюдение; фильтр читает исходные x, v',ha='center',fontsize=8)
    ax.text(5,2.83,'Replay: запрос a; журнал: запрос, команда, фактически исполненное u',ha='center',fontsize=8)
    save(fig,'command_path')

    hist=read('main/validation_history.csv')
    fig,axs=plt.subplots(3,2,figsize=(6.7,6.0),sharex=True,sharey=True,layout='constrained')
    methods=[('SAC','on'),('TQC','on'),('DQN','on'),('DDPG','on'),('SAC','off')]
    for ax,(alg,mode) in zip(axs.flat,methods):
        for seed in range(3):
            rows=[r for r in hist if r['algorithm']==alg and r['training_filter']==mode and int(r['seed'])==seed and r['measured']=='True']
            rows.sort(key=lambda r:int(r['transitions']));assert len(rows)==11
            ax.plot([int(r['transitions'])/1000 for r in rows],[int(r['validation_successes']) for r in rows],color=COLORS[seed],ls=STYLES[seed],marker=['o','s','^'][seed],ms=3,label=f'seed {seed}')
        ax.set_title(f'{alg}, обучение {mode}');ax.set(ylim=(-.6,20.8),yticks=[0,5,10,15,20],xticks=[0,20,40,60,80,100]);ax.grid(alpha=.18)
    axs[2,1].axis('off');h,l=axs[0,0].get_legend_handles_labels();axs[2,1].legend(h,l,loc='center',frameon=False)
    fig.supxlabel('Обучающие переходы, тыс.');fig.supylabel('Успехи на validation20')
    save(fig,'main_learning')

    seeds=read('confirmation/seeds_0_5.csv')
    fig,axs=plt.subplots(1,2,figsize=(6.7,2.7),sharey=True,layout='constrained')
    for ax,alg,mode in zip(axs,['DDPG','SAC'],['on','off']):
        rows=sorted([r for r in seeds if r['algorithm']==alg and r['evaluation_filter']==mode],key=lambda r:int(r['seed']));assert len(rows)==6
        x=np.arange(6)
        ax.bar(x-.16,[int(r['best_successes']) for r in rows],.32,color=COLORS[0],label='best')
        ax.bar(x+.16,[int(r['last_successes']) for r in rows],.32,color=COLORS[1],hatch='//',label='last')
        ax.axhline(16,color='.3',ls=':',lw=1);ax.axvline(2.5,color='.6',lw=.8)
        ax.set(title=f'{alg}-{mode}, warmup5000',xticks=x,ylim=(0,21),yticks=[0,5,10,15,20],xlabel='Seed: 0–2 отбор; 3–5 confirmation')
    axs[0].set_ylabel('Успехи на validation20');axs[1].legend(*axs[0].get_legend_handles_labels(),loc='upper right',ncol=2,fontsize=7)
    save(fig,'confirmation')

    trace=read('confirmation/ddpg_seed5_trajectory.csv')
    fig,axs=plt.subplots(2,1,figsize=(6.7,3.05),sharex=True,layout='constrained')
    for label,c,ls in [('best',COLORS[0],'-'),('last',COLORS[1],'--')]:
        r=[r for r in trace if r['label']==label and float(r['time_s'])>=8]
        t=[float(r['time_s']) for r in r]
        axs[0].plot(t,[float(r['v_m_s']) for r in r],color=c,ls=ls,label=label)
        axs[1].plot(t,[math.degrees(math.atan2(math.sin(float(r['theta_rad'])-math.pi),math.cos(float(r['theta_rad'])-math.pi))) for r in r],color=c,ls=ls)
    for ax,lim in zip(axs,[.2,10]):
        ax.axhspan(-lim,lim,color='#397B4A',alpha=.08)
        ax.axhline(lim,color='.4',ls=':',lw=.8);ax.axhline(-lim,color='.4',ls=':',lw=.8);ax.grid(alpha=.12)
    axs[0].set(ylabel='v, м/с',ylim=(-.4,.4));axs[0].legend(ncol=2,loc='upper center')
    axs[1].set(ylabel='Ошибка угла, °',xlabel='Время, с',ylim=(-12,12),xlim=(8,10))
    save(fig,'ddpg_criterion')

    rr=read('robustness/summary_controller_group.csv')
    models=[f'{a}_seed{s}' for a in ['SAC','TQC'] for s in range(3)]+['classical']
    groups=['nominal','position','velocity','angle','angular_velocity','boundary','joint']
    labels=['Nominal','Координата','Скорость','Угол','Угл. скорость','Граница','Совместно']
    fig,axs=plt.subplots(1,2,figsize=(6.9,4.1),sharey=True,layout='constrained')
    for ax,mode in zip(axs,['off','on']):
        a=np.array([[int(next(r['successes'] for r in rr if r['controller']==m and r['group']==g and r['filter']==mode)) for g in groups] for m in models])
        im=ax.imshow(a,vmin=0,vmax=20,cmap='Blues',aspect='equal')
        for i in range(7):
            for j in range(7):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white' if a[i,j]>=12 else 'black',fontsize=8)
        ax.set_xticks(range(7),labels,rotation=60,ha='right',fontsize=8)
        ax.set_yticks(range(7),[m.replace('_seed',' s') if m!='classical' else 'Классика' for m in models],fontsize=8)
        ax.set_title(f'Исполнение {mode}: успехи из 20')
    save(fig,'robustness_matrix')

    overall=read('robustness/summary_overall.csv');o={r['filter']:r for r in overall}
    fig,axs=plt.subplots(1,2,figsize=(6.7,2.8),layout='constrained')
    cats=[('successes','Успех'),('successes_without_resolved_exceedance','Успех без\nпревышения'),('horizons','Горизонт'),('resolved_exceedances','Превышение\nс допуском')]
    for k,mode in enumerate(['off','on']):
        vals=[int(o[mode][key])/980*100 for key,_ in cats]
        bars=axs[0].bar(np.arange(4)+(k-.5)*.35,vals,.35,label=mode,color=COLORS[k],hatch='//' if k else '')
        for b,v in zip(bars,vals):axs[0].text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}',ha='center',fontsize=7)
    axs[0].set_xticks(range(4),[v for _,v in cats],fontsize=7);axs[0].set(ylabel='% от 980 эпизодов режима',ylim=(0,113));axs[0].legend(ncol=2,loc='upper left',fontsize=7)
    p=next(r for r in read('robustness/paired_outcomes.csv') if r['scope']=='overall')
    keys=['both_success','on_only_success','off_only_success','neither_success']
    vals=[int(p[k]) for k in keys]
    bars=axs[1].barh(range(4),vals,color=['#176C91','#397B4A','#B35820','#888888'])
    axs[1].set_yticks(range(4),['Оба успешны','Только on','Только off','Оба неуспешны'],fontsize=8);axs[1].invert_yaxis();axs[1].set(xlim=(0,520),xlabel='Число пар (всего 980)')
    for b,v in zip(bars,vals):axs[1].text(v+7,b.get_y()+b.get_height()/2,str(v),va='center',fontsize=8)
    save(fig,'robustness_outcomes')

    # Deterministic LaTeX table for the full main validation matrix.
    summary=read('main/evaluation_summary.csv')
    lines=[]
    for alg,mode in methods:
        for seed in range(3):
            row=[]
            for label,ef in [('best','on'),('best','off'),('last','on'),('last','off')]:
                rs=[r for r in summary if r['category']=='external' and r['algorithm']==alg and r['training_filter']==mode and r['seed']==str(seed) and r['checkpoint_label']==label and r['evaluation_filter']==ef and r['subset']=='validation']
                assert len(rs)==1,(alg,mode,seed,label,ef)
                row.append(rs[0]['successes'])
            lines.append(f'{alg}-{mode} & {seed} & '+' & '.join(row)+r' \\')
    (OUT/'main_table.tex').write_text('\n'.join(lines)+'\n'+r'\bottomrule'+'\n')
    (ROOT/'build'/'figure_build.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'numpy':np.__version__,'figures':7,'source_files_verified':len(manifest['files']),'new_rollouts':0},indent=2)+'\n')
    print('PASS: sources verified; 7 figures and main matrix generated without simulation')

if __name__=='__main__':main()
