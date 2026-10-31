"""Standalone scientific figures from completed, audited JSON outputs.

Run in a plotting environment. Does not import the inference stack or models.
"""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)

def csv_file(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(a):
    run=Path(a.run_dir);audit=Path(a.audit_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    checked=json.loads((audit/'audit.json').read_text());assert checked['passed']
    result=json.loads((run/'main_results.json').read_text());regional=json.loads((run/'region_effects.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    models=['P_base','P_rich'];conditions=['repeat4','nested16','dense16'];colors=['#0072B2','#D55E00']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),gridspec_kw={'width_ratios':[1,1.4]},layout='constrained')
    x=np.arange(3)
    rows=[]
    for mi,m in enumerate(models):
        values=[]
        for c in conditions:
            s=result['summary'][m+'__'+c];values.append(s['accuracy_percent'])
            rows.append(dict(model=m,condition=c,n=s['n'],correct=s['correct'],accuracy_percent=s['accuracy_percent'],mean_margin=s['mean_margin'],mean_rank=s['mean_rank']))
        bars=axes[0].bar(x+(mi-.5)*.36,values,width=.36,color=colors[mi],label=m)
        axes[0].bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=8,padding=3)
    axes[0].set_xticks(x,['Repeated 4\n16 slots','Nested real\n16 slots','Historical\nuniform 16'])
    axes[0].set_ylabel('Accuracy (%)');axes[0].set_ylim(0,max(70,max(r['accuracy_percent'] for r in rows)+8));axes[0].legend(frameon=False)
    axes[0].set_title('Fixed checkpoints; all 20 candidates recomputed')
    names=['G_base','G_rich','S_nested','I']
    labels=['Added observations: base','Added observations: rich','Rich - base: nested','Supervision x observations']
    contrasts=[]
    for y,name in enumerate(names):
        c=result['contrasts'][name+'__margin']
        axes[1].hlines(y,*c['ci9875'],color='#9ecae1',linewidth=2)
        axes[1].hlines(y,*c['ci95'],color='#0072B2',linewidth=4)
        axes[1].plot(c['mean'],y,'o',color='black',markersize=5)
        for measure in ('margin','correct'):
            v=result['contrasts'][name+'__'+measure]
            contrasts.append(dict(contrast=name,measure=measure,mean=v['mean'],ci95_low=v['ci95'][0],ci95_high=v['ci95'][1],ci9875_low=v['ci9875'][0],ci9875_high=v['ci9875'][1],query_video_clusters=v['query_clusters']))
    axes[1].axvline(0,color='0.5',linestyle='--',linewidth=1)
    axes[1].set_yticks(range(4),labels);axes[1].invert_yaxis();axes[1].set_ylim(3.6,-.6)
    axes[1].set_xlabel('Paired cosine-margin difference')
    axes[1].set_title('Query-source clustered descriptive intervals')
    axes[1].legend(handles=[Line2D([0],[0],color='#0072B2',lw=4,label='95%'),Line2D([0],[0],color='#9ecae1',lw=2,label='98.75% (4 primary contrasts)')],frameon=False,loc='lower right',fontsize=8)
    title='Known development VAL: 800 questions, epoch 4, training seed 17'
    if not result['lineage_passed']:title='NUMERICAL LINEAGE GATE FAILED - diagnostic values only'
    fig.suptitle(title,fontsize=12);save(fig,out/'main_comparison')
    csv_file(out/'main_table.csv',rows);csv_file(out/'primary_contrasts.csv',contrasts)
    conditional=[]
    for model in models:
        s=result['summary']
        cells=[s[model+'__repeat4'],s[model+'__query_nested16__candidates_repeat4'],
               s[model+'__query_repeat4__candidates_nested16'],s[model+'__nested16']]
        for measure in ('mean_margin','accuracy_percent'):
            a00,a10,a01,a11=[c[measure] for c in cells]
            values=dict(query_gain_candidates_repeat=a10-a00,query_gain_candidates_nested=a11-a01,
                        candidate_gain_query_repeat=a01-a00,candidate_gain_query_nested=a11-a10,
                        both_gain=a11-a00,query_candidate_interaction=a11-a10-a01+a00)
            conditional.extend(dict(model=model,measure=measure,contrast=name,value=value) for name,value in values.items())
    csv_file(out/'conditional_input_effects.csv',conditional)

    pairs=regional['paired_supervision_effects'];random=[r for r in pairs if r['cohort']=='random_query' and r['matched_analysis_eligible']]
    known=[r for r in pairs if r['cohort']=='known_case' and r['matched_analysis_eligible']]
    random_keys=sorted({(r['record_id'],r['uid'],r['role']) for r in random});known_keys=sorted({(r['record_id'],r['uid'],r['role']) for r in known})
    fig,axes=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[2,1]},layout='constrained')
    for ax,subset,keys in [(axes[0],random,random_keys),(axes[1],known,known_keys)]:
        for j,sigma in enumerate([4,8]):
            values=[]
            for key in keys:
                rows=[r for r in subset if (r['record_id'],r['uid'],r['role'])==key and r['sigma']==sigma]
                values.append(rows[0]['D_rich_minus_base'] if rows else np.nan)
            ax.plot(np.arange(len(keys))+(j-.5)*.12,values,'o',color=colors[j],label=f'Blur sigma {sigma}',markersize=5)
        ax.axhline(0,color='0.5',linestyle='--',lw=1)
        ax.set_ylabel('Regional selectivity D(rich) - D(base)')
        ax.legend(frameon=False,fontsize=8)
    axes[0].set_xticks(range(len(random_keys)),[f'r{r[0]:02d}' for r in random_keys]);axes[0].set_xlabel('Frozen region record')
    axes[0].set_title(f'Random queries with matched controls (n={len(random_keys)})')
    axes[1].set_xticks(range(len(known_keys)),[f'UID {r[1]}\n{r[2]}' for r in known_keys]);axes[1].set_title('Known shared clip: selected cases')
    fig.suptitle('Exploratory region sensitivity; assistant boxes, not human localization truth',fontsize=11)
    save(fig,out/'regional_selectivity')
    csv_file(out/'regional_paired_effects.csv',pairs)
    software=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
                  plotting_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
                  primary_commit=checked['primary_commit'],audit_commit=checked['audit_commit'])
    (out/'software.json').write_text(json.dumps(software,indent=2)+'\n')
    print(json.dumps(dict(figures=['main_comparison','regional_selectivity'],output_dir=str(out))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run-dir','audit-dir','output-dir'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
