"""CPU-only figures and descriptor diagnostics derived from the frozen run."""
from degradation_core import *


def main():
    torch.set_num_threads(2)
    result=read(REP/'退化原因诊断.json');assert result['protocol_sha256']==digest(POST/'degradation_protocol.json')
    archives=[np.load(DEG/f'seed{s}/features.npz') for s in SEEDS]
    keys=[k for k in archives[0].files if k!='ids'];metrics={}
    for key in keys:
        rows=[]
        for archive in archives:
            f=archive[key].astype(np.float64);n=len(f)
            centered=f-f.mean(0,keepdims=True);cov=centered.T@centered
            spectrum=np.maximum(np.linalg.eigvalsh(cov),0);spectrum/=max(spectrum.sum(),1e-30)
            base=archive['blocks_R0__readout_R0__dense'].astype(np.float64)
            rows.append(dict(mean_cosine_to_R0_dense=float((f*base).sum(-1).mean()),
                             mean_pairwise_cosine=float(((f.sum(0)**2).sum()-(f*f).sum())/(n*(n-1))),
                             centered_effective_rank=float(np.exp(-(spectrum*np.log(np.maximum(spectrum,1e-30))).sum())),
                             mean_norm=float(np.linalg.norm(f,axis=1).mean())))
        metrics[key]=dict(per_seed=rows,mean={m:float(np.mean([r[m] for r in rows])) for m in rows[0]})
    atomic(REP/'退化_描述符诊断.json',dict(metrics=metrics,clips=len(archives[0]['ids']),
          source_protocol_sha256=result['protocol_sha256'],analysis_source_sha256=digest(Path(__file__)),
          caveat='Descriptive fixed3510-clip statistics, not independent samples; no collapse threshold was selected.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8.4),constrained_layout=True)
    colors={'R0':'#2369a0','R1':'#d28a13','R2':'#bf4747','R5':'#26836b'}
    ax=axes[0,0]
    for arm in ('R0','R1','R2','R5'):
        y=np.array([[r['caption_holdout'] for r in result['histories'][arm][str(s)]] for s in SEEDS])
        ax.plot(range(5),y.mean(0),marker='o',color=colors[arm],label=arm)
        ax.fill_between(range(5),y.min(0),y.max(0),color=colors[arm],alpha=.12)
    ax.set(xlabel='Epoch (0 = before new training)',ylabel='Internal caption loss',title='A. Loss gap exists before adaptation',xticks=range(5))
    ax.legend(ncol=4,frameon=False)
    ax=axes[0,1];positions=np.arange(2);width=.34
    for offset,policy,label,color in ((-width/2,'dense','Dense memory','#2369a0'),(width/2,'r2_memory','Fixed R2 memory','#bf4747')):
        mean=[];seed_values=[]
        for arm in ('R0','R2'):
            r=result['results'][f'blocks_{arm}__readout_{arm}__{policy}'];mean.append(r['mean_accuracy']);seed_values.append(r['per_seed_accuracy'])
        bars=ax.bar(positions+offset,mean,width,label=label,color=color,alpha=.83)
        for j,v in enumerate(seed_values):
            ax.scatter(positions[j]+offset+np.linspace(-.035,.035,3),v,s=17,color='black',zorder=4)
            ax.text(positions[j]+offset,max(v)+.5,f'{mean[j]:.2f}',ha='center',fontsize=9)
    ax.set(xticks=positions,xticklabels=['R0 weights','R2 weights'],ylabel='Accuracy (%)',title='B. Fixed256 cohort: weights x memory')
    ax.set_ylim(0,60)
    ax.legend(frameon=False)
    ax=axes[1,0];labels=[f'{arm}/B{j}' for arm in ('R0','R2') for j in (10,11)];x=np.arange(4)
    dense=[];compact=[]
    for arm in ('R0','R2'):
        for j in (10,11):
            m=result['local_approximation'][arm][str(j)];dense.append(m['dense_context_attention_mass']['mean']);compact.append(m['compact_context_attention_mass']['mean'])
    ax.bar(x-.17,dense,.34,color='#2369a0',label='Native attention')
    ax.bar(x+.17,compact,.34,color='#bf4747',label='Compressed attention')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel='CLS probability assigned to context',title='C. Context probability is undercounted')
    ax.legend(frameon=False)
    ax=axes[1,1]
    for offset,metric,label,color in ((-.24,'CLS_relative_error','Compact K/V','#bf4747'),(0,'exact_group_mass_CLS_relative_error','Exact group mass','#d28a13'),(.24,'exact_group_values_CLS_relative_error','Exact group values','#26836b')):
        values=[result['local_approximation'][arm][str(j)][metric]['mean'] for arm in ('R0','R2') for j in (10,11)]
        ax.bar(x+offset,values,.24,label=label,color=color)
    ax.set(xticks=x,xticklabels=labels,ylabel='Relative L2 error of CLS spatial output',title='D. Probability and V errors interact')
    ax.legend(frameon=False,fontsize=9)
    fig.suptitle('EgoExoLearn: exploratory degradation diagnosis (3 paired seeds)',fontsize=14)
    fig.savefig(REP/'退化原因诊断图.png',dpi=220)
    fig.savefig(REP/'退化原因诊断图.pdf')
    plt.close(fig)
    print('FIGURES_AND_DESCRIPTOR_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':main()
