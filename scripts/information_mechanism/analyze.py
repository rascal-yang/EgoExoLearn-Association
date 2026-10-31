"""Paired descriptive statistics, archived lineage check, and region summaries."""
import argparse
import json
from pathlib import Path
import numpy as np

from info_common import atomic,load_data,sha

def evaluate(features,indices,truth):
    x=features/np.linalg.norm(features,axis=-1,keepdims=True)
    s=np.einsum('nd,njd->nj',x[indices[:,0]],x[indices[:,1:]])
    return score_metrics(s,truth)

def score_metrics(s,truth):
    negative=s.copy();negative[np.arange(len(s)),truth]=-np.inf
    positive=s[np.arange(len(s)),truth];pred=s.argmax(1)
    return dict(scores=s,margin=positive-negative.max(1),correct=(pred==truth).astype(float),
                rank=1+(s>positive[:,None]).sum(1),pred=pred,positive=positive,negative=negative.max(1))

def intervals(values,groups,positive_groups,seed,repeats):
    """Query source clusters; two-way pigeonhole weights are sensitivity only."""
    values=np.asarray(values);unique,inverse=np.unique(groups,return_inverse=True)
    unique_p,inv_p=np.unique(positive_groups,return_inverse=True)
    n=len(unique);rng=np.random.default_rng(seed);boot=[];two=[]
    for _ in range(repeats):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[inverse]
        wp=np.bincount(rng.integers(len(unique_p),size=len(unique_p)),minlength=len(unique_p))[inv_p]
        boot.append((values*w).sum()/w.sum())
        w2=w*wp
        if w2.sum():two.append((values*w2).sum()/w2.sum())
    return dict(mean=float(values.mean()),query_clusters=n,positive_clusters=len(unique_p),
                ci95=np.quantile(boot,[.025,.975]).tolist(),ci9875=np.quantile(boot,[.00625,.99375]).tolist(),
                two_way_sensitivity_ci95=np.quantile(two,[.025,.975]).tolist())

def main(a):
    out=Path(a.run_dir);cfg=json.loads((out/'config.json').read_text())
    assert (out/'inference_complete.json').exists()
    root,ep,plan,ann,meta=load_data(a.snapshot,a.annotations,a.metadata)
    ids=np.load(out/'clip_ids.npy');ix={int(v):i for i,v in enumerate(ids)}
    indices=np.array([[ix[int(c)] for c in row] for row in ep['indices']]);truth=ep['truth']
    groups=np.array([plan['keys'][int(row[0])][0] for row in ep['indices']])
    positive_groups=np.array([plan['keys'][int(row[int(t)+1])][0] for row,t in zip(ep['indices'],truth)])
    views=np.array([meta[v]['view'] for v in groups])
    feature={c:np.load(out/'features'/f'{c}.npy',mmap_mode='r') for c in cfg['conditions']}
    metrics={};scores={};lineage=[]
    for mi,model in enumerate(cfg['models']):
        for condition in cfg['conditions']:
            name=model+'__'+condition;value=evaluate(feature[condition][mi],indices,truth)
            metrics[name]=value;scores[name]=value['scores']
        archived=np.load(root/'experiments/egoexo_training_stability_20261029/fits'/model/'seed17/val_curves/raw_features.npy',mmap_mode='r')[3]
        ref=evaluate(archived,indices,truth);cur=metrics[model+'__dense16']
        cosine=np.einsum('nd,nd->n',archived/np.linalg.norm(archived,axis=-1,keepdims=True),feature['dense16'][mi]/np.linalg.norm(feature['dense16'][mi],axis=-1,keepdims=True))
        gate=bool(cosine.min()>.999 and abs(cur['correct'].sum()-ref['correct'].sum())<=4)
        lineage.append(dict(model=model,archived_correct=int(ref['correct'].sum()),current_correct=int(cur['correct'].sum()),prediction_changes=int((ref['pred']!=cur['pred']).sum()),cosine_min=float(cosine.min()),gate_passed=gate,gate='cosine>0.999, accuracy count difference<=4/800; not bitwise equality'))
        for qcondition,ccondition in [('repeat4','nested16'),('nested16','repeat4')]:
            q=feature[qcondition][mi];q=q/np.linalg.norm(q,axis=-1,keepdims=True)
            c=feature[ccondition][mi];c=c/np.linalg.norm(c,axis=-1,keepdims=True)
            sc=np.einsum('nd,njd->nj',q[indices[:,0]],c[indices[:,1:]])
            name=model+'__query_'+qcondition+'__candidates_'+ccondition
            metrics[name]=score_metrics(sc,truth);scores[name]=sc
    atomic(out/'lineage_check.json',lineage)
    np.savez_compressed(out/'scores.npz',uids=ep['uids'],truth=truth,**scores)
    summary={}
    for name,m in metrics.items():
        summary[name]=dict(correct=int(m['correct'].sum()),n=len(truth),accuracy_percent=float(100*m['correct'].mean()),mean_margin=float(m['margin'].mean()),mean_rank=float(m['rank'].mean()),by_query_view={view:dict(n=int((views==view).sum()),accuracy_percent=float(100*m['correct'][views==view].mean()),mean_margin=float(m['margin'][views==view].mean())) for view in ('ego','exo')})
    contrasts={}
    for measure in ('margin','correct'):
        bn=metrics['P_base__nested16'][measure];br=metrics['P_base__repeat4'][measure]
        rn=metrics['P_rich__nested16'][measure];rr=metrics['P_rich__repeat4'][measure]
        for name,values in dict(G_base=bn-br,G_rich=rn-rr,S_nested=rn-bn,I=(rn-rr)-(bn-br)).items():
            values=values*(100 if measure=='correct' else 1)
            contrasts[name+'__'+measure]=intervals(values,groups,positive_groups,cfg['bootstrap_seed'],cfg['bootstrap_repeats'])
    atomic(out/'main_results.json',dict(lineage_passed=all(r['gate_passed'] for r in lineage),summary=summary,contrasts=contrasts,interpretation='single training seed; repeatedly used development VAL; descriptive, not independent confirmation'))
    with (out/'per_question.jsonl').open('w') as f:
        for j,uid in enumerate(ep['uids']):
            f.write(json.dumps(dict(uid=str(uid),query_source=groups[j],positive_source=positive_groups[j],query_view=views[j],conditions={name:dict(correct=bool(v['correct'][j]),margin=float(v['margin'][j]),rank=int(v['rank'][j]),positive=float(v['positive'][j]),strongest_negative=float(v['negative'][j])) for name,v in metrics.items()}))+'\n')
    regional={}
    if (out/'region_results.json').exists():
        rows=json.loads((out/'region_results.json').read_text())['rows'];by={}
        for r in rows:by.setdefault((r['record_id'],r['uid'],r['role'],r['model']),{})[r['condition']]=r
        effects=[]
        for key,v in by.items():
            clean=v['clean']
            for sigma in cfg['blur_sigmas']:
                bg=[r for n,r in v.items() if n.startswith('bg') and n.endswith(f'sigma{sigma}')]
                roi=v[f'roi_sigma{sigma}'];drop=clean['margin']-roi['margin']
                effects.append(dict(**{k:clean[k] for k in ('record_id','uid','role','model','cohort','view','roi_type','source_video','matched_analysis_eligible')},sigma=sigma,clean_margin=clean['margin'],whole_frame_drop=clean['margin']-v['replace_whole']['margin'],roi_drop=drop,background_drops=[clean['margin']-r['margin'] for r in bg],D=(drop-float(np.mean([clean['margin']-r['margin'] for r in bg]))) if bg else None))
        for model in cfg['models']:
            for sigma in cfg['blur_sigmas']:
                sample=[r for r in effects if r['model']==model and r['sigma']==sigma and r['cohort']=='random_query' and r['matched_analysis_eligible']]
                regional[model+f'__sigma{sigma}']=dict(n=len(sample),mean_D=float(np.mean([r['D'] for r in sample])) if sample else None,hand_object_n=sum(r['roi_type']=='hand_object' for r in sample),limitation='assistant boxes, matched-background availability selection, descriptive subset')
        paired=[]
        pairs={}
        for r in effects:pairs.setdefault((r['record_id'],r['uid'],r['role'],r['sigma']),{})[r['model']]=r
        for key,v in pairs.items():
            base,rich=v['P_base'],v['P_rich']
            paired.append(dict(record_id=key[0],uid=key[1],role=key[2],sigma=key[3],cohort=base['cohort'],
                               matched_analysis_eligible=base['matched_analysis_eligible'],
                               D_rich_minus_base=(rich['D']-base['D']) if base['D'] is not None and rich['D'] is not None else None))
        atomic(out/'region_effects.json',dict(summary=regional,rows=effects,paired_supervision_effects=paired,
                                             limitation='Small matched subset, exploratory, no population representativeness or independent human annotations'))
    lines=['信息机制实验：完成报告',f"运行：{out.name}",'全部数字来自单种子、反复开发VAL；区域为助手标注，不是独立人工真值。','',f"原模型血缘门槛通过：{all(r['gate_passed'] for r in lineage)}",'']
    for name,s in summary.items():lines.append(f"{name}: {s['correct']}/800 = {s['accuracy_percent']:.3f}%; margin={s['mean_margin']:.6f}")
    lines+=['','事前比较（correct单位pp，其余为余弦margin）：']
    for name,c in contrasts.items():lines.append(f"{name}: {c['mean']:.6f}, 95%={c['ci95']}, 98.75%={c['ci9875']}")
    lines+=['','区域匹配子集：',json.dumps(regional,ensure_ascii=False,indent=2),'','完整逐题、候选分数、区域效应与不确定性保留在JSON/NPZ；不宣称唯一因果机制或跨种子泛化。']
    if not all(r['gate_passed'] for r in lineage):lines+=['','血缘门槛失败：本次输出仅用于排错，不作信息机制解释。']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    atomic(out/'analysis_complete.json',dict(passed=True,lineage_passed=all(r['gate_passed'] for r in lineage),main_results_sha256=sha(out/'main_results.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run-dir','snapshot','annotations','metadata'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
