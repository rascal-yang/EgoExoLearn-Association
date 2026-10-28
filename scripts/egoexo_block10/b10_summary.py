"""Frozen primary comparisons against the reused R0 and matched S1 control."""
import numpy as np
import torch
from b10_common import *
from summarize import bootstrap


def main():
    torch.set_num_threads(2);p=verify_protocol();signature=digest(PROTOCOL)
    annotation=read(ROOT/'data/association_annotations/association_val_hard.json')
    metadata={r['video_uid']:r for r in read(ROOT/'repos/EgoExoLearn/video_meta_downsampled.json')}
    results={};correct={};secondary={};ref=None;training=[]
    for arm in ('R0','S1','S2'):
        folder_root=OLD_EXP if arm=='R0' else EXP
        result={};cs=[];bs=[]
        for seed in SEEDS:
            folder=folder_root/f'fits/{arm}/seed{seed}'
            ck=read(folder/'complete.json');assert ck['passed'] and ck['steps']==2676 and ck['frozen_prefix_exact']
            assert ck['exposure_sha256']==p['baseline_reuse']['exposure_sha256']
            if arm!='R0':assert ck['protocol_sha256']==signature;training.append(dict(arm=arm,seed=seed,numerical_retries=ck['numerical_retries'],elapsed_s=ck['elapsed_s']))
            score=np.load(folder/'val_final/scores.npz');uids=score['uids'].astype(str)
            if ref is None:ref=uids
            assert np.array_equal(uids,ref)
            c=score['scores'].argmax(1)==score['truth'];cs.append(c)
            b=np.load(folder/'val_best/scores.npz');bs.append(b['scores'].argmax(1)==b['truth'])
            result[str(seed)]=dict(correct=int(c.sum()),accuracy=float(c.mean()*100),
                    internal_best_secondary=read(folder/'val_best/complete.json')['accuracy'])
        correct[arm]=np.stack(cs);secondary[arm]=np.stack(bs)
        result['mean_accuracy']=float(correct[arm].mean()*100);results[arm]=result
    source=np.array([annotation[u]['query']['video_uid'] for u in ref]);candidate=np.array([annotation[u]['choices'][str(annotation[u]['answer'])]['video_uid'] for u in ref])
    tasks=np.array([metadata[v]['task'] for v in source]);views=np.array([metadata[v]['view'] for v in source])
    comparisons={}
    for left,right,level in (('S2','R0',.975),('S2','S1',.975),('S1','R0',.95)):
        a,b=correct[left],correct[right];delta=(a.astype(float)-b).mean(0)
        values=bootstrap(delta,source,np.random.default_rng(20261028));alpha=(1-level)/2
        unique,ix=np.unique(source,return_inverse=True);counts=np.bincount(ix);sums=np.bincount(ix,weights=delta);largest=counts.argmax()
        leave=(delta.sum()-sums)/(len(delta)-counts)
        comparisons[f'{left}-{right}']=dict(gain_pp=float(delta.mean()*100),per_seed_gain_pp=((a.mean(1)-b.mean(1))*100).tolist(),
             interval_level=level,source_cluster_interval=(np.quantile(values,[alpha,1-alpha])*100).tolist(),
             correct_candidate_cluster_interval=(np.quantile(bootstrap(delta,candidate,np.random.default_rng(20261029)),[alpha,1-alpha])*100).tolist(),
             macro_video_gain_pp=float((sums/counts).mean()*100),macro_task_gain_pp=float(np.mean([delta[tasks==t].mean() for t in np.unique(tasks)])*100),
             delete_largest_source_gain_pp=float(delta[source!=unique[largest]].mean()*100),
             leave_one_source_gain_range_pp=[float(leave.min()*100),float(leave.max()*100)],
             repaired_per_seed=(a&~b).sum(1).tolist(),broken_per_seed=(~a&b).sum(1).tolist(),
             internal_best_secondary_gain_pp=float((secondary[left].mean()-secondary[right].mean())*100))
        if left=='S2':
            physical=np.unique(np.concatenate([source,candidate]));lookup={v:i for i,v in enumerate(physical)}
            qi=np.array([lookup[v] for v in source]);ci=np.array([lookup[v] for v in candidate])
            values=[];rng=np.random.default_rng(20261030)
            while len(values)<20000:
                w=rng.poisson(1,size=(500,len(physical)));weights=w[:,qi]*w[:,ci];total=weights.sum(1);take=total>0
                values.extend((weights[take]@delta)/total[take])
            stratified=np.zeros(20000);denominator=np.zeros(20000);rng=np.random.default_rng(20261031)
            for task in np.unique(tasks):
                take=tasks==task;u,j=np.unique(source[take],return_inverse=True);ct=np.bincount(j);sm=np.bincount(j,weights=delta[take])
                w=rng.multinomial(len(u),np.ones(len(u))/len(u),size=20000)
                stratified+=w@sm;denominator+=w@ct
            comparisons[f'{left}-{right}'].update(shared_physical_video_Poisson_interval=(np.quantile(values[:20000],[alpha,1-alpha])*100).tolist(),
                task_stratified_source_interval=(np.quantile(stratified/denominator,[alpha,1-alpha])*100).tolist())
    for arm,c in correct.items():
        results[arm]['direction']={v:float(c[:,views==v].mean()*100) for v in np.unique(views)}
        results[arm]['metadata_task']={t:dict(questions=int((tasks==t).sum()),accuracy=float(c[:,tasks==t].mean()*100)) for t in np.unique(tasks)}
    c0,c1=comparisons['S2-R0'],comparisons['S2-S1']
    success=c0['gain_pp']>=1 and all(x>0 for x in c0['per_seed_gain_pp']) and c0['source_cluster_interval'][0]>0 and c1['source_cluster_interval'][0]>0
    report=dict(version='3.0_block10_only',exploratory_after_v2=True,primary_endpoint='epoch4_final',models=results,
         comparisons=comparisons,project_target_met=bool(success),baseline_reused_read_only=True,training=training,
         TEST_evaluated=False,human_grounding_evidence=False,protocol_sha256=signature,
         caveat='Known VAL used for exploratory development; fixed candidate pool and three seeds; source clustering incompletely treats candidate reuse. Historical TEST was not reused for tuning this run.')
    atomic(REP/'结果汇总.json',report)
    lines=['block10单层压缩：6次新训练与复用R0的完整800题VAL结果','仅block10空间K/V压缩；block11原生稠密。主终点epoch4，三种子整体报告。','']
    for arm,r in results.items():lines.append(f"{arm}: "+', '.join(f"seed{s} {r[str(s)]['accuracy']:.3f}%" for s in SEEDS)+f"；均值 {r['mean_accuracy']:.3f}%")
    for key,v in comparisons.items():lines.append(f"{key}: {v['gain_pp']:+.3f}pp；{v['interval_level']*100:.1f}%源视频聚类区间 {v['source_cluster_interval']}")
    lines.extend(['',f'预定提升目标满足：{success}',report['caveat'],'未新增TEST评测；未完成的人工核验不作为真值。'])
    (REP/'结果汇总.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
