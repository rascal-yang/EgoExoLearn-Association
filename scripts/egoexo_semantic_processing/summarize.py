"""Predeclared paired cluster statistics for the fixed800-question VAL."""
import numpy as np
from sp_common import *


def ci(values,level=.975):
    alpha=(1-level)/2
    return (np.quantile(values,[alpha,1-alpha])*100).tolist()


def bootstrap(diff,clusters,rng,n=20000):
    unique,idx=np.unique(clusters,return_inverse=True)
    counts=np.bincount(idx)
    sums=np.bincount(idx,weights=diff)
    values=[]
    for start in range(0,n,1000):
        w=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=min(1000,n-start))
        values.extend((w@sums)/(w@counts))
    return np.asarray(values)


def main():
    protocol=read(EXP/'protocol_v2_frozen.json')
    annotation=read(ROOT/protocol['evaluation']['annotation'])
    meta={r['video_uid']:r for r in read(ROOT/protocol['evaluation']['direction_and_task_source'])}
    result={};all_correct={};reference_uids=None
    for arm in ARMS:
        result[arm]={};correct=[]
        for seed in SEEDS:
            folder=EXP/f'fits/{arm}/seed{seed}/val_final'
            ck=read(folder/'complete.json');assert ck['passed'] and ck['epoch']==4
            score=np.load(folder/'scores.npz')
            uids=score['uids'].astype(str)
            if reference_uids is None:reference_uids=uids
            assert np.array_equal(uids,reference_uids)
            c=score['scores'].argmax(1)==score['truth'];correct.append(c)
            result[arm][str(seed)]=dict(correct=int(c.sum()),accuracy=float(c.mean()*100))
            secondary=read(EXP/f'fits/{arm}/seed{seed}/val_best/complete.json')
            result[arm][str(seed)]['internal_best_secondary']=dict(epoch=secondary['epoch'],accuracy=secondary['accuracy'])
        all_correct[arm]=np.stack(correct)
        result[arm]['mean_accuracy']=float(all_correct[arm].mean()*100)
        result[arm]['internal_best_mean_accuracy']=float(np.mean([result[arm][str(seed)]['internal_best_secondary']['accuracy'] for seed in SEEDS]))
    rows=[annotation[u] for u in reference_uids]
    source=np.array([r['query']['video_uid'] for r in rows])
    candidate=np.array([r['choices'][str(r['answer'])]['video_uid'] for r in rows])
    tasks=np.array([meta[v]['task'] for v in source]);directions=np.array([meta[v]['view'] for v in source])
    dishes=np.array([r['query']['dish'] for r in rows]);duration=np.array([r['query']['end_sec']-r['query']['start_sec'] for r in rows])
    feature_data=np.load(EXP/'fits/R0/seed17/val_final/features.npz')
    frame_lookup={int(i):int(n) for i,n in zip(feature_data['indices'],feature_data['unique_PTS_frames'])}
    episode_indices=np.load(BASE/'val_episodes.npz')['indices']
    unique_frames=np.array([frame_lookup[int(i)] for i in episode_indices[:,0]])
    for arm in ARMS:
        result[arm]['subgroups']={}
        for name,group in [('direction',directions),('metadata_task',tasks),('dish',dishes),('unique_PTS_frames',unique_frames),('duration',np.where(duration<=5,'le5',np.where(duration<=15,'5to15','gt15')))]:
            result[arm]['subgroups'][name]={str(v):dict(questions=int((group==v).sum()),
                per_seed=(all_correct[arm][:,group==v].mean(1)*100).tolist(),mean_accuracy=float(all_correct[arm][:,group==v].mean()*100)) for v in np.unique(group)}
    comparisons={}
    for other in ('R0','R1','R3','R4','R5'):
        key=f'R2-{other}';a=all_correct['R2'];b=all_correct[other]
        diff=(a.astype(float)-b.astype(float)).mean(0)
        values=bootstrap(diff,source,np.random.default_rng(20261027))
        unique,idx=np.unique(source,return_inverse=True);counts=np.bincount(idx);sums=np.bincount(idx,weights=diff)
        biggest=int(counts.argmax());keep=source!=unique[biggest]
        leave_one=(diff.sum()-sums)/(len(diff)-counts)
        paired=[dict(seed=seed,repaired=int((a[j]&~b[j]).sum()),broken=int((~a[j]&b[j]).sum()),
                     gain_pp=float((a[j].mean()-b[j].mean())*100)) for j,seed in enumerate(SEEDS)]
        item=dict(gain_pp=float(diff.mean()*100),per_seed=paired,
             source_cluster_interval=ci(values,.975 if other in ('R0','R1') else .95),
             interval_level=.975 if other in ('R0','R1') else .95,
             macro_video_gain_pp=float((sums/counts).mean()*100),
             macro_task_gain_pp=float(np.mean([diff[tasks==t].mean() for t in np.unique(tasks)])*100),
             correct_candidate_cluster_interval=ci(bootstrap(diff,candidate,np.random.default_rng(20261028))),
             delete_largest_source=dict(video=str(unique[biggest]),questions=int(counts[biggest]),gain_pp=float(diff[keep].mean()*100)),
             leave_one_source_gain_range_pp=[float(leave_one.min()*100),float(leave_one.max()*100)],
             internal_best_secondary_gain_pp=result['R2']['internal_best_mean_accuracy']-result[other]['internal_best_mean_accuracy'],
             interpretation='primary_Bonferroni_adjusted' if other in ('R0','R1') else 'exploratory_no_significance_claim')
        if other in ('R0','R1'):
            physical=np.unique(np.concatenate([source,candidate]));mapping={v:i for i,v in enumerate(physical)}
            qi=np.array([mapping[v] for v in source]);ci_idx=np.array([mapping[v] for v in candidate]);shared=[]
            rng=np.random.default_rng(20261029)
            while len(shared)<20000:
                w=rng.poisson(1,size=(500,len(physical)));weight=w[:,qi]*w[:,ci_idx];total=weight.sum(1);take=total>0
                shared.extend((weight[take]@diff)/total[take])
            item['shared_physical_video_weight_interval']=ci(np.asarray(shared[:20000]))
            stratified=np.zeros(20000);denom=np.zeros(20000);rng=np.random.default_rng(20261030)
            for task in np.unique(tasks):
                take=tasks==task;u,ix=np.unique(source[take],return_inverse=True);ct=np.bincount(ix);sm=np.bincount(ix,weights=diff[take])
                w=rng.multinomial(len(u),np.ones(len(u))/len(u),size=20000);stratified+=w@sm;denom+=w@ct
            item['task_stratified_source_interval']=ci(stratified/denom)
        comparisons[key]=item
    target=protocol['success_target'];main0=comparisons['R2-R0'];main1=comparisons['R2-R1']
    success=main0['gain_pp']>=target['R2_R0_mean_gain_pp_min'] and all(x['gain_pp']>0 for x in main0['per_seed']) and main0['source_cluster_interval'][0]>0 and main1['source_cluster_interval'][0]>0
    report=dict(primary_endpoint='epoch4_final',models=result,comparisons=comparisons,project_target_met=bool(success),
          caveat=protocol['evaluation']['scope_of_intervals'],TEST_access=False,human_grounding_evidence=False)
    atomic(REP/'v2_VAL结果.json',report)
    lines=['v2固定第4轮：800题官方VAL结果','所有结果为三种子，不选择最佳种子。','']
    for arm in ARMS:
        lines.append(f"{arm}: "+', '.join(f"seed{s} {result[arm][str(s)]['accuracy']:.3f}%" for s in SEEDS)+f"；均值 {result[arm]['mean_accuracy']:.3f}%")
    for pair in ('R2-R0','R2-R1'):
        value=comparisons[pair];lines.append(f"{pair}: {value['gain_pp']:+.3f}个百分点；97.5%源视频聚类区间 {value['source_cluster_interval']}")
    lines.extend(['',f'项目预声明目标满足：{success}',report['caveat'],'人工物体定位及精细动作核验尚未完成；不能据此宣称。'])
    (REP/'v2_VAL结果.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
