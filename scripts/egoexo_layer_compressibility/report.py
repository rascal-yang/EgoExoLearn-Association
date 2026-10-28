"""Paired source-cluster descriptive results and measured speed/quality plots."""
import csv
import numpy as np
from lc_common import *
from summarize import bootstrap


def timing_summary(archive):
    result = {}
    rng = np.random.default_rng(20261028)
    for batch, values in archive['batches'].items():
        result[batch] = {}
        for name, samples in values['samples'].items():
            base = np.array([s['baseline_cuda_ms'] for s in samples])
            candidate = np.array([s['candidate_cuda_ms'] for s in samples])
            ratios = candidate/base
            resamples = ratios[rng.integers(len(ratios),size=(10000,len(ratios)))].mean(1)
            result[batch][name] = dict(baseline_cuda_median_ms=float(np.median(base)),
                candidate_cuda_median_ms=float(np.median(candidate)),
                paired_median_latency_ratio=float(np.median(ratios)),
                mean_latency_ratio=float(ratios.mean()),
                descriptive_within_run_ratio_ci95=np.quantile(resamples,[.025,.975]).tolist(),
                candidate_clips_per_second=float(int(batch)*1000/np.median(candidate)),
                median_latency_reduction_percent=float((1-np.median(ratios))*100),
                baseline_wall_median_ms=float(np.median([s['baseline_wall_ms'] for s in samples])),
                candidate_wall_median_ms=float(np.median([s['candidate_wall_ms'] for s in samples])),
                memory=values['memory'][name])
    return result


def main():
    p=verify()
    protocol_sha=digest(EXP/'protocol_frozen.json')
    ep,rows,all_ids=cohort()
    configs=p['conditions']
    features=np.empty((len(configs),3,len(all_ids),256),np.float32)
    for shard in (0,1):
        folder=EXP/f'scan/shard{shard}'
        done=read(folder/'complete.json')
        assert done['passed'] and done['protocol_sha256']==protocol_sha
        assert done['features_sha256']==digest(folder/'features.npy')
        ids=np.load(folder/'ids.npy')
        positions=np.searchsorted(all_ids,ids)
        assert np.array_equal(all_ids[positions],ids)
        features[:,:,positions]=np.load(folder/'features.npy',mmap_mode='r')
    indices=np.searchsorted(all_ids,ep['indices'][rows])
    assert np.array_equal(all_ids[indices],ep['indices'][rows])
    truth=ep['truth'][rows]
    uids=ep['uids'][rows].astype(str)
    annotation=read(ROOT/'data/association_annotations/association_val_hard.json')
    source=np.array([annotation[u]['query']['video_uid'] for u in uids])
    correct=[];margins=[];scores={}
    for c,config in enumerate(configs):
        f=features[c]
        sc=np.stack([np.einsum('qd,qkd->qk',fs[indices[:,0]],fs[indices[:,1:]]) for fs in f])
        scores[config['name']]=sc
        correct.append(sc.argmax(-1)==truth[None])
        wrong=sc.copy();wrong[:,np.arange(len(truth)),truth]=-np.inf
        margins.append(sc[:,np.arange(len(truth)),truth]-wrong.max(-1))
    correct=np.stack(correct);margins=np.stack(margins)
    timing_path=EXP/'benchmark_gpu3.json'
    archive=read(timing_path)
    assert archive['completed'] and archive['protocol_sha256']==protocol_sha
    timings=timing_summary(archive)
    result={}
    for j,c in enumerate(configs):
        diff=(correct[j].astype(float)-correct[0]).mean(0)
        md=(margins[j]-margins[0]).mean(0)
        ci=np.quantile(bootstrap(diff,source,np.random.default_rng(20261028)),[.025,.975])*100
        mci=np.quantile(bootstrap(md,source,np.random.default_rng(20261028)),[.025,.975])
        result[c['name']]=dict(**c,per_seed_accuracy=(correct[j].mean(1)*100).tolist(),
            mean_accuracy=float(correct[j].mean()*100),gain_pp=float(diff.mean()*100),
            per_seed_gain_pp=((correct[j].mean(1)-correct[0].mean(1))*100).tolist(),
            descriptive_source_cluster_ci95_pp=ci.tolist(),mean_margin=float(margins[j].mean()),
            margin_change=float(md.mean()),descriptive_margin_ci95=mci.tolist(),
            mean_descriptor_cosine_to_dense=float((features[j]*features[0]).sum(-1).mean()),
            repaired_per_seed=(correct[j]&~correct[0]).sum(1).tolist(),
            broken_per_seed=(~correct[j]&correct[0]).sum(1).tolist())
        if j:
            result[c['name']]['timing']={b:values[c['name']] for b,values in timings.items()}
    # Fixed screening rules, not a noninferiority/significance claim.
    candidates=[]
    for name,r in result.items():
        if name=='dense':continue
        t=r['timing']['8']
        quality=(r['gain_pp']>=p['screen']['min_gain_pp'] and
            min(r['per_seed_gain_pp'])>=p['screen']['min_per_seed_gain_pp'] and
            r['margin_change']>=p['screen']['min_margin_change'])
        speed=(t['mean_latency_ratio']<1 and t['descriptive_within_run_ratio_ci95'][1]<1)
        r['screen_quality_pass']=quality
        r['screen_speed_pass']=speed
        if quality and speed:candidates.append(name)
    candidates.sort(key=lambda n:(-result[n]['gain_pp'],-result[n]['margin_change'],
        result[n]['timing']['8']['paired_median_latency_ratio'],result[n]['layer'],result[n]['operator']))
    selected=[];layers=set()
    for name in candidates:
        if result[name]['layer'] not in layers and len(selected)<3:
            selected.append(name);layers.add(result[name]['layer'])
    out=dict(protocol_sha256=protocol_sha,questions=len(rows),unique_clips=len(all_ids),seeds=list(SEEDS),
        conditions=result,selected_for_possible_budget_curves=selected,
        benchmark=str(timing_path.relative_to(ROOT)),training=False,TEST_access=False,
        limits=['Exploratory fixed256 VAL; intervals unadjusted for24 comparisons; no noninferiority claim',
            'All3 seeds paired; query-source clusters jointly resampled',
            'RGB FP32 baseline differs from historical FP16-prefix boundary; do not substitute54.2917%',
            'Timing uses fixed8 clips, one GPU3 run, seed17; within-run CIs do not cover hardware/input variability',
            'Multi-layer combinations and other budgets remain unmeasured in this stage'])
    atomic(REP/'逐层96槽结果.json',out)
    np.savez(REP/'逐层96槽逐题分数.npz',uids=uids,truth=truth,**scores)
    fields=['name','mean_accuracy','gain_pp','margin_change','b1_latency_reduction_percent',
            'b8_latency_reduction_percent','screen_quality_pass','screen_speed_pass']
    with (REP/'逐层96槽结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,r in result.items():
            row={k:r.get(k,'') for k in fields[:4]+fields[6:]}
            for b in ('1','8'):
                row[f'b{b}_latency_reduction_percent']=r.get('timing',{}).get(b,{}).get('median_latency_reduction_percent',0)
            w.writerow(row)
    lines=['逐层96槽K/V压缩：冻结R0、RGB FP32、固定256题×三种子',
        f'稠密基线 {result["dense"]["mean_accuracy"]:.3f}%；独立clip {len(all_ids)}；不读取TEST。',
        '速度包含完整编码器、分组与聚合；正数表示延迟下降。计时为GPU3固定8条视频的本次运行。',
        '条件                         准确率    相对dense(pp)   margin变化    B1延迟下降  B8延迟下降']
    for name,r in result.items():
        if name=='dense':continue
        lines.append(f'{name:28s} {r["mean_accuracy"]:7.3f} {r["gain_pp"]:+9.3f} {r["margin_change"]:+12.5f} '
            f'{r["timing"]["1"]["median_latency_reduction_percent"]:+9.2f}% {r["timing"]["8"]["median_latency_reduction_percent"]:+9.2f}%')
    lines.extend(['',f'满足预设精度+分差+实测速度筛选规则的候选：{selected}',
        '该筛选不是无损/非劣证明；预算曲线与多层组合尚未测量。',
        '若候选为空，本阶段报告否定结果，不自动增加训练或方法搜索。'])
    (REP/'逐层96槽结果.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for op,marker in [('random','o'),('similarity','s')]:
        rs=[r for r in result.values() if r['operator']==op]
        x=[r['layer'] for r in rs]
        axes[0].plot(x,[r['gain_pp'] for r in rs],marker=marker,label=op)
        axes[1].plot(x,[r['timing']['8']['median_latency_reduction_percent'] for r in rs],marker=marker,label=op)
        axes[2].scatter([r['timing']['8']['median_latency_reduction_percent'] for r in rs],
            [r['gain_pp'] for r in rs],marker=marker,label=op)
        for r in rs:axes[2].annotate(str(r['layer']),
            (r['timing']['8']['median_latency_reduction_percent'],r['gain_pp']),fontsize=7)
    axes[0].set(xlabel='Block (zero based)',ylabel='Accuracy change (pp)')
    axes[1].set(xlabel='Block (zero based)',ylabel='Full-encoder latency reduction (%)')
    axes[2].set(xlabel='Full-encoder latency reduction (%)',ylabel='Accuracy change (pp)')
    for ax in axes:
        ax.axhline(0,color='gray',lw=.7);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('96 spatial K/V slots: fixed256 VAL / 3 seeds; FP32 batch8 timing')
    fig.tight_layout();fig.savefig(REP/'逐层精度与实测速度.png',dpi=180);fig.savefig(REP/'逐层精度与实测速度.pdf')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
