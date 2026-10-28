"""Paired factorial effects and approximation errors; exploratory only."""
from degradation_core import *
from summarize import bootstrap
import csv


def name(blocks,policy,readout=None):return f'blocks_{blocks}__readout_{readout or blocks}__{policy}'


def main():
    torch.set_num_threads(2)
    verify_degradation();ep,rows=diagnostic()
    annotation=read(ROOT/'data/association_annotations/association_val_hard.json')
    metadata={r['video_uid']:r for r in read(ROOT/'repos/EgoExoLearn/video_meta_downsampled.json')}
    archives=[];errors=[];replay=[]
    for seed in SEEDS:
        folder=DEG/f'seed{seed}';done=read(folder/'complete.json');assert done['passed']
        assert done['stage_protocol_sha256']==digest(POST/'degradation_protocol.json')
        assert digest(folder/'scores.npz')==done['scores_sha256']
        archives.append(np.load(folder/'scores.npz'));errors.append(np.load(folder/'local_errors.npz'))
        replay.append(done['replay_max_abs'])
    uids=archives[0]['uids'].astype(str);truth=archives[0]['truth']
    assert np.array_equal(uids,ep['uids'][rows].astype(str))
    source=np.array([annotation[u]['query']['video_uid'] for u in uids])
    stratum=np.array([metadata[v]['view']+'_'+metadata[v]['task'] for v in source])
    cohort=read(REP/'v2_固定256题诊断队列.json');weights=np.zeros(len(uids))
    for s in cohort['strata']:
        take=stratum==s['direction']+'_'+s['task'];assert take.sum()==s['quota'];weights[take]=s['population']/take.sum()
    weights/=weights.sum()
    names=[k for k in archives[0].files if k not in ('uids','truth')]
    scores={k:np.stack([a[k] for a in archives]) for k in names}
    correct={k:s.argmax(-1)==truth[None] for k,s in scores.items()}
    margins={}
    for k,s in scores.items():
        wrong=s.copy();wrong[:,np.arange(len(truth)),truth]=-np.inf
        margins[k]=s[:,np.arange(len(truth)),truth]-wrong.max(-1)
    results={k:dict(per_seed_accuracy=(c.mean(1)*100).tolist(),mean_accuracy=float(c.mean()*100),
                 population_weighted_accuracy=float((c.mean(0)*weights).sum()*100),
                 mean_margin=float(margins[k].mean()),
                 mean_candidate_score_std=float(scores[k].std(-1).mean()),
                 mean_correct_similarity=float(scores[k][:,np.arange(len(truth)),truth].mean()),
                 strata={s:dict(questions=int((stratum==s).sum()),accuracy=float(c[:,stratum==s].mean()*100)) for s in np.unique(stratum)}) for k,c in correct.items()}
    comparisons={}
    def compare(key,left,right):
        diff=(correct[left].astype(float)-correct[right]).mean(0)
        ci=np.quantile(bootstrap(diff,source,np.random.default_rng(20261027)),[.025,.975])*100
        md=(margins[left]-margins[right]).mean(0)
        mci=np.quantile(bootstrap(md,source,np.random.default_rng(20261027)),[.025,.975])
        comparisons[key]=dict(left=left,right=right,gain_pp=float(diff.mean()*100),descriptive_ci95_pp=ci.tolist(),
            per_seed_gain_pp=((correct[left].mean(1)-correct[right].mean(1))*100).tolist(),
            margin_change=float(md.mean()),descriptive_margin_ci95=mci.tolist(),
            correct_similarity_change=float((scores[left][:,np.arange(len(truth)),truth]-scores[right][:,np.arange(len(truth)),truth]).mean()),
            hardest_negative_similarity_change=float((scores[left][:,np.arange(len(truth)),truth]-scores[right][:,np.arange(len(truth)),truth]-(margins[left]-margins[right])).mean()),
            repaired_per_seed=(correct[left]&~correct[right]).sum(1).tolist(),broken_per_seed=(~correct[left]&correct[right]).sum(1).tolist(),
            prediction_change_rate=float((scores[left].argmax(-1)!=scores[right].argmax(-1)).mean()))
    for arm in ('R0','R2'):
        compare(f'{arm}_compression_vs_dense',name(arm,'r2_memory'),name(arm,'dense'))
        compare(f'{arm}_random_vs_R2_selection',name(arm,'random_memory'),name(arm,'r2_memory'))
    compare('weight_effect_dense',name('R2','dense'),name('R0','dense'))
    compare('weight_effect_compressed',name('R2','r2_memory'),name('R0','r2_memory'))
    for layer in (10,11):compare(f'R0_compact_block{layer}_only',name('R0',f'compact_block{layer}_only'),name('R0','dense'))
    for policy in ('dense','r2_memory'):
        compare(f'R0_blocks_R2_readout__{policy}',name('R0',policy,'R2'),name('R0',policy))
        compare(f'R2_blocks_R0_readout__{policy}',name('R2',policy,'R0'),name('R2',policy))
    with (REP/'退化逐题对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        cells=[name(a,p) for a in ('R0','R2') for p in ('dense','r2_memory')]
        writer.writerow(['UID','查询源视频','视角','任务','查询描述','正确候选描述',
            *[f'{c}_三种子预测' for c in cells],*[f'{c}_三种子对错' for c in cells],
            '压缩效应_margin_R0','权重效应_margin_压缩读取','人工错误类别_留空','人工备注_留空'])
        for j,uid in enumerate(uids):
            q=annotation[uid]
            writer.writerow([uid,source[j],metadata[source[j]]['view'],metadata[source[j]]['task'],q['query']['narration_en'],
                q['choices'][str(q['answer'])]['narration_en'],
                *[scores[c][:,j].argmax(-1).tolist() for c in cells],*[correct[c][:,j].astype(int).tolist() for c in cells],
                float((margins[name('R0','r2_memory')][:,j]-margins[name('R0','dense')][:,j]).mean()),
                float((margins[name('R2','r2_memory')][:,j]-margins[name('R0','r2_memory')][:,j]).mean()),'',''])
    interaction=(correct[name('R2','r2_memory')].astype(float)-correct[name('R2','dense')]
                 -correct[name('R0','r2_memory')]+correct[name('R0','dense')]).mean(0)
    ici=np.quantile(bootstrap(interaction,source,np.random.default_rng(20261027)),[.025,.975])*100
    local={}
    queryids=ep['indices'][rows,0]
    for arm in ('R0','R2'):
        local[arm]={}
        for layer in (10,11):
            prefix=f'{arm}__block{layer}__';measurements={}
            for k in errors[0].files:
                if not k.startswith(prefix):continue
                values=[]
                for archive in errors:
                    lookup={int(i):j for j,i in enumerate(archive['ids'])}
                    values.append(archive[k][[lookup[int(i)] for i in queryids]])
                v=np.stack(values);metric=k[len(prefix):]
                measurements[metric]=dict(per_seed_mean=v.mean(1).tolist(),mean=float(v.mean()),
                      population_weighted_mean=float((v.mean(0)*weights).sum()),median=float(np.median(v)),
                      p90=float(np.quantile(v,.9)),maximum=float(v.max()))
            local[arm][str(layer)]=measurements
    histories={};updates={}
    for arm in ('R0','R1','R2','R3','R4','R5'):
        histories[arm]={};updates[arm]={}
        for seed in SEEDS:
            folder=EXP/f'fits/{arm}/seed{seed}';histories[arm][str(seed)]=read(folder/'history.json')
            ck=torch.load(folder/'final.pt',map_location='cpu')['updates']
            initial=torch.load(MBASE/f'heads/M/seed{seed}/best.pt',map_location='cpu')['updates']
            groups={'block10':lambda k:k.startswith('visual.blocks.10.'),'block11':lambda k:k.startswith('visual.blocks.11.'),
                    'final_norm':lambda k:k.startswith('visual.norm.'),'projection':lambda k:k=='projection'}
            group_stats={}
            for group,take in groups.items():
                keys=[k for k in initial if take(k)]
                base=sum(float(initial[k].double().square().sum()) for k in keys)
                delta=sum(float((ck[k].double()-initial[k].double()).square().sum()) for k in keys)
                group_stats[group]=dict(relative_parameter_L2=float(np.sqrt(delta/max(base,1e-30))),keys=len(keys))
            updates[arm][str(seed)]=group_stats
    result=dict(stage='exploratory_post_result_degradation_diagnosis',questions=len(uids),seeds=list(SEEDS),
          results=results,comparisons=comparisons,weight_memory_interaction=dict(gain_pp=float(interaction.mean()*100),ci95_pp=ici.tolist()),
          local_approximation=local,histories=histories,relative_updates=updates,replay_max_abs=replay,
          protocol_sha256=digest(POST/'degradation_protocol.json'),TEST_access=False,training=False,human_labels_used=False,
          limits=['Fixed256 cohort, not full800 VAL; source-cluster CIs descriptive and unadjusted',
                  'Crossing learned weights/readout/membership creates distribution shifts; no exclusive causal claim',
                  'Jensen/value decomposition is local CLS attention on native trajectories, not a repaired-model accuracy',
                  'Final/best/resume snapshots do not reconstruct per-step soft/hard loss trajectories or prove ST gradient failure'])
    atomic(REP/'退化原因诊断.json',result)
    lines=['语义路由退化原因诊断：冻结模型的探索性分析','固定256题、种子17/29/43；复用保存的R2分组，查询与全部候选同条件计算。','未训练、未读取或重评TEST、未改变原结果。','',
           '一、主干权重×记忆组成（准确率%，三种子均值）','权重    稠密记忆    R2记忆    随机32+上下文']
    for arm in ('R0','R2'):lines.append(f"{arm}      "+'    '.join(f"{results[name(arm,p)]['mean_accuracy']:.3f}" for p in ('dense','r2_memory','random_memory')))
    lines.extend(['','二、配对差值及描述性95%源视频聚类区间'])
    for k,v in comparisons.items():lines.append(f"{k}: {v['gain_pp']:+.3f}pp [{v['descriptive_ci95_pp'][0]:+.3f}, {v['descriptive_ci95_pp'][1]:+.3f}]；margin {v['margin_change']:+.5f}")
    lines.append(f"权重×记忆交互: {interaction.mean()*100:+.3f}pp；区间 {ici.tolist()}")
    lines.extend(['','三、局部注意力近似：同一个输入、不同K/V读取；只在固定查询片段上统计'])
    for arm in ('R0','R2'):
        for layer in (10,11):
            m=local[arm][str(layer)]
            lines.append(f"{arm} block{layer}: CLS相对L2 {m['CLS_relative_error']['mean']:.4f}；仅修正分组概率后 {m['exact_group_mass_CLS_relative_error']['mean']:.4f}；仅修正组内值后 {m['exact_group_values_CLS_relative_error']['mean']:.4f}；两项精确重构 {m['exact_both_CLS_relative_error']['maximum']:.3g}")
            lines.append(f"  上下文注意力质量 {m['dense_context_attention_mass']['mean']:.4f} → {m['compact_context_attention_mass']['mean']:.4f}；证据质量 {m['dense_evidence_attention_mass']['mean']:.4f} → {m['compact_evidence_attention_mass']['mean']:.4f}；加权Jensen缺口 {m['context_Jensen_gap_weighted']['mean']:.4f}")
    lines.extend(['','四、训练曲线与参数位移'])
    for arm in histories:
        means=[np.mean([histories[arm][str(s)][e]['caption_holdout'] for s in SEEDS]) for e in range(5)]
        lines.append(f"{arm} 内部caption损失epoch0–4: "+', '.join(f'{v:.4f}' for v in means))
    lines.extend(['','局部概率/值精确修正仅用于误差分解，不是新的正式模型成绩。',
                  '不能将三种子的配对差值拆成相互独立的因果贡献百分比。',
                  '现有保存点不足以回放完整训练轨迹；优化/直通梯度问题只能据曲线和权重交叉提出线索，不能声称已证实。'])
    (REP/'退化原因诊断.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
