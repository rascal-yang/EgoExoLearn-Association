from posthoc_core import *
from summarize import bootstrap


def main():
    archives=[];windows=[]
    for seed in SEEDS:
        path=POST/f'pixels/seed{seed}';assert read(path/'complete.json')['passed']
        archives.append(np.load(path/'scores.npz'));windows.extend(read(path/'windows.json'))
    uids=archives[0]['uids'].astype(str);truth=archives[0]['truth']
    assert all(np.array_equal(a['uids'].astype(str),uids) for a in archives)
    annotation=read(ROOT/'data/association_annotations/association_val_hard.json')
    source=np.array([annotation[u]['query']['video_uid'] for u in uids])
    metadata={r['video_uid']:r for r in read(ROOT/'repos/EgoExoLearn/video_meta_downsampled.json')}
    cohort=read(REP/'v2_固定256题诊断队列.json')
    strata=np.array([metadata[v]['view']+'_'+metadata[v]['task'] for v in source]);weights=np.zeros(len(uids))
    for s in cohort['strata']:
        take=strata==s['direction']+'_'+s['task'];weights[take]=s['population']/take.sum()
    weights/=weights.sum()
    levels=collections.Counter(level for r in windows for f in r['frames'] for level in f['control_matching_levels'])
    # Map each query to its matching quality for separate exact-match summaries.
    keys=read(BASE/'plan.json')['keys'];key_to_id={tuple(k):i for i,k in enumerate(keys)}
    queryids=[key_to_id[(annotation[u]['query']['video_uid'],float(annotation[u]['query']['start_sec']),float(annotation[u]['query']['end_sec']))] for u in uids]
    exact=[]
    for seed in SEEDS:
        w={r['record']:r for r in read(POST/f'pixels/seed{seed}/windows.json')}
        exact.append(np.array([all(level=='brightness_and_gradient' for f in w[i]['frames'] for level in f['control_matching_levels']) for i in queryids]))
    exact=np.stack(exact)
    results={}
    for arm in ('R0','R1','R2'):
        results[arm]={}
        for mode in ('adaptive','fixed'):
            scores=np.stack([a[arm+'_'+mode] for a in archives])
            wrong=scores.copy()
            # scores: seed, query, variant, candidate.
            for q,t in enumerate(truth):wrong[:,q,:,t]=-np.inf
            margin=np.stack([scores[:,q,:,t] for q,t in enumerate(truth)],axis=1)-wrong.max(-1)
            pred=scores.argmax(-1);correct=pred==truth[None,:,None]
            selected_drop=margin[:,:,0]-margin[:,:,1]
            control_drop=margin[:,:,0,None]-margin[:,:,2:]
            excess=selected_drop-control_drop.mean(-1)
            interval=np.quantile(bootstrap(excess.mean(0),source,np.random.default_rng(20261027)),[.025,.975])
            exact_values=excess[exact]
            results[arm][mode]=dict(clean_accuracy=float(correct[:,:,0].mean()*100),
                selected_accuracy=float(correct[:,:,1].mean()*100),control_accuracies=(correct[:,:,2:].mean((0,1))*100).tolist(),
                selected_margin_drop=float(selected_drop.mean()),control_mean_margin_drop=float(control_drop.mean()),
                selected_minus_control_margin_drop=float(excess.mean()),descriptive_source_cluster_ci95=interval.tolist(),
                population_weighted_excess_margin_drop=float((excess.mean(0)*weights).sum()),
                exact_matching_seed_query_observations=int(exact.sum()),exact_matching_excess_margin_drop=float(exact_values.mean()) if len(exact_values) else None,
                selected_prediction_change_rate=float((pred[:,:,1]!=pred[:,:,0]).mean()),
                control_prediction_change_rate=float((pred[:,:,2:]!=pred[:,:,0,None]).mean()),
                strata={s:dict(questions=int((strata==s).sum()),excess_margin_drop=float(excess[:,strata==s].mean())) for s in np.unique(strata)})
    report=dict(questions=len(uids),seeds=list(SEEDS),scope='query_only; unchanged candidates; full encoder rerun',
                results=results,control_frame_matching_counts=dict(levels),blur_kernel=49,blur_sigma_pixels=8,window_pixels=[48,48],
                inference='Descriptive local sensitivity; no human object localization claim; exact matching separated from fallback.',human_grounding_evidence=False)
    atomic(REP/'v2_像素干预结果.json',report)
    lines=['固定256题RGB查询干预：三种子；清洁/模糊都重算全部12层。','selected−control为处理窗margin下降减去3个对照窗平均下降，正数表示处理窗更敏感。','']
    for arm in results:
        for mode,r in results[arm].items():lines.append(f"{arm}/{mode}: 清洁 {r['clean_accuracy']:.3f}%，处理窗 {r['selected_accuracy']:.3f}%，超额margin下降 {r['selected_minus_control_margin_drop']:+.6f}，区间 {r['descriptive_source_cluster_ci95']}")
    lines.extend(['',f'对照匹配层级（帧×对照窗）: {dict(levels)}','人工区域核验未完成，不能把局部敏感性解释成人工物体定位。'])
    (REP/'v2_像素干预结果.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
