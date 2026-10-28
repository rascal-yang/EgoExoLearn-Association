"""Descriptive fixed-cohort effects, with paired source-video resampling."""
from posthoc_core import *
from summarize import bootstrap


def summarize_internal():
    annotation = read(ROOT/'data/association_annotations/association_val_hard.json')
    metadata = {r['video_uid']:r for r in read(ROOT/'repos/EgoExoLearn/video_meta_downsampled.json')}
    cohort = read(REP/'v2_固定256题诊断队列.json')
    archives = []
    for seed in SEEDS:
        path = POST/f'internal/seed{seed}'
        assert read(path/'complete.json')['passed']
        archives.append(np.load(path/'scores.npz'))
    uids, truth = archives[0]['uids'].astype(str), archives[0]['truth']
    assert all(np.array_equal(a['uids'].astype(str), uids) and np.array_equal(a['truth'], truth) for a in archives)
    source = np.array([annotation[u]['query']['video_uid'] for u in uids])
    stratum = np.array([metadata[v]['view']+'_'+metadata[v]['task'] for v in source])
    weights = np.zeros(len(uids))
    for s in cohort['strata']:
        take = stratum == s['direction']+'_'+s['task']
        assert take.sum() == s['quota']
        weights[take] = s['population']/take.sum()
    weights /= weights.sum()
    all_scores = {c:np.stack([a[c] for a in archives]) for c in CONDITIONS}
    def metrics(scores):
        pred = scores.argmax(-1)
        correct = pred == truth[None]
        wrong = scores.copy()
        wrong[:, np.arange(len(truth)), truth] = -np.inf
        margin = scores[:, np.arange(len(truth)), truth] - wrong.max(-1)
        return pred, correct, margin
    bp, bc, bm = metrics(all_scores['native'])
    result = {}
    for condition, scores in all_scores.items():
        pred, correct, margin = metrics(scores)
        delta = (correct.astype(float)-bc).mean(0)
        margin_delta = (margin-bm).mean(0)
        interval = np.quantile(bootstrap(delta, source, np.random.default_rng(20261027)), [.025,.975])*100
        item = dict(per_seed_accuracy=(correct.mean(1)*100).tolist(), mean_accuracy=float(correct.mean()*100),
                    population_weighted_accuracy=float((correct.mean(0)*weights).sum()*100),
                    gain_vs_native_pp=float(delta.mean()*100), descriptive_source_cluster_ci95_pp=interval.tolist(),
                    margin_change_vs_native=float(margin_delta.mean()), prediction_change_rate=float((pred!=bp).mean()),
                    score_rms_change=float(np.sqrt(np.mean((scores-all_scores['native'])**2))),
                    repaired_per_seed=(correct & ~bc).sum(1).tolist(), broken_per_seed=(~correct & bc).sum(1).tolist(),
                    strata={s:dict(questions=int((stratum==s).sum()), mean_accuracy=float(correct[:,stratum==s].mean()*100),
                                  margin_change=float(margin_delta[stratum==s].mean())) for s in np.unique(stratum)})
        if condition in ('semantic_off', 'random_32_plus_context', 'donor_video_concepts'):
            mean_j = []
            for seed in SEEDS:
                a = np.load(POST/f'internal/seed{seed}/routing_overlap.npz')
                lookup = {int(i):j for j,i in enumerate(a['ids'])}
                ep, rows = diagnostic()
                ix = [lookup[int(i)] for i in ep['indices'][rows,0]]
                mean_j.append(float(a[condition][ix].mean()))
            item['query_evidence_jaccard_vs_native_by_seed'] = mean_j
        result[condition] = item
    report = dict(stage='internal', questions=len(uids), source_videos=len(set(source)), seeds=list(SEEDS),
                  scope='Query and all20 candidates recomputed under each intervention; same frozen R2 final weights.',
                  results=result, inference='Descriptive mechanism diagnosis, unadjusted intervals, no confirmatory significance claim.',
                  human_grounding_evidence=False)
    atomic(REP/'v2_内部干预结果.json', report)
    lines = ['固定256题同权重内部干预（3种子均值，非完整VAL成绩）', '查询与全部候选同时按条件重计算；区间为描述性源视频聚类区间。', '']
    for c,item in result.items():
        lines.append(f"{c}: {item['mean_accuracy']:.3f}%，较原生 {item['gain_vs_native_pp']:+.3f}pp，margin变化 {item['margin_change_vs_native']:+.5f}，预测改变 {item['prediction_change_rate']*100:.2f}%")
    lines.extend(['', '这些是固定权重下的分布替换，不能单独证明唯一因果必要性，不能证明人工物体定位。'])
    (REP/'v2_内部干预结果.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    summarize_internal()
