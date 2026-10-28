import csv
import numpy as np
from rel_common import *

def main():
    scores={};metrics={};exposures=[]
    for arm in ARMS:
        folder=EXP/f'fits/{arm}/seed17';m=read(folder/'evaluation_complete.json');assert m['passed']
        data=np.load(folder/'scores.npz');scores[arm]=data['scores'];metrics[arm]=m
        if arm==ARMS[0]:uids=data['uids'];truth=data['truth']
        else:assert np.array_equal(uids,data['uids']) and np.array_equal(truth,data['truth'])
        exposures.append(np.load(folder/'exposures.npy'))
    for row in exposures[1:]:assert np.array_equal(row,exposures[0])
    original=np.load(CACHE/'scores.npz');assert np.array_equal(uids,original['uids']) and np.array_equal(truth,original['truth'])
    scores['M16_initial']=original['M16_17_normal']
    metrics['M16_initial']=dict(correct=int((scores['M16_initial'].argmax(1)==truth).sum()),accuracy=float((scores['M16_initial'].argmax(1)==truth).mean()*100))
    questions=read(ROOT/'data/association_annotations/association_val_hard.json')
    videos=np.array([questions[str(u)]['query']['video_uid'] for u in uids]);unique=np.unique(videos)
    counts=np.array([(videos==v).sum() for v in unique]);rng=np.random.default_rng(20261026)
    draw=rng.integers(len(unique),size=(5000,len(unique)))
    good={k:v.argmax(1)==truth for k,v in scores.items()};paired={}
    for control in ['A_continue_M','B_slots','D_unbound','E_scrambled','M16_initial']:
        d=good['C_bound'].astype(float)-good[control];sums=np.array([d[videos==v].sum() for v in unique])
        boot=sums[draw].sum(1)/counts[draw].sum(1)*100
        paired['C_vs_'+control]=dict(delta_pp=float(d.mean()*100),video_cluster_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
            repaired=int((good['C_bound']&~good[control]).sum()),broken=int((~good['C_bound']&good[control]).sum()))
    result=dict(metrics=metrics,paired=paired,seeds=[17],matched_exposures=True,TEST_access=False,
        human_relation_review_pending=True,interpretation='Single-seed exploratory validation; not proof of relation grounding or final three-seed results')
    atomic(REP/'seed17_results.json',result)
    with (REP/'seed17_逐题变化.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['题号','查询描述','正确候选描述',*scores.keys()])
        for i,u in enumerate(uids):
            q=questions[str(u)];w.writerow([u,q['query']['narration_ch'],q['choices'][str(int(truth[i]))]['narration_ch'],*[int(scores[k][i].argmax()) for k in scores]])
    lines=['关系绑定实验：seed17探索性结果（不是三个种子的最终结论）','']
    for arm,m in metrics.items():lines.append(f'{arm}: {m["accuracy"]:.3f}% ({m["correct"]}/800)')
    lines += ['',json.dumps(paired,ensure_ascii=False,indent=2),'','所有关系标签为自动解析弱监督；人工机制集未核验，不报告其准确率。']
    (REP/'seed17_结果阅读版.txt').write_text('\n'.join(lines)+'\n')
    atomic(EXP/'seed17_complete.json',dict(passed=True,finished=time.time(),next='Inspect controlled effects and parser/grounding diagnostics before seed29/43 replication'))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
