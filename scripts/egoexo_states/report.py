from state_common import *
import numpy as np,csv,html

def main():
 for seed in SEEDS:
  assert read(EXP/f'fits/temporal_mean/seed{seed}/exposures.json')==read(EXP/f'fits/ordered_states/seed{seed}/exposures.json'), 'Unmatched training exposures'
 assert read(EXP/'evaluation_complete.json')['passed'];a=np.load(EXP/'scores.npz');features=np.load(EXP/'features.npz');truth=a['truth'];uids=a['uids'].tolist();q=read(ROOT/'data/association_annotations/association_val_hard.json');good={arm:np.stack([a[f'{arm}_{s}_normal'].argmax(1)==truth for s in SEEDS]) for arm in ARMS};delta=(good['ordered_states'].astype(float)-good['temporal_mean']).mean(0);videos=np.array([q[u]['query']['video_uid'] for u in uids]);unique=np.unique(videos);sums=np.array([delta[videos==v].sum() for v in unique]);counts=np.array([(videos==v).sum() for v in unique]);rng=np.random.default_rng(20261025);ix=rng.integers(len(unique),size=(5000,len(unique)));interval=np.quantile(sums[ix].sum(1)/counts[ix].sum(1)*100,[.025,.975]).tolist()
 pairs=read(ROOT/'reports/egoexo_mechanism_20261025/direction_pairs.json')['pairs'];lookup={int(i):j for j,i in enumerate(features['indices'])};pairmetrics={}
 for arm in ARMS:
  vals=[]
  for seed in SEEDS:
   f=features[f'{arm}_{seed}'];hits=[]
   for p in pairs:
    qi,pi,ni=[lookup[p[k]] for k in ['query_index','positive_index','negative_index']];hits.append(float(f[qi]@(f[pi]-f[ni]))>0)
   vals.append(int(sum(hits)))
  pairmetrics[arm]=dict(counts=vals,total=len(pairs),mean=float(np.mean(vals)/len(pairs)*100),human_verified=False)
 repaired=good['ordered_states']&~good['temporal_mean'];broken=~good['ordered_states']&good['temporal_mean']
 with (REP/'逐题变化.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['原始题号','查询动作','正确答案','修复种子数','退化种子数'])
  for i,u in enumerate(uids):
   if repaired[:,i].any() or broken[:,i].any():w.writerow([u,q[u]['query']['narration_ch'],q[u]['choices'][str(int(truth[i]))]['narration_ch'],int(repaired[:,i].sum()),int(broken[:,i].sum())])
 stage=np.load(REP/'query_stage_weights.npz');selected=read(ROOT/'reports/egoexo_mechanism_20261025/direction_rendered.json')['uids'];cards=[]
 for u in selected:
  i=uids.index(u);curves=stage['ordered_states_17'][i];svg=[];colors=['#2979ff','#e67e22','#2e8b57']
  for j,v in enumerate(curves):
   points=' '.join(f'{20+t*36},{180-float(x)*160}' for t,x in enumerate(v));svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[j]}" stroke-width="3"/>')
  cards.append(f'<article><h2>题号 {u}</h2><p>{html.escape(q[u]["query"]["narration_ch"])}</p><svg viewBox="0 0 600 200">{"".join(svg)}</svg><p>横轴：输入16帧的先后位置；纵轴：阶段内聚合权重。蓝/橙/绿分别为隐含阶段1/2/3，不代表已经确认的真实动作阶段。</p></article>')
 (REP/'阶段权重浏览.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:900px;margin:auto;font:17px sans-serif;line-height:1.6}article{border-bottom:1px solid #ccc;padding:20px}svg{max-width:100%}</style><h1>状态模型隐含阶段</h1><p>这些例子按先前固定名单选取，没有按本轮模型成绩挑选。阶段未经过人工定位核验。</p>'+''.join(cards))
 result=dict(ordered_minus_control_pp=float(delta.mean()*100),video_cluster_ci95=interval,weak_direction_pairs=pairmetrics,stable_repaired=int(repaired.all(0).sum()),stable_broken=int(broken.all(0).sum()),TEST_access=False)
 atomic(REP/'paired_analysis.json',result)
 with (REP/'结果阅读版.txt').open('a') as f:f.write('\n完整模型相对普通时序对照：'+json.dumps(result,ensure_ascii=False,indent=2)+'\n方向对照87题仍是弱标注，只作辅助分析，不作为正式动作方向能力结论。\n')
 atomic(EXP/'complete.json',dict(passed=True,report_sha256=digest(REP/'results.json'),TEST_access=False))
if __name__=='__main__':main()
