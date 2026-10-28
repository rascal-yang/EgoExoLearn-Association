import csv,json,time
from tm_common import *
import numpy as np

def main():
 ep=np.load(BASE/'val_episodes.npz');ids=np.unique(ep['indices']);lookup={int(i):j for j,i in enumerate(ids)};ix=np.vectorize(lookup.get)(ep['indices']);allf=np.empty((len(CONFIGS),len(SEEDS),len(ids),256),np.float32);diagnostics=np.empty((len(ids),3),np.float32)
 for g in (2,3):
  folder=EXP/f'gpu{g}';done=read(folder/'complete.json');assert done['passed'] and digest(folder/'features.npy')==done['features_sha256']
  local=np.load(folder/'ids.npy');where=np.array([lookup[int(i)] for i in local]);allf[:,:,where,:]=np.load(folder/'features.npy');diagnostics[where]=np.load(folder/'diagnostics.npy')
 qs=read(ROOT/'data/association_annotations/association_val_hard.json');scores={};good={};metrics={}
 for ci,name in enumerate(CONFIGS):
  for sj,seed in enumerate(SEEDS):
   f=allf[ci,sj];scores[f'{name}_{seed}']=np.einsum('nd,njd->nj',f[ix[:,0]],f[ix[:,1:]])
  good[name]=np.stack([scores[f'{name}_{s}'].argmax(1)==ep['truth'] for s in SEEDS]);counts=good[name].sum(1)
  metrics[name]=dict(counts=counts.tolist(),mean_accuracy=float(counts.mean()/len(ep['truth'])*100),tokens=1+16*CONFIGS[name][0]+196-CONFIGS[name][0])
 reference=np.load(DENSE/'scores.npz');replay={}
 for seed in SEEDS:
  a=scores[f'dense16_{seed}'];b=reference[f'M16_{seed}_normal'];replay[str(seed)]=dict(max_score_error=float(np.abs(a-b).max()),prediction_changes=int((a.argmax(1)!=b.argmax(1)).sum()),current_correct=int((a.argmax(1)==ep['truth']).sum()),historical_correct=int((b.argmax(1)==ep['truth']).sum()),note='Current full float32 encoder; historical cache rounded block9 activations to float16')
 videos=np.array([qs[str(u)]['query']['video_uid'] for u in ep['uids']]);unique=np.unique(videos);n=np.array([(videos==v).sum() for v in unique]);pick=np.random.default_rng(20261026).integers(len(unique),size=(5000,len(unique)))
 pairs=[(name,'dense16') for name in CONFIGS if name!='dense16']+[('merge49','random49'),('merge49','haar49'),('exact49','merge49')];comparisons={}
 for a,b in pairs:
  d=(good[a].astype(float)-good[b]).mean(0);sums=np.array([d[videos==v].sum() for v in unique]);boot=sums[pick].sum(1)/n[pick].sum(1)*100
  comparisons[a+'_vs_'+b]=dict(delta_pp=float(d.mean()*100),ci95_video_cluster_pp=np.quantile(boot,[.025,.975]).tolist(),repairs_per_seed=((good[a])&(~good[b])).sum(1).tolist(),breaks_per_seed=((~good[a])&(good[b])).sum(1).tolist())
 durations=np.array([min(qs[str(u)]['query']['end_sec']-qs[str(u)]['query']['start_sec'],qs[str(u)]['choices'][str(int(t))]['end_sec']-qs[str(u)]['choices'][str(int(t))]['start_sec']) for u,t in zip(ep['uids'],ep['truth'])]);camera=diagnostics[ix[:,0],1]
 query_types=np.array([qs[str(u)]['types'] for u in ep['uids']]);assert set(query_types)=={1,2}
 groups={'short_pair_le1s':durations<=1,'pair_1to3s':(durations>1)&(durations<=3),'pair_gt3s':durations>3,'query_ego':query_types==1,'query_exo':query_types==2,'camera_proxy_low':camera<=np.median(camera),'camera_proxy_high':camera>np.median(camera)}
 strata={k:dict(n=int(mask.sum()),accuracy={a:float(g[:,mask].mean()*100) if mask.any() else None for a,g in good.items()}) for k,mask in groups.items()}
 with (REP/'共同退化与修复清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.writer(f);writer.writerow(['策略','类别','题号','查询动作','正确动作','压缩模型选择的动作（种子17/29/43）','较短片段秒数','相机运动代理值'])
  for a in CONFIGS:
   if a=='dense16':continue
   broken=np.all(good['dense16']&~good[a],axis=0);repaired=np.all(~good['dense16']&good[a],axis=0)
   for label,mask in [('三个种子共同退化',broken),('三个种子共同修复',repaired)]:
    for j in np.flatnonzero(mask):
     uid=str(ep['uids'][j]);q=qs[uid];pred=[int(scores[f'{a}_{s}'][j].argmax()) for s in SEEDS]
     writer.writerow([a,label,uid,q['query'].get('narration_ch',''),q['choices'][str(int(ep['truth'][j]))].get('narration_ch',''),' / '.join(q['choices'][str(p)].get('narration_ch','') for p in pred),round(float(durations[j]),2),round(float(camera[j]),2)])
 np.savez_compressed(EXP/'scores.npz',uids=ep['uids'],truth=ep['truth'],**scores)
 result=dict(metrics=metrics,comparisons=comparisons,strata=strata,dense_reference_replay=replay,TEST_access=False,training_performed=False,score='cosine',routing_uses_labels=False,camera_proxy_is_not_ground_truth=True)
 atomic(REP/'results.json',result)
 names={'dense16':'完整16帧','merge98':'共享背景＋98个动态位置','merge49':'共享背景＋49个动态位置','haar49':'49个动态位置，取消局部运动补偿','random49':'49个随机动态位置，同等token预算','exact49':'49个动态位置，背景逐帧注意力后平均'}
 lines=['16帧背景token合并：固定权重验证','800题、20选1、3个种子平均；相同解码和余弦评分；无测试集评测。','']
 for name,v in metrics.items():lines.append(f'{names[name]}：{v["mean_accuracy"]:.3f}%；各种子答对{v["counts"]}题；含CLS共{v["tokens"]}个token。')
 lines+=['','相对当前完整16帧的差异：']
 for a,b in pairs:
  c=comparisons[a+'_vs_'+b];lines.append(f'{a} 对 {b}：{c["delta_pp"]:+.3f}个百分点，95%视频分组区间{c["ci95_video_cluster_pp"]}；修复{c["repairs_per_seed"]}题，损坏{c["breaks_per_seed"]}题。')
 lines+=['','速度见latency.json：完整编码器，包含路由和合并；不含解码。单条与批处理分开报告。','相机运动是粗粒度局部匹配代理值，不能当成人工标注。','静态空间查询通过时间平均的键分布更新，是对原计算的近似；这可能带来语义变化，不能把所有退化归因于路由。','已保留每题每个种子的分数、每片段的动态位置与频率分数，供进一步机制分析。']
 (REP/'结果阅读版.txt').write_text('\n'.join(lines)+'\n')
 atomic(EXP/'complete.json',dict(passed=True,TEST_access=False,training_performed=False,completed_at=time.time(),scores_sha256=digest(EXP/'scores.npz')))
 print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
