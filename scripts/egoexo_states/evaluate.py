import os,time
from state_common import *
import numpy as np,torch
from state_model import StateModel
from state_data import Data

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.85);ep=np.load(BASE/'val_episodes.npz');ids=np.unique(ep['indices']);lookup={int(i):j for j,i in enumerate(ids)};ix=np.vectorize(lookup.get)(ep['indices']);data=Data();inter=EXP/'interventions';tids=np.load(inter/'ids.npy');tlookup={int(i):j for j,i in enumerate(tids)};qi=np.array([tlookup[int(i)] for i in ep['indices'][:,0]]);orders=read(inter/'complete.json')['orders'];scores={};features={};stage_reports={};heads={};rawM={};start=time.time()
 for arm in ARMS:
  for seed in SEEDS:
   folder=EXP/f'fits/{arm}/seed{seed}';done=read(folder/'complete.json');assert done['passed'] and digest(folder/'best.pt')==done['checkpoint_sha256'];m=StateModel(arm,seed).eval();state=torch.load(folder/'best.pt',map_location='cpu');assert set(state['updates'])==set(m.updates());m.load_state_dict(state['updates'],strict=False);allf=[];allstages=[]
   with torch.inference_mode():
    for begin in range(0,len(ids),16):
     cool();d=data.batch(ids[begin:begin+16],False);g,p,f,di,rev,w=m(d['x'].float(),d['mask']);allf.append(g.cpu().numpy());allstages.append(w.cpu().numpy())
    fs=np.concatenate(allf);features[f'{arm}_{seed}']=fs;stage_reports[f'{arm}_{seed}']=np.concatenate(allstages)[ix[:,0]];candidate=fs[ix[:,1:]]
    for cond in ['normal',*orders]:
     if cond=='normal':query=fs[ix[:,0]]
     else:
      cache=np.load(inter/(cond+'.npy'),mmap_mode='r');outputs=[]
      for begin in range(0,len(tids),16):
       cool();d=data.batch(tids[begin:begin+16],False);mask=d['mask'][:,orders[cond]]
       if cond=='repeat':mask=torch.ones_like(mask) # one repeated observation in every slot; input sensitivity, not natural supervision
       x=torch.tensor(np.array(cache[begin:begin+16]),device='cuda').float();g,*_=m(x,mask);outputs.append(g.cpu().numpy())
      query=np.concatenate(outputs)[qi]
     scores[f'{arm}_{seed}_{cond}']=np.einsum('nd,njd->nj',query,candidate)
   heads[f'{arm}_{seed}']=dict(epoch=state['epoch'],gate=float(.5*m.gate_logit.sigmoid()));del m;torch.cuda.empty_cache();atomic(EXP/'evaluation_progress.json',dict(completed=f'{arm}_{seed}',models=len(heads),total=6,elapsed_s=time.time()-start))
 # Unmodified M at16 frames, before new heads/training. Report raw dot and cosine separately.
 for seed in SEEDS:
  m=StateModel('temporal_mean',seed).eval();allf=[]
  with torch.inference_mode():
   for begin in range(0,len(ids),16):
    cool();d=data.batch(ids[begin:begin+16],False);g,_,_=m.encode_tail(d['x'].float());allf.append(g.cpu().numpy())
  f=np.concatenate(allf);rawM[f'M16_{seed}']=np.einsum('nd,njd->nj',f[ix[:,0]],f[ix[:,1:]]);f=f/np.linalg.norm(f,axis=1,keepdims=True);scores[f'M16_{seed}_normal']=np.einsum('nd,njd->nj',f[ix[:,0]],f[ix[:,1:]]);del m;torch.cuda.empty_cache()
 np.savez_compressed(EXP/'scores.npz',uids=ep['uids'],truth=ep['truth'],**scores);np.savez_compressed(EXP/'M16_raw_scores.npz',uids=ep['uids'],truth=ep['truth'],**rawM);np.savez_compressed(EXP/'features.npz',indices=ids,**features);np.savez_compressed(REP/'query_stage_weights.npz',uids=ep['uids'],**stage_reports)
 metrics={}
 for arm in [*ARMS,'M16']:
  metrics[arm]={}
  for cond in (['normal'] if arm=='M16' else ['normal',*orders]):
   counts=[int((scores[f'{arm}_{s}_{cond}'].argmax(1)==ep['truth']).sum()) for s in SEEDS];metrics[arm][cond]=dict(counts=counts,mean_accuracy=float(np.mean(counts)/8))
 atomic(REP/'results.json',dict(metrics=metrics,heads=heads,TEST_access=False,stage_labels='latent only, not human ground truth',direction_labels='caption rule weak labels',score='dot of normalized final embeddings; M16 cosine for matched scoring, raw saved separately'))
 lines=['完整状态变化方案：验证集结果','', '两模型相同16帧、相同训练数据/目标/步数；参数量小幅不同，见preflight。阶段是学习的隐变量，没有人工阶段真值。']
 for arm,v in metrics.items():
  for cond,d in v.items():lines.append(f'{arm} {cond}: {d["mean_accuracy"]:.3f}% {d["counts"]}/800')
 lines+=['','干预分数仅是原答案保留率，不是反转视频动作真值。未评测测试集。']
 (REP/'结果阅读版.txt').write_text('\n'.join(lines)+'\n');atomic(EXP/'evaluation_complete.json',dict(passed=True,TEST_access=False));print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
