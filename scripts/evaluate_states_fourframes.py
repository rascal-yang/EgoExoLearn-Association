"""Evaluate frozen16-frame-trained state checkpoints with the historical four inputs."""
import os,sys,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/egoexo_states'))
from state_common import EXP,BASE,SEEDS,ARMS,read,atomic,digest,cool
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from state_model import StateModel
from state_data import Data as Data16
from data import Prefixes
OUT=ROOT/'reports/egoexo_states_fourframes_20261026'

class VariableFramesStateModel(StateModel):
 def set_frames(self,frames):
  assert self.position.shape[1]==16
  position=F.interpolate(self.position.detach().transpose(1,2),size=frames,mode='linear',align_corners=False).transpose(1,2).contiguous()
  self.position=nn.Parameter(position,requires_grad=False)
 def encode_tail(self,x):
  frames=(x.shape[1]-1)//196;assert x.shape[1]==1+frames*196
  for block in self.visual.blocks[10:]:x=block(x,self.visual.einops_from_space,self.visual.einops_to_space,self.visual.einops_from_time,self.visual.einops_to_time,time_n=196,space_f=frames,use_checkpoint=False)
  x=self.visual.norm(x);g=x[:,0]@self.projection;p=(x[:,1:]@self.projection).reshape(-1,frames,196,256)
  return g,p,p.mean(2)

def run(gpu):
 assert os.environ['CUDA_VISIBLE_DEVICES']==gpu and gpu in ('2','3');torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75)
 arm=ARMS[int(gpu)-2];ep=np.load(BASE/'val_episodes.npz');ids=np.unique(ep['indices']);lookup={int(i):j for j,i in enumerate(ids)};index=np.vectorize(lookup.get)(ep['indices']);a=Prefixes();scores={};start=time.time();integrity={}
 reference=np.load(EXP/'features.npz');reflookup={int(i):j for j,i in enumerate(reference['indices'])};data16=Data16()
 for seed in SEEDS:
  folder=EXP/f'fits/{arm}/seed{seed}';done=read(folder/'complete.json');assert done['passed'] and digest(folder/'best.pt')==done['checkpoint_sha256'];state=torch.load(folder/'best.pt',map_location='cpu');m=VariableFramesStateModel(arm,seed).eval();assert set(state['updates'])==set(m.updates());m.load_state_dict(state['updates'],strict=False)
  with torch.no_grad():
   # Verify unmodified16-frame path against previously saved features before adapting.
   testids=ids[:16];d=data16.batch(testids,False);g,*_=m(d['x'].float(),d['mask']);expected=reference[f'{arm}_{seed}'][[reflookup[int(i)] for i in testids]];err=float(np.max(np.abs(g.cpu().numpy()-expected)));assert err<2e-5,err
   integrity[f'{arm}_{seed}']=dict(checkpoint_sha256=done['checkpoint_sha256'],replay16_max_feature_error=err,epoch=state['epoch'])
   m.set_frames(4);features=[]
   for begin in range(0,len(ids),64):
    cool();x,mask=a.batch(ids[begin:begin+64],False);g,*_=m(x.float(),mask);features.append(g.cpu().numpy())
    if begin%512==0:atomic(OUT/f'gpu{gpu}_progress.json',dict(arm=arm,seed=seed,clips=begin+len(g),total_clips=len(ids),elapsed_s=time.time()-start))
   f=np.concatenate(features);assert np.isfinite(f).all();assert np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5);scores[f'{arm}_{seed}']=np.einsum('nd,njd->nj',f[index[:,0]],f[index[:,1:]])
  del m;torch.cuda.empty_cache()
 np.savez_compressed(OUT/f'gpu{gpu}_scores.npz',uids=ep['uids'],truth=ep['truth'],**scores);atomic(OUT/f'gpu{gpu}_complete.json',dict(passed=True,integrity=integrity,elapsed_s=time.time()-start,TEST_access=False));print('complete',gpu,flush=True)

def summarize():
 scores={};integrity={}
 for gpu in ('2','3'):
  d=read(OUT/f'gpu{gpu}_complete.json');assert d['passed'];integrity.update(d['integrity']);a=np.load(OUT/f'gpu{gpu}_scores.npz');uids=a['uids'];truth=a['truth']
  scores.update({k:a[k] for k in a.files if k not in ('uids','truth')})
 baseline=np.load(ROOT/'experiments/egoexo_mechanism_20261025/cosine_scores.npz');assert np.array_equal(uids,baseline['uids']) and np.array_equal(truth,baseline['truth'])
 for seed in SEEDS:scores[f'M_{seed}']=baseline[f'M_seed{seed}_normal']
 metrics={};good={}
 for arm in ['M',*ARMS]:
  good[arm]=np.stack([scores[f'{arm}_{s}'].argmax(1)==truth for s in SEEDS]);counts=good[arm].sum(1);metrics[arm]=dict(correct_counts=counts.tolist(),mean_accuracy=float(np.mean(counts)/8))
 q=read(ROOT/'data/association_annotations/association_val_hard.json');videos=np.array([q[u]['query']['video_uid'] for u in uids]);unique=np.unique(videos);ns=np.array([(videos==v).sum() for v in unique]);rng=np.random.default_rng(20261026);pick=rng.integers(len(unique),size=(5000,len(unique)));comparisons={}
 for other in ['M','temporal_mean']:
  delta=(good['ordered_states'].astype(float)-good[other]).mean(0);sums=np.array([delta[videos==v].sum() for v in unique]);bs=sums[pick].sum(1)/ns[pick].sum(1)*100
  comparisons['states_vs_'+other]=dict(delta_pp=float(delta.mean()*100),video_cluster_ci95_pp=np.quantile(bs,[.025,.975]).tolist())
 report=dict(metrics=metrics,comparisons=comparisons,integrity=integrity,frames=4,training_frames=16,trained_new_weights=False,position_adaptation='Linear interpolate learned16 temporal positions to4, align_corners=False; no repeat padding',sampling='Same existing historical bounded4-frame prefix cache and actual PTS masks',score='dot of normalized embeddings == cosine; M baseline cosine',TEST_access=False)
 atomic(OUT/'results.json',report);np.savez_compressed(OUT/'scores.npz',uids=uids,truth=truth,**scores)
 lines=['新状态模型4帧推理：验证集结果','', '已有16帧训练权重直接改为4帧推理，没有重新进行4帧训练。全部使用历史同一4帧输入、800题、三个种子、余弦评分。']
 for arm,v in metrics.items():lines.append(f'{arm}: {v["mean_accuracy"]:.3f}% {v["correct_counts"]}/800')
 lines+=['',str(comparisons),'','16位置编码线性插值为4个；不重复帧填充成16帧。16帧原路径与历史特征回放检查已通过。未使用测试集。']
 (OUT/'结果阅读版.txt').write_text('\n'.join(lines)+'\n');atomic(OUT/'complete.json',dict(passed=True,TEST_access=False));print('\n'.join(lines),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','summarize']);p.add_argument('--gpu',choices=['2','3']);a=p.parse_args();run(a.gpu) if a.mode=='run' else summarize()
