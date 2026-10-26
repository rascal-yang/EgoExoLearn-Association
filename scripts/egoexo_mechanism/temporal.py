import os,sys,time,argparse
from mech_common import *
import numpy as np,torch
import backend as b
from model import VideoTail
from data import Prefixes
CONDITIONS={'reverse':[3,2,1,0],'shuffle':[2,0,3,1],'repeat':[2,2,2,2]}
def extract():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='3';torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28);b.ref.init_worker('egoexo')
 ep=np.load(BASE/'val_episodes.npz');ids=np.unique(ep['indices'][:,0]);plan=read(BASE/'plan.json');folder=EXP/'temporal';folder.mkdir(exist_ok=True);np.save(folder/'indices.npy',ids)
 mm={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype='float16',shape=(len(ids),785,768)) for k in CONDITIONS}
 m=b.load_model().eval();captured=[];h=m.visual.blocks[9].register_forward_hook(lambda mod,i,o:captured.append(o));start=time.time()
 with torch.inference_mode():
  decoded=b.ref.bounded_map(b.decode,[plan['keys'][int(i)] for i in ids],workers=4,initializer=b.ref.init_worker,initargs=('egoexo',))
  for begin in range(0,len(ids),4):
   cool();items=[next(decoded) for _ in ids[begin:begin+4]];x=torch.tensor(np.stack([v[0] for v in items]),device='cuda');
   for k,order in CONDITIONS.items():
    captured.clear();m.encode_image(x[:,:,order]);p=captured.pop();mm[k][begin:begin+len(x)]=p.cpu().numpy().astype('float16')
   atomic(folder/'progress.json',dict(completed=begin+len(x),total=len(ids),elapsed_s=time.time()-start))
  decoded.close()
 h.remove()
 for v in mm.values():v.flush()
 atomic(folder/'complete.json',dict(passed=True,unique_queries=len(ids),conditions=CONDITIONS,intervention='query frames only; reorder original sampled pixels before full encoder; candidates unchanged',sources={str(Path(__file__)):digest(__file__)},TEST_access=False))

def evaluate():
 assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3');torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.55)
 ep=np.load(BASE/'val_episodes.npz');ids=np.unique(ep['indices']);lookup={int(i):j for j,i in enumerate(ids)};index=np.vectorize(lookup.get)(ep['indices']);a=Prefixes();folder=EXP/'temporal';assert read(folder/'complete.json')['passed'];tids=np.load(folder/'indices.npy');tl={int(i):j for j,i in enumerate(tids)};qindex=np.array([tl[int(i)] for i in ep['indices'][:,0]])
 cache={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in CONDITIONS};scores={};cos={};features={};checks={}
 old=np.load(BASE/'val_scores.npz')
 for arm in ['B','C','D','N','V','M']:
  for seed in SEEDS:
   path=(BASE if arm in 'BCD' else EXP)/f'heads/{arm}/seed{seed}';done=read(path/'complete.json');assert done['passed'] and digest(path/'best.pt')==done['checkpoint_sha256']
   m=VideoTail().eval();state=torch.load(path/'best.pt',map_location='cpu');assert set(state['updates'])==set(m.updates());m.load_state_dict(state['updates'],strict=False);fs=[]
   with torch.inference_mode():
    for begin in range(0,len(ids),64):
     cool();x,_=a.batch(ids[begin:begin+64],False);g,_,_=m(x.float(),False);fs.append(g.cpu().numpy())
    f=np.concatenate(fs);candidates=f[index[:,1:]];cn=candidates/np.linalg.norm(candidates,axis=-1,keepdims=True)
    for cond in ['normal',*CONDITIONS]:
     if cond=='normal':q=f[index[:,0]]
     else:
      output=[]
      for begin in range(0,len(tids),64):
       cool();x=torch.tensor(np.array(cache[cond][begin:begin+64]),device='cuda').float();g,_,_=m(x,False);output.append(g.cpu().numpy())
      q=np.concatenate(output)[qindex]
     name=f'{arm}_seed{seed}_{cond}';scores[name]=np.einsum('nd,njd->nj',q,candidates);cos[name]=np.einsum('nd,njd->nj',q/np.linalg.norm(q,axis=-1,keepdims=True),cn)
     if cond=='normal' and arm in 'BCD':
      err=float(np.max(np.abs(scores[name]-old[f'{arm}_seed{seed}'])));assert np.array_equal(scores[name].argmax(1),old[f'{arm}_seed{seed}'].argmax(1));checks[name]=err
    features[f'{arm}_seed{seed}']=f
   del m;torch.cuda.empty_cache();atomic(EXP/'evaluation_progress.json',dict(last=f'{arm}_{seed}',configs=len(features),total=18))
 np.savez_compressed(EXP/'scores.npz',uids=ep['uids'],truth=ep['truth'],**scores);np.savez_compressed(EXP/'cosine_scores.npz',uids=ep['uids'],truth=ep['truth'],**cos);np.savez_compressed(EXP/'normal_features.npz',indices=ids,**features)
 atomic(EXP/'evaluation_complete.json',dict(passed=True,checks=checks,TEST_access=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','evaluate']);args=p.parse_args();extract() if args.mode=='extract' else evaluate()
