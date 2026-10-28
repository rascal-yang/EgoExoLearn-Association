import os,time,argparse
from tm_common import *
import numpy as np,torch
import backend as backend
from model import VideoTail
from tm_model import stem,dense_block,packed_block,pack,routing_scores

def memmap(path,shape,dtype):
 if path.exists():
  a=np.load(path,mmap_mode='r+');assert a.shape==shape and a.dtype==np.dtype(dtype);return a
 return np.lib.format.open_memmap(path,mode='w+',shape=shape,dtype=dtype)

def main(gpu,batch):
 assert os.environ['CUDA_VISIBLE_DEVICES']==gpu;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.86)
 protocol=read(EXP/'protocol.json')
 for path,h in protocol['sources'].items():assert digest(path)==h,path
 ep=np.load(BASE/'val_episodes.npz');ids=np.unique(ep['indices'])[int(gpu)-2::2];keys=read(BASE/'plan.json')['keys']
 folder=EXP/f'gpu{gpu}';folder.mkdir(exist_ok=True)
 if (folder/'complete.json').exists():return
 if (folder/'ids.npy').exists():assert np.array_equal(np.load(folder/'ids.npy'),ids)
 else:np.save(folder/'ids.npy',ids)
 features=memmap(folder/'features.npy',(len(CONFIGS),len(SEEDS),len(ids),256),'float32')
 times=memmap(folder/'pts.npy',(len(ids),16),'float64')
 selected=memmap(folder/'selected.npy',(len(CONFIGS)-1,len(ids),98),'int16')
 haar=memmap(folder/'haar.npy',(len(ids),196),'float32');comp=memmap(folder/'compensated.npy',(len(ids),196),'float32')
 diagnostic=memmap(folder/'diagnostics.npy',(len(ids),3),'float32')
 cursor=read(folder/'progress.json')['completed'] if (folder/'progress.json').exists() else 0
 models=[]
 for j,seed in enumerate(SEEDS):
  m=VideoTail().eval();state=torch.load(MBASE/f'heads/M/seed{seed}/best.pt',map_location='cpu');assert set(state['updates'])==set(m.updates());m.load_state_dict(state['updates'],strict=False);m.requires_grad_(False)
  if j:
   for layer in range(10):m.visual.blocks[layer]=torch.nn.Identity()
  models.append(m)
 shared=models[0];start=time.time();resumed_from=cursor
 # Each clip is decoded once. Shared blocks2..9 are evaluated once per policy,
 # and only the final2blocks/projection are repeated for the three seeds.
 decoded=backend.ref.bounded_map(decode16,[keys[int(i)] for i in ids[cursor:]],workers=4,initializer=backend.ref.init_worker,initargs=('egoexo',))
 try:
  with torch.no_grad():
   for begin in range(cursor,len(ids),batch):
    cool();n=min(batch,len(ids)-begin);items=[next(decoded) for _ in range(n)]
    video=torch.tensor(np.stack([a[0] for a in items]),device='cuda');prefix,content=stem(shared.visual,video)
    raw,adjusted,diags=routing_scores(content);del content
    for block in shared.visual.blocks[:2]:prefix=dense_block(block,prefix)
    haar[begin:begin+n]=raw.cpu().numpy();comp[begin:begin+n]=adjusted.cpu().numpy();diagnostic[begin:begin+n]=torch.stack(list(diags.values()),1).cpu().numpy()
    times[begin:begin+n]=[a[1] for a in items]
    for ci,(name,(K,mode)) in enumerate(CONFIGS.items()):
     cool()
     if mode=='dense':x=prefix
     else:
      if mode=='random':score=torch.tensor(np.stack([np.random.default_rng(20261026+int(i)).random(196) for i in ids[begin:begin+n]]),device='cuda',dtype=torch.float32)
      else:score=adjusted if mode in ('compensated','exact') else raw
      x,dynamic,static=pack(prefix,score,K);selected[ci-1,begin:begin+n]=-1;selected[ci-1,begin:begin+n,:K]=dynamic.cpu().numpy()
     for block in shared.visual.blocks[2:10]:x=dense_block(block,x) if mode=='dense' else packed_block(block,x,16,K,static_exact=mode=='exact')
     for sj,m in enumerate(models):
      h=x
      for block in m.visual.blocks[10:]:h=dense_block(block,h) if mode=='dense' else packed_block(block,h,16,K,static_exact=mode=='exact')
      z=m.visual.norm(h)[:,0]@m.projection;features[ci,sj,begin:begin+n]=torch.nn.functional.normalize(z.float(),dim=-1).cpu().numpy()
      del h,z
     del x
    del prefix,video
    for array in (features,times,selected,haar,comp,diagnostic):array.flush()
    atomic(folder/'progress.json',dict(completed=begin+n,total=len(ids),elapsed_s=time.time()-start,resumed_from=resumed_from,pid=os.getpid(),gpu=gpu,batch=batch,configs=len(CONFIGS),seeds=list(SEEDS),peak_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
 finally:decoded.close()
 assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=-1),1,atol=2e-5)
 atomic(folder/'complete.json',dict(passed=True,rows=len(ids),elapsed_s=time.time()-start,features_sha256=digest(folder/'features.npy'),configs=CONFIGS,seeds=SEEDS,TEST_access=False))
 print('complete',gpu,len(ids),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',choices=['2','3'],required=True);p.add_argument('--batch',type=int,default=16);a=p.parse_args();main(a.gpu,a.batch)
