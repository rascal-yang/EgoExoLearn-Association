import os,time
from state_common import *
import numpy as np,torch
import backend as b
from cache import decode16

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8);b.ref.init_worker('egoexo');ep=np.load(BASE/'val_episodes.npz');ids=np.unique(ep['indices'][:,0]);keys=read(BASE/'plan.json')['keys'];folder=EXP/'interventions';folder.mkdir(exist_ok=False);np.save(folder/'ids.npy',ids);orders={'reverse':list(range(15,-1,-1)),'shuffle':np.random.default_rng(20261025).permutation(16).tolist(),'repeat':[8]*16};arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype='float16',shape=(len(ids),3137,768)) for k in orders};m=b.load_model().eval();capture=[];hook=m.visual.blocks[9].register_forward_hook(lambda mod,i,o:capture.append(o));start=time.time()
 with torch.inference_mode():
  decoded=b.ref.bounded_map(decode16,[keys[int(i)] for i in ids],workers=4,initializer=b.ref.init_worker,initargs=('egoexo',))
  for begin in range(0,len(ids),2):
   cool();items=[next(decoded) for _ in ids[begin:begin+2]];x=torch.tensor(np.stack([v[0] for v in items]),device='cuda')
   for name,order in orders.items():
    capture.clear();m.encode_image(x[:,:,order]);arrays[name][begin:begin+len(x)]=capture.pop().cpu().numpy().astype('float16')
   atomic(folder/'progress.json',dict(completed=begin+len(x),total=len(ids),elapsed_s=time.time()-start))
  decoded.close()
 hook.remove()
 for a in arrays.values():a.flush()
 atomic(folder/'complete.json',dict(passed=True,orders=orders,rows=len(ids),input_level_intervention=True))
if __name__=='__main__':main()
