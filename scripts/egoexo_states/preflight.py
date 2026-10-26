import os,time
from state_common import *
import torch,numpy as np
from torch.nn import functional as F
from state_model import StateModel,caption_loss,semantic_losses
from state_data import Data

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='2';torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.9);a=Data(smoke=True);ids=list(a.locations)*12;d=a.batch(ids);results={}
 for arm in ARMS:
  torch.manual_seed(17);m=StateModel(arm,17);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5);scaler=torch.cuda.amp.GradScaler();torch.cuda.reset_peak_memory_stats();start=time.time()
  for _ in range(2):
   opt.zero_grad(set_to_none=True)
   with torch.cuda.amp.autocast():
    g,p,f,di,rev,w=m(d['x'],d['mask'],True);gl,lo=semantic_losses(g,p,f,d['phrase'],d['pos'],d['types'],d['mask'],m.scale);state=m.latest_state;pair=F.softplus(((state[0]*state[2]).sum()-(state[0]*state[1]).sum()+.1)/.1)*.1;loss=.2*pair+caption_loss(g,d['caption'],d['cid'],m.scale)+.1*(gl+lo)+.1*(F.cross_entropy(di.float(),torch.arange(len(ids),device='cuda')%2)+F.cross_entropy(rev.float(),1-torch.arange(len(ids),device='cuda')%2))
   assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad)
   assert m.visual.blocks[11].mlp.fc1.weight.grad.abs().sum()>0
   assert all(p.grad is None for p in m.visual.blocks[0].parameters());scaler.step(opt);scaler.update()
  if arm=='ordered_states':
   centers=(w.float()*torch.arange(16,device='cuda')[None,None,:]).sum(-1);assert (centers[:,1:]>centers[:,:-1]).all()
  m.eval()
  with torch.no_grad(),torch.cuda.amp.autocast():g,*_=m(d['x'][:4],d['mask'][:4]);assert torch.isfinite(g).all()
  results[arm]=dict(passed=True,trainable_parameters=sum(p.numel() for p in params),max_batch=48,peak_mib=torch.cuda.max_memory_allocated()/2**20,seconds_per_step=(time.time()-start)/2,reverse_proxy_labels_are_synthetic_only_for_gradient_check=True)
  del m,opt,scaler,params,g,p,f,di,rev,w,loss;torch.cuda.empty_cache()
 atomic(EXP/'preflight.json',dict(passed=True,results=results,cache_reconstruction_passed=read(EXP/'smoke2/complete.json')['passed']));print(results,flush=True)
if __name__=='__main__':main()
