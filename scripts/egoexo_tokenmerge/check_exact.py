import time,os
from tm_common import *
import numpy as np,torch
from model import VideoTail
from tm_model import *

def unpack(x,selected,static,T,N=196):
 B,L,D=x.shape;K=selected.shape[1];p=x.new_empty(B,T,N,D)
 p.scatter_(2,selected[:,None,:,None].expand(-1,T,-1,D),x[:,1:1+T*K].reshape(B,T,K,D))
 p.scatter_(2,static[:,None,:,None].expand(-1,T,-1,D),x[:,1+T*K:][:,None].expand(-1,T,-1,-1))
 return torch.cat([x[:,:1],p.flatten(1,2)],1)

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='2';torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.86);torch.manual_seed(117)
 m=VideoTail().eval();m.load_state_dict(torch.load(MBASE/'heads/M/seed17/best.pt',map_location='cpu')['updates'],strict=False)
 with torch.no_grad():
  score=torch.rand(1,196,device='cuda');x=torch.randn(1,3137,768,device='cuda');px,di,si=pack(x,score,49);repeated=unpack(px,di,si,16)
  module=m.visual.blocks[11].attn
  dense=module(repeated,'b (f n) d','(b f) n d',{'f':16});expected,_,_=pack(dense,score,49)
  actual=packed_attention(module,px,16,49,196,False,True)
  err=float((actual-expected).abs().max());assert err<2e-5,err
 atomic(EXP/'exact_preflight.json',dict(passed=True,mixed_dynamic_static_spatial_attention_max_error=err,scope='Spatial attention only. Shared MLP and merging still approximate evolving dense background states.'))
 latency=read(REP/'latency.json')
 for batch in (1,8):
  v=torch.randn(batch,3,16,224,224,device='cuda')
  for precision in ('float32','amp16'):
   cool();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
   with torch.no_grad():
    for _ in range(3):
     with torch.cuda.amp.autocast(enabled=precision=='amp16'):out=full_forward(m,v,49,'exact')
    torch.cuda.synchronize();pairs=[];start=time.perf_counter()
    for _ in range(12):
     a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
     with torch.cuda.amp.autocast(enabled=precision=='amp16'):out=full_forward(m,v,49,'exact')
     b.record();pairs.append((a,b))
    torch.cuda.synchronize()
   row=dict(batch=batch,precision=precision,dynamic_locations=49,static_mode='exact',packed_tokens=932,gpu_median_ms=float(np.median([a.elapsed_time(b) for a,b in pairs])),wall_mean_ms=(time.perf_counter()-start)/12*1000,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
   latency['inference'].append(row);print(row,flush=True)
 v=torch.randn(4,3,16,224,224,device='cuda');times=[];torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 for step in range(8):
  cool();m.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
  with torch.cuda.amp.autocast():out=full_forward(m,v,49,'exact');loss=out.float().square().mean()
  loss.backward();torch.cuda.synchronize();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
  if step>=2:times.append((time.perf_counter()-start)*1000)
 latency['training_forward_backward'].append(dict(batch=4,dynamic_locations=49,static_mode='exact',step_median_ms=float(np.median(times)),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
 atomic(REP/'latency.json',latency);print('exact spatial reference checked and benchmarked',flush=True)
if __name__=='__main__':main()
