import os,time,copy
from tm_common import *
import numpy as np,torch
from model import VideoTail
from tm_model import *

def unpack(x,selected,static,T,N=196):
 B,L,D=x.shape;K=selected.shape[1]
 p=x.new_empty(B,T,N,D)
 p.scatter_(2,selected[:,None,:,None].expand(-1,T,-1,D),x[:,1:1+T*K].reshape(B,T,K,D))
 p.scatter_(2,static[:,None,:,None].expand(-1,T,-1,D),x[:,1+T*K:][:,None].expand(-1,T,-1,-1))
 return torch.cat([x[:,:1],p.flatten(1,2)],1)

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='2';torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.86);torch.manual_seed(1717)
 EXP.mkdir(parents=True,exist_ok=True);REP.mkdir(parents=True,exist_ok=True)
 m=VideoTail().eval();m.load_state_dict(torch.load(MBASE/'heads/M/seed17/best.pt',map_location='cpu')['updates'],strict=False);checks={}
 with torch.no_grad():
  double_block=copy.deepcopy(m.visual.blocks[11]).double()
  for T in (4,16):
   rep=torch.randn(1,1,7,768,device='cuda',dtype=torch.float64).expand(-1,T,-1,-1)
   rep=torch.cat([torch.randn(1,1,768,device='cuda',dtype=torch.float64),rep.flatten(1,2)],1)
   for K in (0,3,7):
    px,di,si=pack(rep,torch.rand(1,7,device='cuda'),K,T,7)
    out=unpack(packed_block(double_block,px,T,K,7),di,si,T,7)
    err=float((out-dense_block(double_block,rep,T,7)).abs().max());assert err<1e-10,('float64 equivalence',T,K,err)
    checks[f'float64_static_T{T}_K{K}']=err
  del double_block
  for T in (4,16):
   cool();x=torch.randn(2,1+T*196,768,device='cuda');block=m.visual.blocks[11]
   dense=dense_block(block,x,T);packed=packed_block(block,x,T,196)
   err=float((dense-packed).abs().max());assert err<2e-5,('no compression',T,err);checks[f'identity_T{T}']=err
   patches=torch.randn(2,1,196,768,device='cuda').expand(-1,T,-1,-1)
   repeated=torch.cat([torch.randn(2,1,768,device='cuda'),patches.flatten(1,2)],1)
   for K in (0,49,98):
    px,di,si=pack(repeated,torch.rand(2,196,device='cuda'),K,T)
    out=unpack(packed_block(block,px,T,K),di,si,T)
    reference=dense_block(block,repeated,T);err=float((out-reference).abs().max());relative=float((out-reference).norm()/reference.norm());assert err<1e-4 and relative<1e-5,('static equivalence',T,K,err,relative);checks[f'static_equivalence_T{T}_K{K}']=dict(max_error=err,relative_l2=relative)
  content=torch.randn(1,1,196,768,device='cuda').expand(-1,16,-1,-1).clone()
  assert haar_score(content).max()==0
  content[:,8,10]+=2;raw,comp,_=routing_scores(content)
  assert raw.argmax(1).item()==10 and comp.argmax(1).item()==10
  checks['isolated_brief_event_detected']=True
  v=torch.randn(1,3,16,224,224,device='cuda');expected=m.visual(v)@m.projection;got=full_forward(m,v,196)
  err=float((expected-got).abs().max());assert err<2e-5,('full replay',err);checks['full_encoder_replay_error']=err
 x=torch.randn(2,3137,768,device='cuda',requires_grad=True);px,di,si=pack(x,torch.rand(2,196,device='cuda'),49)
 result=packed_block(m.visual.blocks[11],px,16,49);result.square().mean().backward()
 assert torch.isfinite(x.grad).all() and (x.grad.abs().sum(-1)>0).all()
 assert m.visual.blocks[11].mlp.fc1.weight.grad.abs().sum()>0
 checks['finite_backward_and_all_input_tokens_receive_gradient']=True
 m.zero_grad(set_to_none=True);del x,px,result;torch.cuda.empty_cache()
 atomic(EXP/'preflight.json',dict(passed=True,checks=checks));print(checks,flush=True)
 rows=[]
 for batch in (1,8):
  v=torch.randn(batch,3,16,224,224,device='cuda')
  for precision in ('float32','amp16'):
   for K in (196,98,49):
    cool();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
     for _ in range(3):
      with torch.cuda.amp.autocast(enabled=precision=='amp16'):out=full_forward(m,v,K)
     torch.cuda.synchronize();pairs=[];start=time.perf_counter()
     for _ in range(12):
      a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
      with torch.cuda.amp.autocast(enabled=precision=='amp16'):out=full_forward(m,v,K)
      b.record();pairs.append((a,b))
     torch.cuda.synchronize();ms=[a.elapsed_time(b) for a,b in pairs]
    row=dict(batch=batch,precision=precision,dynamic_locations=K,packed_tokens=1+16*K+196-K,gpu_median_ms=float(np.median(ms)),wall_mean_ms=(time.perf_counter()-start)/12*1000,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    rows.append(row);print(row,flush=True)
 # Full-encoder forward/backward benchmark with the same last-two-block
 # trainability as existing M. No optimizer update, cache, or gradient accumulation.
 training=[];v=torch.randn(4,3,16,224,224,device='cuda')
 for K in (196,98,49):
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
  for step in range(8):
   cool();m.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
   with torch.cuda.amp.autocast():out=full_forward(m,v,K);loss=out.float().square().mean()
   loss.backward();torch.cuda.synchronize()
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
   if step>=2:times.append((time.perf_counter()-start)*1000)
  training.append(dict(batch=4,dynamic_locations=K,step_median_ms=float(np.median(times)),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
  m.zero_grad(set_to_none=True)
 atomic(REP/'latency.json',dict(inference=rows,training_forward_backward=training,scope='Full encoder, routing/merging included, synthetic224 input, no video decoding or optimizer update; last2blocks trainable for backward benchmark',GPU=2))
 atomic(EXP/'benchmark_complete.json',dict(passed=True));print('benchmark complete',flush=True)

if __name__=='__main__':main()
