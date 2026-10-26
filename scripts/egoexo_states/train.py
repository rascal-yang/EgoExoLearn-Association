import argparse,os,time
from state_common import *
import torch,numpy as np
from torch.nn import functional as F
from state_model import StateModel,caption_loss,semantic_losses
from state_data import Data

def main(arm,seed):
 assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3');torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.88);torch.manual_seed(seed)
 protocol=read(EXP/'protocol.json')
 for p,h in protocol['sources'].items():assert digest(p)==h,p
 out=EXP/'fits'/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return
 data=Data();split=np.load(EXP/'splits.npz');m=StateModel(arm,seed)
 trunk=[p for n,p in m.named_parameters() if p.requires_grad and (n.startswith('visual.') or n=='projection')];head=[p for n,p in m.named_parameters() if p.requires_grad and not(n.startswith('visual.') or n=='projection')];params=trunk+head
 opt=torch.optim.AdamW([dict(params=trunk,lr=1e-5),dict(params=head,lr=1e-4)],weight_decay=.01);scaler=torch.cuda.amp.GradScaler();initial=m.projection.detach().cpu().clone();frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad};start=time.time();steps=0;exposures={};best_epoch=0
 atomic(out/'manifest.json',dict(arm=arm,seed=seed,trainable_parameters=sum(p.numel() for p in params),protocol_sha256=digest(EXP/'protocol.json'),initial_M_sha256=digest(MBASE/f'heads/M/seed{seed}/best.pt')))
 def validate():
  m.eval();vals=[];counts=[]
  with torch.no_grad():
   for begin in range(0,len(split['holdout']),16):
    cool();d=data.batch(split['holdout'][begin:begin+16])
    with torch.cuda.amp.autocast():g,*_=m(d['x'],d['mask']);loss=caption_loss(g,d['caption'],d['cid'],m.scale)
    vals.append(float(loss));counts.append(len(g))
  return float(np.average(vals,weights=counts))
 def savebest(epoch):torch.save(dict(updates=m.updates(),arm=arm,seed=seed,epoch=epoch,steps=steps),out/'best.pt')
 first=1;resume_batch=None;numerical_events=0
 if (out/'resume.pt').exists():
  ck=torch.load(out/'resume.pt',map_location='cpu');m.load_state_dict(ck['updates'],strict=False);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);first=ck['epoch'] if ck.get('cursor') is not None else ck['epoch']+1;best=ck['best'];best_epoch=ck['best_epoch'];steps=ck['steps'];history=ck['history'];exposures=ck['exposures'];resume_batch=ck if ck.get('cursor') is not None else None;numerical_events=ck.get('numerical_events',0)
  if 'torch_rng' in ck:torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
 else:
  best=validate();history=[dict(epoch=0,caption_holdout=best)];savebest(0)
 for epoch in range(first,5):
  m.train();rng=np.random.default_rng(seed*100+epoch);order=rng.permutation(split['train']);total=[];pair_count=0;direction_count=0
  cursor=0
  if resume_batch is not None and epoch==first:
   order=resume_batch['order'];rng.bit_generator.state=resume_batch['rng_state'];cursor=resume_batch['cursor'];total=resume_batch['total'];pair_count=resume_batch['pair_count'];direction_count=resume_batch['direction_count'];resume_batch=None
  for begin in range(cursor,len(order),16):
   cool();ids,triples=data.augment(order[begin:begin+16],rng);d=data.batch(ids);opt.zero_grad(set_to_none=True)
   for i in ids:exposures[str(i)]=exposures.get(str(i),0)+1
   active=d['direction']>=0
   updated=False
   for retry in range(10):
    use_amp=retry<9;opt.zero_grad(set_to_none=True);scale_before=scaler.get_scale()
    with torch.cuda.amp.autocast(enabled=use_amp):
     g,p,f,di,rev,w=m(d['x'] if use_amp else d['x'].float(),d['mask'],bool(active.any()));base=caption_loss(g,d['caption'],d['cid'],m.scale);gl,lo=semantic_losses(g,p,f,d['phrase'],d['pos'],d['types'],d['mask'],m.scale);pair=base*0;dire=base*0
     if triples:
      ix=torch.tensor(triples,device='cuda');state=m.latest_state;sp=(state[ix[:,0]]*state[ix[:,1]]).sum(1);sn=(state[ix[:,0]]*state[ix[:,2]]).sum(1);pair=F.softplus((sn-sp+.1)/.1).mean()*.1
     if active.any():dire=.5*(F.cross_entropy(di[active].float(),d['direction'][active])+F.cross_entropy(rev[active].float(),1-d['direction'][active]))
     loss=base+.1*gl+.1*lo+.2*pair+.1*dire
    scaled_loss=scaler.scale(loss);loss_finite=bool(torch.isfinite(loss));reason=None
    if loss_finite:
     scaled_loss.backward();scaler.unscale_(opt)
     try:grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
     except RuntimeError as exc:
      if 'non-finite' not in str(exc):raise
      reason='nonfinite_gradient_norm'
     if reason is None:
      scaler.step(opt);scaler.update();updated=True;break
    else:reason='nonfinite_forward_loss'
    numerical_events+=1
    event=dict(epoch=epoch,step=steps+1,cursor=begin,retry=retry,precision='amp16' if use_amp else 'float32',reason=reason,scale=scale_before,ids=ids,valid_frames=d['mask'].sum(1).tolist(),components=[float(v) for v in [base,gl,lo,pair,dire]])
    with (out/'numerical_events.jsonl').open('a') as log:log.write(json.dumps(event)+'\n')
    # No parameter update and no skipped data. Reset scaler bookkeeping before replay.
    scaler.update(new_scale=max(scale_before*.5,1.))
    if not use_amp:raise FloatingPointError('Nonfinite values persist in float32; see numerical_events.jsonl')
    del scaled_loss,loss,g,p,f,di,rev,w,base,gl,lo,pair,dire
    state=sp=sn=None;m.latest_state=None;opt.zero_grad(set_to_none=True)
   assert updated
   steps+=1;pair_count+=len(triples);direction_count+=int(active.sum());total.append([float(v) for v in [base,gl,lo,pair,dire]])
   if steps%200==0:
    torch.save(dict(updates=m.updates(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,cursor=begin+len(order[begin:begin+16]),order=order,rng_state=rng.bit_generator.state,total=total,pair_count=pair_count,direction_count=direction_count,best=best,best_epoch=best_epoch,steps=steps,history=history,exposures=exposures,numerical_events=numerical_events,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'resume.tmp');(out/'resume.tmp').replace(out/'resume.pt')
   if steps%20==0:atomic(out/'progress.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],epoch=epoch,steps=steps,total_steps=4*int(np.ceil(len(order)/16)),loss=float(loss),components=total[-1],elapsed_s=time.time()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,gate=float(.5*m.gate_logit.sigmoid()),numerical_events=numerical_events,amp_scale=scaler.get_scale()))
  held=validate();history.append(dict(epoch=epoch,caption_holdout=held,training_losses=np.mean(total,axis=0).tolist(),triplet_examples=pair_count,direction_examples=direction_count))
  if held<best:best=held;best_epoch=epoch;savebest(epoch)
  atomic(out/'history.json',history);torch.save(dict(updates=m.updates(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,steps=steps,history=history,exposures=exposures,numerical_events=numerical_events,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'resume.tmp');(out/'resume.tmp').replace(out/'resume.pt')
 for n,p in m.named_parameters():
  if n in frozen:torch.testing.assert_close(p.detach().cpu(),frozen[n],atol=0,rtol=0)
 assert (m.projection.detach().cpu()-initial).norm()>0
 atomic(out/'exposures.json',exposures);atomic(out/'complete.json',dict(passed=True,arm=arm,seed=seed,best_epoch=best_epoch,steps=steps,checkpoint_sha256=digest(out/'best.pt'),frozen_prefix_exact=True,elapsed_s=time.time()-start,gate=float(.5*m.gate_logit.sigmoid()),numerical_events=numerical_events,amp_scale=scaler.get_scale()));atomic(out/'progress.json',dict(complete=True,steps=steps));print('complete',arm,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args();main(a.arm,a.seed)
