"""Shared static spatial tokens plus dense dynamic spatial tubes.

Packed layout: CLS, T*K dynamic tokens, N-K shared static tokens.
All weights are the original TimeSformer weights. No learned gate or scorer.
Static spatial-query updates use attention to the time-averaged key measure;
this is an approximation to averaging dense attention outputs and is tested.
"""
import math
import torch
from torch.nn import functional as F

def attention(q,k,v,bias=None):
 score=q@k.transpose(-1,-2)
 if bias is not None:score=score+bias
 return score.softmax(-1)@v

def packed_attention(module,x,T,K,N,temporal,static_exact=False):
 B,L,D=x.shape;H=module.num_heads;d=D//H;S=N-K
 qkv=module.qkv(x).reshape(B,L,3,H,d).permute(2,0,3,1,4)
 q,k,v=qkv.unbind(0);q=q*module.scale
 n_dynamic=T*K;boundary=1+n_dynamic
 # A shared static token represents T original keys for global CLS attention.
 mass=x.new_zeros(L)
 if S:mass[boundary:]=math.log(T)
 co=attention(q[:,:,:1],k,v,mass[None,None,None,:])
 dq=q[:,:,1:boundary].reshape(B,H,T,K,d)
 dk=k[:,:,1:boundary].reshape(B,H,T,K,d)
 dv=v[:,:,1:boundary].reshape(B,H,T,K,d)
 sq,sk,sv=q[:,:,boundary:],k[:,:,boundary:],v[:,:,boundary:]
 if temporal:
  if K:
   keys=torch.cat([k[:,:,:1].unsqueeze(2).expand(-1,-1,K,-1,-1),dk.transpose(2,3)],3)
   vals=torch.cat([v[:,:,:1].unsqueeze(2).expand(-1,-1,K,-1,-1),dv.transpose(2,3)],3)
   do=attention(dq.transpose(2,3),keys,vals).transpose(2,3).reshape(B,H,n_dynamic,d)
  else:do=q[:,:,1:1]
  if S:
   # Each static temporal group has CLS and T copies of its own token.
   logits=torch.stack([(sq*k[:,:,:1]).sum(-1),(sq*sk).sum(-1)+math.log(T)],-1)
   weights=logits.softmax(-1)
   so=weights[...,:1]*v[:,:,:1]+weights[...,1:]*sv
  else:so=sq
 else:
  if K:
   keys=torch.cat([k[:,:,:1].unsqueeze(2).expand(-1,-1,T,-1,-1),dk,sk.unsqueeze(2).expand(-1,-1,T,-1,-1)],3)
   vals=torch.cat([v[:,:,:1].unsqueeze(2).expand(-1,-1,T,-1,-1),dv,sv.unsqueeze(2).expand(-1,-1,T,-1,-1)],3)
   do=attention(dq,keys,vals).reshape(B,H,n_dynamic,d)
  else:do=q[:,:,1:1]
  if S and K and static_exact:
   so=attention(sq.unsqueeze(2).expand(-1,-1,T,-1,-1),keys,vals).mean(2)
  elif S:
   # Average frame key measure: each dynamic frame contributes weight1/T.
   spatial_mass=x.new_zeros(L);spatial_mass[1:boundary]=-math.log(T)
   so=attention(sq,k,v,spatial_mass[None,None,None,:])
  else:so=sq
 out=torch.cat([co,do,so],2).transpose(1,2).reshape(B,L,D)
 return module.proj_drop(module.proj(out))

def packed_block(block,x,T,K,N=196,static_exact=False):
 to=packed_attention(block.timeattn,block.norm3(x),T,K,N,True)
 if hasattr(block,'alpha_timeattn'):to=block.alpha_timeattn.tanh()*to
 tr=x+to
 so=packed_attention(block.attn,block.norm1(tr),T,K,N,False,static_exact)
 assert block.attention_style=='frozen-in-time'
 sr=x+block.drop_path(so)
 return sr+block.drop_path(block.mlp(block.norm2(sr)))

def dense_block(block,x,T=16,N=196):
 return block(x,'b (f n) d','(b f) n d','b (f n) d','(b n) f d',time_n=N,space_f=T,use_checkpoint=False)

def stem(visual,video):
 # video is B,C,T,H,W, exactly as the original visual.forward input.
 video=video.permute(0,2,1,3,4).contiguous();B,T=video.shape[:2]
 p=visual.patch_embed(video).flatten(2).transpose(1,2).reshape(B,T,196,768)
 cls=visual.cls_token.expand(B,-1,-1)
 x=torch.cat([cls,p.flatten(1,2)],1)
 pos=visual.pos_embed[:,1:].repeat(1,T,1)+visual.inflate_temporal_embeds(T).repeat_interleave(196,1)
 x=x+torch.cat([visual.pos_embed[:,:1],pos],1)
 if visual.ln_pre is not None:x=visual.ln_pre(x)
 return visual.pos_drop(x),p

def haar_score(content):
 # Score content before positional embeddings. Multiscale maxima retain brief
 # events and accumulated slow changes rather than averaging them away.
 z=F.normalize(content.float(),dim=-1);energies=[]
 while z.shape[1]>1:
  a,b=z[:,0::2],z[:,1::2]
  energies.append((a-b).square().mean(-1).amax(1))
  z=(a+b)*.5
 return torch.stack(energies).amax(0)

def routing_scores(content):
 B,T,N,D=content.shape;raw=haar_score(content)
 z=F.normalize(content.float(),dim=-1).reshape(B,T,14,14,D)
 previous=z[:,:-1];current=z[:,1:];similarities=[];offsets=[]
 for dy in (-1,0,1):
  for dx in (-1,0,1):
   rolled=torch.roll(previous,(dy,dx),(2,3));sim=(current*rolled).sum(-1)
   if dy<0:sim[:,:,dy:]=-2
   if dy>0:sim[:,:,:dy]=-2
   if dx<0:sim[:,:,:,dx:]=-2
   if dx>0:sim[:,:,:,:dx]=-2
   similarities.append(sim.flatten(2));offsets.append((dy,dx))
 sim=torch.stack(similarities,-1);best,which=sim.max(-1)
 histogram=F.one_hot(which,9).sum(2);global_offset=histogram.argmax(-1)
 coordinates=content.new_tensor(offsets)
 displacement=(coordinates[which]-coordinates[global_offset][:,:,None]).square().sum(-1)/8
 novelty=(1-best).clamp(0,1)
 local_motion=(novelty+displacement*best.clamp(0,1)).clamp(0,1).amax(1)
 # Retain a floor for low-motion changes; no hard background classification.
 compensated=raw*(.25+.75*local_motion)
 diagnostics=dict(mean_novelty=novelty.mean((1,2)),nonzero_global_offset=(global_offset!=4).float().mean(1),mean_haar=raw.mean(1))
 return raw,compensated,diagnostics

def pack(x,score,K,T=16,N=196):
 B,L,D=x.shape
 selected=score.topk(K,dim=1).indices.sort(1).values
 mask=torch.ones(B,N,device=x.device,dtype=torch.bool);mask.scatter_(1,selected,False)
 static=torch.arange(N,device=x.device).expand(B,-1)[mask].reshape(B,N-K)
 p=x[:,1:].reshape(B,T,N,D)
 dynamic=p.gather(2,selected[:,None,:,None].expand(-1,T,-1,D)).flatten(1,2)
 shared=p.mean(1).gather(1,static[:,:,None].expand(-1,-1,D))
 return torch.cat([x[:,:1],dynamic,shared],1),selected,static

def full_forward(model,video,K=196,mode='compensated',merge_layer=2):
 x,content=stem(model.visual,video)
 if K==196:
  for block in model.visual.blocks:x=dense_block(block,x,video.shape[2])
 else:
  raw,comp,_=routing_scores(content);score=comp if mode in ('compensated','exact') else raw
  for block in model.visual.blocks[:merge_layer]:x=dense_block(block,x,video.shape[2])
  x,_,_=pack(x,score,K,T=video.shape[2])
  for block in model.visual.blocks[merge_layer:]:x=packed_block(block,x,video.shape[2],K,static_exact=mode=='exact')
 return model.visual.norm(x)[:,0]@model.projection
