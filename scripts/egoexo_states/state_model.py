from state_common import *
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from model import VideoTail,caption_loss,phrase_ce

class StateModel(VideoTail):
 def __init__(self,arm,seed,m_checkpoint=None):
  super().__init__();self.arm=arm
  state=torch.load(m_checkpoint if m_checkpoint is not None else MBASE/f'heads/M/seed{seed}/best.pt',map_location='cpu');self.load_state_dict(state['updates'],strict=False)
  self.frame_in=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128))
  self.position=nn.Parameter(torch.randn(1,16,128)*.01)
  layer=nn.TransformerEncoderLayer(128,4,512,dropout=0.,batch_first=True,norm_first=True)
  self.temporal=nn.TransformerEncoder(layer,2);self.step=nn.Linear(256,1)
  nn.init.zeros_(self.step.weight);nn.init.zeros_(self.step.bias)
  self.width=nn.Parameter(torch.tensor(-1.1));self.gate_logit=nn.Parameter(torch.tensor(-2.1972246))
  self.state_out=nn.Sequential(nn.LayerNorm(640 if arm=='ordered_states' else 128),nn.Linear(640 if arm=='ordered_states' else 128,256))
  self.direction_head=nn.Linear(128,2)
  self.cuda()
 def encode_tail(self,x):
  if self.training and torch.is_grad_enabled():x=x.detach().requires_grad_(True)
  for block in self.visual.blocks[10:]:
   def forward(z,blk=block):return blk(z,self.visual.einops_from_space,self.visual.einops_to_space,self.visual.einops_from_time,self.visual.einops_to_time,time_n=196,space_f=16,use_checkpoint=False)
   x=checkpoint(forward,x) if self.training and torch.is_grad_enabled() else forward(x)
  x=self.visual.norm(x);g=x[:,0]@self.projection;p=(x[:,1:]@self.projection).reshape(-1,16,196,256)
  return g,p,p.mean(2)
 def summarize(self,frames,mask):
  h=self.temporal(self.frame_in(frames)+self.position,src_key_padding_mask=~mask)
  if self.arm=='ordered_states':
   delta=torch.cat([torch.zeros_like(h[:,:1]),h[:,1:]-h[:,:-1]],1);inc=(F.softplus(self.step(torch.cat([h,delta],-1)).float()).squeeze(-1)+.1)*mask
   cs=inc.cumsum(1);first=cs.masked_fill(~mask,float('inf')).min(1).values;span=(cs[:,-1]-first).clamp_min(1e-6);progress=(cs-first[:,None])/span[:,None]
   centers=torch.tensor([0.,.5,1.],device=h.device);sigma=.1+.4*self.width.sigmoid();logits=-(progress[:,None,:]-centers[None,:,None]).square()/(2*sigma.square());logits=logits.masked_fill(~mask[:,None,:],-1e4);weights=logits.softmax(-1)
   stages=torch.einsum('bst,btd->bsd',weights.to(h.dtype),h);d1=stages[:,1]-stages[:,0];d2=stages[:,2]-stages[:,1];z=torch.cat([stages.flatten(1),d1,d2],-1);direct=stages[:,2]-stages[:,0]
  else:
   direct=(h*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None];z=direct;weights=mask[:,None].float()/mask.sum(1).clamp_min(1)[:,None,None]
  return F.normalize(self.state_out(z).float(),dim=-1),self.direction_head(direct),weights
 def forward(self,x,mask,reverse=False):
  g,p,f=self.encode_tail(x);state,direction,weights=self.summarize(f,mask);gate=.5*self.gate_logit.sigmoid();embedding=F.normalize((1-gate)*F.normalize(g.float(),dim=-1)+gate*state,dim=-1)
  self.latest_state=state
  reversed_logits=self.summarize(f.flip(1),mask.flip(1))[1] if reverse else None
  return embedding,p,f,direction,reversed_logits,weights
 def updates(self):return {n:v.detach().cpu() for n,v in self.state_dict().items() if not n.startswith('visual.') or n.startswith(('visual.blocks.10.','visual.blocks.11.','visual.norm.'))}

def semantic_losses(g,p,f,text,pos,types,mask,scale):
 gl=g.sum()*0;lo=g.sum()*0;groups=0
 for typ in (0,1):
  take=types==typ
  if not take.any():continue
  groups+=1;t=text[take].float();labels=pos[:,take];gl=gl+phrase_ce(scale.float()*(F.normalize(g.float(),dim=-1)@t.T),labels)
  features=F.normalize(f.float(),dim=-1) if typ==0 else F.normalize(p.flatten(1,2).float(),dim=-1)
  valid=mask if typ==0 else mask[:,:,None].expand(-1,-1,196).reshape(len(g),3136)
  logits=torch.einsum('bkd,pd->bkp',features,t).float();score=(logits*valid[:,:,None]).sum(1)/(valid.sum(1).float()+1)[:,None]
  lo=lo+phrase_ce(scale.float()*score,labels)
 return gl/max(groups,1),lo/max(groups,1)
