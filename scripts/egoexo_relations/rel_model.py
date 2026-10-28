import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from rel_common import MBASE
from model import VideoTail,caption_loss,phrase_ce


def binding_scores(role_tokens,role_text,valid_roles,mode,tau=.07):
    """C: same slot must explain all roles; D: roles choose slots independently.

    Both have identical role embeddings, parameters, text, normalization and
    one neutral zero-valued slot. No candidate caption is used at inference.
    """
    dots=torch.einsum('bkhd,rhd->bkrh',role_tokens.float(),role_text.float())
    weight=valid_roles.float();denom=weight.sum(-1).clamp_min(1)
    if mode=='unbound':
        dots=torch.cat([dots,torch.zeros_like(dots[:,:1])],1)
        perrole=tau*(torch.logsumexp(dots/tau,1)-torch.log(dots.new_tensor(dots.shape[1])))
        return (perrole*weight[None]).sum(-1)/denom[None]
    joint=(dots*weight[None,None]).sum(-1)/denom[None,None]
    joint=torch.cat([joint,torch.zeros_like(joint[:,:1])],1)
    return tau*(torch.logsumexp(joint/tau,1)-torch.log(joint.new_tensor(joint.shape[1])))


class RelationModel(VideoTail):
    def __init__(self,arm,seed):
        super().__init__();self.arm=arm
        self.load_state_dict(torch.load(MBASE/f'heads/M/seed{seed}/best.pt',map_location='cpu')['updates'],strict=False)
        if arm!='A_continue_M':
            self.queries=nn.Parameter(torch.randn(1,8,256)*.02)
            self.input_norm=nn.LayerNorm(256)
            self.time_position=nn.Parameter(torch.zeros(1,16,1,256))
            self.cross=nn.MultiheadAttention(256,4,dropout=0.,batch_first=True)
            self.slot_norm=nn.LayerNorm(256)
            self.slot_ff=nn.Sequential(nn.Linear(256,512),nn.GELU(),nn.Linear(512,256))
            self.role_heads=nn.ModuleList([nn.Linear(256,256,bias=False) for _ in range(3)])
            for head in self.role_heads:nn.init.eye_(head.weight)
            self.readout=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256))
            self.gate_logit=nn.Parameter(torch.tensor(-2.1972246))
        self.cuda()

    def forward(self,x,mask,return_attention=False):
        if self.training and torch.is_grad_enabled():x=x.detach().requires_grad_(True)
        for block in self.visual.blocks[10:]:
            def run(z,b=block):
                return b(z,'b (f n) d','(b f) n d','b (f n) d','(b n) f d',time_n=196,space_f=16,use_checkpoint=False)
            x=checkpoint(run,x) if self.training and torch.is_grad_enabled() else run(x)
        x=self.visual.norm(x)
        g=x[:,0]@self.projection
        p=(x[:,1:]@self.projection).reshape(-1,16,196,256)
        frames=p.mean(2)
        if self.arm=='A_continue_M':return dict(g=F.normalize(g.float(),dim=-1),patches=p,frames=frames,roles=None,attention=None)
        keys=self.input_norm(p+self.time_position).flatten(1,2)
        queries=self.queries.expand(len(x),-1,-1)
        blocked=(~mask[:,:,None]).expand(-1,-1,196).flatten(1)
        slots,attention=self.cross(queries,keys,keys,key_padding_mask=blocked,need_weights=return_attention)
        slots=slots+queries;slots=slots+self.slot_ff(self.slot_norm(slots))
        local=F.normalize(self.readout(slots.mean(1)).float(),dim=-1)
        gate=.5*self.gate_logit.sigmoid()
        embedding=F.normalize((1-gate)*F.normalize(g.float(),dim=-1)+gate*local,dim=-1)
        roles=torch.stack([F.normalize(head(slots).float(),dim=-1) for head in self.role_heads],2)
        return dict(g=embedding,patches=p,frames=frames,roles=roles,attention=attention,slots=slots)

    def updates(self):
        return {n:v.detach().cpu() for n,v in self.state_dict().items() if not n.startswith('visual.') or n.startswith(('visual.blocks.10.','visual.blocks.11.','visual.norm.'))}


def m_losses(out,data,scale):
    g,p,frames=out['g'],out['patches'],out['frames']
    base=caption_loss(g,data['caption'],data['cid'],scale)
    gl=g.sum()*0;local=g.sum()*0;groups=0
    for kind in (0,1):
        take=data['types']==kind
        if not take.any():continue
        groups+=1;text=data['phrase'][take].float();labels=data['pos'][:,take]
        gl+=phrase_ce(scale.float()*(F.normalize(g.float(),dim=-1)@text.T),labels)
        features=F.normalize(frames.float(),dim=-1) if kind==0 else F.normalize(p.flatten(1,2).float(),dim=-1)
        valid=data['mask'] if kind==0 else data['mask'][:,:,None].expand(-1,-1,196).flatten(1)
        logits=torch.einsum('bkd,pd->bkp',features,text)
        average=(logits*valid[:,:,None]).sum(1)/(valid.sum(1).float()+1)[:,None]
        local+=phrase_ce(scale.float()*average,labels)
    return base,gl/max(groups,1),local/max(groups,1)


def objective(model,out,data):
    base,gl,local=m_losses(out,data,model.scale)
    relation=base*0
    if model.arm in ('C_bound','D_unbound','E_scrambled') and data['relation_text'].shape[0]:
        scores=binding_scores(out['roles'],data['relation_text'],data['relation_valid'],
                              'unbound' if model.arm=='D_unbound' else 'bound')
        target=data['scrambled_pos'] if model.arm=='E_scrambled' else data['relation_pos']
        allowed=data['scrambled_allowed'] if model.arm=='E_scrambled' else data['relation_allowed']
        logits=(model.scale.float()*scores).masked_fill(~allowed,-1e4)
        relation=phrase_ce(logits,target)
    loss=base+.1*gl+.1*local+.1*relation
    return loss,dict(caption=base,global_phrase=gl,local_phrase=local,relation=relation)
