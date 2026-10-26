import torch
from torch import nn
from torch.nn import functional as F
import backend as b

class VideoTail(nn.Module):
    def __init__(self):
        super().__init__();net=b.load_model();net.requires_grad_(False)
        self.visual=net.visual;self.projection=net.image_projection
        for block in self.visual.blocks[10:]:block.requires_grad_(True)
        self.visual.norm.requires_grad_(True);self.projection.requires_grad_(True)
        self.register_buffer('scale',net.logit_scale.exp().detach().clone())
        del net
    def forward(self,prefix,local=True):
        x=b.tail(self.visual,prefix)
        global_=x[:,0]@self.projection
        if not local:return global_,None,None
        patches=(x[:,1:]@self.projection).reshape(-1,4,196,256)
        frames=patches.mean(2)
        return global_,F.normalize(patches.flatten(1,2).float(),dim=-1),F.normalize(frames.float(),dim=-1)
    def updates(self):return {k:v.detach().cpu() for k,v in self.state_dict().items() if k=='projection' or k.startswith(('visual.blocks.10.','visual.blocks.11.','visual.norm.'))}

def caption_loss(video,text,ids,scale):
    logits=(scale.float()*F.normalize(video.float(),dim=-1)@text.float().T).float()
    same=ids[:,None]==ids[None,:];target=same.float()/same.sum(-1,keepdim=True)
    return -.5*((target*logits.log_softmax(-1)).sum(-1).mean()+(target.T*logits.log_softmax(0)).sum(0).mean())

def phrase_ce(logits,positive):
    active=positive.any(-1)
    if not active.any():return logits.sum()*0
    target=positive[active].float();target=target/target.sum(-1,keepdim=True)
    return -(target*logits[active].float().log_softmax(-1)).sum(-1).mean()

def auxiliary(global_,patches,frames,phrase_features,positive,types,frame_mask,scale,permutation):
    global_loss=global_.sum()*0;local_loss=global_.sum()*0;groups=0
    g=F.normalize(global_.float(),dim=-1)
    for typ in (0,1):
        take=types==typ
        if not take.any():continue
        text=phrase_features[take].float();labels=positive[:,take];groups+=1
        global_loss=global_loss+phrase_ce(scale.float()*(g@text.T),labels)
        if patches is not None:
            feature=frames if typ==0 else patches
            mask=frame_mask if typ==0 else frame_mask[:,:,None].expand(-1,-1,196).reshape(len(g),784)
            logits=torch.einsum('bkd,pd->bkp',feature.float(),text).float()/.07
            logits=logits.masked_fill(~mask[:,:,None],-1e4)
            # A fixed neutral unmatched slot avoids forcing every phrase onto a patch.
            logits=torch.cat([logits,torch.zeros_like(logits[:,:1])],dim=1)
            score=.07*(logits.logsumexp(1)-(mask.sum(-1).float()+1).log()[:,None])
            local_loss=local_loss+phrase_ce(scale.float()*score,labels[permutation])
    return global_loss/max(groups,1),local_loss/max(groups,1)
