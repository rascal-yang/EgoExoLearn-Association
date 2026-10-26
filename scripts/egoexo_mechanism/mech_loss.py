import torch
from torch.nn import functional as F
from model import auxiliary as original,phrase_ce

def auxiliary_variant(arm,g,p,f,text,labels,types,mask,scale,permutation):
    global_loss,_=original(g,None,None,text,labels,types,mask,scale,permutation)
    groups=sum(bool((types==t).any()) for t in (0,1));local=g.sum()*0
    for typ in (0,1):
        if arm=='N' and typ==0:continue
        if arm=='V' and typ==1:continue
        take=types==typ
        if not take.any():continue
        if arm!='M':
            _,part=original(g,p,f,text[take],labels[:,take],types[take],mask,scale,permutation)
        else:
            feature=f if typ==0 else p
            valid=mask if typ==0 else mask[:,:,None].expand(-1,-1,196).reshape(len(g),784)
            logits=torch.einsum('bkd,pd->bkp',feature.float(),text[take].float()).float()
            # Matched neutral slot of value zero and same valid+1 denominator.
            score=(logits*valid[:,:,None]).sum(1)/(valid.sum(1).float()+1)[:,None]
            part=phrase_ce(scale.float()*score,labels[:,take][permutation])
        local=local+part
    return global_loss,local/max(groups,1)
