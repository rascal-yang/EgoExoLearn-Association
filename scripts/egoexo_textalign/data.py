import numpy as np
import torch
from common import EXP,read

class Prefixes:
    def __init__(self):
        self.arrays=[np.load(EXP/f'gpu{g}/prefix.npy',mmap_mode='r') for g in (2,3)]
        self.pts=[np.load(EXP/f'gpu{g}/pts.npy',mmap_mode='r') for g in (2,3)]
        self.records=read(EXP/'records.json');self.phrases=read(EXP/'phrases.json')
        self.text=torch.tensor(np.load(EXP/'text_features.npy'),device='cuda')
        self.phrase_labels=np.array([p['label']+(1000 if p['type']=='noun' else 0) for p in self.phrases])
    def batch(self,ids,text=True):
        ids=np.asarray(ids);pixels=np.stack([self.arrays[int(i)%2][int(i)//2] for i in ids]);pts=np.stack([self.pts[int(i)%2][int(i)//2] for i in ids]);mask=np.ones((len(ids),4),bool)
        for j in range(1,4):mask[:,j]=np.all(pts[:,j,None]!=pts[:,:j],axis=1)
        prefix=torch.from_numpy(pixels).cuda(non_blocking=True);mask=torch.tensor(mask,device='cuda')
        if not text:return prefix,mask
        records=[self.records[int(i)] for i in ids];captions=torch.tensor([r['caption_id'] for r in records],device='cuda',dtype=torch.long)
        pp=sorted({p for r in records for p in r['phrases']})
        features=self.text[torch.tensor([self.phrases[p]['text_id'] for p in pp],device='cuda',dtype=torch.long)]
        positive=np.zeros((len(ids),len(pp)),bool)
        for j,r in enumerate(records):positive[j]=np.isin(self.phrase_labels[pp],self.phrase_labels[r['phrases']])
        types=torch.tensor([0 if self.phrases[p]['type']=='verb' else 1 for p in pp],device='cuda')
        return prefix,mask,self.text[captions],captions,features,torch.tensor(positive,device='cuda'),types,records
