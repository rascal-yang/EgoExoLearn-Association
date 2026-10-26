from state_common import *
import numpy as np,torch

class Data:
 def __init__(self,smoke=False):
  self.smoke=smoke;self.folders=[EXP/'smoke2'] if smoke else [EXP/'cache2',EXP/'cache3'];self.arrays=[np.load(p/'prefix.npy',mmap_mode='r') for p in self.folders];self.pts=[np.load(p/'pts.npy',mmap_mode='r') for p in self.folders];self.locations={int(i):(k,j) for k,p in enumerate(self.folders) for j,i in enumerate(np.load(p/'ids.npy'))};self.records=read(BASE/'records.json');self.phrases=read(BASE/'phrases.json');self.text=torch.tensor(np.load(BASE/'text_features.npy'),device='cuda');self.labels=read(EXP/'direction_labels.json');self.triplets=read(EXP/'triplets.json');self.phrase_labels=np.array([p['label']+(1000 if p['type']=='noun' else 0) for p in self.phrases])
 def augment(self,anchors,rng):
  ids=list(map(int,anchors));triples=[]
  for j,i in enumerate(anchors):
   if str(i) not in self.triplets:continue
   pool=self.triplets[str(i)];p=int(rng.choice(pool['positive']));n=int(rng.choice(pool['negative']));triples.append((j,len(ids),len(ids)+1));ids.extend([p,n])
  return ids,triples
 def batch(self,ids,text=True):
  positions=[self.locations[int(i)] for i in ids];x=torch.tensor(np.stack([self.arrays[k][j] for k,j in positions]),device='cuda');pts=np.stack([self.pts[k][j] for k,j in positions]);mask=np.ones((len(ids),16),bool)
  for j in range(1,16):mask[:,j]=(pts[:,j,None]!=pts[:,:j]).all(1)
  mask=torch.tensor(mask,device='cuda');result={'x':x,'mask':mask}
  if not text:return result
  records=[self.records[int(i)] for i in ids];cid=torch.tensor([r['caption_id'] for r in records],device='cuda');pp=sorted({v for r in records for v in r['phrases']});features=self.text[torch.tensor([self.phrases[v]['text_id'] for v in pp],device='cuda',dtype=torch.long)];positive=np.zeros((len(ids),len(pp)),bool)
  for j,r in enumerate(records):positive[j]=np.isin(self.phrase_labels[pp],self.phrase_labels[r['phrases']])
  direction=torch.tensor([self.labels.get(str(i),{}).get('direction',-1) for i in ids],device='cuda');direction[mask.sum(1)<4]=-1
  result.update(cid=cid,caption=self.text[cid],phrase=features,pos=torch.tensor(positive,device='cuda'),types=torch.tensor([0 if self.phrases[v]['type']=='verb' else 1 for v in pp],device='cuda'),direction=direction)
  return result
