import numpy as np
import torch
from rel_common import *


class Data:
    def __init__(self,allow_hdd=False):
        self.arrays=[];self.pts=[];self.locations={};self._read_fds={}
        ready=read(EXP/'ssd_ready.json') if (EXP/'ssd_ready.json').exists() else None
        if not ready and not allow_hdd:raise RuntimeError('Verified SSD copy required for full training')
        for k,g in enumerate((2,3)):
            folder=CACHE/f'cache{g}'
            path=SSD/f'cache{g}_prefix.npy' if ready else folder/'prefix.npy'
            self.arrays.append(np.load(path,mmap_mode='r'));self.pts.append(np.load(folder/'pts.npy',mmap_mode='r'))
            self.locations.update({int(i):(k,j) for j,i in enumerate(np.load(folder/'ids.npy'))})
        # Two persistent host buffers support exactly one CPU lookahead while
        # the main thread performs a blocking CUDA copy of the previous batch.
        # Reusing locked pages avoids repeated large allocations/page faults.
        self._cpu_buffers=[torch.empty((48,)+self.arrays[0].shape[1:],dtype=torch.float16,
                                      pin_memory=True).numpy() for _ in range(2)]
        self._buffer_cursor=0
        self.records=read(BASE/'records.json');self.phrases=read(BASE/'phrases.json')
        self.text=torch.tensor(np.load(BASE/'text_features.npy'),device='cuda')
        self.rel=read(EXP/'relations.json')
        self.relation_text=torch.tensor(np.load(EXP/'relation_text.npy'),device='cuda')
        self.phrase_labels=np.array([p['label']+(1000 if p['type']=='noun' else 0) for p in self.phrases])
        self.role_ids=np.asarray(self.rel['role_text_ids'],np.int64)

    def cpu_batch(self,ids):
        # Copy worker publishes each file by atomic rename only AFTER its two
        # hashes pass. Switch read-only mappings without changing any examples.
        for k,g in enumerate((2,3)):
            fast=SSD/f'cache{g}_prefix.npy'
            if fast.exists() and str(self.arrays[k].filename)!=str(fast):
                new=np.load(fast,mmap_mode='r')
                assert new.shape==self.arrays[k].shape and new.dtype==self.arrays[k].dtype
                self.arrays[k]=new
        ids=list(map(int,ids));positions=[self.locations[i] for i in ids]
        # Avoid faulting the entire activation mmap into this CUDA process.
        # Read the exact same FP16 rows into one contiguous CPU batch instead.
        if len(ids)>48:raise ValueError('CPU staging capacity is48 clips')
        x=self._cpu_buffers[self._buffer_cursor][:len(ids)]
        self._buffer_cursor=(self._buffer_cursor+1)%2
        for b,(k,j) in enumerate(positions):
            array=self.arrays[k];path=str(array.filename)
            if path not in self._read_fds:self._read_fds[path]=os.open(path,os.O_RDONLY)
            view=memoryview(x[b]).cast('B');offset=array.offset+j*array.strides[0];done=0
            while done<len(view):
                received=os.preadv(self._read_fds[path],[view[done:]],offset+done)
                if received<=0:raise IOError(f'Unexpected end of activation cache: {path}')
                done+=received
        pts=np.stack([self.pts[k][j] for k,j in positions]);mask=np.ones((len(ids),16),bool)
        for j in range(1,16):mask[:,j]=(pts[:,j,None]!=pts[:,:j]).all(1)
        return ids,x,mask

    def batch(self,ids,prepared=None,training=True):
        ids,x,mask=self.cpu_batch(ids) if prepared is None else prepared
        result=dict(x=torch.from_numpy(x).cuda(),mask=torch.tensor(mask,device='cuda'),ids=ids)
        # Official validation indices are intentionally not valid training records.
        if not training:return result
        records=[self.records[i] for i in ids]
        cid=torch.tensor([r['caption_id'] for r in records],device='cuda')
        phrases=sorted({p for r in records for p in r['phrases']})
        feat=self.text[torch.tensor([self.phrases[p]['text_id'] for p in phrases],device='cuda',dtype=torch.long)]
        positive=np.stack([np.isin(self.phrase_labels[phrases],self.phrase_labels[r['phrases']]) for r in records])
        result.update(cid=cid,caption=self.text[cid],phrase=feat,pos=torch.tensor(positive,device='cuda'),
            types=torch.tensor([0 if self.phrases[p]['type']=='verb' else 1 for p in phrases],device='cuda'))
        normal=[self.rel['labels'].get(str(i),[]) for i in ids]
        corrupt=[self.rel['scrambled'].get(str(i),p) for i,p in zip(ids,normal)]
        candidates=sorted({r for rows in (normal,corrupt) for row in rows for r in row})
        roles=self.role_ids[candidates] if candidates else np.empty((0,3),np.int64)
        result['relation_text']=self.relation_text[torch.tensor(roles.clip(min=0),device='cuda')]
        result['relation_valid']=torch.tensor(roles>=0,device='cuda')
        for name,labels in [('relation',normal),('scrambled',corrupt)]:
            target=np.array([[r in row for r in candidates] for row in labels],bool).reshape(len(ids),len(candidates))
            allowed=np.ones_like(target)
            for b,row in enumerate(labels):
                for c,relid in enumerate(candidates):
                    action,obj,ref=self.rel['tuples'][relid]
                    # Missing target annotation is unknown, not a negative target.
                    ambiguous=any((action,obj)==tuple(self.rel['tuples'][p][:2]) and (not ref or not self.rel['tuples'][p][2]) for p in row)
                    if ambiguous and not target[b,c]:allowed[b,c]=False
            result[name+'_pos']=torch.tensor(target,device='cuda')
            result[name+'_allowed']=torch.tensor(allowed,device='cuda')
        return result
