import time,json
import numpy as np,torch
from torch.nn import functional as F
from common import EXP,read,atomic,cool
from model import VideoTail,caption_loss,auxiliary

def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(17)
    assert read(EXP/'text_complete.json')['passed']
    prefix=np.load(EXP/'smoke_gpu2/prefix.npy');records=read(EXP/'records.json');phrases=read(EXP/'phrases.json');ids=read(EXP/'plan.json')['smoke_indices'];text=torch.tensor(np.load(EXP/'text_features.npy'),device='cuda')
    teacher=text[torch.tensor([records[i]['caption_id'] for i in ids],device='cuda')]
    phrase=text[torch.tensor([p['text_id'] for p in phrases[:256]],device='cuda')];types=torch.tensor([0 if p['type']=='verb' else 1 for p in phrases[:256]],device='cuda')
    results={}
    for batch in (32,):
        cool();m=VideoTail();m.train();params=[p for p in m.parameters() if p.requires_grad];names=[n for n,p in m.named_parameters() if p.requires_grad]
        assert all(n=='projection' or n.startswith(('visual.blocks.10.','visual.blocks.11.','visual.norm.')) for n in names)
        frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
        opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01);scaler=torch.cuda.amp.GradScaler()
        xx=torch.tensor(np.tile(prefix,(batch//8,1,1)),device='cuda');tt=teacher.repeat(batch//8,1);cid=torch.arange(batch,device='cuda')%8;mask=torch.ones(batch,4,dtype=torch.bool,device='cuda')
        target=torch.zeros(batch,len(phrase),dtype=torch.bool,device='cuda');target[torch.arange(batch),torch.arange(batch)%len(phrase)]=True;perm=torch.arange(batch,device='cuda').roll(1)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
        for step in range(4):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                g,p,f=m(xx);base=caption_loss(g,tt,cid,m.scale);glob,local=auxiliary(g,p,f,phrase,target,types,mask,m.scale,torch.arange(batch,device='cuda'));loss=base+.1*glob+.1*local
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt)
            assert m.projection.grad is not None and torch.isfinite(m.projection.grad).all()
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.visual.blocks[10].parameters())
            assert all(p.grad is None for p in m.visual.blocks[0].parameters())
            torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update()
        torch.cuda.synchronize();elapsed=time.time()-start;mem=torch.cuda.max_memory_allocated()/2**20
        for n,p in m.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],atol=0,rtol=0)
        # Inspect local-only gradient and corruption isolation independently.
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            g,p,f=m(xx);ga,la=auxiliary(g,p,f,phrase,target,types,mask,m.scale,torch.arange(batch,device='cuda'));gd,ld=auxiliary(g,p,f,phrase,target,types,mask,m.scale,perm)
        torch.testing.assert_close(ga,gd,atol=0,rtol=0);assert abs(float(la-ld))>1e-7
        la.backward();assert m.projection.grad is not None and m.projection.grad.abs().sum()>0
        results[str(batch)]=dict(peak_mib=mem,seconds_per_step=elapsed/4,clips_per_second=4*batch/elapsed,trainable_parameters=sum(p.numel() for p in params))
        del m,opt,scaler,frozen,xx,g,p,f,loss,ga,la,gd,ld;torch.cuda.empty_cache()
    eligible=[int(k) for k,v in results.items() if v['peak_mib']<8000];assert eligible
    chosen=max(eligible,key=lambda k:results[str(k)]['clips_per_second'])
    atomic(EXP/'resource_preflight.json',dict(passed=True,measurements=results,batch=chosen,precision='AMP float16 training; float32 losses',prefix_frozen_exact=True,last_two_blocks_gradients_nonzero=True,local_only_gradients_nonzero=True,D_global_phrase_loss_identical=True,D_local_target_corruption_changes_loss=True,selection_basis='Throughput/memory only; no validation accuracy used'))
    print(results,'chosen',chosen,flush=True)

if __name__=='__main__':main()
