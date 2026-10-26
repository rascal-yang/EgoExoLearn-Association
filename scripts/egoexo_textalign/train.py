import argparse,os,time
import numpy as np,torch
from common import EXP,ARMS,SEEDS,read,atomic,digest,cool
from model import VideoTail,caption_loss,auxiliary
from data import Prefixes

def main(arm,seed):
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2','3');torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.44);torch.manual_seed(seed)
    protocol=read(EXP/'protocol.json');batch=protocol['batch'];assert read(EXP/'resource_preflight.json')['passed']
    for p,h in protocol['sources'].items():assert digest(p)==h,p
    out=EXP/'heads'/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True);assert not (out/'best.pt').exists()
    arrays=Prefixes();split=np.load(EXP/'splits.npz');train=split['train'];hold=split['holdout'];rng=np.random.default_rng(seed)
    m=VideoTail();params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01);scaler=torch.cuda.amp.GradScaler()
    initial={k:v.clone() for k,v in m.updates().items()};frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    atomic(out/'manifest.json',dict(arm=arm,seed=seed,protocol_sha256=digest(EXP/'protocol.json'),batch=batch,epochs=2,objective='caption' if arm=='A' else 'caption+.1global_phrase'+('+.1local_phrase' if arm in ('C','D') else ''),only_D_local_targets_corrupted=arm=='D'))
    start=time.time();steps=0;shuffled=0
    def validate():
        m.eval();values=[];counts=[]
        with torch.no_grad():
            for b in range(0,len(hold),batch):
                cool();args=arrays.batch(hold[b:b+batch]);x,mask,text,cid=args[:4]
                with torch.cuda.amp.autocast():g,_,_=m(x,False);loss=caption_loss(g,text,cid,m.scale)
                values.append(float(loss));counts.append(len(x))
        return float(np.average(values,weights=counts))
    def save(epoch):torch.save(dict(updates=m.updates(),epoch=epoch,steps=steps,arm=arm,seed=seed,manifest_sha256=digest(out/'manifest.json')),out/'best.pt')
    best=validate();best_epoch=0;history=[dict(epoch=0,held_caption_loss=best)];save(0)
    for epoch in (1,2):
        m.train();order=rng.permutation(train);totals=[]
        for b in range(0,len(order),batch):
            cool();args=arrays.batch(order[b:b+batch]);x,mask,text,cid,phrase,positive,types,records=args;opt.zero_grad(set_to_none=True)
            permutation=np.arange(len(x))
            if arm=='D':
                for view in ('ego','exo'):
                    take=np.array([i for i,r in enumerate(records) if r['view']==view],dtype=np.int64);permutation[take]=np.roll(take,1)
                shuffled+=int((permutation!=np.arange(len(x))).sum())
            permutation=torch.tensor(permutation,device='cuda',dtype=torch.long)
            with torch.cuda.amp.autocast():
                g,p,f=m(x,arm in ('C','D'));base=caption_loss(g,text,cid,m.scale);loss=base
                gl=lo=base*0
                if arm!='A':gl,lo=auxiliary(g,p,f,phrase,positive,types,mask,m.scale,permutation);loss=loss+.1*gl
                if arm in ('C','D'):loss=loss+.1*lo
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad);scaler.step(opt);scaler.update();steps+=1
            totals.append([float(base),float(gl),float(lo)])
            if steps%20==0:atomic(out/'progress.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],arm=arm,seed=seed,epoch=epoch,steps=steps,total_steps=2*int(np.ceil(len(train)/batch)),caption_loss=float(base),global_phrase_loss=float(gl),local_phrase_loss=float(lo),elapsed_s=time.time()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20))
        held=validate();history.append(dict(epoch=epoch,held_caption_loss=held,training_losses=np.mean(totals,axis=0).tolist()))
        if held<best:best=held;best_epoch=epoch;save(epoch)
        atomic(out/'history.json',history)
    delta={k:float((v-initial[k]).norm()) for k,v in m.updates().items()};assert delta['projection']>0
    for n,p in m.named_parameters():
        if n in frozen:torch.testing.assert_close(p,frozen[n],atol=0,rtol=0)
    atomic(out/'complete.json',dict(passed=True,arm=arm,seed=seed,steps=steps,best_epoch=best_epoch,best_held_loss=best,elapsed_s=time.time()-start,checkpoint_sha256=digest(out/'best.pt'),frozen_prefix_exact=True,update_norms=delta,shuffled_local_rows=shuffled,peak_mib=torch.cuda.max_memory_allocated()/2**20))
    atomic(out/'progress.json',dict(complete=True,arm=arm,seed=seed,steps=steps));print('complete',arm,seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args();main(a.arm,a.seed)
