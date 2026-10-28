import argparse
import math
import signal
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from rel_common import *
from rel_model import RelationModel,objective,caption_loss
from rel_data import Data

def frozen_signature(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main(arm,seed):
    guard();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.90)
    protocol=verify_protocol();torch.manual_seed(seed)
    outdir=EXP/f'fits/{arm}/seed{seed}';outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'complete.json').exists():return
    data=Data(allow_hdd=True);splits=np.load(BASE/'splits.npz');train=splits['train'];hold=np.load(EXP/'selection_holdout.npy')
    batch=protocol['batch'];epochs=protocol['epochs'];total_steps=epochs*math.ceil(len(train)/batch)
    model=RelationModel(arm,seed)
    trunk=[p for n,p in model.named_parameters() if p.requires_grad and (n.startswith('visual.') or n=='projection')]
    head=[p for n,p in model.named_parameters() if p.requires_grad and not(n.startswith('visual.') or n=='projection')]
    params=trunk+head;groups=[dict(params=trunk,lr=1e-5)]
    if head:groups.append(dict(params=head,lr=1e-4))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01);scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    initial_frozen=frozen_signature(model);start=time.time();steps=0;epoch=1;cursor=0;best=float('inf');history=[];numerical=0
    exposures=np.zeros(len(data.records),np.int32);elapsed_offset=0.;stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def updates():return model.updates()
    def save_resume(epoch,cursor):
        path=outdir/'resume.tmp'
        torch.save(dict(updates=updates(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            epoch=epoch,cursor=cursor,steps=steps,best=best,history=history,exposures=exposures,numerical=numerical,
            elapsed_s=elapsed_offset+time.time()-start,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            protocol_sha256=digest(EXP/'protocol.json')),path)
        path.replace(outdir/'resume.pt')
    def validate():
        model.eval();values=[];counts=[]
        with torch.inference_mode():
            for begin in range(0,len(hold),batch):
                guard();d=data.batch(hold[begin:begin+batch])
                with torch.cuda.amp.autocast():z=model(d['x'],d['mask']);loss=caption_loss(z['g'],d['caption'],d['cid'],model.scale)
                assert torch.isfinite(loss);values.append(float(loss));counts.append(len(d['ids']))
        return float(np.average(values,weights=counts))
    if (outdir/'resume.pt').exists():
        ck=torch.load(outdir/'resume.pt',map_location='cpu')
        assert ck['protocol_sha256'] in [digest(EXP/'protocol.json')]+protocol.get('compatible_resume_protocols',[])
        model.load_state_dict(ck['updates'],strict=False);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        epoch,cursor,steps,best,history,exposures,numerical=[ck[k] for k in ['epoch','cursor','steps','best','history','exposures','numerical']]
        elapsed_offset=ck['elapsed_s'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        del ck
    else:
        baseline=validate();best=baseline;history=[dict(epoch=0,caption_holdout=baseline)]
        torch.save(dict(updates=updates(),arm=arm,seed=seed,epoch=0,steps=0),outdir/'best.pt')
    atomic(outdir/'manifest.json',dict(arm=arm,seed=seed,batch=batch,epochs=epochs,train_clips=len(train),holdout_clips=len(hold),
        initial_M_sha256=digest(MBASE/f'heads/M/seed{seed}/best.pt'),protocol_sha256=digest(EXP/'protocol.json'),
        trainable_parameters=sum(p.numel() for p in params),frozen_sha256=initial_frozen,
        head_learning_rate=1e-4,trunk_learning_rate=1e-5,relation_weight=.1))
    torch.cuda.reset_peak_memory_stats();step_times=[]
    try:
        while epoch<=epochs:
            order=np.random.default_rng(seed*100+epoch).permutation(train);model.train();totals=[]
            # One CPU lookahead; GPU transfer remains on the training thread.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(data.cpu_batch,order[cursor:cursor+batch]) if cursor<len(order) else None
                while cursor<len(order):
                    if stop[0]:save_resume(epoch,cursor);raise InterruptedError('Owned task stopped; checkpoint saved')
                    guard();tick=time.perf_counter();prepared=future.result();end=cursor+len(prepared[0])
                    future=pool.submit(data.cpu_batch,order[end:end+batch]) if end<len(order) else None
                    d=data.batch(prepared[0],prepared=prepared);success=False
                    for retry in range(10):
                        amp=retry<9;optimizer.zero_grad(set_to_none=True);scale=scaler.get_scale()
                        with torch.cuda.amp.autocast(enabled=amp):
                            result=model(d['x'] if amp else d['x'].float(),d['mask']);loss,parts=objective(model,result,d)
                        reason=None
                        if torch.isfinite(loss):
                            scaler.scale(loss).backward();scaler.unscale_(optimizer)
                            try:torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                            except RuntimeError as exc:
                                if 'non-finite' not in str(exc):raise
                                reason='nonfinite_gradient'
                            if reason is None:scaler.step(optimizer);scaler.update();success=True;break
                        else:reason='nonfinite_loss'
                        numerical+=1
                        with (outdir/'numerical.jsonl').open('a') as f:f.write(json.dumps(dict(step=steps,retry=retry,reason=reason,scale=scale))+'\n')
                        scaler.update(new_scale=max(1.,scale*.5))
                        if not amp:raise FloatingPointError('Float32 retry failed')
                        del result,loss,parts
                    assert success
                    exposures[d['ids']]+=1;steps+=1;cursor=end
                    values={k:float(v) for k,v in parts.items()};totals.append(values)
                    torch.cuda.synchronize();step_times.append(time.perf_counter()-tick)
                    if steps%10==0 or steps==1:
                        atomic(outdir/'progress.json',dict(phase='training',arm=arm,seed=seed,gpu=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getpid(),
                            epoch=epoch,steps=steps,total_steps=total_steps,components=values,elapsed_s=elapsed_offset+time.time()-start,
                            median_recent_step_s=float(np.median(step_times[-50:])),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,numerical_retries=numerical,
                            ssd_shards=sum(str(a.filename).startswith(str(SSD)) for a in data.arrays)))
                    if steps%100==0:save_resume(epoch,cursor)
                    del d,result,loss,parts
            held=validate();history.append(dict(epoch=epoch,caption_holdout=held,
                training_losses={k:float(np.mean([v[k] for v in totals])) for k in totals[0]} if totals else {}))
            if held<best:
                best=held;torch.save(dict(updates=updates(),arm=arm,seed=seed,epoch=epoch,steps=steps),outdir/'best.pt')
            atomic(outdir/'history.json',history);epoch+=1;cursor=0;save_resume(epoch,cursor)
        assert frozen_signature(model)==initial_frozen
        assert (exposures[train]==epochs).all() and exposures.sum()==len(train)*epochs
        np.save(outdir/'exposures.npy',exposures)
        atomic(outdir/'complete.json',dict(passed=True,arm=arm,seed=seed,steps=steps,best_caption_holdout=best,
            checkpoint_sha256=digest(outdir/'best.pt'),exposure_sha256=digest(outdir/'exposures.npy'),
            frozen_prefix_exact=True,elapsed_s=elapsed_offset+time.time()-start,numerical_retries=numerical,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        atomic(outdir/'progress.json',dict(phase='complete',steps=steps,total_steps=total_steps))
    except BaseException as exc:
        atomic(outdir/'failed.json',dict(reason=repr(exc),epoch=epoch,cursor=cursor,steps=steps,time=time.time()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=[17,29,43],default=17)
    a=p.parse_args();main(a.arm,a.seed)
