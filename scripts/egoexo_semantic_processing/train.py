"""Fixed-budget training with exact exposure accounting and safe stop/resume."""
import argparse
import math
import random
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import numpy as np
import torch
from sp_common import *
from sp_data import Data
from sp_model import SemanticModel, make_noise
from route_loss import objective, caption_loss
from gpu_prefetch import CudaBatches


def frozen_signature(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state(generator):
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                route=generator.get_state(), python=random.getstate(), numpy=np.random.get_state())


def restore_rng(state, generator):
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'])
    generator.set_state(state['route']);random.setstate(state['python']);np.random.set_state(state['numpy'])


def main(arm, seed, engineering_steps=None):
    guard(True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.90)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if engineering_steps is None:
        protocol = verify_protocol()
        protocol_hash = digest(EXP / 'protocol_v2_frozen.json')
    else:
        assert not (EXP/'protocol_v2_frozen.json').exists(), 'Engineering fixtures must precede freeze'
        protocol=read(EXP/'design_protocol_v2_draft.json')
        protocol['training']['batch']=48
        protocol_hash=digest(EXP/'design_protocol_v2_draft.json')
    runtime=protocol.get('runtime_amendment',{})
    if runtime:
        torch.cuda.set_per_process_memory_fraction(runtime['allocator_cap_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
    folder = EXP / f'{"fits" if engineering_steps is None else "engineering_trainer"}/{arm}/seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        assert read(folder / 'complete.json')['protocol_sha256'] == protocol_hash
        return
    if (folder / 'unsafe_optimizer_failure.json').exists():
        raise RuntimeError('Partial optimizer failure requires audit; will not silently resume')
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    route_generator = torch.Generator(device='cuda').manual_seed(70000 + seed)
    data = Data()
    train = np.load(BASE / 'splits.npz')['train']
    hold = np.load(REL / 'selection_holdout.npy')
    if engineering_steps is not None:hold=hold[:32]
    batch = protocol['training']['batch']
    epochs = protocol['training']['epochs']
    per_epoch = math.ceil(len(train) / batch)
    model = SemanticModel(arm, seed, data.prototypes, data.types)
    if runtime:
        policy=runtime['checkpoint_policy_by_arm'][arm]
        model.checkpoint_blocks=tuple(policy['full_blocks'])
        model.spatial_checkpoint_blocks=tuple(policy['spatial_only_blocks'])
    trunk, head = model.parameter_groups()
    groups = [dict(params=trunk, lr=1e-5)]
    if head:
        groups.append(dict(params=head, lr=1e-4))
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    initial_frozen = frozen_signature(model)
    start, elapsed_offset = time.time(), 0.
    state = dict(epoch=1,cursor=0,steps=0,best=float('inf'),history=[],numerical=0,
                 exposures=np.zeros(len(data.records),np.int32),epoch_sums={},epoch_samples=0)
    stop = [False]
    phase = 'idle'
    step_start_rng = None

    def handle_signal(*args):
        stop[0] = True

    signal.signal(signal.SIGTERM,handle_signal);signal.signal(signal.SIGINT,handle_signal)

    def check_stop():
        if stop[0]:
            raise StudyStop('Owned task received stop signal')
        guard()

    def checkpoint(kind, epoch=None):
        value = dict(updates=model.updates(),arm=arm,seed=seed,epoch=state['epoch'] if epoch is None else epoch,
                     steps=state['steps'],protocol_sha256=protocol_hash)
        if runtime:value['runtime_amendment_sha256']=protocol['runtime_amendment_sha256']
        if kind == 'resume':
            value.update(state=state,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                         rng=rng_state(route_generator),initial_frozen_sha256=initial_frozen,
                         elapsed_s=elapsed_offset+time.time()-start)
        atomic_torch_save(folder / f'{kind}.pt',value)

    def validate(epoch):
        model.eval();total=0.;count=0
        with torch.inference_mode():
            for begin in range(0,len(hold),32):
                check_stop()
                d=data.batch(hold[begin:begin+32],arm=arm)
                with torch.cuda.amp.autocast():
                    out=model(d['x'],d['mask'])
                    loss=caption_loss(out['g'],d['caption'],d['cid'],model.scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite internal caption loss')
                total+=float(loss)*len(d['ids']);count+=len(d['ids'])
                if begin % 256 == 0:
                    atomic(folder/'progress.json',dict(phase='internal_validation',arm=arm,seed=seed,epoch=epoch,
                          completed=count,total=len(hold),steps=state['steps'],pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
        return total/count

    try:
        if (folder/'resume.pt').exists():
            ck=torch.load(folder/'resume.pt',map_location='cpu')
            assert ck['protocol_sha256']==protocol_hash and ck['initial_frozen_sha256']==initial_frozen
            model.load_state_dict(ck['updates'],strict=False)
            optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
            state=ck['state'];elapsed_offset=ck['elapsed_s'];restore_rng(ck['rng'],route_generator)
            del ck
        atomic(folder/'manifest.json',dict(arm=arm,seed=seed,batch=batch,epochs=epochs,train_clips=len(train),holdout_clips=len(hold),
              protocol_sha256=protocol_hash,initial_M_sha256=digest(MBASE/f'heads/M/seed{seed}/best.pt'),
              frozen_sha256=initial_frozen,trunk_parameters=sum(p.numel() for p in trunk),
              router_parameters=sum(p.numel() for p in head),primary_checkpoint='epoch4_final',
              scoring='FP32_cosine_20way',pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
        if runtime:
            manifest=read(folder/'manifest.json');manifest.update(runtime_amendment_sha256=protocol['runtime_amendment_sha256'],
                checkpoint_policy=runtime['checkpoint_policy_by_arm'][arm],allocator_cap_gib=runtime['allocator_cap_gib'])
            atomic(folder/'manifest.json',manifest)
        if not state['history']:
            state['best']=validate(0)
            state['history']=[dict(epoch=0,caption_holdout=state['best'])]
            checkpoint('best',0);checkpoint('resume')
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        while state['epoch']<=epochs:
            epoch=state['epoch']
            order=np.random.default_rng(seed*100+epoch).permutation(train)
            model.train()
            with ThreadPoolExecutor(max_workers=1) as pool,ExitStack() as contexts:
                cursor=state['cursor']
                depth=runtime.get('gpu_prefetch_depth_by_arm',{}).get(arm,0)
                loader=contexts.enter_context(CudaBatches(data,order[cursor:],batch,arm,depth)) if depth and cursor<len(order) else None
                future=pool.submit(data.cpu_batch,order[cursor:cursor+batch]) if cursor<len(order) and loader is None else None
                while state['cursor']<len(order):
                    check_stop();tick=time.perf_counter()
                    if loader is not None:
                        d=loader.next();end=state['cursor']+len(d['ids'])
                    else:
                        prepared=future.result();end=state['cursor']+len(prepared[0])
                        future=pool.submit(data.cpu_batch,order[end:end+batch]) if end<len(order) else None
                        d=data.batch(prepared=prepared,arm=arm)
                    progress=min(1.,state['steps']/per_epoch)
                    step_start_rng=rng_state(route_generator)
                    noise=make_noise(model,len(d['ids']),route_generator)
                    before_rng=rng_state(route_generator)
                    success=False
                    for retry in range(10):
                        restore_rng(before_rng,route_generator)
                        optimizer.zero_grad(set_to_none=True)
                        amp=retry<9;phase='forward'
                        with torch.cuda.amp.autocast(enabled=amp):
                            out=model(d['x'] if amp else d['x'].float(),d['mask'],progress,noise)
                            loss,parts=objective(model,out,d)
                        scaled=scaler.scale(loss)
                        reason=None
                        if torch.isfinite(loss):
                            phase='backward';scaled.backward();scaler.unscale_(optimizer)
                            try:
                                trunk_norm=float(torch.nn.utils.clip_grad_norm_(trunk,1.,error_if_nonfinite=True))
                                head_norm=float(torch.nn.utils.clip_grad_norm_(head,1.,error_if_nonfinite=True)) if head else 0.
                            except RuntimeError as exc:
                                if 'non-finite' not in str(exc):raise
                                reason='nonfinite_gradient'
                            if reason is None:
                                phase='optimizer';scaler.step(optimizer);scaler.update();success=True;phase='idle';break
                        else:reason='nonfinite_loss'
                        state['numerical']+=1
                        with (folder/'numerical.jsonl').open('a') as f:
                            f.write(json.dumps(dict(step=state['steps'],retry=retry,reason=reason,scale=scaler.get_scale()))+'\n')
                        scaler.update(new_scale=max(1.,scaler.get_scale()*.5))
                        del out,loss,parts,scaled
                        if not amp:raise FloatingPointError('FP32 retry failed')
                    assert success
                    state['exposures'][d['ids']]+=1;state['steps']+=1;state['cursor']=end
                    values={k:float(v) for k,v in parts.items()}
                    count=len(d['ids']);state['epoch_samples']+=count
                    for k,v in values.items():state['epoch_sums'][k]=state['epoch_sums'].get(k,0.)+count*v
                    if loader is not None:torch.cuda.current_stream().synchronize()
                    else:torch.cuda.synchronize()
                    timings.append(time.perf_counter()-tick)
                    if state['steps']%10==0 or state['steps']==1:
                        atomic(folder/'progress.json',dict(phase='training',arm=arm,seed=seed,epoch=epoch,steps=state['steps'],
                              total_steps=epochs*per_epoch,cursor=end,train_clips=len(train),components=values,
                              trunk_gradient_norm=trunk_norm,router_gradient_norm=head_norm,
                              elapsed_s=elapsed_offset+time.time()-start,median_recent_step_s=float(np.median(timings[-50:])),
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,numerical_retries=state['numerical'],
                              pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],gpu_prefetch_depth=depth))
                        print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,step=state['steps'],loss=float(loss),seconds=timings[-1])),flush=True)
                    if state['steps']%100==0:checkpoint('resume')
                    del d,out,loss,parts,scaled,noise
                    if engineering_steps is not None and state['steps']>=engineering_steps:
                        raise StudyStop('ENGINEERING_STOP_AFTER_STEPS')
            held=validate(epoch)
            state['history'].append(dict(epoch=epoch,caption_holdout=held,
                training_losses={k:v/max(1,state['epoch_samples']) for k,v in state['epoch_sums'].items()}))
            if held<state['best']:
                state['best']=held;checkpoint('best',epoch)
            if epoch==epochs:checkpoint('final',epoch)
            atomic(folder/'history.json',state['history'])
            state.update(epoch=epoch+1,cursor=0,epoch_sums={},epoch_samples=0)
            checkpoint('resume')
        assert frozen_signature(model)==initial_frozen
        assert (state['exposures'][train]==epochs).all() and state['exposures'].sum()==len(train)*epochs
        np.save(folder/'exposures.npy',state['exposures'])
        atomic(folder/'complete.json',dict(passed=True,arm=arm,seed=seed,steps=state['steps'],
              protocol_sha256=protocol_hash,final_sha256=digest(folder/'final.pt'),best_sha256=digest(folder/'best.pt'),
              exposure_sha256=digest(folder/'exposures.npy'),frozen_prefix_exact=True,
              elapsed_s=elapsed_offset+time.time()-start,numerical_retries=state['numerical'],
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        if runtime:
            completion=read(folder/'complete.json');completion.update(runtime_amendment_sha256=protocol['runtime_amendment_sha256'],
                checkpoint_policy=runtime['checkpoint_policy_by_arm'][arm],allocator_cap_gib=runtime['allocator_cap_gib'])
            atomic(folder/'complete.json',completion)
        atomic(folder/'progress.json',dict(phase='complete',arm=arm,seed=seed,steps=state['steps'],total_steps=epochs*per_epoch))
    except BaseException as exc:
        safe=phase!='optimizer'
        saved=False;save_error=None
        try:
            if safe:
                if phase in ('forward','backward') and step_start_rng is not None:
                    restore_rng(step_start_rng,route_generator)
                checkpoint('resume');saved=True
            else:
                atomic(folder/'unsafe_optimizer_failure.json',dict(reason=repr(exc),step=state['steps']))
        except BaseException as save_exc:save_error=repr(save_exc)
        atomic(folder/'failed.json',dict(reason=repr(exc),phase=phase,epoch=state['epoch'],cursor=state['cursor'],
              steps=state['steps'],resume_saved=saved,resume_safe=safe,save_error=save_error,time=time.time()))
        if engineering_steps is not None and str(exc)=='ENGINEERING_STOP_AFTER_STEPS' and saved:
            atomic(EXP/'trainer_engineering.json',dict(passed=True,actual_training_steps=state['steps'],
                  cursor=state['cursor'],stop_exception_checkpoint_saved=True,optimizer_and_scaler_serialized=True,
                  numerical_retries=state['numerical'],formal_fits_modified=False))
            return
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--engineering-steps',type=int,default=None)
    args=p.parse_args();main(args.arm,args.seed,args.engineering_steps)
