"""Measure activation retention at unchanged batch/precision and compare updates."""
import copy
import argparse
import gc
import time
import numpy as np
import torch
from sp_common import *
from sp_data import Data
from sp_model import SemanticModel,make_noise
from route_loss import objective


def cpu_clone(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_clone(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_clone(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_clone(v) for v in value)
    return copy.deepcopy(value)


def main(arms,cap_gib):
    guard(True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    total=torch.cuda.get_device_properties(0).total_memory
    # Driver/context overhead is about0.9GiB; a19GiB allocator cap targets20GiB total.
    torch.cuda.set_per_process_memory_fraction(cap_gib*2**30/total)
    data=Data();train=np.load(BASE/'splits.npz')['train'];d=data.batch(train[100:148])
    gen=torch.Generator(device='cuda').manual_seed(70029)
    report=dict(batch=48,precision='unchanged_FP16_AMP_FP32_routing',allocator_cap_gib=cap_gib,
                purpose='retain useful activations to reduce recomputation; no dummy allocations',arms={})
    target=EXP/f'runtime_amendments/memory20g_20261027/probe_{"_".join(arms)}.json'
    for arm in arms:
        model=SemanticModel(arm,29,data.prototypes,data.types).train()
        trunk,head=model.parameter_groups();params=trunk+head
        groups=[dict(params=trunk,lr=1e-5)]
        if head:groups.append(dict(params=head,lr=1e-4))
        opt=torch.optim.AdamW(groups,weight_decay=.01)
        resume=EXP/f'fits/{arm}/seed29/resume.pt'
        if resume.exists():
            ck=torch.load(resume,map_location='cpu');model.load_state_dict(ck['updates'],strict=False);opt.load_state_dict(ck['optimizer']);del ck
        initial=cpu_clone(model.updates());initial_opt=cpu_clone(opt.state_dict())
        noise=make_noise(model,48,gen)
        outcomes=[];baseline=None

        def run(mode,spatial_mode=()):
            nonlocal baseline
            model.load_state_dict(initial,strict=False);opt.load_state_dict(cpu_clone(initial_opt))
            opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model.checkpoint_blocks=mode
            model.spatial_checkpoint_blocks=spatial_mode
            tick=time.perf_counter()
            with torch.cuda.amp.autocast():
                out=model(d['x'],d['mask'],.25,noise)
                loss,_=objective(model,out,d)
            assert torch.isfinite(loss);loss.backward()
            gradients=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in params]
            for group in (trunk,head):
                if group:torch.nn.utils.clip_grad_norm_(group,1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize()
            result=dict(checkpoint_blocks=list(mode),spatial_checkpoint_blocks=list(spatial_mode),passed=True,seconds_with_cpu_comparison=time.perf_counter()-tick,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                loss=float(loss))
            output=out['g'].detach().cpu().clone()
            selected=None if out['route'] is None else out['route']['indices'].detach().cpu().clone()
            updated=[p.detach().cpu().clone() for p in params]
            if baseline is None:baseline=(output,selected,gradients,updated)
            else:
                result['output_max_error']=float((output-baseline[0]).abs().max())
                result['indices_exact']=selected is None or torch.equal(selected,baseline[1])
                result['gradient_max_error']=max(float((a-b).abs().max()) for a,b in zip(gradients,baseline[2]) if a is not None and b is not None)
                result['update_max_error']=max(float((a-b).abs().max()) for a,b in zip(updated,baseline[3]))
                assert result['output_max_error']==0 and result['indices_exact']
                assert result['gradient_max_error']==0 and result['update_max_error']==0,result
            del out,loss,gradients,updated,output,selected
            # Timing after equivalence check; same weights and noise for each timed step.
            times=[]
            for repeat in range(4):
                opt.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter()
                with torch.cuda.amp.autocast():
                    out=model(d['x'],d['mask'],.25,noise);loss,_=objective(model,out,d)
                loss.backward()
                for group in (trunk,head):
                    if group:torch.nn.utils.clip_grad_norm_(group,1.,error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-tick)
                del out,loss
            result['median_update_seconds']=float(np.median(times))
            result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            result['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
            return result

        for mode,spatial_mode in (((10,11),()),((11,),()),((11,),(10,)),((10,),())):
            try:result=run(mode,spatial_mode)
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower():raise
                result=dict(checkpoint_blocks=list(mode),spatial_checkpoint_blocks=list(spatial_mode),passed=False,reason=f'OOM_at{cap_gib}GiB_allocator_cap')
                opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            outcomes.append(result);report['arms'][arm]=outcomes;atomic(target,report)
            print(arm,json.dumps(result),flush=True)
        del model,opt,params,trunk,head,groups,noise,initial,initial_opt,baseline
        gc.collect();torch.cuda.empty_cache()
    report['passed']=True;atomic(target,report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['R0','R5','R2']);p.add_argument('--cap-gib',type=float,default=19.)
    args=p.parse_args();main(args.arms,args.cap_gib)
