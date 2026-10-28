"""Exercise real queued batches at the intended20GiB working set."""
import argparse
import hashlib
import numpy as np
import torch
import time
from sp_common import *
from sp_data import Data
from sp_model import SemanticModel,make_noise
from gpu_prefetch import CudaBatches
from route_loss import objective


def main(arms):
    guard(True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(19*2**30/torch.cuda.get_device_properties(0).total_memory)
    report={};data=Data();train=np.load(BASE/'splits.npz')['train'];order=train[100:100+32*48]
    reference={}
    for step in (0,15,31):
        ids,x,mask=data.cpu_batch(order[step*48:(step+1)*48])
        reference[step]=hashlib.sha256(memoryview(x).cast('B')).hexdigest()
    for arm in arms:
        model=SemanticModel(arm,29,data.prototypes,data.types).train();model.checkpoint_blocks=(10,)
        trunk,head=model.parameter_groups();groups=[dict(params=trunk,lr=1e-5)]
        if head:groups.append(dict(params=head,lr=1e-4))
        opt=torch.optim.AdamW(groups,weight_decay=.01)
        depth=7 if arm=='R0' else 13
        gen=torch.Generator(device='cuda').manual_seed(70029)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        times=[];memory=[];rng_before=torch.cuda.get_rng_state()
        with CudaBatches(data,order,48,arm,depth) as batches:
            for step in range(32):
                d=batches.next();assert d['ids']==list(map(int,order[step*48:(step+1)*48]))
                if step in reference:
                    actual=hashlib.sha256(memoryview(d['x'].cpu().numpy()).cast('B')).hexdigest()
                    assert actual==reference[step],(arm,step)
                torch.cuda.current_stream().synchronize();tick=time.perf_counter()
                opt.zero_grad(set_to_none=True);noise=make_noise(model,48,gen)
                with torch.cuda.amp.autocast():
                    out=model(d['x'],d['mask'],.25,noise);loss,_=objective(model,out,d)
                assert torch.isfinite(loss);loss.backward()
                for group in (trunk,head):
                    if group:torch.nn.utils.clip_grad_norm_(group,1.,error_if_nonfinite=True)
                opt.step();torch.cuda.current_stream().synchronize();times.append(time.perf_counter()-tick)
                if step in (8,16,24):
                    used=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
                    memory.append(int(used.strip()))
                del d,out,loss,noise
        assert torch.equal(rng_before,torch.cuda.get_rng_state()),'Input prefetch changed global CUDA RNG'
        report[arm]=dict(passed=True,depth=depth,checkpoints=[10],allocator_cap_gib=19,
              real_batches=32,input_byte_checks_passed=True,sample_order_exact=True,global_cuda_rng_unchanged=True,
              median_update_seconds=float(np.median(times[4:24])),observed_device_memory_mib=memory,
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        atomic(EXP/f'runtime_amendments/memory20g_20261027/prefetch_{"_".join(arms)}.json',report)
        print(arm,json.dumps(report[arm]),flush=True)
        del model,opt,trunk,head,groups,gen;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['R0','R5','R2']);main(p.parse_args().arms)
