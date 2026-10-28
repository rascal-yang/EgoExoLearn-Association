"""Engineering gates on TRAIN only; no new VAL/TEST model selection."""
import argparse
import copy
import gc
import random
import numpy as np
import torch
from b10_common import *
from b10_model import Block10Model,SemanticModel,make_noise
from sp_data import Data
from route_loss import objective,m_losses,route_losses
from gpu_prefetch import CudaBatches


def setup():
    guard(True);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(19*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old_common.verify_protocol()


def clone(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
    if isinstance(value,list):return [clone(v) for v in value]
    if isinstance(value,tuple):return tuple(clone(v) for v in value)
    return copy.deepcopy(value)


def optimizer(model):
    trunk,head=model.parameter_groups()
    return torch.optim.AdamW([dict(params=trunk,lr=1e-5),dict(params=head,lr=1e-4)],betas=(.9,.999),eps=1e-8,weight_decay=.01)


def algebra():
    setup();data=Data();ids=np.load(BASE/'splits.npz')['train'][:2];d=data.batch(ids)
    dense=SemanticModel('R0',17,data.prototypes,data.types).train()
    routed=Block10Model('S2',17,data.prototypes,data.types).train()
    generator=torch.Generator(device='cuda').manual_seed(70017)
    a=dense(d['x'].float(),d['mask'],use_checkpoint=False)
    b=routed(d['x'].float(),d['mask'],.1,make_noise(routed,2,generator),dense=True,use_checkpoint=False)
    assert torch.equal(a['g'],b['g']) and torch.equal(a['patches'],b['patches'])
    ma,mb=m_losses(a,d,dense.scale),m_losses(b,d,routed.scale)
    assert all(torch.equal(x,y) for x,y in zip(ma,mb))
    prefix,evidence=route_losses(routed,b,d)
    ta,_=dense.parameter_groups();tb,hb=routed.parameter_groups()
    aux_grad=torch.autograd.grad(prefix+evidence,tb,allow_unused=True,retain_graph=True)
    assert all(v is None for v in aux_grad)
    (ma[0]+.1*ma[1]+.1*ma[2]).backward()
    (mb[0]+.1*mb[1]+.1*mb[2]+.05*(prefix+evidence)).backward()
    gradient_error=max(float((x.grad-y.grad).abs().max()) for x,y in zip(ta,tb))
    assert gradient_error==0
    for group in (ta,tb,hb):torch.nn.utils.clip_grad_norm_(group,1.,error_if_nonfinite=True)
    oa=torch.optim.AdamW(ta,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    ob=optimizer(routed);oa.step();ob.step()
    update_error=max(float((x-y).abs().max()) for x,y in zip(ta,tb));assert update_error==0
    routed.zero_grad(set_to_none=True);routed.eval();seen=[]
    hook=routed.visual.blocks[11].attn.qkv.register_forward_hook(lambda module,inputs,output:seen.append(list(inputs[0].shape)))
    with torch.inference_mode():
        out=routed(d['x'].float(),d['mask'])
        hook.remove();assert seen==[[2,3137,768]],seen
        route=routed.router(d['x'].float(),d['mask'],'R2')
        from memory_attention import run_block
        x=run_block(routed.visual.blocks[10],d['x'].float(),route['assignment'],route['mass'])
        x=run_block(routed.visual.blocks[11],x)
        g=torch.nn.functional.normalize((routed.visual.norm(x)[:,0]@routed.projection).float(),dim=-1)
        reference_error=float((g-out['g']).abs().max());assert reference_error==0
        assert bool((route['assignment'].sum(1)==1).all()) and bool((route['mass'].sum(1)==196).all())
    routed.train();noise=make_noise(routed,2,generator)
    out=routed(d['x'].float(),d['mask'],.4,noise,use_checkpoint=False)
    components=m_losses(out,d,routed.scale);task=components[0]+.1*(components[1]+components[2])
    parameters=[routed.router.content.weight,routed.router.projection,routed.visual.blocks[11].attn.qkv.weight]
    grads=torch.autograd.grad(task,parameters)
    norms=[float(v.norm()) for v in grads];assert all(np.isfinite(norms)) and all(v>0 for v in norms)
    atomic(EXP/'algebra_gate.json',dict(passed=True,dense_descriptor_and_patch_exact=True,M_losses_exact=True,
           dense_bypass_trunk_gradient_max_error=gradient_error,dense_bypass_trunk_update_max_error=update_error,
           auxiliary_direct_trunk_path=False,block11_native_full_qkv_input=seen,
           explicit_block10_compact_block11_dense_max_error=reference_error,
           task_grad_norms=dict(router_content=norms[0],router_projection=norms[1],dense_block11_qkv=norms[2]),
           partition_exact196=True,TEST_access=False,new_VAL_evaluation=False))
    print('ALGEBRA_GATE_PASSED',norms,flush=True)


def worst_batch(data):
    path=EXP/'worst_training_batch.json'
    if path.exists():return np.array(read(path)['ids'])
    train=np.load(BASE/'splits.npz')['train'];best=(-1,None)
    for seed in SEEDS:
        for epoch in range(1,5):
            order=np.random.default_rng(seed*100+epoch).permutation(train)
            for begin in range(0,len(order),48):
                ids=order[begin:begin+48]
                if len(ids)!=48:continue
                phrases={p for i in ids for p in data.records[int(i)]['phrases']}
                nouns=sum(data.phrases[p]['type']=='noun' for p in phrases)
                score=nouns*3136+(len(phrases)-nouns)*16
                if score>best[0]:best=(score,dict(ids=ids.tolist(),seed=seed,epoch=epoch,cursor=begin,
                                                noun_phrases=nouns,verb_phrases=len(phrases)-nouns,score=score))
    atomic(path,best[1]);return np.array(best[1]['ids'])


def memory(arm):
    setup();data=Data();ids=worst_batch(data);d=data.batch(ids,arm=arm)
    model=Block10Model(arm,29,data.prototypes,data.types).train();opt=optimizer(model)
    initial=clone(model.updates());initial_opt=clone(opt.state_dict())
    gen=torch.Generator(device='cuda').manual_seed(70029);noise=make_noise(model,48,gen)
    baseline=None;outcomes=[]
    for policy in ((10,11),(10,)):
        model.load_state_dict(initial,strict=False);opt.load_state_dict(clone(initial_opt));model.checkpoint_blocks=policy
        opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        with torch.cuda.amp.autocast():out=model(d['x'],d['mask'],.25,noise);loss,_=objective(model,out,d)
        assert torch.isfinite(loss);loss.backward()
        params=[p for p in model.parameters() if p.requires_grad]
        grads=[p.grad.detach().cpu().clone() for p in params]
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        for group in model.parameter_groups():torch.nn.utils.clip_grad_norm_(group,1.,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize()
        current=(out['g'].detach().cpu().clone(),out['route']['indices'].detach().cpu().clone(),grads,[p.detach().cpu().clone() for p in params])
        row=dict(policy=list(policy),passed=True,loss=float(loss),seconds=time.perf_counter()-tick,
                 peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if baseline is None:baseline=current
        else:
            row.update(output_error=float((current[0]-baseline[0]).abs().max()),indices_exact=torch.equal(current[1],baseline[1]),
                       gradient_error=max(float((a-b).abs().max()) for a,b in zip(current[2],baseline[2])),
                       update_error=max(float((a-b).abs().max()) for a,b in zip(current[3],baseline[3])))
            assert row['output_error']==0 and row['indices_exact'] and row['gradient_error']==0 and row['update_error']==0,row
        outcomes.append(row);del out,loss,current,grads
    atomic(EXP/f'memory_{arm}.json',dict(passed=True,arm=arm,batch=48,worst_training_batch=read(EXP/'worst_training_batch.json'),policies=outcomes))
    print('MEMORY_GATE_PASSED',arm,outcomes,flush=True)


def prefetch(arm,depth):
    setup();data=Data();train=np.load(BASE/'splits.npz')['train'];order=np.random.default_rng(2901).permutation(train)
    model=Block10Model(arm,29,data.prototypes,data.types).train();model.checkpoint_blocks=(10,)
    opt=optimizer(model);scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    gen=torch.Generator(device='cuda').manual_seed(70029);timings=[];resident=[]
    torch.cuda.reset_peak_memory_stats()
    with CudaBatches(data,order[:48*40],48,arm,depth) as loader:
        for step in range(32):
            guard();d=loader.next();assert d['ids']==list(map(int,order[step*48:(step+1)*48]))
            if step<2:
                expected=[]
                for i in d['ids']:
                    k,j=data.locations[i];expected.append(np.array(data.arrays[k][j]))
                assert np.array_equal(d['x'].cpu().numpy(),np.stack(expected))
            tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            noise=make_noise(model,48,gen)
            with torch.cuda.amp.autocast():out=model(d['x'],d['mask'],.25,noise);loss,_=objective(model,out,d)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt)
            for group in model.parameter_groups():torch.nn.utils.clip_grad_norm_(group,1.,error_if_nonfinite=True)
            scaler.step(opt);scaler.update();torch.cuda.current_stream().synchronize();timings.append(time.perf_counter()-tick)
            if step in (10,20,31):resident.append(int(subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
            del out,loss,noise,d
    atomic(EXP/f'prefetch_{arm}.json',dict(passed=True,arm=arm,batch=48,depth=depth,steps=32,
           input_bytes_exact=True,sample_order_exact=True,finite_loss_and_gradients=True,observed_device_memory_mib=resident,
           median_step_seconds=float(np.median(timings[-20:])),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
           peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,TEST_access=False,new_VAL_evaluation=False))
    print('PREFETCH_GATE_PASSED',arm,depth,resident,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('algebra','memory','prefetch'),required=True)
    p.add_argument('--arm',choices=ARMS,default='S2');p.add_argument('--depth',type=int,default=6);a=p.parse_args()
    EXP.mkdir(exist_ok=True);REP.mkdir(exist_ok=True)
    if a.mode=='algebra':algebra()
    elif a.mode=='memory':memory(a.arm)
    else:prefetch(a.arm,a.depth)
