import argparse
import copy
import numpy as np
import torch
from rel_common import *
from rel_model import RelationModel,binding_scores,objective
from rel_data import Data

def logic_checks():
    # Both role words are present, but on DIFFERENT slots.
    split=torch.tensor([[[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]])
    bound=split.clone();bound[:,:,1]=bound[:,[1,0],1]
    text=torch.tensor([[[1.,0.],[1.,0.]]]);mask=torch.ones(1,2,dtype=torch.bool)
    a=binding_scores(split,text,mask,'bound');b=binding_scores(bound,text,mask,'bound')
    u=binding_scores(split,text,mask,'unbound');v=binding_scores(bound,text,mask,'unbound')
    assert float(b)>float(a)+.2
    torch.testing.assert_close(u,v)
    assert float(u)>float(a)+.2
    z=split.clone().requires_grad_(True);binding_scores(z,text,mask,'bound').sum().backward()
    assert torch.isfinite(z.grad).all()
    # An unannotated target role cannot affect a score.
    role3=torch.cat([split,torch.randn(1,2,1,2)],2)
    txt3=torch.cat([text,torch.randn(1,1,2)],1)
    valid=torch.tensor([[True,True,False]])
    torch.testing.assert_close(binding_scores(role3,txt3,valid,'bound'),a)
    return dict(bound_discriminates_binding=True,unbound_invariant_to_role_slot_permutation=True,
                missing_role_is_masked=True,finite_binding_gradient=True)

def main(gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.90)
    checks=logic_checks();guard()
    data=Data(allow_hdd=True);relation=read(EXP/'relations.json')
    selected=[int(i) for i in relation['scrambled']][:8]
    assert len(selected)==8,'Not enough controllable relation permutations'
    small=data.cpu_batch(selected)
    prepared=(selected*6,np.tile(small[1],(6,1,1)),np.tile(small[2],(6,1)))
    batch=data.batch(prepared[0],prepared=prepared)
    assert not torch.equal(batch['relation_pos'],batch['scrambled_pos'])
    measurements=[];reference=None;head_signature=None
    for arm in ['A_continue_M','C_bound','D_unbound','E_scrambled']:
        guard();torch.manual_seed(17);m=RelationModel(arm,17).train()
        m.eval()
        with torch.no_grad():
            if arm=='A_continue_M':reference=m(batch['x'][:2].float(),batch['mask'][:2])['g'].clone()
            else:
                h=hashlib.sha256()
                for name,param in m.named_parameters():
                    if param.requires_grad:h.update(name.encode());h.update(param.detach().cpu().numpy().tobytes())
                if head_signature is None:head_signature=h.hexdigest()
                else:assert h.hexdigest()==head_signature,'C/D/E initialization differs'
                old=m.gate_logit.detach().clone();m.gate_logit.fill_(-100.)
                plain=m(batch['x'][:2].float(),batch['mask'][:2])['g']
                torch.testing.assert_close(plain,reference,atol=2e-5,rtol=2e-5)
                m.gate_logit.copy_(old)
        checks['zero_gate_replays_M']=True;checks['C_D_E_initialization_identical']=True
        m.train()
        params=[p for p in m.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-5)
        scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
        torch.cuda.reset_peak_memory_stats();times=[]
        for step in range(3):
            guard();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            with torch.cuda.amp.autocast():out=m(batch['x'],batch['mask']);loss,parts=objective(m,out,batch)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            assert m.visual.blocks[11].mlp.fc1.weight.grad.abs().sum()>0
            assert all(p.grad is None for p in m.visual.blocks[0].parameters())
            if arm!='A_continue_M':
                assert m.queries.grad.abs().sum()>0
                assert m.role_heads[0].weight.grad.abs().sum()>0 and m.role_heads[1].weight.grad.abs().sum()>0
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            if step:times.append(time.perf_counter()-start)
        measurements.append(dict(arm=arm,batch=48,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
             peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,step_median_s=float(np.median(times)),
             trainable_parameters=sum(p.numel() for p in params),components={k:float(v) for k,v in parts.items()}))
        if arm=='C_bound':
            m.eval()
            with torch.no_grad():
                diagnostic=m(batch['x'][:2].float(),batch['mask'][:2],True)
                attention=diagnostic['attention'];assert attention.shape==(2,8,3136)
                assert torch.isfinite(attention).all()
                torch.testing.assert_close(attention.sum(-1),torch.ones_like(attention.sum(-1)),atol=1e-5,rtol=1e-5)
                checks['attention_probability_normalized']=True
        print(measurements[-1],flush=True)
        del m,params,optimizer,scaler,out,loss,parts
        if arm=='C_bound':del diagnostic,attention
        torch.cuda.empty_cache()
    atomic(EXP/'preflight.json',dict(passed=True,checks=checks,measurements=measurements,
        nominal_batch=48,training_epochs=4,implementation_only=True,changed_real_checkpoints=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',choices=['2','3'],default='3');a=p.parse_args();main(a.gpu)
