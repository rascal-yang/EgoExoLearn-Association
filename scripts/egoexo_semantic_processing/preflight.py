"""Real-model engineering checks only; does not read official scores."""
import argparse
import copy
import math
import time
import torch
import numpy as np
from torch.nn import functional as F
from sp_common import *
from sp_data import Data
from sp_model import SemanticModel, make_noise
from memory_attention import compact_spatial, run_block
from route_loss import objective, m_losses


def close(a, b, atol=1e-5, rtol=1e-5):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (float((a-b).abs().max()), float(a.abs().max()))
    return float((a-b).abs().max())


def optimizer_for(model):
    trunk, head = model.parameter_groups()
    groups = [dict(params=trunk, lr=1e-5)]
    if head:
        groups.append(dict(params=head, lr=1e-4))
    return torch.optim.AdamW(groups, weight_decay=.01)


def update(model, data, optimizer, generator, progress=.1, checkpoint=True):
    optimizer.zero_grad(set_to_none=True)
    noise = make_noise(model, len(data['ids']), generator)
    with torch.cuda.amp.autocast():
        out = model(data['x'], data['mask'], progress, noise, use_checkpoint=checkpoint)
        loss, parts = objective(model, out, data)
    assert torch.isfinite(loss)
    loss.backward()
    trunk, head = model.parameter_groups()
    norms = [float(torch.nn.utils.clip_grad_norm_(group, 1., error_if_nonfinite=True)) for group in (trunk, head) if group]
    optimizer.step()
    return float(loss), norms, out


def main(role):
    guard(True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.90)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Data()
    train = np.load(BASE / 'splits.npz')['train']
    gen = torch.Generator(device='cuda').manual_seed(70017)
    records = dict(role=role, gpu=os.environ['CUDA_VISIBLE_DEVICES'], started=time.time(), checks={}, implementation_only=True, official_scores_read=False)
    target = EXP / f'preflight_{role}.json'

    def record(name, value):
        records['checks'][name] = value
        atomic(target, records)
        print(name, json.dumps(value, ensure_ascii=False), flush=True)

    if role == 'main':
        model = SemanticModel('R2', 17, data.prototypes, data.types).eval()
        batch = data.batch(train[:2])
        block = model.visual.blocks[10]
        normalized = block.norm1(batch['x'].float())
        identity = torch.eye(196, device='cuda')[None].expand(32, -1, -1).clone()
        mass = torch.ones(32, 196, device='cuda')
        with torch.no_grad():
            native = block.attn(normalized, 'b (f n) d', '(b f) n d', {'f':16})
            compact = compact_spatial(block.attn, normalized, identity, mass)
            error = close(native, compact)
            native_block = block(batch['x'].float(), 'b (f n) d', '(b f) n d', 'b (f n) d', '(b n) f d', time_n=196, space_f=16)
            compact_block = run_block(block, batch['x'].float(), identity, mass)
            block_error = close(native_block, compact_block)
        record('dense_identity_replay_FP32', dict(attention_max_error=error, block_max_error=block_error))
        del native, compact, native_block, compact_block, normalized, identity, mass

        # FP64 multiplicity identity including linear projection bias.
        q = torch.randn(2, 13, 9, dtype=torch.float64)
        k = torch.randn(2, 5, 9, dtype=torch.float64)
        v = torch.randn(2, 5, 9, dtype=torch.float64)
        counts = torch.tensor([1, 3, 2, 4, 3])
        dense = (q @ k.repeat_interleave(counts, 1).transpose(1, 2) / 3).softmax(-1) @ v.repeat_interleave(counts, 1)
        compact = (q @ k.transpose(1, 2) / 3 + counts.log()).softmax(-1) @ v
        # Counts must be logged in float64, not rounded float32.
        compact = (q @ k.transpose(1, 2) / 3 + counts.double().log()).softmax(-1) @ v
        record('multiplicity_FP64', dict(max_error=close(dense, compact, 1e-10, 1e-10)))

        for arm in ('R1', 'R2', 'R3', 'R4', 'R5'):
            model.arm = arm
            model.train()
            noise = make_noise(model, 2, gen)
            route = model.router(batch['x'], batch['mask'], arm, .1, noise)
            a, m = route['assignment'], route['mass']
            expected_mass = 48 if arm == 'R5' else 196
            assert torch.equal(m.sum(1), torch.full((32,), expected_mass, device='cuda'))
            if arm != 'R5':
                assert torch.equal(a.sum(1), torch.ones(32, 196, device='cuda'))
            idx = route['indices'].sort(-1).values
            assert (idx[..., 1:] != idx[..., :-1]).all()
            assert torch.equal(a[:, :32].sum(-1), torch.ones(32, 32, device='cuda'))
            record(f'partition_{arm}', dict(mass=expected_mass, unique=True, finite=bool(torch.isfinite(a).all())))
            del route, a, m, noise
        model.arm = 'R2'
        optimizer = optimizer_for(model)
        noise = make_noise(model, 2, gen)
        with torch.cuda.amp.autocast():
            out = model(batch['x'], batch['mask'], .1, noise, use_checkpoint=False)
            parts = m_losses(out, batch, model.scale)
            main_loss = parts[0] + .1 * parts[1] + .1 * parts[2]
        named = [(n,p) for n,p in model.named_parameters() if n.startswith('router.') and p.requires_grad]
        grads = torch.autograd.grad(main_loss, [p for n,p in named], retain_graph=False, allow_unused=True)
        gradient_norms = {n: None if g is None else float(g.float().norm()) for (n,p),g in zip(named,grads)}
        assert gradient_norms['router.content.weight'] > 0 and gradient_norms['router.projection'] > 0
        assert all(g is None or torch.isfinite(g).all() for g in grads)
        record('main_task_router_gradients', gradient_norms)
        del out, grads, main_loss, parts

        # Checkpoint versus direct outputs and router/main gradients, identical routing noise.
        results = []
        for ckpt in (False, True):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(batch['x'], batch['mask'], .1, noise, use_checkpoint=ckpt)
                loss, _ = objective(model, out, batch)
            loss.backward()
            results.append((out['g'].detach().clone(), [p.grad.detach().clone() for p in model.parameters() if p.requires_grad and p.grad is not None]))
            del out, loss
        output_error = close(results[0][0], results[1][0], .005, .005)
        gradient_error = max(close(a,b,.005,.005) for a,b in zip(results[0][1], results[1][1]))
        record('checkpoint_recomputation', dict(output_max_error=output_error, gradient_max_error=gradient_error))
        del results, noise

        # Synthetic mutation of the per-layer input must change freshly rebuilt memory.
        model.eval()
        with torch.no_grad():
            route = model.router(batch['x'], batch['mask'], 'R2')
            z = batch['x'].float()
            a, m = route['assignment'], route['mass']
            first = run_block(model.visual.blocks[10], z, a, m)
            h = model.visual.blocks[11].norm1(first)
            changed = model.visual.blocks[11].norm1(first + .1*torch.randn_like(first))
            memory1 = torch.bmm(a, h[:,1:].reshape(32,196,768))
            memory2 = torch.bmm(a, changed[:,1:].reshape(32,196,768))
            assert not torch.equal(memory1,memory2)
        record('layer11_rebuild', dict(memory_changes_with_current_input=True))
        del route, z, a, m, first, h, changed, memory1, memory2

        model.train()
        update_log = []
        for step in range(8):
            d = data.batch(train[step*2:step*2+2])
            loss, norms, out = update(model, d, optimizer, gen)
            update_log.append(dict(step=step, loss=loss, gradient_norms=norms))
            del out, d
        record('real_updates_small', update_log)

        # Save/reload optimizer, model, and independent Gumbel generator; repeat the next update.
        snapshot = copy.deepcopy(model.state_dict())
        optim_state = copy.deepcopy(optimizer.state_dict())
        gen_state, cuda_state, cpu_state = gen.get_state(), torch.cuda.get_rng_state(), torch.get_rng_state()
        d = data.batch(train[20:22])
        loss1, _, out = update(model, d, optimizer, gen)
        expected = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        del out
        model.load_state_dict(snapshot)
        optimizer.load_state_dict(optim_state)
        gen.set_state(gen_state);torch.cuda.set_rng_state(cuda_state);torch.set_rng_state(cpu_state)
        loss2, _, out = update(model, d, optimizer, gen)
        err = max(close(p,expected[n],1e-6,1e-6) for n,p in model.named_parameters() if p.requires_grad)
        assert loss1 == loss2
        record('resume_replay', dict(loss_exact=True, max_parameter_error=err))
        del snapshot, optim_state, expected, out, optimizer, model, batch, d
        torch.cuda.empty_cache()

    arms = ('R2', 'R5') if role == 'main' else ('R0', 'R3')
    memory_records = []
    for batch_size in (48, 32):
        success = True
        for arm in arms:
            model = optimizer = out = d = None
            try:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                model = SemanticModel(arm, 17, data.prototypes, data.types).train()
                optimizer = optimizer_for(model)
                d = data.batch(train[100:100+batch_size], arm=arm)
                timings=[]
                for step in range(3):
                    guard();tick=time.perf_counter()
                    loss,norms,out=update(model,d,optimizer,gen)
                    torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
                    del out;out=None
                memory_records.append(dict(arm=arm,batch=batch_size,passed=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                           peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,update_seconds=timings,loss=loss))
            except torch.cuda.OutOfMemoryError:
                success=False
                memory_records.append(dict(arm=arm,batch=batch_size,passed=False,reason='CUDA_OOM'))
            finally:
                del model,optimizer,out,d
                torch.cuda.empty_cache()
            record('batch_memory',memory_records)
        if success:
            records['supported_batch']=batch_size
            break
    assert 'supported_batch' in records
    records.update(passed=True,finished=time.time(),new_model_accuracy=False)
    atomic(target,records)
    print('PREFLIGHT_PASSED',records['supported_batch'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['main','dense'],required=True)
    main(p.parse_args().role)
