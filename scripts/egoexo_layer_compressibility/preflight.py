"""Numerical and memory gates, without reading question answers or scores."""
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
from lc_model import *
from posthoc_core import decode16
from rgb_gpu_prefetch import ordered_decode
import backend as backend


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    setup(False)
    started = time.time()
    models = load_models()
    _, _, all_ids = cohort()
    ids_np = all_ids[:8]
    keys = read(BASE/'plan.json')['keys']
    decoded = list(ordered_decode(decode16, [keys[int(i)] for i in ids_np], backend.ref.init_worker, workers=4))
    cpu = torch.from_numpy(np.stack([v[0] for v in decoded])).pin_memory()
    image = cpu.cuda()
    ids = torch.as_tensor(ids_np, device='cuda')
    model = models[0]
    capture = []
    handle = model.visual.blocks[0].register_forward_pre_hook(lambda m, inputs: capture.append(inputs[0].clone()))
    native = F.normalize((model.visual(image[:1]) @ model.projection).float(), dim=-1)
    handle.remove()
    x = stem(model.visual, image[:1])
    stem_error = float((capture.pop()-x).abs().max())
    manual = encode(model, image[:1], ids[:1], conditions()[0])
    dense_error = float((native-manual).abs().max())
    assert stem_error == 0 and dense_error <= 2e-6, (stem_error, dense_error)
    identity_errors = []
    for layer, block in enumerate(model.visual.blocks):
        expected = run_block(block, x)
        actual = compressed_block(block, x, ids[:1], 'identity', 196)
        relative = float((expected-actual).norm()/expected.norm())
        identity_errors.append(relative)
        assert relative < 1e-5, (layer, relative)
        x = expected
    invariants = []
    z = stem(model.visual, image[:2])
    for budget in (128, 96, 64, 48):
        for op in ('random', 'similarity'):
            a, m = random_assignment(ids[:2], budget) if op == 'random' else similarity_assignment(z, budget)
            assert a.shape == (32, budget, 196)
            assert torch.equal(a.sum(1), torch.ones_like(a.sum(1)))
            assert torch.equal(m.sum(-1), torch.full_like(m.sum(-1), 196))
            assert bool(((a == 0) | (a == 1)).all())
            again = random_assignment(ids[:1], budget)[0] if op == 'random' else similarity_assignment(z[:1], budget)[0]
            assert torch.equal(a[:16], again), (op, budget, 'batch dependence')
            invariants.append(dict(operator=op, budget=budget, min_mass=float(m.min()), max_mass=float(m.max())))
    # Verify shared-prefix execution against independently executed full encoders.
    replay_errors = {}
    selected = [conditions()[0], conditions()[1], conditions()[12], conditions()[-1]]
    for name, g in shared_scan(models, image[:1], ids[:1], selected):
        config = next(c for c in selected if c['name'] == name)
        direct = torch.stack([encode(m, image[:1], ids[:1], config) for m in models])
        error = float((g-direct).abs().max())
        assert error <= 2e-6, (name, error)
        replay_errors[name] = error
    del x, z, expected, actual, direct, g, a, m, again
    # A complete scan at the intended useful batch, never dummy VRAM padding.
    batch = args.batch
    expanded = image.repeat((batch+7)//8, 1, 1, 1, 1)[:batch].contiguous()
    expanded_ids = ids.repeat((batch+7)//8)[:batch]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    count = 0
    for name, g in shared_scan(models, expanded, expanded_ids, conditions()):
        guard()
        assert bool(torch.isfinite(g).all())
        count += 1
    torch.cuda.synchronize()
    elapsed = time.monotonic()-t0
    result = dict(passed=True, training=False, TEST_access=False, task_scores_computed=False,
        gpu=os.environ['CUDA_VISIBLE_DEVICES'], seconds=time.time()-started,
        stem_max_abs=stem_error, dense_descriptor_max_abs=dense_error,
        identity_block_relative_L2=identity_errors, invariants=invariants,
        shared_vs_full_max_abs=replay_errors, common_first10_weights_equal=True,
        precision='FP32_TF32_disabled', batch=batch, conditions=count,
        complete_scan_batch_seconds=elapsed, clips_per_second=batch/elapsed,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
    atomic(EXP/f'preflight_gpu{os.environ["CUDA_VISIBLE_DEVICES"]}_b{batch}.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
