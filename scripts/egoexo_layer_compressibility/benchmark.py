"""Paired native/full-encoder latency; grouping stays inside timed execution."""
import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from lc_model import *
from posthoc_core import decode16
from rgb_gpu_prefetch import ordered_decode
import backend as backend


def measure(model, image, ids, config, repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(repeats):
        output = encode(model, image, ids, config)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)/repeats, (time.perf_counter()-wall)*1000/repeats


@torch.inference_mode()
def main():
    p = setup()
    protocol_sha = digest(EXP/'protocol_frozen.json')
    output_path = EXP/f'benchmark_gpu{os.environ["CUDA_VISIBLE_DEVICES"]}.json'
    model = load_models((17,))[0]
    _, _, all_ids = cohort()
    keys = read(BASE/'plan.json')['keys']
    ids_np = all_ids[:8]
    sample_keys = [keys[int(i)] for i in ids_np]
    decoded = list(ordered_decode(decode16, sample_keys, backend.ref.init_worker, workers=4))
    host = torch.from_numpy(np.stack([v[0] for v in decoded])).pin_memory()
    image = host.cuda()
    ids = torch.as_tensor(ids_np, device='cuda')
    configs = p['conditions']
    result = dict(protocol_sha256=protocol_sha, gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        device=torch.cuda.get_device_name(), torch_version=torch.__version__,
        precision='FP32_TF32_disabled', seed=17, input_ids=ids_np.tolist(),
        timing_includes=['RGB_tensor_stem', 'all12_blocks', 'selector_matching', 'group_aggregation',
                         'mass_correction', 'final_norm_projection_normalization'],
        timing_excludes=['video_decode', 'host_to_device_transfer', 'file_writes'],
        batches={}, completed=False)
    rng = np.random.default_rng(p['timing']['order_seed'])
    for batch in p['timing']['batches']:
        x, index = image[:batch], ids[:batch]
        for _ in range(10):
            encode(model, x, index, configs[0])
        for config in configs:
            guard()
            for _ in range(p['timing']['warmups_per_config']):
                encode(model, x, index, config)
        torch.cuda.synchronize()
        samples = {c['name']:[] for c in configs[1:]}
        for round_id in range(p['timing']['paired_rounds']):
            for idx in rng.permutation(np.arange(1,len(configs))):
                guard()
                c = configs[int(idx)]
                pair = []
                candidate_first = bool(rng.integers(2))
                order = [c,configs[0]] if candidate_first else [configs[0],c]
                for config in order:
                    pair.append(measure(model,x,index,config,p['timing']['repeats_per_measurement']))
                if candidate_first:
                    pair.reverse()
                samples[c['name']].append(dict(round=round_id,
                    baseline_cuda_ms=pair[0][0], candidate_cuda_ms=pair[1][0],
                    baseline_wall_ms=pair[0][1], candidate_wall_ms=pair[1][1],
                    candidate_first=candidate_first))
            result['batches'][str(batch)] = dict(samples=samples, rounds_done=round_id+1)
            atomic(output_path,result)
            print(dict(stage='paired_latency',batch=batch,rounds_done=round_id+1,
                       rounds_total=p['timing']['paired_rounds']),flush=True)
        memory = {}
        for config in configs:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            encode(model,x,index,config)
            torch.cuda.synchronize()
            memory[config['name']] = dict(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                incremental_peak_GiB=(torch.cuda.max_memory_allocated()-base)/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        result['batches'][str(batch)]['memory'] = memory
        atomic(output_path,result)
    # Native module CUDA timing, with the same batch8 input. Hook overhead is
    # reported separately from the paired full-encoder measurement above.
    modules = {'temporal':[], 'spatial':[], 'mlp':[]}
    handles = []
    events = {k:[] for k in modules}
    for block in model.visual.blocks:
        for group, module in [('temporal',block.timeattn),('spatial',block.attn),('mlp',block.mlp)]:
            def begin(m, inputs, group=group):
                event=torch.cuda.Event(enable_timing=True);event.record();events[group].append([event,None])
            def finish(m, inputs, output, group=group):
                event=torch.cuda.Event(enable_timing=True);event.record();events[group][-1][1]=event
            handles.extend([module.register_forward_pre_hook(begin),module.register_forward_hook(finish)])
    for _ in range(3):
        encode(model,image,ids,configs[0])
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    result['native_module_cuda_ms_per_batch8'] = {
        k:sum(a.elapsed_time(b) for a,b in values)/3 for k,values in events.items()}
    # Decode+H2D+encoder measured directly for three predeclared representatives.
    representatives = [configs[0]] + [c for c in configs if c['layer']==5]
    pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'),
        initializer=backend.ref.init_worker, initargs=('egoexo',))
    e2e = {c['name']:[] for c in representatives}
    try:
        for round_id in range(4):
            for j in rng.permutation(3):
                guard()
                c=representatives[int(j)]
                torch.cuda.synchronize()
                wall=time.perf_counter()
                futures=[pool.submit(decode16,k) for k in sample_keys]
                items=[f.result(timeout=180) for f in futures]
                decoded_host=torch.from_numpy(np.stack([v[0] for v in items])).pin_memory()
                decode_done=time.perf_counter()
                rgb=decoded_host.cuda(non_blocking=True)
                torch.cuda.synchronize()
                transfer_done=time.perf_counter()
                encode(model,rgb,ids,c)
                torch.cuda.synchronize()
                end=time.perf_counter()
                if round_id:
                    e2e[c['name']].append(dict(total_ms=(end-wall)*1000,
                        decode_transform_stack_ms=(decode_done-wall)*1000,
                        transfer_ms=(transfer_done-decode_done)*1000,
                        encoder_wall_ms=(end-transfer_done)*1000))
                del rgb, decoded_host, items, futures
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    result['end_to_end_batch8_fixed8clips_warm_video_cache'] = e2e
    result['completed']=True
    atomic(output_path,result)
    print(dict(stage='benchmark_complete',path=str(output_path)),flush=True)


if __name__ == '__main__':
    main()
