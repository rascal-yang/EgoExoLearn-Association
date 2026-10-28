"""Resumable clip-sharded FP32 inference from newly decoded RGB."""
import argparse
import math
import time
import numpy as np
import torch
from lc_model import *
from posthoc_core import decode16
from rgb_gpu_prefetch import ordered_decode
import backend as backend


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    p = setup()
    shard = args.shard
    assert shard in (0, 1)
    folder = EXP/f'scan/shard{shard}'
    folder.mkdir(parents=True, exist_ok=True)
    protocol_sha = digest(EXP/'protocol_frozen.json')
    if (folder/'complete.json').exists():
        done = read(folder/'complete.json')
        assert done['passed'] and done['protocol_sha256'] == protocol_sha
        return
    configs = p['conditions']
    batch = p['scan_batch']
    models = load_models()
    _, _, all_ids = cohort()
    ids_np = all_ids[shard::2]
    keys = read(BASE/'plan.json')['keys']
    shape = (len(configs), len(SEEDS), len(ids_np), 256)
    path = folder/'features.npy'
    if path.exists():
        output = np.lib.format.open_memmap(path, mode='r+')
        assert output.shape == shape
        progress = read(folder/'progress.json')
        assert progress['protocol_sha256'] == protocol_sha
        offset = progress['clips_done']
    else:
        assert storage_bytes()+int(np.prod(shape))*4+100*2**20 < 12*2**30
        output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
        np.save(folder/'ids.npy', ids_np)
        output.flush()
        offset = 0
        atomic(folder/'progress.json', dict(protocol_sha256=protocol_sha, clips_done=0))
    assert np.array_equal(np.load(folder/'ids.npy'), ids_np)
    decoded = ordered_decode(decode16, [keys[int(i)] for i in ids_np[offset:]],
                             backend.ref.init_worker, workers=p['decode_workers_per_gpu'])
    start = time.time()
    initial_offset = offset
    names = {c['name']:i for i,c in enumerate(configs)}
    try:
        while offset < len(ids_np):
            guard()
            n = min(batch, len(ids_np)-offset)
            t0 = time.monotonic()
            items = [next(decoded) for _ in range(n)]
            host = torch.from_numpy(np.stack([v[0] for v in items])).pin_memory()
            image = host.cuda(non_blocking=True)
            ids = torch.as_tensor(ids_np[offset:offset+n], device='cuda')
            decode_seconds = time.monotonic()-t0
            count = 0
            for name, g in shared_scan(models, image, ids, configs):
                guard()
                assert bool(torch.isfinite(g).all()), name
                output[names[name], :, offset:offset+n] = g.cpu().numpy()
                count += 1
            assert count == len(configs)
            torch.cuda.synchronize()
            offset += n
            output.flush()
            elapsed = time.time()-start
            rate = (offset-initial_offset)/max(elapsed, 1e-6)
            status = dict(protocol_sha256=protocol_sha, gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                shard=shard, clips_done=offset, clips_total=len(ids_np), conditions=len(configs),
                seeds=list(SEEDS), batch=batch, seconds_this_run=elapsed,
                clips_per_second_this_run=rate, remaining_seconds_estimate=(len(ids_np)-offset)/max(rate,1e-6),
                last_decode_wait_seconds=decode_seconds, last_batch_seconds=time.monotonic()-t0,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                training=False, TEST_access=False, updated_at=time.time())
            atomic(folder/'progress.json', status)
            print(status, flush=True)
            del image, host, items, ids, g
    finally:
        decoded.close()
    atomic(folder/'complete.json', dict(passed=True, protocol_sha256=protocol_sha,
        features_sha256=digest(path), ids_sha256=digest(folder/'ids.npy'), shape=list(shape),
        seconds_this_run=time.time()-start, training=False, TEST_access=False))


if __name__ == '__main__':
    main()
