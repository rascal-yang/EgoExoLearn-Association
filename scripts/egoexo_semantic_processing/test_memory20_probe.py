"""Validate real RGB prefetch on all9 final suffixes, without looking at answers."""
from posthoc_core import *
from rgb_gpu_prefetch import RGBPrefetch,choose_depth,ordered_decode
from itertools import chain


def main():
    setup();data=Data(capacity=16)
    pairs=[(arm,seed) for seed in SEEDS for arm in ('R0','R1','R2')]
    models=[load_model(data,*pair) for pair in pairs]
    ep=read(POST/'test_episodes.json');progress=read(POST/'test/shard0/progress.json')
    begin=progress['completed'];ids=np.arange(len(ep['keys']))[::2][begin:]
    import backend as b
    b.ref.init_worker('egoexo')
    decoded=ordered_decode(decode16,[ep['keys'][int(i)] for i in ids],b.ref.init_worker,workers=6)
    first=[next(decoded) for _ in range(16)];rgb=np.stack([v[0] for v in first]);pts=np.stack([v[1] for v in first])
    torch.cuda.reset_peak_memory_stats()
    def encode(image,frame_pts):
        valid=mask_from_pts(frame_pts);prefix=prefix_from_rgb(models[0].visual,image).half().float()
        return np.stack([suffix(model,prefix,valid).cpu().numpy() for model in models])
    with torch.inference_mode():
        image=torch.as_tensor(rgb,device='cuda');expected=encode(image,pts)
        peak=torch.cuda.max_memory_allocated();batch_bytes=image.numel()*image.element_size();del image
        depth=choose_depth(batch_bytes,peak)
        loader=RGBPrefetch(iter(chain(first,decoded)),[16]*(depth+4),depth)
        try:
            # The entire physical ring is already allocated: this tests peak
            # storage even before the decoder fills all its future batches.
            loader.ready.wait(timeout=3)
            image,actual_pts=loader.get()
            assert np.array_equal(image.cpu().numpy(),rgb) and np.array_equal(actual_pts,pts)
            actual=encode(image,actual_pts)
            assert np.array_equal(expected,actual),float(np.abs(expected-actual).max())
            for _ in range(3):
                other,other_pts=loader.get();assert np.isfinite(encode(other,other_pts)).all()
            resident=int(subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
                  '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            result=dict(passed=True,all9_model_feature_max_abs=0.,input_RGB_bytes_exact=True,PTS_exact=True,
                        batch_size_unchanged=16,precision_unchanged=True,queue_depth=depth,
                        queue_capacity_clips=depth*16,full_physical_ring_allocated=True,reused_ring_three_more_batches=True,native_peak_allocated_mib=peak/2**20,
                        prefetched_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        reserved_mib=torch.cuda.memory_reserved()/2**20,device_memory_used_mib=resident,
                        TEST_answers_or_accuracy_inspected=False,primary_results_written=False)
        finally:
            loader.close();decoded.close()
    atomic(POST/'test_memory20g/ring_revision/probe.json',result);print(result,flush=True)


if __name__=='__main__':main()
