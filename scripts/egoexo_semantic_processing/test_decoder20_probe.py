"""Byte/order equivalence and graceful-close check for the decode scheduler."""
from posthoc_core import *
from rgb_gpu_prefetch import ordered_decode


def main():
    import backend as b
    ep=read(POST/'test_episodes.json');begin=read(POST/'test/shard0/progress.json')['completed']
    ids=np.arange(len(ep['keys']))[::2][begin:begin+8]
    keys=[ep['keys'][int(i)] for i in ids]
    old=list(b.ref.bounded_map(decode16,keys,workers=2,initializer=b.ref.init_worker,initargs=('egoexo',)))
    new=list(ordered_decode(decode16,keys,b.ref.init_worker,workers=2))
    assert all(np.array_equal(a[0],c[0]) and np.array_equal(a[1],c[1]) for a,c in zip(old,new))
    partial=ordered_decode(decode16,keys,b.ref.init_worker,workers=2)
    next(partial);partial.close()
    atomic(POST/'test_memory20g/decoder_probe.json',dict(passed=True,clips=8,RGB_bytes_exact=True,PTS_exact=True,order_exact=True,early_close_passed=True))
    print('DECODER_EQUIVALENCE_AND_CLOSE_PASSED',flush=True)


if __name__=='__main__':main()
