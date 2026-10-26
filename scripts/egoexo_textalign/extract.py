import argparse,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from common import ROOT,EXP,CKPT,read,atomic,digest,cool
import backend as b

def main(gpu,smoke):
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu and gpu in ('2','3')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
    plan=read(EXP/'plan.json');assert digest(CKPT)==plan['checkpoint_sha256'];indices=plan['smoke_indices'] if smoke else plan['remaining'][gpu]
    folder=EXP/(('smoke_gpu' if smoke else 'gpu')+gpu);folder.mkdir();n=len(indices)
    sources=[Path(__file__),Path(b.__file__),ROOT/'scripts/sampling_corrected/bounded_decode.py',ROOT/'repos/EgoExoLearn/association_benchmark/models/timesformer.py',ROOT/'repos/EgoExoLearn/association_benchmark/models/retriever.py']
    atomic(folder/'manifest.json',dict(plan_sha256=digest(EXP/'plan.json'),sources={str(p):digest(p) for p in sources},checkpoint_sha256=plan['checkpoint_sha256'],rows=n,dtype='float16 prefix; float32 globals',prefix_boundary='after block9, before block10'))
    prefix=np.lib.format.open_memmap(folder/'prefix.npy',mode='w+',dtype=np.float16,shape=(n,785,768))
    globals_=np.lib.format.open_memmap(folder/'global.npy',mode='w+',dtype=np.float32,shape=(n,256))
    actual=np.lib.format.open_memmap(folder/'pts.npy',mode='w+',dtype=np.float64,shape=(n,4));np.save(folder/'indices.npy',np.array(indices,np.int64))
    b.ref.init_worker('egoexo');model=b.load_model();model.requires_grad_(False)
    captured=[];hook=model.visual.blocks[9].register_forward_hook(lambda m,i,o:captured.append(o))
    keys=[plan['keys'][i] for i in indices];decoded=b.ref.bounded_map(b.decode,keys,workers=8,initializer=b.ref.init_worker,initargs=('egoexo',))
    start=time.time();errors=[]
    try:
        with torch.inference_mode():
            for begin in range(0,n,8):
                cool();chunk=keys[begin:begin+8];items=[next(decoded) for _ in chunk];pixels=torch.from_numpy(np.stack([x[0] for x in items])).cuda();captured.clear();g=model.encode_image(pixels);assert len(captured)==1
                p=captured.pop();assert p.shape[1:]==(785,768)
                if smoke:
                    exact=b.tail(model.visual,p)[:,0]@model.image_projection;torch.testing.assert_close(g,exact,atol=2e-4,rtol=2e-4)
                    reconstructed=b.tail(model.visual,p.half().float())[:,0]@model.image_projection
                    rel=float((reconstructed-g).norm()/g.norm());assert rel<.002,rel;errors.append(rel)
                prefix[begin:begin+len(chunk)]=p.cpu().numpy().astype(np.float16);globals_[begin:begin+len(chunk)]=g.cpu().numpy();actual[begin:begin+len(chunk)]=[x[1] for x in items]
                for k,(_,pts) in zip(chunk,items):assert all(k[1]<=t<k[2] for t in pts)
                if begin%64==0:atomic(folder/'progress.json',dict(pid=os.getpid(),gpu=gpu,completed=begin+len(chunk),total=n,elapsed_s=time.time()-start))
    finally:hook.remove();decoded.close()
    prefix.flush();globals_.flush();actual.flush()
    atomic(folder/'complete.json',dict(passed=True,rows=n,indices_sha256=digest(folder/'indices.npy'),prefix_sha256=digest(folder/'prefix.npy'),global_sha256=digest(folder/'global.npy'),elapsed_s=time.time()-start,quantization_relative_errors=errors,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))
    atomic(folder/'progress.json',dict(pid=os.getpid(),gpu=gpu,completed=n,total=n,complete=True,elapsed_s=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.gpu,a.smoke)
