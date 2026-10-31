"""Frozen-checkpoint information interventions; all mutable outputs stay in run_dir."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from info_common import atomic, decode, load_data, perturb, roi_mask, sha

def gpu():
    row=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.free,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    return dict(uuid=row[0].strip(),used_MiB=int(row[1]),free_MiB=int(row[2]),temperature_C=int(row[3]),utilization=int(row[4]))

def guard(out,cfg):
    if (out/'STOP').exists(): raise RuntimeError('STOP marker; no automatic restart')
    r=gpu()
    if r['temperature_C']>cfg['max_temperature_C']: raise RuntimeError('Temperature stop; no automatic restart')
    if r['free_MiB']<2048: raise RuntimeError('GPU headroom exhausted')
    return r

def outputs(models,image,native):
    from ec_model import stem
    x,raw=stem(models[0].visual,image)
    del raw
    for block in models[0].visual.blocks[:2]: x=native(block,x)
    result=[]
    for model in models:
        z=x
        for block in model.visual.blocks[2:]: z=native(block,z)
        g=model.visual.norm(z)[:,0]@model.projection
        result.append(g.float())
    return torch.stack(result)

def verify_sources(a,out,cfg):
    root=Path(a.snapshot)/'workspace/EgoExoLearn-Association'
    protocol=json.loads((root/'experiments/egoexo_training_stability_20261029/protocol_frozen.json').read_text())
    files={}
    for name,expected in protocol['sources'].items():
        actual=sha(root/name);assert actual==expected,(name,actual,expected);files[name]=actual
    initial=root/'checkpoints/egoexo_colleague_best.pt'
    assert sha(initial)==protocol['initial_checkpoint_sha256']
    files['checkpoints/egoexo_colleague_best.pt']=protocol['initial_checkpoint_sha256']
    for arm in cfg['models']:
        folder=root/'experiments/egoexo_training_stability_20261029/fits'/arm/'seed17'
        done=json.loads((folder/'complete.json').read_text())
        assert sha(folder/'final.pt')==done['final_sha256']
        files[str((folder/'final.pt').relative_to(root))]=done['final_sha256']
    for name in ('plan.json','val_episodes.npz'):
        rel='experiments/egoexo_textalign_20261025/'+name
        assert sha(root/rel)==protocol['inputs'][rel]
        files[rel]=sha(root/rel)
    for name in ('portable_encoder.py',):files['migration/'+name]=sha(Path(a.snapshot)/'migration'/name)
    manifest=dict(files=files,python=sys.version,environment={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','av','timm','einops','opencv-python')},
                  paths=vars(a),precision='FP32',TF32=False,device=gpu(),training=False,TEST_access=False)
    atomic(out/'environment.json',manifest)
    return manifest

def decoder_check(a,out):
    root=Path(a.snapshot)/'workspace/EgoExoLearn-Association'
    rows=json.loads((root/'reports/egoexo_embedding_compression_20261029/TRAIN_256对应审计队列.json').read_text())['rows']
    rows=[r for r in rows if r['shard']==0]
    rgb=np.load(Path(a.snapshot)/'assets/engineering_rgb/rgb0.npy',mmap_mode='r')
    pts=np.load(Path(a.snapshot)/'assets/engineering_rgb/pts0.npy')
    checks=[]
    for i in (0,len(rows)//2,len(rows)-1):
        value,times=decode(rows[i]['key'],a.videos,a.snapshot)
        error=float(abs(value['dense16']-rgb[i]).max());ptserr=float(abs(np.array(times['dense16'])-pts[i]).max())
        assert error==0 and ptserr==0,(i,error,ptserr)
        # Independent pixel checks of repeated content, not only tensor shapes.
        assert all(np.array_equal(value['repeat4'][:,4*k],value['repeat4'][:,4*k+j]) for k in range(4) for j in range(4))
        for k in range(4):assert any(np.array_equal(value['repeat4'][:,4*k],value['nested16'][:,j]) for j in range(16))
        frame=value['nested16'][:,0]
        assert np.array_equal(perturb(frame,[30,40,80,100],0),frame)
        blurred=perturb(frame,[30,40,80,100],4)
        outside=roi_mask([30,40,80,100])==0
        assert np.array_equal(blurred[:,outside],frame[:,outside])
        checks.append(dict(index=i,rgb_max_abs=error,pts_max_abs=ptserr,repeated_and_nested_pixel_checks=True,perturbation_boundary=True))
    atomic(out/'decoder_checks.json',checks)

def calibrate(models,base,a,cfg,out,native,lineage_files):
    attempts=[];chosen=None;best=0
    reuse=a.reuse_calibration and (out/'preflight_complete.json').exists()
    if reuse:
        old=json.loads((out/'manifest.json').read_text())
        assert old['files']==lineage_files and old['config_sha256']==sha(out/'config.json')
        assert old['device']['uuid']==gpu()['uuid']
        attempts=json.loads((out/'batch_calibration.json').read_text())['attempts']
    for size in ([] if reuse else cfg['batch_candidates']):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        before=gpu()
        try:
            image=torch.from_numpy(np.repeat(base[None],size,axis=0)).cuda()
            for _ in range(2):value=outputs(models,image,native)
            torch.cuda.synchronize();start=time.monotonic()
            for _ in range(3):value=outputs(models,image,native)
            torch.cuda.synchronize();seconds=(time.monotonic()-start)/3
            stats=gpu();peak=torch.cuda.max_memory_allocated()/2**20
            ok=stats['free_MiB']>=cfg['gpu_memory_headroom_MiB']
            attempt=dict(batch=size,seconds=seconds,clips_per_second=size/seconds,peak_allocated_MiB=peak,resource=stats,eligible=ok)
            if ok and size/seconds>best:chosen=size;best=size/seconds
            del image,value
        except torch.cuda.OutOfMemoryError:
            attempt=dict(batch=size,OOM=True,eligible=False)
            if 'image' in locals():del image
            if 'value' in locals():del value
            gc.collect();torch.cuda.empty_cache()
        attempts.append(attempt);atomic(out/'batch_calibration.json',dict(attempts=attempts,selected_batch=chosen))
        print(json.dumps(dict(event='batch_calibration',**attempt)),flush=True)
        if not attempt['eligible']:break
    eligible=[r for r in attempts if r['eligible']]
    assert eligible,'No batch retained required GPU headroom'
    best=max(r['clips_per_second'] for r in eligible)
    # Timing differences below 2% are treated as a throughput tie. Honor the
    # user's preference for higher useful memory/batching within that tie.
    chosen=max(r['batch'] for r in eligible if r['clips_per_second']>=.98*best)
    atomic(out/'batch_calibration.json',dict(attempts=attempts,selected_batch=chosen,reused_preflight_measurements=reuse,
                                           selection_rule='largest batch within 2% of fastest eligible throughput'))
    gc.collect();torch.cuda.empty_cache()
    # All models are compared to their full archived encoder at the same selected batch.
    image=torch.from_numpy(np.repeat(base[None],chosen,axis=0)).cuda()
    timestamps=torch.arange(16,device='cuda')[None].expand(chosen,-1)
    shared=outputs(models,image,native);checks=[]
    for i,m in enumerate(models):
        full=m(image,timestamps,local=False)['g'];err=float((full-shared[i]).abs().max())
        assert err<1e-5,(cfg['models'][i],err)
        checks.append(dict(model=cfg['models'][i],batch=chosen,shared_vs_full_max_abs=err))
    del image,timestamps,full,shared
    atomic(out/'encoder_checks.json',checks)
    return chosen

def main(a):
    out=Path(a.run_dir);cfg=json.loads((out/'config.json').read_text())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.90)
    manifest=verify_sources(a,out,cfg);decoder_check(a,out)
    sys.path.insert(0,str(Path(a.snapshot)/'migration'))
    sys.path.insert(0,str(Path(a.snapshot)/'workspace/EgoExoLearn-Association/scripts/egoexo_embedding_compression'))
    from portable_encoder import load_encoder
    from ec_model import native
    models=[]
    for arm in cfg['models']:
        model,kwargs,info=load_encoder(a.snapshot,arm=arm,device='cuda:0')
        assert kwargs==dict(frame_pair_mean=False)
        models.append(model.eval().requires_grad_(False))
        print(json.dumps(dict(event='model_loaded',model=arm,resource=gpu())),flush=True)
    common=[n for n in models[0].state_dict() if n.startswith(('visual.patch_embed.','visual.blocks.0.','visual.blocks.1.','visual.ln_pre.')) or n in ('visual.cls_token','visual.pos_embed','visual.temporal_embed')]
    states=[m.state_dict() for m in models]
    assert all(torch.equal(states[0][n],states[1][n]) for n in common)
    del states
    root,ep,plan,ann,meta=load_data(a.snapshot,a.annotations,a.metadata)
    ids=np.unique(ep['indices']);assert len(ids)==8564
    first,_=decode(plan['keys'][int(ids[0])],a.videos,a.snapshot)
    with torch.inference_mode():
        batch=calibrate(models,first['dense16'],a,cfg,out,native,manifest['files'])
    manifest.update(batch=batch,source_sha256=sha(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),config_sha256=sha(out/'config.json'))
    atomic(out/'manifest.json',manifest)
    if a.preflight_only:
        atomic(out/'preflight_complete.json',dict(passed=True,batch=batch));return
    features=out/'features';features.mkdir(exist_ok=True)
    cursor=0
    key=dict(source_sha256=sha(__file__),common_sha256=sha(Path(__file__).with_name('info_common.py')),config_sha256=sha(out/'config.json'),batch=batch,
             archived_files=manifest['files'])
    if (out/'progress.json').exists():
        old=json.loads((out/'progress.json').read_text());assert old['resume_key']==key
        cursor=old['completed_clips']
    arrays={}
    for condition in cfg['conditions']:
        file=features/(condition+'.npy')
        arrays[condition]=np.load(file,mmap_mode='r+') if file.exists() else np.lib.format.open_memmap(file,mode='w+',dtype='float32',shape=(2,len(ids),256))
        assert arrays[condition].shape==(2,len(ids),256)
    np.save(out/'clip_ids.npy',ids)
    started=time.monotonic()
    def get_clip(cid):return decode(plan['keys'][int(cid)],a.videos,a.snapshot)
    def decode_batch(start):
        with ThreadPoolExecutor(max_workers=cfg['decoder_workers']) as pool:return list(pool.map(get_clip,ids[start:start+batch]))
    with ThreadPoolExecutor(max_workers=1) as prefetch,torch.inference_mode():
        pending=prefetch.submit(decode_batch,cursor) if cursor<len(ids) else None
        with (out/'sampling.jsonl').open('a') as sampling:
            while cursor<len(ids):
                stats=guard(out,cfg);current=pending.result();n=len(current);begin=cursor
                next_cursor=cursor+n
                pending=prefetch.submit(decode_batch,next_cursor) if next_cursor<len(ids) else None
                for condition in cfg['conditions']:
                    host=np.stack([r[0][condition] for r in current])
                    if n<batch:host=np.concatenate((host,np.repeat(host[:1],batch-n,axis=0)),axis=0)
                    image=torch.from_numpy(host).pin_memory().to('cuda',non_blocking=True)
                    value=outputs(models,image,native)
                    arrays[condition][:,begin:next_cursor]=value[:,:n].cpu().numpy()
                    arrays[condition].flush()
                    del host,image,value
                for cid,(_,times) in zip(ids[begin:next_cursor],current):sampling.write(json.dumps(dict(clip_id=int(cid),**times))+'\n')
                sampling.flush();cursor=next_cursor
                state=dict(completed_clips=cursor,total_clips=len(ids),batch=batch,elapsed_seconds=time.monotonic()-started,resource=gpu(),resume_key=key,pid=os.getpid())
                atomic(out/'progress.json',state)
                print(json.dumps(dict(event='progress',**state)),flush=True)
                del current
    atomic(out/'features_complete.json',dict(clips=len(ids),conditions=cfg['conditions'],models=cfg['models'],elapsed_seconds=time.monotonic()-started))
    # Region intervention features use the same full batch size and same forward path.
    if (out/'regions_frozen.json').exists():
        from regions import run_regions
        run_regions(a,out,cfg,models,batch,native,outputs,guard)
    atomic(out/'inference_complete.json',dict(passed=True,finished_at=time.time(),resource=gpu()))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run-dir','snapshot','videos','annotations','metadata'):p.add_argument('--'+key,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--reuse-calibration',action='store_true')
    args=p.parse_args()
    try:main(args)
    except BaseException as e:
        atomic(Path(args.run_dir)/'failed.json',dict(type=type(e).__name__,message=str(e),time=time.time()))
        raise
