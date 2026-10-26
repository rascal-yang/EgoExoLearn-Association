"""Portable raw-video evaluation for original EgoExo, M and both state heads.

Data and weights must be supplied separately. Historical prefix FP16 rounding is
explicit; this wrapper is not a replacement for archived training protocols.
"""
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['egoexo','M','temporal_mean','ordered_states'],required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--m-checkpoint',type=Path)
    p.add_argument('--state-checkpoint',type=Path)
    p.add_argument('--seed',type=int,choices=[17,29,43],default=17)
    p.add_argument('--frames',type=int,choices=[4,16],required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--video-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--memory-fraction',type=float,default=.16)
    p.add_argument('--score',choices=['cosine','dot'],default='cosine')
    p.add_argument('--prefix-precision',choices=['float16','float32'],default='float16')
    return p.parse_args()

def load_model(args):
    import torch
    from torch import nn
    from torch.nn import functional as F
    os.environ['EGOEXO_BASE_CHECKPOINT']=str(args.base_checkpoint.resolve())
    sys.path.insert(0,str(ROOT/'scripts/egoexo_states'))
    import state_common
    from model import VideoTail
    from state_model import StateModel
    if args.method in ('egoexo','M'):
        model=VideoTail().eval()
        if args.method=='M':
            model.load_state_dict(torch.load(args.m_checkpoint,map_location='cpu')['updates'],strict=False)
    else:
        model=StateModel(args.method,args.seed,m_checkpoint=args.m_checkpoint).eval()
        updates=torch.load(args.state_checkpoint,map_location='cpu')['updates']
        assert set(updates)==set(model.updates()),'State checkpoint does not match architecture'
        model.load_state_dict(updates,strict=False)
        if args.frames==4:
            model.position=nn.Parameter(F.interpolate(model.position.detach().transpose(1,2),size=4,mode='linear',align_corners=False).transpose(1,2).contiguous(),requires_grad=False)
    return model.requires_grad_(False)

def encode(model, video, mask, method, prefix_precision):
    import torch
    from torch.nn import functional as F
    visual=model.visual
    B,_,T,_,_=video.shape
    patch=visual.patch_embed(video.permute(0,2,1,3,4).contiguous()).flatten(2).transpose(1,2).reshape(B,T*196,768)
    x=torch.cat([visual.cls_token.expand(B,-1,-1),patch],1)
    pos=visual.pos_embed[:,1:].repeat(1,T,1)+visual.inflate_temporal_embeds(T).repeat_interleave(196,1)
    x=x+torch.cat([visual.pos_embed[:,:1],pos],1)
    if visual.ln_pre is not None:x=visual.ln_pre(x)
    x=visual.pos_drop(x)
    for layer,block in enumerate(visual.blocks):
        x=block(x,'b (f n) d','(b f) n d','b (f n) d','(b n) f d',time_n=196,space_f=T,use_checkpoint=False)
        if layer==9 and prefix_precision=='float16':x=x.half().float()
    x=visual.norm(x)
    g=x[:,0]@model.projection
    if method in ('egoexo','M'):return g
    patches=(x[:,1:]@model.projection).reshape(B,T,196,256)
    state,_,_=model.summarize(patches.mean(2),mask)
    gate=.5*model.gate_logit.sigmoid()
    return F.normalize((1-gate)*F.normalize(g.float(),dim=-1)+gate*state,dim=-1)

def main():
    args=arguments()
    assert args.batch_size>0 and 0<args.memory_fraction<=1
    required=[args.base_checkpoint,args.annotations,args.video_root]
    if args.method!='egoexo':required.append(args.m_checkpoint)
    if args.method not in ('egoexo','M'):required.append(args.state_checkpoint)
    if any(p is None or not p.exists() for p in required):raise SystemExit('Required data/checkpoint path missing')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    model=load_model(args)
    import backend
    backend.ref.init_worker('egoexo')
    questions=json.loads(args.annotations.read_text())
    def key(c):return (c.get('video_uid',c.get('video_id')),float(c['start_sec']),float(c['end_sec']))
    keys=list(dict.fromkeys(key(c) for q in questions.values() for c in [q['query']]+[q['choices'][str(j)] for j in range(20)]))
    index={k:i for i,k in enumerate(keys)}
    features=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for start in range(0,len(keys),args.batch_size):
            temp=int(subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
            if temp>83:raise RuntimeError('GPU exceeds83C; stop, no automatic restart')
            videos=[];masks=[]
            for uid,lo,hi in keys[start:start+args.batch_size]:
                times=tuple(lo+(i+.5)*(hi-lo)/args.frames for i in range(args.frames))
                rgb,pts,seq=backend.decode_views(args.video_root/(uid+'.mp4'),[(times,(lo,hi))])
                take=seq[0];actual=[pts[i] for i in take]
                videos.append(backend.ref.transform(torch.from_numpy(rgb[list(take)].copy()).float()).numpy())
                masks.append([t not in actual[:i] for i,t in enumerate(actual)])
            video=torch.tensor(np.stack(videos),device='cuda')
            mask=torch.tensor(masks,device='cuda')
            z=encode(model,video,mask,args.method,args.prefix_precision)
            if args.score=='cosine':z=F.normalize(z.float(),dim=-1)
            features.append(z.cpu().numpy())
            print(f'{min(start+args.batch_size,len(keys))}/{len(keys)} clips',flush=True)
    f=np.concatenate(features)
    indices=np.array([[index[key(q['query'])]]+[index[key(q['choices'][str(j)])] for j in range(20)] for q in questions.values()])
    scores=np.einsum('nd,njd->nj',f[indices[:,0]],f[indices[:,1:]])
    truth=np.array([q['answer'] for q in questions.values()])
    np.savez_compressed(args.output,uids=np.array(list(questions)),truth=truth,scores=scores)
    summary=dict(method=args.method,frames=args.frames,seed=args.seed,score=args.score,prefix_precision=args.prefix_precision,
                 questions=len(truth),correct=int((scores.argmax(1)==truth).sum()),accuracy=float((scores.argmax(1)==truth).mean()*100))
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)

if __name__=='__main__':main()
