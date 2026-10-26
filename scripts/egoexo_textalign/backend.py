import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,CKPT
sys.path.insert(0,str(ROOT/'scripts/time_origin_reeval'))
import run as ref
sys.path.insert(0,str(ROOT/'scripts/sampling_corrected'))
from bounded_decode import decode_views

def load_model():return ref.model_load('egoexo',CKPT,None)[0]
def decode(k):
    import numpy as np,torch
    times=tuple(k[1]+(i+.5)*(k[2]-k[1])/4 for i in range(4))
    images,pts,sequences=decode_views(ROOT/'data/EgoExoLearn/videos'/(k[0]+'.mp4'),[(times,(k[1],k[2]))])
    seq=sequences[0];return ref.transform(torch.from_numpy(images[list(seq)].copy()).float()).numpy(),[pts[i] for i in seq]

def tail(visual,x):
    for block in visual.blocks[10:]:
        x=block(x,visual.einops_from_space,visual.einops_to_space,visual.einops_from_time,visual.einops_to_time,time_n=196,space_f=4,use_checkpoint=False)
    return visual.norm(x)
