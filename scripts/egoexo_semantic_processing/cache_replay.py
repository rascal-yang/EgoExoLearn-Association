"""Compare freshly decoded RGB block9 against the existing FP16 prefix."""
import numpy as np
import torch
from sp_common import *
from sp_data import Data
from sp_model import SemanticModel
import backend as b


def main():
    guard(True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Data();train=np.load(BASE/'splits.npz')['train'];hold=np.load(REL/'selection_holdout.npy')
    ids=[int(train[0]),int(train[1]),int(hold[0]),int(hold[1])]
    b.ref.init_worker('egoexo')
    model=SemanticModel('R0',17,data.prototypes,data.types).eval().requires_grad_(False)
    capture=[]
    hook=model.visual.blocks[9].register_forward_hook(lambda module,inputs,output:capture.append(output))
    rows=[]
    with torch.inference_mode():
        for i in ids:
            guard();key=data.records[i]['key'];start,end=key[1:]
            times=tuple(start+(t+.5)*(end-start)/16 for t in range(16))
            images,pts,seq=b.decode_views(ROOT/'data/EgoExoLearn/videos'/(key[0]+'.mp4'),[(times,(start,end))])
            image=b.ref.transform(torch.from_numpy(images[list(seq[0])].copy()).float()).unsqueeze(0).cuda()
            model.visual(image);fresh=capture.pop().half().float()
            cached=data.batch([i],training=False)['x'].float()
            relative=float((fresh-cached).norm()/cached.norm())
            maximum=float((fresh-cached).abs().max())
            assert relative<.002,(i,relative,maximum)
            k,j=data.locations[i]
            assert np.array_equal(np.array([pts[t] for t in seq[0]]),data.pts[k][j])
            rows.append(dict(record=i,relative_L2=relative,max_abs=maximum,PTS_exact=True))
    hook.remove()
    # Initial M updates contain only the common frozen-prefix-compatible suffix.
    for seed in SEEDS:
        state=torch.load(MBASE/f'heads/M/seed{seed}/best.pt',map_location='cpu')['updates']
        assert not any(k.startswith(tuple(f'visual.blocks.{j}.' for j in range(10))) for k in state)
    atomic(EXP/'cache_replay.json',dict(passed=True,rows=rows,all_seed_initializers_suffix_only=True,
           cache_FP16_boundary=True,native_full_FP32_lossless=False,TEST_access=False))
    print('CACHE_REPLAY_PASSED',rows,flush=True)


if __name__=='__main__':main()
