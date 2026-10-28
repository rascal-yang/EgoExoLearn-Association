import argparse
import numpy as np
import torch
from rel_common import *
from rel_model import RelationModel
from rel_data import Data

def main(arm,seed):
    guard();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.90);verify_protocol()
    folder=EXP/f'fits/{arm}/seed{seed}';done=read(folder/'complete.json')
    assert done['passed'] and digest(folder/'best.pt')==done['checkpoint_sha256']
    if (folder/'evaluation_complete.json').exists():return
    model=RelationModel(arm,seed).eval().requires_grad_(False)
    state=torch.load(folder/'best.pt',map_location='cpu');assert set(state['updates'])==set(model.updates())
    model.load_state_dict(state['updates'],strict=False)
    data=Data(allow_hdd=True);episodes=np.load(BASE/'val_episodes.npz');ids=np.unique(episodes['indices']);lookup={int(i):j for j,i in enumerate(ids)}
    index=np.vectorize(lookup.get)(episodes['indices']);features=[];diagnostics=[];batch=32
    with torch.inference_mode():
        for begin in range(0,len(ids),batch):
            guard();d=data.batch(ids[begin:begin+batch],training=False)
            # FP32 inference matches historical reported scoring precision.
            out=model(d['x'].float(),d['mask'],return_attention=begin==0)
            features.append(out['g'].cpu().numpy())
            if out.get('attention') is not None:
                a=out['attention'];s=out['slots'].float()
                entropy=-(a.clamp_min(1e-12)*a.clamp_min(1e-12).log()).sum(-1).mean()
                sim=torch.nn.functional.normalize(s,dim=-1)@torch.nn.functional.normalize(s,dim=-1).transpose(1,2)
                off=~torch.eye(8,device=s.device,dtype=torch.bool)
                diagnostics.append(dict(attention_entropy=float(entropy),mean_offdiagonal_slot_cosine=float(sim[:,off].mean()),
                                        scope='First32 fixed validation clips; no grounding truth'))
                np.savez_compressed(folder/'attention_sample.npz',indices=ids[:len(a)],attention=a.cpu().numpy())
            if begin%256==0:atomic(folder/'evaluation_progress.json',dict(completed=begin+len(out['g']),total=len(ids),pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    features=np.concatenate(features);assert np.isfinite(features).all()
    assert np.allclose(np.linalg.norm(features,axis=-1),1,atol=1e-5)
    scores=np.einsum('nd,njd->nj',features[index[:,0]],features[index[:,1:]])
    np.savez_compressed(folder/'features.npz',indices=ids,features=features)
    np.savez_compressed(folder/'scores.npz',uids=episodes['uids'],truth=episodes['truth'],scores=scores)
    correct=int((scores.argmax(1)==episodes['truth']).sum())
    atomic(folder/'evaluation_complete.json',dict(passed=True,arm=arm,seed=seed,epoch=state['epoch'],correct=correct,questions=len(scores),
        accuracy=correct/len(scores)*100,score='cosine',frames=16,precision='FP32',diagnostics=diagnostics,
        TEST_access=False,human_grounding_evaluated=False))
    print(arm,seed,correct,len(scores),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args();main(a.arm,a.seed)
