"""All VAL scores at the fixed final endpoint and internal-best secondary."""
import argparse
import numpy as np
import torch
from sp_common import *
from sp_data import Data
from sp_model import SemanticModel


def evaluate(arm,seed,endpoint):
    guard(True);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.90)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    verify_protocol();protocol_hash=digest(EXP/'protocol_v2_frozen.json')
    folder=EXP/f'fits/{arm}/seed{seed}'
    complete=read(folder/'complete.json');assert complete['passed'] and complete['protocol_sha256']==protocol_hash
    target=folder/f'val_{endpoint}'
    if (target/'complete.json').exists():return
    target.mkdir(exist_ok=True)
    assert digest(folder/f'{endpoint}.pt')==complete[f'{endpoint}_sha256']
    ck=torch.load(folder/f'{endpoint}.pt',map_location='cpu')
    if endpoint=='final':assert ck['epoch']==4
    # An identical internal-best state has exactly the same deterministic FP32 evaluation.
    # Compare actual tensors, never choose reuse based on accuracy.
    final_dir=folder/'val_final'
    if endpoint=='best' and (final_dir/'complete.json').exists():
        final=torch.load(folder/'final.pt',map_location='cpu')
        identical=set(ck['updates'])==set(final['updates']) and all(torch.equal(v,final['updates'][k]) for k,v in ck['updates'].items())
        del final
        if identical:
            source_complete=read(final_dir/'complete.json')
            assert source_complete['passed'] and source_complete['protocol_sha256']==protocol_hash
            assert digest(final_dir/'scores.npz')==source_complete['scores_sha256']
            for name in ('features.npz','scores.npz'):
                tmp=target/(name+f'.{os.getpid()}.linktmp')
                os.link(final_dir/name,tmp);tmp.replace(target/name)
            result=dict(source_complete,endpoint='best',epoch=ck['epoch'],
                        checkpoint_sha256=digest(folder/'best.pt'),reused_from='val_final_exact_state_tensor_equality')
            atomic(target/'complete.json',result)
            atomic(target/'progress.json',dict(completed=8564,total=8564,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],reused_identical_final=True))
            print(json.dumps(dict(arm=arm,seed=seed,endpoint='best',exact_state_reuse=True)),flush=True)
            return
    data=Data()
    model=SemanticModel(arm,seed,data.prototypes,data.types).eval().requires_grad_(False)
    assert set(ck['updates'])==set(model.updates())
    model.load_state_dict(ck['updates'],strict=False)
    episode=np.load(BASE/'val_episodes.npz')
    ids=np.unique(episode['indices']);lookup={int(i):j for j,i in enumerate(ids)}
    index=np.vectorize(lookup.get)(episode['indices'])
    features=[];indices=[];concepts=[];unique_frames=[]
    with torch.inference_mode():
        for begin in range(0,len(ids),32):
            guard();d=data.batch(ids[begin:begin+32],training=False)
            out=model(d['x'].float(),d['mask'])
            features.append(out['g'].cpu().numpy())
            unique_frames.append(d['mask'].sum(1).cpu().numpy().astype(np.uint8))
            route=out['route']
            if route is not None and endpoint=='final':
                indices.append(route['indices'].cpu().numpy().astype(np.uint8))
                concepts.append(route['concepts'].cpu().numpy().astype(np.uint16))
            atomic(target/'progress.json',dict(completed=min(begin+32,len(ids)),total=len(ids),pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    features=np.concatenate(features)
    assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=-1),1,atol=1e-5)
    scores=np.einsum('nd,njd->nj',features[index[:,0]],features[index[:,1:]])
    truth=episode['truth'];uids=episode['uids'].astype(str)
    assert scores.shape==(800,20)
    np.savez_compressed(target/'features.npz',indices=ids,features=features,unique_PTS_frames=np.concatenate(unique_frames))
    np.savez_compressed(target/'scores.npz',uids=uids,truth=truth,scores=scores)
    if indices:
        np.savez_compressed(target/'routing.npz',indices=ids,selected=np.concatenate(indices),concepts=np.concatenate(concepts))
    correct=int((scores.argmax(1)==truth).sum())
    atomic(target/'complete.json',dict(passed=True,arm=arm,seed=seed,endpoint=endpoint,epoch=ck['epoch'],
           correct=correct,questions=800,accuracy=correct/8,precision='FP32',score='cosine20',
           protocol_sha256=protocol_hash,checkpoint_sha256=digest(folder/f'{endpoint}.pt'),
           scores_sha256=digest(target/'scores.npz'),TEST_access=False))
    print(json.dumps(dict(arm=arm,seed=seed,endpoint=endpoint,correct=correct,accuracy=correct/8)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--endpoint',choices=['final','best'],default='final');args=p.parse_args()
    evaluate(args.arm,args.seed,args.endpoint)
