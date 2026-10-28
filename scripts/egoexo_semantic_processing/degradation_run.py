"""Frozen three-seed factorial and local approximation error analysis."""
from degradation_core import *
from degradation_prefetch import PrefixRing


def main(seed,smoke=False,prefetch_depth=None):
    setup()
    if not smoke:
        protocol=verify_degradation()
        prefetch_depth=protocol['runtime']['prefetch_depth']
    target=DEG/(f'ring_smoke_seed{seed}' if smoke and prefetch_depth else (f'smoke_seed{seed}' if smoke else f'seed{seed}'))
    target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():return
    data=Data(capacity=32);models={arm:load_model(data,arm,seed) for arm in ('R0','R2')}
    ep,rows=diagnostic();ids=np.unique(ep['indices'][rows])
    if smoke:ids=np.unique(ep['indices'][rows,0])[:128 if prefetch_depth else 32]
    rg=np.load(EXP/f'fits/R2/seed{seed}/val_final/routing.npz')
    saved_indices=rg['selected'];lookup={int(i):j for j,i in enumerate(rg['indices'])}
    selections={int(i):saved_indices[lookup[int(i)]] for i in ids}
    refs={}
    for arm in models:
        a=np.load(EXP/f'fits/{arm}/seed{seed}/val_final/features.npz')
        refs[arm]={int(i):v for i,v in zip(a['indices'],a['features'])}
    features=collections.defaultdict(list);local=collections.defaultdict(list)
    reconstruction_max=0.;replay_max={arm:0. for arm in models};start=time.time()
    query_ids=set(map(int,ep['indices'][rows,0]))
    query_local_ids=[]
    ring=None
    try:
      with torch.inference_mode():
        for begin in range(0,len(ids),32):
            guard();batch_ids=ids[begin:begin+32]
            d=data.batch(batch_ids,training=False) if ring is None else ring.get()
            assert np.array_equal(np.asarray(d['ids']),batch_ids)
            if smoke and ring is not None:
                expected=[]
                for i in batch_ids:
                    k,j=data.locations[int(i)];expected.append(np.array(data.arrays[k][j]))
                assert np.array_equal(d['x'].cpu().numpy(),np.stack(expected))
            x=d['x'].float()
            route=saved_route(np.stack([selections[int(i)] for i in batch_ids]),models['R2'].router.cells)
            if smoke:
                live=models['R2'].router(x,d['mask'],'R2')
                assert torch.equal(live['assignment'],route['assignment']) and torch.equal(live['mass'],route['mass'])
            random_route=make_random_route(models['R2'].router,batch_ids)
            for arm,model in models.items():
                policies=POLICIES if arm=='R0' else POLICIES[:3]
                for policy in policies:
                    compact_blocks=() if policy=='dense' else ((10,) if policy=='compact_block10_only' else ((11,) if policy=='compact_block11_only' else (10,11)))
                    r=random_route if policy=='random_memory' else route
                    out,_=crossed_forward(model,models,x,r,compact_blocks)
                    for norm_arm,g in out.items():
                        name=f'blocks_{arm}__readout_{norm_arm}__{policy}'
                        values=g.cpu().numpy();features[name].append(values)
                        if norm_arm==arm and ((arm=='R0' and policy=='dense') or (arm=='R2' and policy=='r2_memory')):
                            error=float(np.abs(values-np.stack([refs[arm][int(i)] for i in batch_ids])).max())
                            replay_max[arm]=max(replay_max[arm],error)
                            assert error<3e-5,(arm,policy,error)
            positions=[j for j,i in enumerate(batch_ids) if int(i) in query_ids]
            if smoke and not positions:positions=list(range(min(2,len(batch_ids))))
            if positions:
                qi=[int(batch_ids[j]) for j in positions];query_local_ids.extend(qi)
                xx=x[positions];rr=saved_route(np.stack([selections[i] for i in qi]),models['R2'].router.cells)
                for arm,model in models.items():
                    diag=local_error_diagnostics(model,xx,rr)
                    for layer,values in diag.items():
                        for name,v in values.items():local[f'{arm}__block{layer}__{name}'].append(v)
            atomic(target/'progress.json',dict(completed=min(begin+32,len(ids)),total=len(ids),seed=seed,
                   elapsed_s=time.time()-start,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],replay_max_abs=replay_max,
                   peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,prefetch_depth=prefetch_depth,
                   reserved_mib=torch.cuda.memory_reserved()/2**20))
            del x,d,route,random_route,out
            if positions:del xx,rr,diag
            if ring is None and prefetch_depth and begin+32<len(ids):
                ring=PrefixRing(data,ids[begin+32:],32,prefetch_depth)
    finally:
        if ring is not None:ring.close()
    all_features={k:np.concatenate(v) for k,v in features.items()}
    assert all(np.isfinite(v).all() for v in all_features.values())
    np.savez_compressed(target/'features.npz',ids=ids,**all_features)
    np.savez_compressed(target/'local_errors.npz',ids=np.array(query_local_ids),**{k:np.concatenate(v) for k,v in local.items()})
    if not smoke:
        order={int(i):j for j,i in enumerate(ids)};index=np.vectorize(order.get)(ep['indices'][rows])
        scores={k:np.einsum('nd,njd->nj',v[index[:,0]],v[index[:,1:]]) for k,v in all_features.items()}
        np.savez_compressed(target/'scores.npz',uids=ep['uids'][rows],truth=ep['truth'][rows],**scores)
        old=np.load(POST/f'internal/seed{seed}/scores.npz')
        for current,prior in (('blocks_R2__readout_R2__r2_memory','native'),('blocks_R2__readout_R2__dense','restore_dense_both'),('blocks_R2__readout_R2__random_memory','random_32_plus_context')):
            assert float(np.abs(scores[current]-old[prior]).max())<3e-5,(current,prior)
    atomic(target/'complete.json',dict(passed=True,seed=seed,smoke=smoke,clips=len(ids),conditions=list(all_features),
           local_query_clips=len(query_local_ids),replay_max_abs=replay_max,elapsed_s=time.time()-start,
           stage_protocol_sha256=None if smoke else digest(POST/'degradation_protocol.json'),
           features_sha256=digest(target/'features.npz'),scores_sha256=None if smoke else digest(target/'scores.npz'),
           peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,prefetch_depth=prefetch_depth,TEST_access=False,training=False))
    print('DEGRADATION_JOB_COMPLETE',seed,smoke,replay_max,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--prefetch-depth',type=int);a=p.parse_args();main(a.seed,a.smoke,a.prefetch_depth)
