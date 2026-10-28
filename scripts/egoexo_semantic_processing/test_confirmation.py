"""Once-only fixed final TEST confirmation; prefix shared across all nine models."""
from posthoc_core import *
import signal
from rgb_gpu_prefetch import RGBPrefetch,choose_depth,ordered_decode


def prepare():
    path=POST/'test_episodes.json'
    assert not path.exists()
    annotation=read(ROOT/'data/association_annotations/association_test_hard.json')
    keys=[];lookup={};indices=[]
    def key(c):return (c.get('video_uid',c.get('video_id')),float(c['start_sec']),float(c['end_sec']))
    for q in annotation.values():
        row=[]
        for c in [q['query']]+[q['choices'][str(j)] for j in range(20)]:
            k=key(c)
            if k not in lookup:lookup[k]=len(keys);keys.append(k)
            row.append(lookup[k])
        indices.append(row)
    records=read(BASE/'records.json');train_ids=np.load(BASE/'splits.npz')['train']
    intervals=collections.defaultdict(list)
    for i in train_ids:
        v,a,b=records[int(i)]['key'];intervals[v].append((a,b))
    overlap=sum(any(max(a,s)<min(b,t) for s,t in intervals[v]) for v,a,b in keys)
    atomic(path,dict(uids=list(annotation),truth=[q['answer'] for q in annotation.values()],keys=keys,indices=indices,
                    annotation_sha256=digest(ROOT/'data/association_annotations/association_test_hard.json'),
                    interval_audit=dict(unique_clips=len(keys),shared_train_long_videos=len(set(k[0] for k in keys)&set(intervals)),
                                        positive_duration_overlap_clips=overlap,annotation_times_only=True),
                    caveat='Project has historical TEST exposure. This confirmation is not a pristine blind test.'))
    print('TEST_PREPARED',len(annotation),len(keys),'overlap',overlap,flush=True)


def run(shard):
    setup();protocol=verify_stage('test')
    target=POST/f'test/shard{shard}';target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():return
    ep=read(POST/'test_episodes.json')
    ids=np.arange(len(ep['keys']),dtype=np.int64)[shard::2]
    data=Data(capacity=16)
    combinations=[(arm,seed) for seed in SEEDS for arm in ('R0','R1','R2')]
    models=[load_model(data,*pair) for pair in combinations]
    progress_path=target/'progress.json'
    begin=read(progress_path)['completed'] if progress_path.exists() else 0
    signature=digest(POST/'test_protocol.json')
    if progress_path.exists():assert read(progress_path)['stage_protocol_sha256'] in [signature]+protocol.get('accepted_resume_protocol_sha256',[])
    mode='r+' if begin else 'w+'
    features=np.lib.format.open_memmap(target/'features.npy',mode=mode,dtype='float32',shape=(len(ids),9,256))
    frames=np.lib.format.open_memmap(target/'unique_PTS.npy',mode=mode,dtype='uint8',shape=(len(ids),))
    np.save(target/'ids.npy',ids)
    import backend as b
    b.ref.init_worker('egoexo')
    decoded=ordered_decode(decode16,[ep['keys'][int(i)] for i in ids[begin:]],b.ref.init_worker,workers=6)
    start=time.time()
    prefetch=None;depth=0;stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *args:stopping.__setitem__(0,True))
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for cursor in range(begin,len(ids),16):
                if stopping[0]:raise StudyStop('TEST stopped after the last committed batch')
                guard();count=min(16,len(ids)-cursor)
                if prefetch is None:
                    items=[next(decoded) for _ in range(count)]
                    image=torch.as_tensor(np.stack([v[0] for v in items]),device='cuda')
                    batch_pts=np.stack([v[1] for v in items]);del items
                else:
                    image,batch_pts=prefetch.get()
                mask=mask_from_pts(batch_pts)
                # Same batch16, FP32 computation, and original FP16 boundary.
                prefix=prefix_from_rgb(models[0].visual,image).half().float()
                for j,model in enumerate(models):
                    features[cursor:cursor+count,j]=suffix(model,prefix,mask).cpu().numpy()
                frames[cursor:cursor+count]=mask.sum(1).cpu().numpy()
                features.flush();frames.flush()
                peak=torch.cuda.max_memory_allocated();batch_bytes=image.numel()*image.element_size()
                del image,prefix,mask
                atomic(progress_path,dict(completed=cursor+count,total=len(ids),elapsed_s=time.time()-start,
                       shard=shard,pid=os.getpid(),stage_protocol_sha256=signature,
                       resume_begin=begin,processed_this_session=cursor+count-begin,
                       RGB_GPU_prefetch_depth=depth,allocated_mib=torch.cuda.memory_allocated()/2**20,
                       prefetch_batches_ready=0 if prefetch is None else prefetch.queue.qsize(),
                       reserved_mib=torch.cuda.memory_reserved()/2**20))
                if prefetch is None and cursor+count<len(ids):
                    sizes=[min(16,len(ids)-offset) for offset in range(cursor+count,len(ids),16)]
                    depth=min(choose_depth(batch_bytes,peak),len(sizes))
                    prefetch=RGBPrefetch(decoded,sizes,depth)
                    prefetch.ready.wait(timeout=3)
    finally:
        if prefetch is not None:prefetch.close()
        decoded.close()
    assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=-1),1,atol=1e-5)
    atomic(target/'complete.json',dict(passed=True,shard=shard,clips=len(ids),models=[list(x) for x in combinations],
           elapsed_s=time.time()-start,stage_protocol_sha256=signature,features_sha256=digest(target/'features.npy'),
           precision='FP32 with original matched FP16 boundary after block9',endpoint='epoch4_final',TEST_access=True))


def summarize():
    from summarize import bootstrap
    ep=read(POST/'test_episodes.json');index=np.asarray(ep['indices']);truth=np.asarray(ep['truth'])
    features=np.empty((len(ep['keys']),9,256),np.float32)
    combinations=None
    for shard in (0,1):
        folder=POST/f'test/shard{shard}';done=read(folder/'complete.json');assert done['passed']
        assert digest(folder/'features.npy')==done['features_sha256']
        combinations=done['models']
        features[np.load(folder/'ids.npy')]=np.load(folder/'features.npy',mmap_mode='r')
    assert np.isfinite(features).all()
    annotation=read(ROOT/'data/association_annotations/association_test_hard.json')
    metadata={r['video_uid']:r for r in read(ROOT/'repos/EgoExoLearn/video_meta_downsampled.json')}
    source=np.array([annotation[u]['query'].get('video_uid',annotation[u]['query'].get('video_id')) for u in ep['uids']])
    direction=np.array([metadata[v]['view'] for v in source]);task=np.array([metadata[v]['task'] for v in source])
    correct={arm:[] for arm in ('R0','R1','R2')};scores={};results={}
    for j,(arm,seed) in enumerate(combinations):
        f=features[:,j];score=np.einsum('nd,njd->nj',f[index[:,0]],f[index[:,1:]])
        scores[f'{arm}_seed{seed}']=score;c=score.argmax(1)==truth;correct[arm].append(c)
    for arm in correct:
        c=np.stack(correct[arm]);correct[arm]=c
        results[arm]=dict(per_seed_accuracy=(c.mean(1)*100).tolist(),per_seed_correct=c.sum(1).tolist(),
                          mean_accuracy=float(c.mean()*100),direction={d:float(c[:,direction==d].mean()*100) for d in np.unique(direction)},
                          metadata_task={t:float(c[:,task==t].mean()*100) for t in np.unique(task)})
    comparisons={}
    for arm in ('R0','R1'):
        diff=(correct['R2'].astype(float)-correct[arm]).mean(0)
        interval=np.quantile(bootstrap(diff,source,np.random.default_rng(20261027)),[.0125,.9875])*100
        comparisons[f'R2-{arm}']=dict(gain_pp=float(diff.mean()*100),source_cluster_interval97_5=interval.tolist(),
             per_seed_gain_pp=((correct['R2'].mean(1)-correct[arm].mean(1))*100).tolist())
    np.savez_compressed(POST/'test/scores.npz',uids=np.asarray(ep['uids']),truth=truth,**scores)
    report=dict(endpoint='epoch4_final',questions=len(truth),models=results,comparisons=comparisons,
                training_VAL_protocol_unchanged=True,TEST_used_for_selection=False,interval_audit=ep['interval_audit'],
                precision='FP32 with matched FP16 block9 boundary',historical_TEST_exposure=True,
                caveat='Fixed candidate pool and three fixed seeds; historical TEST exposure; no pristine blind-test claim.',human_grounding_evidence=False)
    atomic(REP/'v2_TEST确认结果.json',report)
    lines=['固定epoch4：2200题TEST确认，三种子均值；历史TEST接触已披露。','']
    for arm,r in results.items():lines.append(f"{arm}: {r['per_seed_accuracy']}%；均值 {r['mean_accuracy']:.3f}%")
    for key,r in comparisons.items():lines.append(f"{key}: {r['gain_pp']:+.3f}pp；97.5%源视频聚类区间 {r['source_cluster_interval97_5']}")
    lines.extend(['','未用TEST选模型/参数；不能宣称从未接触的盲测泛化或人工物体定位。'])
    (REP/'v2_TEST确认结果.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--shard',type=int,choices=(0,1));a=p.parse_args()
    if a.prepare:prepare()
    elif a.summarize:summarize()
    else:assert a.shard is not None;run(a.shard)
