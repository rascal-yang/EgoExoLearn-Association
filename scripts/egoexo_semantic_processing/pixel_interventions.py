"""Query-only RGB intervention; all12 layers rerun, candidates unchanged."""
from posthoc_core import *

PIXEL_CONDITIONS = ('clean', 'selected', 'control0', 'control1', 'control2')
MEAN = np.array([122.7709393,116.7460125,104.09373615], np.float32)
STD = np.array([68.5005327,66.6321579,70.32316305], np.float32)


def window_features(frame):
    rgb = frame * STD[:,None,None] + MEAN[:,None,None]
    gray = rgb.mean(0)
    gy, gx = np.gradient(gray)
    energy = gx**2 + gy**2
    centers = [(r,c) for r in range(1,13) for c in range(1,13)]
    values = []
    for r,c in centers:
        sl = (slice((r-1)*16,(r+2)*16), slice((c-1)*16,(c+2)*16))
        values.append((float(gray[sl].mean()),float(energy[sl].mean())))
    values = np.asarray(values)
    quartiles = np.stack([np.minimum(np.searchsorted(np.quantile(values[:,j],[.25,.5,.75]), values[:,j], side='right'),3) for j in (0,1)],1)
    return centers, values, quartiles


def iou(a,b):
    intersection = max(0,3-abs(a[0]-b[0])) * max(0,3-abs(a[1]-b[1]))
    return intersection/(18-intersection)


def choose_windows(image, pts, route, record, seed):
    first = np.unique(pts, return_index=True)[1]
    first.sort()
    times = first[np.rint(np.linspace(0,len(first)-1,min(4,len(first)))).astype(int)]
    all_windows, rows = [], []
    for t in times:
        evidence = route['indices'][0,int(t)].cpu().numpy()
        selected = (int(np.clip(evidence[0]//14,1,12)),int(np.clip(evidence[0]%14,1,12)))
        centers, values, quartiles = window_features(image[:,t])
        target = centers.index(selected)
        candidates = [j for j,c in enumerate(centers) if iou(selected,c) <= .2]
        rng = np.random.default_rng(np.random.SeedSequence([20261027,seed,int(record),int(t)]))
        chosen, levels = [], []
        for _ in range(3):
            pool = [j for j in candidates if j not in chosen and np.array_equal(quartiles[j],quartiles[target])]
            level = 'brightness_and_gradient'
            if not pool:
                pool = [j for j in candidates if j not in chosen and quartiles[j,1] == quartiles[target,1]]
                level = 'gradient_only'
            if not pool:
                pool = [j for j in candidates if j not in chosen]
                level = 'geometry_only'
            assert pool
            chosen.append(int(rng.choice(pool)))
            levels.append(level)
        windows = [selected]+[centers[j] for j in chosen]
        counts = [int(sum(abs(int(e)//14-r)<=1 and abs(int(e)%14-c)<=1 for e in evidence)) for r,c in windows]
        positions = np.flatnonzero(pts == pts[t]).tolist()
        all_windows.append((positions,windows))
        rows.append(dict(frame_index=int(t), PTS=float(pts[t]), repeated_input_positions=positions,
                         centers_patch_rc=windows, control_matching_levels=levels,
                         selected_patches_in_window=counts, appearance_values=values[[target]+chosen].tolist()))
    return all_windows, rows


def blurred_variants(image, windows):
    # Separable Gaussian equals the specified 49x49 sigma8 kernel.
    clean = torch.as_tensor(image,device='cuda')
    variants = clean[None].expand(5,-1,-1,-1,-1).clone()
    coordinate = torch.arange(-24,25,device='cuda',dtype=torch.float32)
    kernel = torch.exp(-coordinate**2/(2*8**2));kernel /= kernel.sum()
    for positions, centers in windows:
        frame = clean[:,positions[0]][None]
        blur = F.conv2d(F.pad(frame,(24,24,0,0),mode='reflect'),kernel[None,None,None,:].expand(3,1,1,49),groups=3)
        blur = F.conv2d(F.pad(blur,(0,0,24,24),mode='reflect'),kernel[None,None,:,None].expand(3,1,49,1),groups=3)[0]
        for variant,(r,c) in enumerate(centers,1):
            ys,xs = slice((r-1)*16,(r+2)*16),slice((c-1)*16,(c+2)*16)
            for t in positions:
                variants[variant,:,t,ys,xs] = blur[:,ys,xs]
    return variants


def main(seed):
    setup();protocol=verify_stage('pixels')
    target=POST/f'pixels/seed{seed}';target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():return
    data=Data(capacity=8)
    models={arm:load_model(data,arm,seed) for arm in ('R0','R1','R2')}
    ep,rows=diagnostic();ids=np.unique(ep['indices'][rows,0]);keys=read(BASE/'plan.json')['keys']
    features={};window_rows=[];start=time.time();decode_errors=[]
    import backend as b
    b.ref.init_worker('egoexo')
    decoded=b.ref.bounded_map(decode16,[keys[int(i)] for i in ids],workers=4,initializer=b.ref.init_worker,initargs=('egoexo',))
    with torch.inference_mode():
        for j,i in enumerate(ids):
            guard()
            image,pts=next(decoded)
            mask=mask_from_pts(pts[None])
            clean_prefix=prefix_from_rgb(models['R2'].visual,torch.as_tensor(image[None],device='cuda'))
            clean_routes={arm:None if arm=='R0' else m.router(clean_prefix,mask,arm) for arm,m in models.items()}
            windows,window_info=choose_windows(image,pts,clean_routes['R2'],int(i),seed)
            variants=blurred_variants(image,windows)
            # Hard ranking can amplify batch-size-dependent FP32 ties. Every
            # RGB variant and router uses exactly the same one-clip path.
            fresh=torch.cat([clean_prefix]+[prefix_from_rgb(models['R2'].visual,variants[v:v+1]) for v in range(1,5)],0)
            masks=mask.expand(5,-1)
            row={}
            for arm,m in models.items():
                adaptive_route=None
                if arm!='R0':
                    routes=[clean_routes[arm]]+[m.router(fresh[v:v+1],mask,arm) for v in range(1,5)]
                    adaptive_route={name:torch.cat([r[name] for r in routes],0) for name in ('assignment','mass')}
                g=suffix(m,fresh,masks,adaptive_route).cpu().numpy()
                row[arm+'_adaptive']=g
                if arm=='R0':
                    row[arm+'_fixed']=g.copy()
                else:
                    r=clean_routes[arm]
                    override=dict(assignment=r['assignment'].reshape(1,16,48,196).expand(5,-1,-1,-1).reshape(80,48,196),
                                  mass=r['mass'].reshape(1,16,48).expand(5,-1,-1).reshape(80,48))
                    row[arm+'_fixed']=suffix(m,fresh,masks,override).cpu().numpy()
                    assert np.max(np.abs(row[arm+'_fixed'][0]-g[0]))<3e-5
            features[int(i)]=row
            window_rows.append(dict(record=int(i),key=keys[int(i)],frames=window_info))
            atomic(target/'progress.json',dict(completed=j+1,total=len(ids),seed=seed,elapsed_s=time.time()-start,pid=os.getpid()))
    decoded.close()
    outputs={}
    query_ids=ep['indices'][rows,0]
    cache_drift={}
    for arm in models:
        f=np.load(EXP/f'fits/{arm}/seed{seed}/val_final/features.npz')
        look={int(i):v for i,v in zip(f['indices'],f['features'])}
        candidates=np.stack([[look[int(i)] for i in r[1:]] for r in ep['indices'][rows]])
        for mode in ('adaptive','fixed'):
            q=np.stack([features[int(i)][arm+'_'+mode] for i in query_ids])
            outputs[arm+'_'+mode]=np.einsum('nvd,njd->nvj',q,candidates)
        clean=np.stack([features[int(i)][arm+'_adaptive'][0] for i in query_ids])
        cached=np.stack([look[int(i)] for i in query_ids])
        cache_drift[arm]=dict(mean_feature_L2=float(np.linalg.norm(clean-cached,axis=1).mean()),max_abs=float(np.abs(clean-cached).max()))
    np.savez_compressed(target/'scores.npz',uids=ep['uids'][rows],truth=ep['truth'][rows],conditions=np.array(PIXEL_CONDITIONS),**outputs)
    atomic(target/'windows.json',window_rows)
    atomic(target/'complete.json',dict(passed=True,seed=seed,questions=len(rows),unique_query_clips=len(ids),
           stage_protocol_sha256=digest(POST/'pixels_protocol.json'),elapsed_s=time.time()-start,
           scores_sha256=digest(target/'scores.npz'),cache_vs_full_encoder_clean=cache_drift,
           full_prefix_recomputed_for_all_variants=True,precision='FP32',candidate_path='unchanged cached FP32 final VAL descriptors',
           repeated_PTS_policy='same blur applied to all copies of each chosen physical frame',human_grounding_evidence=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);main(p.parse_args().seed)
