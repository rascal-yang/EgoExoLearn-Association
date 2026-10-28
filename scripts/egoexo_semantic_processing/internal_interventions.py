"""All input clips on the fixed256 cohort, all three frozen R2 final models."""
from posthoc_core import *


def main(seed):
    setup()
    protocol = verify_stage('internal')
    target = POST / f'internal/seed{seed}'
    if (target/'complete.json').exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    ep, rows = diagnostic()
    ids = np.unique(ep['indices'][rows])
    lookup = {int(i): j for j, i in enumerate(ids)}
    index = np.vectorize(lookup.get)(ep['indices'][rows])
    keys = read(BASE/'plan.json')['keys']
    metadata = {r['video_uid']: r for r in read(ROOT/'repos/EgoExoLearn/video_meta_downsampled.json')}
    donors = donor_mapping(ids, keys, metadata)
    atomic(target/'donors.json', donors)
    data = Data(capacity=32)
    model = load_model(data, 'R2', seed)
    donor_cls = {}
    for k, array in enumerate(data.arrays):
        for i in set(donors.values()):
            kk, j = data.locations[i]
            if kk == k:
                donor_cls[i] = np.array(array[j, 0], copy=True)
    features = {c: [] for c in CONDITIONS}
    overlap = {c: [] for c in ('semantic_off', 'random_32_plus_context', 'donor_video_concepts')}
    start = time.time()
    reference = np.load(EXP/f'fits/R2/seed{seed}/val_final/features.npz')
    ref = {int(i): v for i, v in zip(reference['indices'], reference['features'])}
    maximum_error = 0.
    with torch.inference_mode():
        for begin in range(0, len(ids), 32):
            guard()
            batch_ids = ids[begin:begin+32]
            d = data.batch(batch_ids, training=False)
            x, valid = d['x'].float(), d['mask']
            route = model.router(x, valid, 'R2')
            altered = x.clone()
            altered[:, 0] = torch.as_tensor(np.stack([donor_cls[donors[int(i)]] for i in batch_ids]), device='cuda').float()
            routes = dict(native=route,
                          semantic_off=model.router(x, valid, 'R2', mode='content'),
                          random_32_plus_context=make_random_route(model.router, batch_ids),
                          donor_video_concepts=model.router(altered, valid, 'R2'))
            for condition in CONDITIONS:
                r = routes.get(condition, route)
                g = suffix(model, x, valid, r, condition).cpu().numpy()
                features[condition].append(g)
                if condition == 'native':
                    error = float(np.max(np.abs(g - np.stack([ref[int(i)] for i in batch_ids]))))
                    maximum_error = max(maximum_error, error)
                    assert error < 3e-5, ('Native replay mismatch', error)
                if condition in overlap:
                    a = F.one_hot(route['indices'], 196).sum(2).bool()
                    b = F.one_hot(r['indices'], 196).sum(2).bool()
                    jaccard = (a & b).sum(-1).float() / (a | b).sum(-1)
                    overlap[condition].append(jaccard.cpu().numpy())
            atomic(target/'progress.json', dict(completed=min(begin+32, len(ids)), total=len(ids), seed=seed,
                   conditions=len(CONDITIONS), elapsed_s=time.time()-start, pid=os.getpid()))
    scores = {}
    truth = ep['truth'][rows]
    for condition in CONDITIONS:
        f = np.concatenate(features[condition])
        scores[condition] = np.einsum('nd,njd->nj', f[index[:, 0]], f[index[:, 1:]])
    np.savez_compressed(target/'scores.npz', uids=ep['uids'][rows], truth=truth, **scores)
    np.savez_compressed(target/'routing_overlap.npz', ids=ids, **{k: np.concatenate(v) for k, v in overlap.items()})
    atomic(target/'complete.json', dict(passed=True, seed=seed, questions=len(rows), clips=len(ids),
           conditions=CONDITIONS, precision='FP32', intervention_scope='query_and_all_candidates',
           native_feature_max_abs=maximum_error, elapsed_s=time.time()-start,
           stage_protocol_sha256=digest(POST/'internal_protocol.json'), scores_sha256=digest(target/'scores.npz'),
           caveat='Distribution shifts at fixed trained weights; descriptive mechanism evidence, not unique causal necessity.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    main(p.parse_args().seed)
