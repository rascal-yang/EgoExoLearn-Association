"""Prepare TRAIN-only prototypes and fixed label-bag permutation; no GPU."""
import collections
import numpy as np
from sp_common import *


def main():
    if (EXP / 'protocol_v2_frozen.json').exists():
        raise RuntimeError('Preparation cannot rewrite frozen artifacts')
    records, phrases = read(BASE / 'records.json'), read(BASE / 'phrases.json')
    train = np.load(BASE / 'splits.npz')['train'].astype(int)
    text = np.load(BASE / 'text_features.npy')
    keys = sorted({(phrases[j]['type'], phrases[j]['label']) for i in train for j in records[i]['phrases']})
    vectors, items = [], []
    for kind, label in keys:
        ids = sorted({phrases[j]['text_id'] for i in train for j in records[i]['phrases']
                      if (phrases[j]['type'], phrases[j]['label']) == (kind, label)})
        v = text[ids].astype(np.float32)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        mean = v.mean(0)
        vectors.append(mean / max(np.linalg.norm(mean), 1e-12))
        items.append(dict(type=kind, label=label, text_ids=ids))
    np.save(EXP / 'route_prototypes.npy', np.asarray(vectors, np.float32))
    atomic(EXP / 'route_vocabulary.json', items)
    key_to_id = {key: j for j, key in enumerate(keys)}
    labels = np.zeros((len(records), len(keys)), bool)
    for i, record in enumerate(records):
        for j in record['phrases']:
            key = phrases[j]['type'], phrases[j]['label']
            if key in key_to_id:
                labels[i, key_to_id[key]] = True
    np.save(EXP / 'route_labels.npy', labels)
    verb = np.array([kind == 'verb' for kind, label in keys])
    strata = collections.defaultdict(list)
    for i in train:
        strata[(records[i]['view'], bool(labels[i, verb].any()), bool(labels[i, ~verb].any()))].append(int(i))
    donor = np.arange(len(records), dtype=np.int64)
    rng = np.random.default_rng(20261027)
    reports = []
    for key, ids in sorted(strata.items()):
        ids = np.array(ids, np.int64)
        # Fixed candidate set, selected only on metadata feasibility, never model scores.
        best, best_metric = ids.copy(), (-1., -1.)
        for attempt in range(64):
            candidate = rng.permutation(ids)
            changed = (labels[ids] != labels[candidate]).any(1).mean()
            cross_video = np.mean([records[a]['key'][0] != records[b]['key'][0] for a, b in zip(ids, candidate)])
            metric = (changed, cross_video)
            if metric > best_metric:
                best, best_metric = candidate, metric
        donor[ids] = best
        assert np.array_equal(labels[ids].sum(0), labels[best].sum(0))
        reports.append(dict(stratum=list(key), samples=len(ids), changed_fraction=float(best_metric[0]),
                            different_video_donor_fraction=float(best_metric[1])))
    np.save(EXP / 'route_donors.npy', donor)
    changed = (labels[train] != labels[donor[train]]).any(1)
    active = labels[train].any(1)
    ratio = float(changed[active].mean())
    assert ratio >= .8, ratio
    a, b = labels[train], labels[donor[train]]
    report = dict(passed=True, train_clips=len(train), vocabulary=len(keys),
                  vocabulary_by_type=dict(collections.Counter(k for k, l in keys)),
                  active_clips=int(active.sum()), changed_labeled_bag_fraction=ratio,
                  type_changed_fraction={kind: float((a[:, verb if kind == 'verb' else ~verb] != b[:, verb if kind == 'verb' else ~verb]).any(1).mean()) for kind in ('verb', 'noun')},
                  jaccard_mean=float(((a & b).sum(1) / np.maximum((a | b).sum(1), 1)).mean()),
                  strata=reports, original_M_targets_changed=False, TEST_access=False)
    atomic(EXP / 'preparation.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
