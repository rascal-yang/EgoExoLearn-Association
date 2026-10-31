"""Independent float64 score/mapping checks and regional numerical controls.

Never writes source results. Use a distinct output directory after inference.
"""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np
from PIL import Image

from info_common import atomic,decode,load_data,perturb,raw_rgb,roi_mask,sha

def unit(x):
    x=np.asarray(x,dtype=np.float64)
    assert np.isfinite(x).all()
    norm=np.sqrt(np.sum(x*x,axis=-1,keepdims=True))
    assert (norm>0).all()
    return x/norm

def episode_scores(f,episode_indices):
    """Explicit float64 per-question matrix multiplication, independent of einsum."""
    f=unit(f)
    return np.stack([np.dot(f[row[1:]],f[row[0]]) for row in episode_indices])

def margins(scores,truth):
    result=[]
    for row,t in zip(scores,truth):
        result.append(float(row[t]-max(row[j] for j in range(len(row)) if j!=t)))
    return np.array(result)

def replay_score(q,c,f,truth,role):
    q=unit(q);c=unit(c);f=unit(f)
    if role=='query':q=f
    elif role=='positive':
        c=c.copy();c[truth]=f
    else:raise ValueError(role)
    return np.dot(c,q)

def self_check():
    f=np.array([[1.,0.],[0.,1.],[1.,1.]])
    sc=episode_scores(f,np.array([[0,1,2]]))
    assert np.allclose(sc,[[0.,1/np.sqrt(2)]])
    q=np.array([1.,0.]);c=np.array([[1.,0.],[0.,1.]]);new=np.array([0.,1.])
    assert np.array_equal(replay_score(q,c,new,0,'query'),[0.,1.])
    assert np.array_equal(replay_score(q,c,new,0,'positive'),[0.,0.])
    assert margins(np.array([[.7,.2,.4]]),np.array([0]))[0]>.29
    print('PASS: independent cosine, query/positive intervention roles, margin')

def main(a):
    import torch
    torch.set_num_threads(2)
    state=json.loads(Path(a.state).read_text());run=Path(state['run_dir'])
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    assert out.resolve()!=run.resolve()
    assert (run/'analysis_complete.json').exists(),'Wait for the original analysis to complete'
    root,ep,plan,ann,meta=load_data(state['snapshot'],state['annotations'],state['metadata'])
    cfg=json.loads((run/'config.json').read_text());record=json.loads((run/'run.json').read_text())
    assert record['status']=='complete','Wait for the supervising process to finish'
    manifest=json.loads((run/'artifact_manifest.json').read_text())
    artifact_checks=[]
    for name,want in manifest.items():
        p=run/name
        assert p.stat().st_size==want['bytes'] and sha(p)==want['sha256'],name
        artifact_checks.append(name)
    for name,key in [('config.json','config_sha256'),('plan.txt','plan_sha256'),('regions_frozen.json','regions_sha256')]:
        assert sha(run/name)==record[key],name
    with tarfile.open(run/'code.tar.gz') as tar:
        for name,member in [('plan.txt','docs/INFORMATION_MECHANISM_20261031.txt'),
                            ('config.json','configs/information_mechanism_20261031.json'),
                            ('regions_frozen.json','provenance/information_mechanism/regions_frozen_20261031.json')]:
            assert tar.extractfile(member).read()==(run/name).read_bytes(),member
    heads=subprocess.check_output(['git','bundle','list-heads',str(run/'code.bundle')],text=True)
    assert record['git_commit'] in heads
    ids=np.load(run/'clip_ids.npy');lookup={int(cid):i for i,cid in enumerate(ids)}
    assert len(ids)==8564 and np.array_equal(ids,np.unique(ep['indices']))
    ix=np.array([[lookup[int(cid)] for cid in row] for row in ep['indices']]);truth=ep['truth']
    assert len(truth)==800 and ix.shape==(800,21)
    for uid,row,t in zip(ep['uids'],ep['indices'],truth):
        data=ann[str(uid)];assert int(t)==int(data['answer'])
        for slot,cid in enumerate(row):
            label=data['query'] if slot==0 else data['choices'][str(slot-1)]
            key=plan['keys'][int(cid)]
            assert key[0]==label['video_uid']
            assert max(abs(key[1]-label['start_sec']),abs(key[2]-label['end_sec']))<1e-6
    sampling=[json.loads(x) for x in (run/'sampling.jsonl').read_text().splitlines()]
    assert len(sampling)==len(ids) and {x['clip_id'] for x in sampling}==set(map(int,ids))
    counts=[]
    for row in sampling:
        key=plan['keys'][row['clip_id']];four=np.array(row['four']);nested=np.array(row['nested16'])
        assert len(four)==4 and len(nested)==16
        assert np.array_equal(np.repeat(four,4),row['repeat4'])
        assert all(np.min(abs(nested-t))<1e-7 for t in four)
        assert all(key[1]<=v<key[2] for cond in ('four','dense16','nested16') for v in row[cond])
        counts.append(dict(clip_id=row['clip_id'],view=meta[key[0]]['view'],
                           unique_four=len(set(four)),unique_nested=len(set(nested)),
                           extra_unique=len(set(nested)-set(four))))
    arrays={c:np.load(run/'features'/f'{c}.npy',mmap_mode='r') for c in cfg['conditions']}
    saved=np.load(run/'scores.npz');independent={};numerics=[]
    assert np.array_equal(saved['uids'],ep['uids']) and np.array_equal(saved['truth'],truth)
    for mi,model in enumerate(cfg['models']):
        order=np.load(root/'experiments/egoexo_training_stability_20261029/fits'/model/'seed17/val_curves/indices.npy')
        assert np.array_equal(order,ids)
        for condition,f in arrays.items():
            assert f.shape==(2,8564,256)
            name=model+'__'+condition;scores=episode_scores(f[mi],ix);independent[name]=scores
            delta=float(abs(scores-saved[name]).max());assert delta<=5e-6,(name,delta)
            numerics.append(dict(condition=name,max_abs_score_error=delta,
                                  float64_pred_changes=int((scores.argmax(1)!=saved[name].argmax(1)).sum())))
        for qcondition,ccondition in [('repeat4','nested16'),('nested16','repeat4')]:
            q=unit(arrays[qcondition][mi]);c=unit(arrays[ccondition][mi])
            scores=np.stack([np.dot(c[row[1:]],q[row[0]]) for row in ix])
            name=model+'__query_'+qcondition+'__candidates_'+ccondition
            delta=float(abs(scores-saved[name]).max());assert delta<=5e-6,(name,delta)
            independent[name]=scores
            numerics.append(dict(condition=name,max_abs_score_error=delta,
                                  float64_pred_changes=int((scores.argmax(1)!=saved[name].argmax(1)).sum())))
    assert set(independent)==set(saved.files)-{'uids','truth'}
    mains=json.loads((run/'main_results.json').read_text());comparisons={};conversions={}
    for measure in ('margin','correct'):
        # Near ties may legitimately flip under float64. Verify reported
        # accuracy from the saved original scores; retain float64 as sensitivity.
        values={n:margins(v,truth) if measure=='margin' else (saved[n].argmax(1)==truth).astype(float) for n,v in independent.items()}
        bn,br=values['P_base__nested16'],values['P_base__repeat4']
        rn,rr=values['P_rich__nested16'],values['P_rich__repeat4']
        for name,contrast in dict(G_base=bn-br,G_rich=rn-rr,S_nested=rn-bn,I=(rn-rr)-(bn-br)).items():
            actual=float(contrast.mean())*(100 if measure=='correct' else 1)
            original=mains['contrasts'][name+'__'+measure]['mean']
            assert abs(actual-original)<1e-5,(name,measure,actual,original)
            comparisons[name+'__'+measure]=dict(independent_mean=actual,original_mean=original,
                                               classification_basis='saved original scores' if measure=='correct' else 'independent float64')
    for model in cfg['models']:
        old=saved[model+'__repeat4'].argmax(1)==truth;new=saved[model+'__nested16'].argmax(1)==truth
        conversions[model]=dict(repaired=int((~old&new).sum()),damaged=int((old&~new).sum()),
                                stable_correct=int((old&new).sum()),stable_wrong=int((~old&~new).sum()))
    frozen=json.loads((run/'regions_frozen.json').read_text());regions=frozen['records']
    index=json.loads((run/'region_feature_index.json').read_text());rlookup={(x['record_id'],x['name']):i for i,x in enumerate(index)}
    assert len(rlookup)==len(index)
    rf=np.load(run/'region_features.npy');assert rf.shape==(2,len(index),256)
    unit(rf)
    replay=[];energy=[];regions_by_id={r['record_id']:r for r in regions}
    for r in regions:
        if 'positions' not in r:continue
        rid=r['record_id'];cid=r['clip_id'];mi_clip=lookup[cid]
        for mi,model in enumerate(cfg['models']):
            current=rf[mi,rlookup[(rid,'clean')]];primary=arrays['nested16'][mi,mi_clip]
            replay.append(dict(record_id=rid,model=model,cohort=r['cohort'],view=r['view'],
                               max_abs_raw=float(abs(current-primary).max()),
                               relative_l2=float(np.linalg.norm(current-primary)/np.linalg.norm(primary)),
                               cosine=float(unit(current)@unit(primary))))
        decoded,times=decode(r['key'],state['videos'],state['snapshot'])
        assert np.array_equal(times['nested16'],r['times']['nested16'])
        positions=np.flatnonzero(abs(np.array(times['nested16'])-r['target_pts'])<1e-7).tolist()
        assert positions==r['positions']
        frame=decoded['nested16'][:,positions[0]]
        source_png=run/'region_sources'/f'{rid:02d}.png'
        assert sha(source_png)==r['frame_sha256']
        assert np.array_equal(raw_rgb(frame),np.array(Image.open(source_png)))
        for name,box in [('roi',r['box'])]+[(f'bg{k}',v['box']) for k,v in enumerate(r['controls'])]:
            mask=roi_mask(box)
            assert mask.sum()==roi_mask(r['box']).sum()
            for sigma in cfg['blur_sigmas']:
                changed=perturb(frame,box,sigma);delta=changed-frame
                assert np.array_equal(changed[:,mask==0],frame[:,mask==0])
                physical=delta*np.array([68.5005327,66.6321579,70.32316305])[:,None,None]
                energy.append(dict(record_id=rid,region=name,sigma=sigma,soft_area=float(mask.sum()),
                                   changed_pixel_count=int((mask>0).sum()),normalized_l2=float(np.linalg.norm(delta)),
                                   rgb_l2=float(np.linalg.norm(physical)),
                                   changed_region_rgb_rms=float(np.sqrt((physical**2).sum()/(3*(mask>0).sum())))))
    results=json.loads((run/'region_results.json').read_text())['rows'];region_score_errors=[];margin_replay=[]
    norm_nested={mi:unit(arrays['nested16'][mi]) for mi in range(2)}
    uid_lookup={str(uid):i for i,uid in enumerate(ep['uids'])}
    for r in results:
        mi=cfg['models'].index(r['model']);j=uid_lookup[r['uid']];episode=ix[j]
        assert r['truth']==int(truth[j])
        assert r['pred']==int(np.asarray(r['scores']).argmax())
        assert r['correct']==bool(r['pred']==r['truth'])
        f=rf[mi,rlookup[(r['record_id'],r['condition'])]];norm=norm_nested[mi]
        scores=replay_score(norm[episode[0]],norm[episode[1:]],f,int(truth[j]),r['role'])
        error=float(abs(scores-r['scores']).max());assert error<=5e-6,error
        margin=margins(scores[None],truth[j:j+1])[0]
        assert abs(margin-r['margin'])<=5e-6
        region_score_errors.append(error)
        if r['condition']=='clean':
            original=margins(independent[r['model']+'__nested16'][j:j+1],truth[j:j+1])[0]
            margin_replay.append(dict(record_id=r['record_id'],uid=r['uid'],role=r['role'],model=r['model'],
                                      clean_margin_error=float(r['margin']-original)))
    stats=dict(passed=True,primary_commit=record['git_commit'],audit_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
               source_artifact_manifest_sha256=sha(run/'artifact_manifest.json'),checked_primary_artifacts=len(artifact_checks),
               mapping_slots=800*21,sampling_clips=len(sampling),score_checks=numerics,contrast_checks=comparisons,
               extra_observations=dict(min_extra=min(x['extra_unique'] for x in counts),mean_extra=float(np.mean([x['extra_unique'] for x in counts])),
                                       clips_without_extra=sum(x['extra_unique']==0 for x in counts)),
               conversions=conversions,region_score_max_abs=max(region_score_errors),
               regional_clean_margin_max_abs=max(abs(x['clean_margin_error']) for x in margin_replay),
               region_coverage=dict(random_total=sum(x['cohort']=='random_query' for x in regions),
                                    random_matched=sum(x['cohort']=='random_query' and x['matched_analysis_eligible'] for x in regions),
                                    matched_by_view=dict(Counter(x['view'] for x in regions if x['cohort']=='random_query' and x['matched_analysis_eligible']))),
               time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic(out/'audit.json',stats);atomic(out/'sampling_audit.json',counts)
    atomic(out/'regional_replay.json',dict(features=replay,margins=margin_replay))
    atomic(out/'perturbation_energy.json',energy)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state');p.add_argument('--output-dir');p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:self_check()
    else:
        assert a.state and a.output_dir
        main(a)
