"""Single-observation interventions with other query/candidate clips held fixed."""
import json
from pathlib import Path
import numpy as np
import torch

from info_common import atomic,decode,load_data,perturb,sha

def run_regions(a,out,cfg,models,batch,native,outputs,guard):
    frozen=json.loads((out/'regions_frozen.json').read_text());records=frozen['records']
    ids=np.load(out/'clip_ids.npy');ix={int(c):i for i,c in enumerate(ids)}
    full=np.load(out/'features/nested16.npy',mmap_mode='r')
    root,ep,plan,ann,meta=load_data(a.snapshot,a.annotations,a.metadata)
    jobs=[]
    for row in records:
        if 'positions' not in row:continue
        arrays,times=decode(row['key'],a.videos,a.snapshot)
        image=arrays['nested16'];variants=[('clean',image.copy())]
        z=image.copy();old=np.array(times['four']);t=row['target_pts'];nearest=int(abs(old-t).argmin())
        z[:,row['positions']]=arrays['repeat4'][:,4*nearest:4*nearest+1]
        variants.append(('replace_whole',z))
        for sigma in cfg['blur_sigmas']:
            for name,box in [('roi',row['box'])]+[(f'bg{k}',c['box']) for k,c in enumerate(row['controls'])]:
                z=image.copy()
                for pos in row['positions']:z[:,pos]=perturb(image[:,pos],box,sigma)
                variants.append((f'{name}_sigma{sigma}',z))
        for name,z in variants:jobs.append((row['record_id'],name,z))
    result=[];metadata=[]
    with torch.inference_mode():
        for start in range(0,len(jobs),batch):
            guard(out,cfg);chunk=jobs[start:start+batch];host=np.stack([j[2] for j in chunk]);n=len(host)
            if n<batch:host=np.concatenate((host,np.repeat(host[:1],batch-n,axis=0)))
            features=outputs(models,torch.from_numpy(host).cuda(),native)[:,:n].cpu().numpy()
            result.append(features);metadata.extend([dict(record_id=r,name=name) for r,name,_ in chunk])
    values=np.concatenate(result,axis=1);np.save(out/'region_features.npy',values)
    atomic(out/'region_feature_index.json',metadata)
    rows_by_id={r['record_id']:r for r in records};details=[]
    normalized=full/np.linalg.norm(full,axis=-1,keepdims=True)
    for k,job in enumerate(metadata):
        record=rows_by_id[job['record_id']]
        uses=record['uses'] if record['cohort']=='known_case' else [dict(row=record['row'],uid=record['uid'],role='query')]
        for use in uses:
            episode=int(use['row']);truth=int(ep['truth'][episode]);indices=[ix[int(c)] for c in ep['indices'][episode]]
            for mi,model in enumerate(cfg['models']):
                q=normalized[mi,indices[0]];c=normalized[mi,indices[1:]];scores=q@c.T
                f=values[mi,k];f=f/np.linalg.norm(f)
                if use['role']=='query':scores=f@c.T
                else:scores[truth]=q@f
                negative=scores.copy();negative[truth]=-np.inf
                details.append(dict(record_id=record['record_id'],uid=use['uid'],role=use['role'],model=model,condition=job['name'],cohort=record['cohort'],view=record['view'],roi_type=record['type'],matched_analysis_eligible=record['matched_analysis_eligible'],source_video=record['key'][0],scores=scores.tolist(),margin=float(scores[truth]-negative.max()),positive_score=float(scores[truth]),negative_score=float(negative.max()),pred=int(scores.argmax()),truth=truth,correct=bool(scores.argmax()==truth)))
    atomic(out/'region_results.json',dict(regions_sha256=sha(out/'regions_frozen.json'),rows=details))
