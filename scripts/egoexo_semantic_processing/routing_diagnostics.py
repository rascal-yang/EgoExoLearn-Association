"""CPU analysis of saved routing on the prelocked cohort; no new model selection."""
import ast
from posthoc_core import *


def main():
    ep,rows=diagnostic();queryids=ep['indices'][rows,0];ids=np.unique(queryids)
    cohort=read(REP/'v2_固定256题诊断队列.json');annotation=read(ROOT/'data/association_annotations/association_val_hard.json')
    metadata={r['video_uid']:r for r in read(ROOT/'repos/EgoExoLearn/video_meta_downsampled.json')}
    uids=ep['uids'][rows].astype(str);queries=[annotation[u]['query'] for u in uids]
    direction=np.array([metadata[q['video_uid']]['view'] for q in queries])
    stratum=np.array([metadata[q['video_uid']]['view']+'_'+metadata[q['video_uid']]['task'] for q in queries])
    weights=np.zeros(len(rows))
    for s in cohort['strata']:
        take=stratum==s['direction']+'_'+s['task'];weights[take]=s['population']/take.sum()
    weights/=weights.sum()
    arrays=[];pts=[];locations={}
    for k,gpu in enumerate((2,3)):
        arrays.append(np.load(SSD/f'cache{gpu}_prefix.npy',mmap_mode='r'))
        pts.append(np.load(CACHE/f'cache{gpu}/pts.npy',mmap_mode='r'))
        locations.update({int(i):(k,j) for j,i in enumerate(np.load(CACHE/f'cache{gpu}/ids.npy'))})
    vocab=read(EXP/'route_vocabulary.json');label_lookup={(r['type'],r['label']):i for i,r in enumerate(vocab)}
    train=np.load(BASE/'splits.npz')['train'];frequency=np.load(EXP/'route_labels.npy')[train].sum(0)
    positive_frequency=frequency[frequency>0];cuts=np.quantile(positive_frequency,[1/3,2/3])
    selected={};concepts={};parameters={};stats={};class_confidence={}
    for arm in ('R1','R2','R3','R4','R5'):
        for seed in SEEDS:
            key=f'{arm}_seed{seed}';r=np.load(EXP/f'fits/{arm}/seed{seed}/val_final/routing.npz')
            lookup={int(i):j for j,i in enumerate(r['indices'])}
            saved_selected,saved_concepts=r['selected'],r['concepts']
            selected[key]={int(i):saved_selected[lookup[int(i)]] for i in ids}
            concepts[key]=np.stack([saved_concepts[lookup[int(i)]] for i in queryids])
            ck=torch.load(EXP/f'fits/{arm}/seed{seed}/final.pt',map_location='cpu')['updates']
            parameters[key]={n.removeprefix('router.'):v for n,v in ck.items() if n.startswith('router.') and n in (
                'router.norm.weight','router.norm.bias','router.projection','router.prototypes','router.types')}
            stats[key]={};class_confidence[key]={}
            del ck
    cls=[];frames_used={}
    for count,i in enumerate(ids):
        k,j=locations[int(i)];prefix=np.array(arrays[k][j],dtype=np.float32)
        frame_pts=pts[k][j];first=np.unique(frame_pts,return_index=True)[1];first.sort();frames_used[int(i)]=len(first)
        x=prefix[1:].reshape(16,196,768);x/=np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
        cls.append(prefix[0])
        for key,selection in selected.items():
            indices=selection[int(i)].astype(np.int64)[first]
            feature=x[first[:,None],indices]
            count_evidence=indices.shape[1]
            gram=feature@feature.transpose(0,2,1)
            pair_cos=(gram.sum((1,2))-np.trace(gram,axis1=1,axis2=2))/(count_evidence*(count_evidence-1))
            centered=feature-feature.mean(1,keepdims=True)
            spectrum=np.maximum(np.linalg.eigvalsh(centered@centered.transpose(0,2,1)),0)
            spectrum/=np.maximum(spectrum.sum(-1,keepdims=True),1e-20)
            rank=np.exp(-(spectrum*np.log(np.maximum(spectrum,1e-20))).sum(-1))
            cells=(4*(indices//14)//14)*4+(4*(indices%14)//14)
            coverage=np.array([len(set(row)) for row in cells])
            stats[key][int(i)]=dict(coverage=float(coverage.mean()),pair_cosine=float(pair_cos.mean()),effective_rank=float(rank.mean()),
                normalized_energy_spectrum=spectrum[:,::-1].mean(0),position=np.bincount(indices.ravel(),minlength=196)/indices.size)
        if count%16==0:atomic(POST/'routing_diagnostics_progress.json',dict(completed=count+1,total=len(ids)))
    lookup={int(i):j for j,i in enumerate(ids)};querycls=torch.as_tensor(np.stack(cls)[[lookup[int(i)] for i in queryids]])
    report={};alignments={}
    for arm in ('R1','R2','R3','R4','R5'):
        report[arm]={};arm_stats=[];conf=[]
        for seed in SEEDS:
            key=f'{arm}_seed{seed}';p=parameters[key]
            with torch.inference_mode():
                u=F.normalize(F.layer_norm(querycls,(768,),p['norm.weight'],p['norm.bias'],1e-5)@p['projection'],dim=-1)
                logits=(u@p['prototypes'].T/.07).numpy()
            by_type={};conf_seed={}
            for typ,top in (('verb',2),('noun',4)):
                vi=np.array([j for j,r in enumerate(vocab) if r['type']==typ]);z=logits[:,vi]
                probability=np.exp(z-z.max(1,keepdims=True));probability/=probability.sum(1,keepdims=True)
                conf_seed[typ]=dict(mean_max_probability=float(probability.max(1).mean()),mean_entropy=float(-(probability*np.log(np.maximum(probability,1e-20))).sum(1).mean()))
                valid=[];hits=[];outside=0;missing=0;frequency_groups=collections.defaultdict(lambda:[0,0])
                for q,pred in zip(queries,concepts[key]):
                    raw=ast.literal_eval(q[typ+'s']) if isinstance(q[typ+'s'],str) else q[typ+'s']
                    raw=set(map(int,raw));target={label_lookup[(typ,label)] for label in raw if (typ,label) in label_lookup}
                    outside+=len(raw)-len(target);missing+=not bool(raw)
                    got=target&set(map(int,pred))
                    valid.append(len(target));hits.append(len(got))
                    for label in target:
                        group=('lower','middle','upper')[int(np.searchsorted(cuts,frequency[label],side='right'))]
                        frequency_groups[group][0]+=int(label in got);frequency_groups[group][1]+=1
                valid=np.array(valid);hits=np.array(hits);active=valid>0
                by_type[typ]=dict(topk=top,micro_recall=float(hits.sum()/valid.sum()) if valid.sum() else None,
                    macro_query_recall=float((hits[active]/valid[active]).mean()) if active.any() else None,
                    population_weighted_micro_recall=float((hits*weights).sum()/(valid*weights).sum()) if valid.sum() else None,
                    positive_labels_in_vocab=int(valid.sum()),positive_labels_out_of_vocab=outside,queries_missing_type=int(missing),
                    queries_without_in_vocab_positive=int((~active).sum()),
                    direction={d:dict(hits=int(hits[direction==d].sum()),positive_labels=int(valid[direction==d].sum())) for d in np.unique(direction)},
                    TRAIN_frequency_groups={k:dict(hits=v[0],positive_labels=v[1],recall=v[0]/v[1] if v[1] else None) for k,v in frequency_groups.items()})
            numeric={name:np.array([stats[key][int(i)][name] for i in queryids]) for name in ('coverage','pair_cosine','effective_rank')}
            sm={name:dict(mean=float(a.mean()),population_weighted=float((a*weights).sum())) for name,a in numeric.items()}
            sm['normalized_energy_spectrum']=np.mean([stats[key][int(i)]['normalized_energy_spectrum'] for i in queryids],0).tolist()
            sm['position_probability_14x14']=np.mean([stats[key][int(i)]['position'] for i in queryids],0).reshape(14,14).tolist()
            report[arm][str(seed)]=dict(weak_concept_recall=by_type,prefix_concept_confidence=conf_seed,spatial=sm)
        stable=[]
        for i in queryids:
            k,j=locations[int(i)];first=np.unique(pts[k][j],return_index=True)[1]
            a=[selected[f'{arm}_seed{s}'][int(i)][first] for s in SEEDS]
            for left,right in ((0,1),(0,2),(1,2)):
                stable.extend([len(set(x)&set(y))/len(set(x)|set(y)) for x,y in zip(a[left],a[right])])
        report[arm]['cross_seed_query_evidence_jaccard']=float(np.mean(stable))
    result=dict(cohort_questions=len(rows),unique_query_clips=len(ids),metrics=report,unique_PTS_frames=sum(frames_used.values()),
                rank_definition='exp entropy of normalized squared singular values of centered L2-normalized frozen block9 selected features',
                R5_evidence_count=48,other_evidence_count=32,all_indices_in_same_frame=True,
                source_sha256=digest(Path(__file__)),selection_score_independent=True,
                caveat='Narration-derived weak-label recall is not human object recognition or grounding. Metrics descriptive, no selected significance claims.',human_grounding_evidence=False)
    atomic(REP/'v2_路由诊断.json',result)
    print('ROUTING_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':main()
