"""Fixed video-disjoint review queues, never auto-promoted to ground truth."""
import csv
import collections
import numpy as np
from rel_common import *

def main():
    records=read(BASE/'records.json');rel=read(EXP/'relations.json');s=np.load(BASE/'splits.npz')
    holds=list(map(int,s['holdout']))
    group=lambda i: int(hashlib.sha256(records[i]['key'][0].encode()).hexdigest()[:8],16)%3
    # Locked mechanism-review videos do not select checkpoints for NEW fits.
    # Existing M initialization used the historical holdout; disclose this.
    locked=[i for i in holds if group(i)==0];selection=[i for i in holds if group(i)!=0]
    assert not {records[i]['key'][0] for i in locked}&{records[i]['key'][0] for i in selection}
    np.save(EXP/'selection_holdout.npy',np.array(selection,np.int64))
    rng=np.random.default_rng(20261026)
    report={}
    for name,pool,cap in [('development',selection,100),('locked_review',locked,300)]:
        relations=collections.defaultdict(list)
        for i in pool:
            for ri in rel['labels'].get(str(i),[]):relations[(records[i]['view'],ri)].append(i)
        pairs=[];seen=set()
        for qi in rng.permutation(pool):
            qi=int(qi);other='exo' if records[qi]['view']=='ego' else 'ego'
            options=rel['labels'].get(str(qi),[])
            if not options:continue
            for ri in rng.permutation(options):
                a,o,t=rel['tuples'][int(ri)];positives=relations.get((other,int(ri)),[])
                if not positives:continue
                negatives=[]
                for (view,nri),items in relations.items():
                    if view!=other:continue
                    na,no,nt=rel['tuples'][nri]
                    kind='same_object_other_action' if no==o and na!=a else ('same_action_other_object' if na==a and no!=o else None)
                    if kind and nri not in options:negatives.extend((i,kind) for i in items)
                if not negatives:continue
                pi=int(rng.choice(positives));ni,kind=negatives[int(rng.integers(len(negatives)))];key=(qi,pi,ni)
                if key in seen or pi==ni:continue
                seen.add(key);pairs.append(dict(query=qi,proposed_positive=pi,proposed_negative=int(ni),kind=kind,relation=rel['tuples'][int(ri)],human_verified=False));break
            if len(pairs)>=cap:break
        atomic(REP/f'{name}_candidates.json',dict(pairs=pairs,labels_are_proposals=True,independent_test=False))
        with (REP/f'{name}_待核验.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['序号','类别','查询记录','候选A记录','候选B记录','查询描述','候选A描述','候选B描述','查询视频','查询起止','候选A视频','候选A起止','候选B视频','候选B起止','正确候选/均可/均不可/无法判断','可见证据','人工核验人'])
            for j,row in enumerate(pairs):
                ids=[row['query'],row['proposed_positive'],row['proposed_negative']]
                # Randomize A/B, retaining hidden proposal metadata separately.
                if rng.random()<.5:ids[1],ids[2]=ids[2],ids[1]
                rs=[records[i] for i in ids]
                w.writerow([j,row['kind'],*ids,*[r['caption'] for r in rs],*[value for r in rs for value in [str(ROOT/'data/EgoExoLearn/videos'/(r['key'][0]+'.mp4')),str(r['key'][1:])]],'待核验','',''])
        report[name]=dict(clips=len(pool),videos=len({records[i]['key'][0] for i in pool}),candidate_pairs=len(pairs))
    atomic(EXP/'review_plan.json',dict(passed=True,queues=report,selection_holdout=len(selection),
        prior_initialization_caveat='Existing M was selected using original holdout; these queues are prospective mechanism diagnostics, NOT a never-used independent test',
        official_VAL_access=False,TEST_access=False,human_verification_pending=True))
    print(report,flush=True)

if __name__=='__main__':main()
