"""Official-train-only weak cross-view episodes, not human association labels."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/complementary_fusion_20261021/weak_episodes'
def partition(v):return 'holdout' if int(hashlib.sha256(v.encode()).hexdigest()[:8],16)%5==0 else 'train'
def jaccard(a,b):
 a,b=set(a),set(b);return len(a&b)/len(a|b) if a|b else 0.
def relevance(a,b):
 # Both verb and noun overlap required; harmonic mean avoids one-factor matches.
 v=jaccard(a['verbs'],b['verbs']);n=jaccard(a['nouns'],b['nouns']);return 2*v*n/(v+n) if v+n else 0.
def main():
 OUT.mkdir(parents=True,exist_ok=True);ip=ROOT/'experiments/complementary_fusion_20261021/eligible_training_clips.json';rows=json.loads(ip.read_text());bykey=defaultdict(list)
 for r in rows:bykey[tuple(r['key'])].append(r)
 records=[];conflicts=0
 for k,rs in sorted(bykey.items()):
  signatures={(r['view'],tuple(r['verbs']),tuple(r['nouns'])) for r in rs}
  if len(signatures)!=1:conflicts+=1;continue
  r=rs[0]
  if not r['verbs'] or not r['nouns']:continue
  records.append(dict(key=list(k),view=r['view'],verbs=r['verbs'],nouns=r['nouns'],partition=partition(k[0])))
 groups=defaultdict(list);signatures=defaultdict(list)
 for i,r in enumerate(records):
  groups[(r['partition'],r['view'])].append(i);signatures[(r['partition'],r['view'],tuple(r['verbs']),tuple(r['nouns']))].append(i)
 reports={};rng=np.random.default_rng(20261021)
 for split in ['train','holdout']:
  episodes=[];labels=[];counts=defaultdict(int)
  for i,r in enumerate(records):
   if r['partition']!=split:continue
   other='exo' if r['view']=='ego' else 'ego';pool=np.array(groups[(split,other)]);positives=signatures[(split,other,tuple(r['verbs']),tuple(r['nouns']))]
   if not positives or len(pool)<20:continue
   # Four independently sampled episodes per query; both views share target definition.
   for repeat in range(4):
    positive=int(rng.choice(positives));candidate_pool=rng.choice(pool,size=min(len(pool),400),replace=False);rel=np.array([relevance(r,records[j]) for j in candidate_pool]);sameverb=np.array([jaccard(r['verbs'],records[j]['verbs'])>0 for j in candidate_pool]);samenoun=np.array([jaccard(r['nouns'],records[j]['nouns'])>0 for j in candidate_pool]);chosen=[positive]
    buckets=[candidate_pool[(rel>0)&(rel<1)],candidate_pool[sameverb&~samenoun],candidate_pool[samenoun&~sameverb],candidate_pool]
    for bi,bucket in enumerate(buckets):
     for j in rng.permutation(bucket):
      if int(j) not in chosen:chosen.append(int(j))
      if len(chosen)>=min(20,1+5*(bi+1)):break
     if len(chosen)>=20:break
    if len(chosen)<20:
     for j in rng.permutation(pool):
      if int(j) not in chosen:chosen.append(int(j))
      if len(chosen)==20:break
    chosen=rng.permutation(chosen[:20]).tolist();weights=np.array([relevance(r,records[j]) for j in chosen],np.float32);assert weights.max()==1;weights/=weights.sum();episodes.append([i]+chosen);labels.append(weights);counts[r['view']]+=1
  a=np.array(episodes,np.int32);p=np.stack(labels);np.savez_compressed(OUT/f'{split}.npz',indices=a,targets=p)
  reports[split]=dict(episodes=len(a),query_view_counts=dict(counts),unique_queries=len(set(a[:,0].tolist())),positive_candidates_mean=float((p>0).sum(-1).mean()))
 # Exact source video separation for ALL query and candidate records.
 trainvideos={r['key'][0] for r in records if r['partition']=='train'};holdvideos={r['key'][0] for r in records if r['partition']=='holdout'};assert not trainvideos&holdvideos
 (OUT/'records.json').write_text(json.dumps(records));(OUT/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=20261021,records=len(records),excluded_conflicting_label_clips=conflicts,reports=reports,supervision='Weak harmonic verb/noun Jaccard, multi-positive soft targets; not benchmark ground truth.',video_disjoint_internal_split=True,official_val_test_labels_used=False),indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
