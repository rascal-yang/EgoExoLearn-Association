"""Inventory supervision without inventing unique cross-view ground-truth pairs."""
import csv,json,ast,hashlib
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 ev=defaultdict(list);sources={}
 for split in ['val','test']:
  p=ROOT/f'data/association_annotations/association_{split}_hard.json';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  for r in json.loads(p.read_text()).values():
   for c in [r['query']]+list(r['choices'].values()):ev[c.get('video_uid',c.get('video_id'))].append((float(c['start_sec']),float(c['end_sec'])))
 summary={};allclips=[]
 for view in ['ego','exo']:
  p=ROOT/f'data/association_annotations/{view}_train.csv';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();rows=list(csv.DictReader(p.open()));counts=Counter();sigs=Counter();videos=set()
  for r in rows:
   v=r['video_uid'];start,end=float(r['start_sec']),float(r['end_sec']);overlap=any(max(start,a)<min(end,b) for a,b in ev[v]);vs=sorted(set(ast.literal_eval(r['verb'])));ns=sorted(set(ast.literal_eval(r['noun'])))
   counts['rows']+=1;counts['interval_disjoint']+=not overlap;counts['video_disjoint']+=not bool(ev[v])
   if not overlap:
    sig=(tuple(vs),tuple(ns));sigs[sig]+=1;videos.add(v);allclips.append(dict(view=view,key=[v,start,end],verbs=vs,nouns=ns,caption=r['narration_en_no_hand_prompt'],video_disjoint=not bool(ev[v]),annotation_id=r['annotation_id']))
  summary[view]=dict(counts,eligible_videos=len(videos),signatures=len(sigs));summary[view]['_sigs']=sigs
 shared=set(summary['ego']['_sigs'])&set(summary['exo']['_sigs']);shared={s for s in shared if s[0] and s[1]}
 detail=dict(shared_nonempty_exact_verb_noun_signatures=len(shared),ego_rows_with_match=sum(summary['ego']['_sigs'][s] for s in shared),exo_rows_with_match=sum(summary['exo']['_sigs'][s] for s in shared),caution='Identical verb/noun sets are weak semantic matches, not human-certified association positives; do not label all other clips negative.')
 for d in summary.values():del d['_sigs']
 out=ROOT/'experiments/complementary_fusion_20261021';out.mkdir(parents=True,exist_ok=True)
 (out/'training_inventory.json').write_text(json.dumps(dict(summary=summary,weak_supervision=detail,sources=sources),indent=2))
 (out/'eligible_training_clips.json').write_text(json.dumps(allclips,ensure_ascii=False))
 print(json.dumps(dict(summary=summary,weak_supervision=detail),indent=2))
if __name__=='__main__':main()
