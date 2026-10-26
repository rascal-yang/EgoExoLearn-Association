from state_common import *
import numpy as np,collections,csv

def main():
 EXP.mkdir(exist_ok=True);REP.mkdir(exist_ok=True);assert not (EXP/'plan.json').exists()
 records=read(BASE/'records.json');split=np.load(BASE/'splits.npz');ep=np.load(BASE/'val_episodes.npz');ids=np.unique(np.concatenate([split['train'],split['holdout'],ep['indices'].ravel()]));labels={};pool=collections.defaultdict(list)
 for i in split['train']:
  d=direction(records[int(i)]['caption'])
  if d:
   labels[str(i)]=dict(object=d[0],direction=d[1],view=records[int(i)]['view']);pool[(d[0],d[1],records[int(i)]['view'])].append(int(i))
 triples={}
 for key,v in labels.items():
  other='exo' if v['view']=='ego' else 'ego';pos=pool[(v['object'],v['direction'],other)];neg=pool[(v['object'],1-v['direction'],other)]
  if pos and neg:triples[key]=dict(positive=pos,negative=neg)
 np.save(EXP/'cache_ids.npy',ids);np.savez(EXP/'splits.npz',train=split['train'],holdout=split['holdout'])
 plan=dict(frames=16,ids=ids.tolist(),cache_shape=[len(ids),16*196+1,768],cache_float16_gib=len(ids)*(16*196+1)*768*2/2**30,epochs=4,anchor_batch=16,max_encoded_batch=48,seeds=list(SEEDS),arms=list(ARMS),train_count=len(split['train']),holdout_count=len(split['holdout']),validation_questions=len(ep['truth']),weak_direction_labels=len(labels),triplet_eligible_anchors=len(triples),label_counts={str(k):len(v) for k,v in pool.items() if v},human_stage_labels=False,TEST_access=False,initializer='Existing M checkpoints, matched seed; frozen prefix shared',stage_definition='Learned monotonic latent progress and soft ordered pooling; no true stage annotations',baseline='Same16frames, shared temporal transformer, mean pooling; same data, objectives and steps; head parameter counts reported',selection='Internal caption holdout only, epoch0 allowed; no validation tuning',training_losses=dict(caption=1,global_phrase=.1,local_mean_phrase=.1,crossview_weak_triplet=.2,direction_and_reverse_proxy=.1),sampling='Correct playback PTS,16 bounded uniform midpoints',cache_plan_source_sha256=digest(BASE/'plan.json'))
 atomic(EXP/'plan.json',plan);atomic(EXP/'direction_labels.json',labels);atomic(EXP/'triplets.json',triples)
 with (REP/'训练方向弱标签待核验.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['记录ID','视角','物体类','方向弱标签','原描述','视频','起点','终点','人工核验'])
  for key,v in labels.items():
   r=records[int(key)];w.writerow([key,v['view'],v['object'],'放置' if v['direction'] else '拿起',r['caption'],r['key'][0],r['key'][1],r['key'][2],'待核验'])
 print({k:v for k,v in plan.items() if k!='ids'},flush=True)
if __name__=='__main__':main()
