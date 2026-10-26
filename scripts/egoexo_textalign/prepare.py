import ast,csv,json,re,hashlib,collections
from pathlib import Path
import numpy as np
from common import ROOT,EXP,REP,OLD,CKPT,atomic,digest,read,ARMS,SEEDS

def key(r):return [r.get('video_uid',r.get('video_id')),float(r['start_sec']),float(r['end_sec'])]
def main():
    assert not (EXP/'plan.json').exists()
    records=read(ROOT/'experiments/complementary_fusion_20261021/weak_episodes/records.json')
    bykey=collections.defaultdict(list)
    for view in ('ego','exo'):
        for r in csv.DictReader((ROOT/f'data/association_annotations/{view}_train.csv').open()):bykey[json.dumps(key(r))].append(r)
    tax={}
    for typ in ('verb','noun'):
        rows=list(csv.DictReader((ROOT/f'repos/EgoExoLearn/annotations/narration_{typ}_taxonomy.csv').open()))
        tax[typ]=rows
    texts=[];textids={};phrases=[];phraseids={};prepared=[]
    def textid(text):
        if text not in textids:textids[text]=len(texts);texts.append(text)
        return textids[text]
    for r in records:
        candidates=sorted(bykey[json.dumps(r['key'])],key=lambda x:x['annotation_id']);assert candidates
        raw=candidates[0];caption=raw['narration_en'].strip();assert caption
        r=dict(r,caption=caption,caption_id=textid(caption),annotation_id=raw['annotation_id'],phrases=[])
        for typ,labels in [('verb',r['verbs']),('noun',r['nouns'])]:
            for label in labels:
                row=tax[typ][label]
                try:aliases=ast.literal_eval(row['group'])
                except (ValueError,SyntaxError):aliases=[]
                aliases+= [row['label'].split('_(')[0].replace('_',' ')]
                matches=[]
                for alias in aliases:
                    pattern=r'(?<!\w)'+re.escape(str(alias).replace('_',' '))+r'(?!\w)'
                    m=re.search(pattern,caption,re.I)
                    if m:matches.append(m.group())
                if not matches:continue # No unsupported phrase construction.
                matched=sorted(matches,key=lambda s:(-len(s),s.lower()))[0].lower()
                phrase=('the action: ' if typ=='verb' else 'the object: ')+matched
                sig=(typ,label,phrase)
                if sig not in phraseids:
                    phraseids[sig]=len(phrases);phrases.append(dict(type=typ,label=label,text=phrase,text_id=textid(phrase)))
                r['phrases'].append(phraseids[sig])
        prepared.append(r)
    keys=[r['key'] for r in prepared];assert len({json.dumps(k) for k in keys})==len(keys)
    qs=read(ROOT/'data/association_annotations/association_val_hard.json');valkeys=list(dict.fromkeys(json.dumps(key(c)) for q in qs.values() for c in [q['query']]+list(q['choices'].values())))
    lookup={json.dumps(k):i for i,k in enumerate(keys)}
    for k in valkeys:
        assert k not in lookup,'TRAIN and VAL exact overlap';lookup[k]=len(keys);keys.append(json.loads(k))
    ids=np.array([[lookup[json.dumps(key(c))] for c in [q['query']]+[q['choices'][str(j)] for j in range(20)]] for q in qs.values()],np.int64)
    np.savez_compressed(EXP/'val_episodes.npz',indices=ids,truth=np.array([q['answer'] for q in qs.values()]),uids=np.array(list(qs)))
    rng=np.random.default_rng(20261025);train=np.array([i for i,r in enumerate(prepared) if r['partition']=='train'],np.int64)
    hold=[]
    for view in ('ego','exo'):
        pool=[i for i,r in enumerate(prepared) if r['partition']=='holdout' and r['view']==view];hold.extend(rng.choice(pool,min(1024,len(pool)),replace=False).tolist())
    hold=np.array(sorted(hold),np.int64)
    assert not {prepared[i]['key'][0] for i in train}&{prepared[i]['key'][0] for i in hold}
    # Official validation intervals must not overlap training intervals.
    intervals=collections.defaultdict(list)
    for i in train:intervals[prepared[i]['key'][0]].append(prepared[i]['key'][1:])
    overlap=sum(any(max(a,s)<min(b,t) for s,t in intervals[v]) for v,a,b in [json.loads(k) for k in valkeys]);assert overlap==0
    np.savez_compressed(EXP/'splits.npz',train=train,holdout=hold)
    atomic(EXP/'records.json',prepared);atomic(EXP/'texts.json',texts);atomic(EXP/'phrases.json',phrases)
    atomic(EXP/'plan.json',dict(keys=keys,train_records=len(prepared),remaining={'2':list(range(0,len(keys),2)),'3':list(range(1,len(keys),2))},smoke_indices=[int(i) for i in train[:8]],checkpoint=str(CKPT),checkpoint_sha256=digest(CKPT),arms=list(ARMS),seeds=list(SEEDS),epochs=2,requested_batch=32,video_trainable='blocks10,11 + final norm + 768to256 projection',text_teacher='Frozen existing checkpoint text tower; no new external model',input_frames=4,sampling='Correct playback PTS, bounded uniform4; no time expansion; same fixed frames and spatial transform for all arms',prefix_storage='float16, must pass reconstruction audit before use',object_loss='MIL over non-averaged spatial tokens',action_loss='MIL over four contextual frame tokens',global_phrase_weight=.1,local_phrase_weight=.1,selection='Minimum internal heldout full-caption contrastive loss including epoch0',common_caption_loss='Symmetric normalized video-text contrastive; identical caption multi-positive',D='Only local auxiliary targets permuted across batch; original sentence and global phrase targets remain correct',TEST_access=False))
    audit=dict(train=len(train),holdout=len(hold),train_views=dict(collections.Counter(prepared[i]['view'] for i in train)),records_with_phrases=sum(bool(r['phrases']) for r in prepared),total_records=len(prepared),unique_phrases=len(phrases),train_val_interval_overlap=overlap,train_holdout_video_disjoint=True,external_pretraining_overlap_unresolved=True,initialization='Current wrt epoch2 checkpoint; historical config resume:null and CLIP constructor observed; exact original pretraining lineage not fully proven',human_grounding_labels=False)
    atomic(EXP/'data_audit.json',audit)
    # Separate rule-development and held-out quality review queues, never fabricate human labels.
    chosen=rng.choice(train,240,replace=False)
    with (REP/'短语与可见性待核验.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['用途','视频','起点','终点','原描述','提取短语','物体是否可见','动作是否可判断','短语是否准确','人工备注'])
        for j,i in enumerate(chosen):
            r=prepared[i];w.writerow(['规则核验' if j<120 else '独立质量核验',str(ROOT/'data/EgoExoLearn/videos'/(r['key'][0]+'.mp4')),r['key'][1],r['key'][2],r['caption'],'; '.join(phrases[p]['text'] for p in r['phrases']),'未核验','未核验','未核验',''])
    print(audit,flush=True)

if __name__=='__main__':main()
