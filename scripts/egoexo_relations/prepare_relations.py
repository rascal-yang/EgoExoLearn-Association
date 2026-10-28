"""Conservative caption dependency relations; all labels remain UNVERIFIED."""
import collections
import csv
import itertools
import re
from rel_common import *

SKIP_VERBS={'use','start','begin','continue','finish','try','want','need','let','make','get','have','be','do'}
GENERIC={'hand','finger','side','time','way','thing','something','anything','everything','one','part','piece','left','right'}
TARGET_PREPS={'into','onto','to','on','in','inside','over','under'}

def noun_phrase(token):
    quantity_noun=token.lemma_.lower() in {'spoonful','handful','cupful'}
    if (token.pos_ not in ('NOUN','PROPN') and not quantity_noun) or token.lemma_.lower() in GENERIC:return None
    selected=[token]+[c for c in token.children if c.dep_ in ('compound','amod') and c.pos_ in ('NOUN','PROPN','ADJ')]
    # Keep an explicitly stated material in e.g. "spoonful of oil".
    for child in token.children:
        if child.dep_=='prep' and child.lower_=='of':
            for obj in child.children:
                if obj.dep_=='pobj' and obj.pos_ in ('NOUN','PROPN'):
                    selected += [child,obj]+[c for c in obj.children if c.dep_=='compound']
    selected=sorted(set(selected),key=lambda t:t.i)
    lo,hi=selected[0].idx,selected[-1].idx+len(selected[-1])
    if len(selected)>8:return None
    return dict(text=' '.join(t.lemma_.lower() for t in selected),span=[lo,hi],source=token.doc.text[lo:hi])

def parse_document(doc):
    output=[]
    for verb in doc:
        if verb.pos_!='VERB' or verb.lemma_.lower() in SKIP_VERBS or any(c.dep_=='neg' for c in verb.children):continue
        # Never infer an unstated object or resolve "it" by guessing.
        objects=[c for c in verb.children if c.dep_ in ('dobj','obj')]
        objects += [c for obj in list(objects) for c in obj.conjuncts]
        particles=sorted([c for c in verb.children if c.dep_=='prt'],key=lambda x:x.i)
        action=dict(text=' '.join([verb.lemma_.lower()]+[p.lower_ for p in particles]),
                    spans=[[verb.idx,verb.idx+len(verb)]]+[[p.idx,p.idx+len(p)] for p in particles],
                    source=[verb.text]+[p.text for p in particles])
        targets=[]
        for prep in verb.children:
            if prep.dep_=='prep' and prep.lower_ in TARGET_PREPS:
                for obj in prep.children:
                    if obj.dep_=='pobj':
                        val=noun_phrase(obj)
                        if val:
                            val=dict(val,text=prep.lower_+' '+val['text'],preposition_span=[prep.idx,prep.idx+len(prep)])
                            targets.append(val)
        # Multiple prepositional targets are ambiguous: retain action/object only.
        target=targets[0] if len(targets)==1 else None
        for obj in objects:
            value=noun_phrase(obj)
            if not value:continue
            output.append(dict(action=action,object=value,target=target,parser='spacy_direct_object',human_verified=False))
    unique={}
    for r in output:
        sig=(r['action']['text'],r['object']['text'],r['target']['text'] if r['target'] else '')
        unique.setdefault(sig,r)
    return list(unique.values())

def main():
    import numpy as np
    import spacy
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    from spacy.symbols import ORTH,NORM
    # Domain tokenization fix: "stir-fry" must not become action=stir/object=fry.
    for verb in ('stir-fry','stir-frying','stir-fried','deep-fry','deep-frying','deep-fried','pan-fry','pan-frying','pan-fried'):
        nlp.tokenizer.add_special_case(verb,[{ORTH:verb,NORM:verb}])
    checks=parse_document(nlp('Use your right hand to hold the spatula and stir-fry the sliced meat.'))
    assert any(r['action']['text']=='stir-fry' and 'meat' in r['object']['text'] for r in checks)
    assert not any(r['object']['text']=='fry' for r in parse_document(nlp('With my right hand, I start to stir-fry using a wooden spatula.')))
    records=read(BASE/'records.json');splits=np.load(BASE/'splits.npz')
    train=list(map(int,splits['train']));hold=list(map(int,splits['holdout']))
    assert not {records[i]['key'][0] for i in train}&{records[i]['key'][0] for i in hold}
    wanted=sorted(set(train+hold));captions=list(dict.fromkeys(records[i]['caption'] for i in wanted))
    parsed={};start=time.time()
    for j,(caption,doc) in enumerate(zip(captions,nlp.pipe(captions,batch_size=128,n_process=2))):
        parsed[caption]=parse_document(doc)
        if j%250==0:atomic(EXP/'parse_progress.json',dict(completed=j,total=len(captions),elapsed_s=time.time()-start))
    tuples=[];lookup={};evidence={};labels={}
    for i in wanted:
        evidence[str(i)]=parsed[records[i]['caption']]
        labels[str(i)]=[]
        for row in evidence[str(i)]:
            sig=(row['action']['text'],row['object']['text'],row['target']['text'] if row['target'] else '')
            if sig not in lookup:lookup[sig]=len(tuples);tuples.append(list(sig))
            labels[str(i)].append(lookup[sig])
    train_rel={v for i in train for v in labels[str(i)]}
    scrambled={};examples=[]
    # Preserve every role's word multiset. Only use recombinations that actually
    # occur elsewhere in TRAIN, avoiding invented negative relation vocabulary.
    for i in train:
        original=labels[str(i)];rs=[tuples[j] for j in original]
        if len(rs)<2:continue
        for shift in range(1,len(rs)):
            changed=[(rs[(j+shift)%len(rs)][0],r[1],r[2]) for j,r in enumerate(rs)]
            if len(set(changed))!=len(rs) or set(changed)=={tuple(r) for r in rs}:continue
            if any(sig not in lookup or lookup[sig] not in train_rel for sig in changed):continue
            replacement=[lookup[sig] for sig in changed]
            if set(replacement)&set(original):continue
            for role in range(3):
                assert sorted(r[role] for r in rs)==sorted(r[role] for r in changed)
            scrambled[str(i)]=replacement;examples.append(i);break
    texts=[];tid={};role_ids=[]
    for r in tuples:
        row=[]
        for role,value in zip(['action','object','target relation'],r):
            if not value:row.append(-1);continue
            text='the '+role+': '+value
            if text not in tid:tid[text]=len(texts);texts.append(text)
            row.append(tid[text])
        role_ids.append(row)
    stats={}
    for name,ids in [('train',train),('holdout',hold)]:
        stats[name]={view:dict(clips=sum(records[i]['view']==view for i in ids),
            clips_with_relations=sum(records[i]['view']==view and bool(labels[str(i)]) for i in ids),
            clips_with_multiple_relations=sum(records[i]['view']==view and len(labels[str(i)])>1 for i in ids),
            tuples=sum(len(labels[str(i)]) for i in ids if records[i]['view']==view)) for view in ['ego','exo']}
    audit=dict(passed=True,stats=stats,unique_relations=len(tuples),training_relations=len(train_rel),
        scrambled_training_clips=len(scrambled),scrambled_preserves_role_word_multisets=True,
        supervision='Caption parser weak labels; NOT human-certified visibility or cross-view association',
        human_grounding_labels=False,official_VAL_access=False,TEST_access=False,
        parser_version=spacy.__version__,parser_model=nlp.meta,elapsed_s=time.time()-start)
    atomic(EXP/'relations.json',dict(tuples=tuples,labels=labels,scrambled=scrambled,texts=texts,role_text_ids=role_ids,training_relation_ids=sorted(train_rel)))
    atomic(EXP/'relation_evidence.json',evidence);atomic(EXP/'data_audit.json',audit)
    # Development review: train only. Heldout challenge examples are a separate
    # untouched queue, with no automatic label promoted to human verification.
    rng=np.random.default_rng(20261026)
    candidates=[i for i in train if labels[str(i)]]
    chosen=rng.choice(candidates,min(200,len(candidates)),replace=False)
    with (REP/'训练关系解析待核验.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['记录ID','视角','原描述','自动关系','视频','起点','终点','文字解析准确','视频中可见','人工备注'])
        for i in chosen:
            r=records[int(i)];w.writerow([i,r['view'],r['caption'],json.dumps([tuples[j] for j in labels[str(i)]],ensure_ascii=False),r['key'][0],r['key'][1],r['key'][2],'待核验','待核验',''])
    atomic(EXP/'parse_complete.json',dict(passed=True,relations_sha256=digest(EXP/'relations.json'),human_review_pending=True))
    print(json.dumps({k:v for k,v in audit.items() if k!='parser_model'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
