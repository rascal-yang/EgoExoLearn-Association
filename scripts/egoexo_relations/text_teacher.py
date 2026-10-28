import numpy as np
import torch
from rel_common import *
import backend

def main():
    guard();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    relation=read(EXP/'relations.json');texts=relation['texts']
    m=backend.load_model().eval().requires_grad_(False)
    from models.tokenizer import SimpleTokenizer
    tokenizer=SimpleTokenizer();sot=tokenizer.encoder['<|startoftext|>'];eot=tokenizer.encoder['<|endoftext|>']
    features=[];truncated=0
    with torch.inference_mode():
        for begin in range(0,len(texts),128):
            guard();chunk=texts[begin:begin+128];tokens=torch.zeros(len(chunk),77,dtype=torch.long)
            for j,text in enumerate(chunk):
                ids=tokenizer.encode(text);truncated+=len(ids)>75;seq=[sot]+ids[:75]+[eot]
                tokens[j,:len(seq)]=torch.tensor(seq)
            z=m.encode_text(tokens.cuda()).float();features.append(torch.nn.functional.normalize(z,dim=-1).cpu().numpy())
    features=np.concatenate(features);assert np.isfinite(features).all()
    np.save(EXP/'relation_text.npy',features)
    atomic(EXP/'text_complete.json',dict(passed=True,rows=len(texts),truncated=truncated,
        relations_sha256=digest(EXP/'relations.json'),features_sha256=digest(EXP/'relation_text.npy'),
        source_sha256=digest(__file__),teacher='Existing frozen EgoExo text tower; no external LLM',human_labels=False))
    print('Relation text features complete',len(texts),flush=True)

if __name__=='__main__':main()
