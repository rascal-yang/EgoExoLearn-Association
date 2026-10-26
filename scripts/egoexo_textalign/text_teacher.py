import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,torch
from common import ROOT,EXP,CKPT,read,atomic,digest,cool
import backend as b

def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);model=b.load_model();model.requires_grad_(False)
    from models.tokenizer import SimpleTokenizer
    tokenizer=SimpleTokenizer();texts=read(EXP/'texts.json');sot=tokenizer.encoder['<|startoftext|>'];eot=tokenizer.encoder['<|endoftext|>'];features=[];truncated=0
    with torch.inference_mode():
        for begin in range(0,len(texts),128):
            cool();chunk=texts[begin:begin+128];tokens=torch.zeros((len(chunk),77),dtype=torch.long)
            for i,text in enumerate(chunk):
                ids=tokenizer.encode(text);truncated+=len(ids)>75;seq=[sot]+ids[:75]+[eot];tokens[i,:len(seq)]=torch.tensor(seq)
            f=model.encode_text(tokens.cuda()).float();f=torch.nn.functional.normalize(f,dim=-1);features.append(f.cpu().numpy())
            atomic(EXP/'text_progress.json',dict(completed=min(begin+128,len(texts)),total=len(texts)))
    a=np.concatenate(features);assert a.shape==(len(texts),256) and np.isfinite(a).all();np.save(EXP/'text_features.npy',a)
    atomic(EXP/'text_complete.json',dict(passed=True,texts=len(texts),truncated_captions_or_phrases=truncated,eot_preserved=True,checkpoint_sha256=digest(CKPT),features_sha256=digest(EXP/'text_features.npy'),teacher_frozen=True,source_sha256=digest(__file__)))

if __name__=='__main__':main()
