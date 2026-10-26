import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/egoexo_textalign_20261025'
REP=ROOT/'reports/egoexo_textalign_20261025'
OLD=ROOT/'experiments/egoexo_content_20261025'
CKPT=Path(os.environ.get('EGOEXO_BASE_CHECKPOINT',str(ROOT/'checkpoints/egoexo_base.pt')))
ARMS=('A','B','C','D');SEEDS=(17,29,43)
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
last=0.
def cool():
    global last
    if time.monotonic()-last<5:return
    last=time.monotonic()
    def temp():return int(subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    if temp()>=78:
        import torch
        torch.cuda.synchronize()
        while temp()>72:time.sleep(5)
