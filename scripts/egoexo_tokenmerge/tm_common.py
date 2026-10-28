import os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/egoexo_textalign'))
from common import read,atomic,digest,EXP as BASE
EXP=ROOT/'experiments/egoexo_tokenmerge_20261026'
REP=ROOT/'reports/egoexo_tokenmerge_20261026'
MBASE=ROOT/'experiments/egoexo_mechanism_20261025'
DENSE=ROOT/'experiments/egoexo_states_20261025'
SEEDS=(17,29,43)
CONFIGS={'dense16':(196,'dense'),'merge98':(98,'compensated'),'merge49':(49,'compensated'),'haar49':(49,'haar'),'random49':(49,'random'),'exact49':(49,'exact')}
_last=0.
def cool():
 global _last
 if time.monotonic()-_last<1:return
 _last=time.monotonic();g=os.environ.get('CUDA_VISIBLE_DEVICES');assert g in ('2','3'),g
 t=int(subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
 if t>83:raise RuntimeError(f'GPU{g} temperature {t}C exceeds83C; stop without automatic restart')

def decode16(key):
 import numpy as np,torch,backend as b
 times=tuple(key[1]+(i+.5)*(key[2]-key[1])/16 for i in range(16))
 images,pts,seq=b.decode_views(ROOT/'data/EgoExoLearn/videos'/(key[0]+'.mp4'),[(times,(key[1],key[2]))])
 return b.ref.transform(torch.from_numpy(images[list(seq[0])].copy()).float()).numpy(),np.array([pts[i] for i in seq[0]])
