import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
EXP=ROOT/'experiments/egoexo_relations_20261026'
REP=ROOT/'reports/egoexo_relations_20261026'
BASE=ROOT/'experiments/egoexo_textalign_20261025'
MBASE=ROOT/'experiments/egoexo_mechanism_20261025'
CACHE=ROOT/'experiments/egoexo_states_20261025'
SSD=Path('/var/tmp/yzj-egoexo-relations-20261026')
PY=ROOT/'.venv-sampling-fixed/bin/python'
ARMS=('A_continue_M','B_slots','C_bound','D_unbound','E_scrambled')
sys.path.insert(0,str(ROOT/'scripts/egoexo_textalign'))

def read(p):return json.loads(Path(p).read_text())
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()
_last=0.
def guard(force=False):
    global _last
    if (EXP/'STOP').exists() or (EXP/'THERMAL_STOP.json').exists():raise RuntimeError('STOP marker; no automatic restart')
    if not force and time.monotonic()-_last<1:return
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu in ('2','3'),gpu
    _last=time.monotonic()
    t=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    if t>83:
        atomic(EXP/'THERMAL_STOP.json',dict(gpu=gpu,temperature=t,time=time.time()))
        raise RuntimeError(f'GPU{gpu} exceeded83C; stop without automatic restart')

def verify_protocol():
    p=read(EXP/'protocol.json')
    for path,sha in p['sources'].items():assert digest(path)==sha,path
    return p
