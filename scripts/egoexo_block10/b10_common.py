"""Independent block10-only experiment, anchored to the immutable v2 study."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
OLD_CODE=ROOT/'scripts/egoexo_semantic_processing'
sys.path.insert(0,str(OLD_CODE))
import sp_common as old_common
from sp_common import BASE,MBASE,CACHE,REL,SSD,PY,SEEDS,StudyStop,read,atomic,digest

OLD_EXP=old_common.EXP
OLD_REP=old_common.REP
EXP=ROOT/'experiments/egoexo_block10_20261028'
REP=ROOT/'reports/egoexo_block10_20261028'
ARMS=('S1','S2')
PROTOCOL=EXP/'protocol_v2_frozen.json'
_last_check=0.


def guard(force=False):
    global _last_check
    for marker in ('STOP','THERMAL_STOP.json'):
        if (EXP/marker).exists():raise StudyStop(f'Block10 study stop marker: {marker}')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in ('2','3'):raise RuntimeError(f'Only GPU2/3 authorized: {gpu}')
    if not force and time.monotonic()-_last_check<3:return
    _last_check=time.monotonic()
    temp=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    if temp>83:
        atomic(EXP/'THERMAL_STOP.json',dict(gpu=gpu,temperature=temp,time=time.time()))
        raise StudyStop(f'GPU{gpu} exceeded83C; no automatic restart')


def verify_protocol():
    old_common.verify_protocol()
    p=read(PROTOCOL)
    assert p['status']=='frozen_for_execution'
    assert p['parent_protocol_sha256']==digest(OLD_EXP/'protocol_v2_frozen.json')
    for relative,sha in p['frozen_sources'].items():assert digest(ROOT/relative)==sha,relative
    assert p['training']['batch']==48 and p['training']['epochs']==4
    assert p['modified_spatial_blocks']==[10] and p['block11']=='native_dense'
    assert p['runtime_amendment']==read(EXP/'runtime_choices.json')
    assert p['runtime_amendment_sha256']==digest(EXP/'runtime_choices.json')
    for row in p['cache_file_signatures']:
        st=Path(row['path']).stat()
        assert st.st_size==row['bytes'] and st.st_mtime_ns==row['mtime_ns'],row['path']
    return p


def storage_bytes():
    return sum(p.stat().st_size for folder in (OLD_EXP,OLD_REP,EXP,REP) for p in folder.rglob('*') if p.is_file())


def atomic_torch_save(path,value):
    import torch
    path=Path(path)
    if storage_bytes()+300*2**20>12*2**30:raise StudyStop('Combined artifact budget would exceed12GiB')
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,tmp);tmp.replace(path)


def load_parent_script(name):
    spec=importlib.util.spec_from_file_location('block10_parent_'+name,OLD_CODE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Only experiment bindings change. Parent source files remain byte-identical.
    for key in ('EXP','REP','CODE','PY','ARMS','SEEDS','guard','verify_protocol','atomic_torch_save'):
        setattr(module,key,globals()[key])
    return module
