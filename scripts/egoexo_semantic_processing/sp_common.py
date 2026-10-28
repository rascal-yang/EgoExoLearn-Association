"""Shared paths, immutable protocol and stop handling for the v2 study."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/egoexo_semantic_processing_20261027'
REP = ROOT / 'reports/egoexo_semantic_processing_20261027'
BASE = ROOT / 'experiments/egoexo_textalign_20261025'
MBASE = ROOT / 'experiments/egoexo_mechanism_20261025'
CACHE = ROOT / 'experiments/egoexo_states_20261025'
REL = ROOT / 'experiments/egoexo_relations_20261026'
SSD = Path('/var/tmp/yzj-egoexo-relations-20261026')
PY = ROOT / '.venv-sampling-fixed/bin/python'
ARMS = ('R0', 'R1', 'R2', 'R3', 'R4', 'R5')
SEEDS = (17, 29, 43)
# Do not shadow the historical model/backend/common modules with our filenames.
sys.path.append(str(ROOT / 'scripts/egoexo_textalign'))


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class StudyStop(RuntimeError):
    pass


_last_check = 0.0


def guard(force=False):
    global _last_check
    for marker in ('STOP', 'THERMAL_STOP.json'):
        if (EXP / marker).exists():
            raise StudyStop(f'Study stop marker: {marker}; no automatic restart')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in ('2', '3'):
        raise RuntimeError(f'Only GPU2/3 authorized, got {gpu}')
    if not force and time.monotonic() - _last_check < 3:
        return
    _last_check = time.monotonic()
    value = subprocess.check_output([
        'nvidia-smi', '-i', gpu, '--query-gpu=temperature.gpu',
        '--format=csv,noheader,nounits'], text=True)
    temp = int(value.strip())
    if temp > 83:
        atomic(EXP / 'THERMAL_STOP.json', dict(gpu=gpu, temperature=temp, time=time.time()))
        raise StudyStop(f'GPU{gpu} exceeded83C; checkpoint then stop')


def verify_protocol():
    path = EXP / 'protocol_v2_frozen.json'
    p = read(path)
    assert p['status'] == 'frozen_for_execution'
    overrides={}
    amendment_path=EXP/'runtime_amendment.json'
    if amendment_path.exists():
        amendment=read(amendment_path)
        assert amendment['base_protocol_sha256']==digest(path)
        assert amendment['training_math_unchanged'] and amendment['verification_passed']
        permitted={str(CODE.relative_to(ROOT)/name) for name in (
            'sp_common.py','sp_model.py','sp_data.py','memory_attention.py','train.py','pipeline.py','evaluate.py')}
        for relative,change in amendment['source_overrides'].items():
            assert relative in permitted and change['old_sha256']==p['frozen_sources'][relative],relative
            overrides[relative]=change['new_sha256']
        for relative,expected in amendment['additional_sources'].items():
            assert digest(ROOT/relative)==expected,relative
        p['runtime_amendment']=amendment
        p['runtime_amendment_sha256']=digest(amendment_path)
    for relative, expected in p['frozen_sources'].items():
        assert digest(ROOT / relative) == overrides.get(relative,expected), relative
    for row in p['prefix_cache_source_signatures']:
        stat=Path(row['path']).stat()
        assert stat.st_size==row['bytes'] and stat.st_mtime_ns==row['mtime_ns'],row['path']
    external=p['external_base_checkpoint']
    stat=Path(external['path']).stat()
    assert stat.st_size==external['bytes'] and stat.st_mtime_ns==external['mtime_ns'],'External base checkpoint changed'
    return p


def storage_bytes():
    return sum(p.stat().st_size for folder in (EXP, REP) for p in folder.rglob('*') if p.is_file())


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    # Reserve sufficient space for an additional resumable checkpoint.
    if storage_bytes() + 300 * 2**20 > 12 * 2**30:
        raise StudyStop('New artifact budget would exceed12GiB')
    torch.save(value, tmp)
    tmp.replace(path)
