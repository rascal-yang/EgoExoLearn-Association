"""Isolated, read-only reuse of the completed R0 study."""
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/egoexo_layer_compressibility_20261028'
REP = ROOT / 'reports/egoexo_layer_compressibility_20261028'
OLD_CODE = ROOT / 'scripts/egoexo_semantic_processing'
sys.path.insert(0, str(OLD_CODE))
from sp_common import read, atomic, digest, BASE, PY, SEEDS
import sp_common as old

_last = 0.


def guard(force=False):
    global _last
    if any((EXP / name).exists() for name in ('STOP', 'THERMAL_STOP.json')):
        raise RuntimeError('Layer study stop marker; no automatic restart')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert gpu in ('2', '3'), gpu
    if not force and time.monotonic() - _last < 3:
        return
    _last = time.monotonic()
    temp = int(subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip())
    if temp > 83:
        atomic(EXP / 'THERMAL_STOP.json', dict(gpu=gpu, temperature=temp, time=time.time()))
        raise RuntimeError('GPU exceeded83C; no automatic restart')


def setup(frozen=True):
    import torch
    guard(True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(20 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    old.verify_protocol()
    if frozen:
        return verify()


def verify():
    p = read(EXP / 'protocol_frozen.json')
    assert p['status'] == 'frozen_before_task_scores'
    for relative, sha in p['sources'].items():
        assert digest(ROOT / relative) == sha, relative
    for relative, sha in p['inputs'].items():
        assert digest(ROOT / relative) == sha, relative
    return p


def storage_bytes():
    folders = [old.EXP, old.REP, ROOT/'experiments/egoexo_block10_20261028',
               ROOT/'reports/egoexo_block10_20261028', EXP, REP]
    return sum(p.stat().st_size for folder in folders for p in folder.rglob('*') if p.is_file())


def load_models(seeds=SEEDS):
    from posthoc_core import load_model
    from types import SimpleNamespace
    models = [load_model(SimpleNamespace(prototypes=None, types=None), 'R0', s) for s in seeds]
    # The first ten blocks and embeddings were frozen during all three fits.
    reference = models[0].visual.state_dict()
    import torch
    for model in models[1:]:
        for key, value in model.visual.state_dict().items():
            if key.startswith(('blocks.10.', 'blocks.11.', 'norm.')):
                continue
            assert torch.equal(value, reference[key]), key
    return models


def cohort():
    import numpy as np
    from posthoc_core import diagnostic
    ep, rows = diagnostic()
    return ep, rows, np.unique(ep['indices'][rows])


def conditions():
    return [dict(name='dense', operator='dense', layer=-1, budget=196)] + [
        dict(name=f'{op}_b{layer:02d}_m96', operator=op, layer=layer, budget=96)
        for layer in range(12) for op in ('random', 'similarity')]
