import os,signal,subprocess,time
from pathlib import Path
import numpy as np
from common import ROOT,EXP,REP,CKPT,ARMS,SEEDS,read,atomic,digest
CODE=Path(__file__).parent;PY=ROOT/'.venv-sampling-fixed/bin/python';children=[]
def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    stat=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,state=stat[0],start_ticks=int(stat[19]),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def launch(script,gpu,args,name):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (EXP/(name+'.log')).open('w') as f:p=subprocess.Popen([str(PY),'-u',str(CODE/script),*args],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(p);return p
def freeze():
    if (EXP/'protocol.json').exists():
        protocol=read(EXP/'protocol.json')
        assert read(EXP/'cache_integrity_audit.json')['passed']
        for path,h in protocol['sources'].items():assert digest(path)==h,path
        assert not (EXP/'heads').exists(), 'Restart requires archiving interrupted fits'
        return
    preflight=read(EXP/'resource_preflight.json');assert preflight['passed']
    plan=read(EXP/'plan.json');assert digest(CKPT)==plan['checkpoint_sha256']
    for gpu in (2,3):
        folder=EXP/f'gpu{gpu}';done=read(folder/'complete.json');assert done['passed']
        assert digest(folder/'prefix.npy')==done['prefix_sha256'];assert digest(folder/'global.npy')==done['global_sha256']
        assert np.load(folder/'indices.npy').tolist()==plan['remaining'][str(gpu)]
    source=list(CODE.glob('*.py'))+[CKPT,EXP/'plan.json',EXP/'records.json',EXP/'phrases.json',EXP/'texts.json',EXP/'text_features.npy',EXP/'splits.npz',EXP/'val_episodes.npz',EXP/'resource_preflight.json',EXP/'data_audit.json']
    for gpu in (2,3):source.extend([EXP/f'gpu{gpu}/manifest.json',EXP/f'gpu{gpu}/complete.json'])
    source+=[ROOT/'scripts/time_origin_reeval/run.py',ROOT/'scripts/sampling_corrected/bounded_decode.py']
    source+=list((ROOT/'repos/EgoExoLearn/association_benchmark/models').glob('*.py'))
    source+=[ROOT/f'data/association_annotations/{v}_train.csv' for v in ('ego','exo')]
    source+=[ROOT/f'repos/EgoExoLearn/annotations/narration_{v}_taxonomy.csv' for v in ('verb','noun')]
    atomic(EXP/'protocol.json',dict(batch=preflight['batch'],arms=list(ARMS),seeds=list(SEEDS),epochs=2,total_fits=12,simultaneous_jobs_per_gpu=2,GPUs=[2,3],primary_validation_score='raw dot; cosine reported separately',checkpoint_selection='minimum fixed internal holdout caption loss, epoch0 included',TEST_evaluated=False,human_labels_pending=True,sources={str(p):digest(p) for p in source}))
    atomic(EXP/'cache_integrity_audit.json',dict(passed=True,all_prefix_hashes_verified=True,rows=len(plan['keys']),training_prefix_precision='float16 storage',reconstruction_relative_error=read(EXP/'smoke_gpu2/complete.json')['quantization_relative_errors']))
def main():
    atomic(EXP/'status.json',dict(phase='waiting_for_extraction',owner=identity(os.getpid()),updated=time.time()))
    while not (EXP/'extraction_status.json').exists() or not read(EXP/'extraction_status.json').get('complete'):
        if (EXP/'extraction_failure.json').exists():raise RuntimeError('Extraction failed')
        status=read(EXP/'extraction_status.json') if (EXP/'extraction_status.json').exists() else {}
        for pid in status.get('pids',[]):
            if identity(pid) is None:
                # Child may have just completed; coordinator is authoritative on next update.
                pass
        atomic(EXP/'status.json',dict(phase='waiting_for_extraction',owner=identity(os.getpid()),updated=time.time()));time.sleep(5)
    atomic(EXP/'status.json',dict(phase='verifying_cache_and_freezing_protocol',owner=identity(os.getpid()),updated=time.time()));freeze()
    queue=[(arm,seed) for seed in SEEDS for arm in ARMS];active=[]
    while queue or active:
        for gpu in ('2','3'):
            while queue and sum(g==gpu for p,g,a,s in active)<2:
                arm,seed=queue.pop(0);p=launch('train.py',gpu,['--arm',arm,'--seed',str(seed)],f'train_{arm}_seed{seed}');active.append((p,gpu,arm,seed))
        live=[]
        for p,g,a,s in active:
            if p.poll() is None:live.append((p,g,a,s))
            else:assert p.returncode==0,(g,a,s)
        active=live;atomic(EXP/'status.json',dict(phase='training',owner=identity(os.getpid()),active=[dict(gpu=g,arm=a,seed=s,**identity(p.pid)) for p,g,a,s in active],remaining=queue,updated=time.time()));time.sleep(5)
    p=launch('evaluate.py','2',[],'evaluation')
    while p.poll() is None:atomic(EXP/'status.json',dict(phase='validation_evaluation',owner=identity(os.getpid()),evaluator=identity(p.pid),updated=time.time()));time.sleep(5)
    assert p.returncode==0,'Evaluation failed'
    atomic(EXP/'status.json',dict(phase='complete',complete=True,TEST_evaluated=False,updated=time.time()))
if __name__=='__main__':
    def stop(*_):raise InterruptedError('Stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main()
    except BaseException as e:
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        atomic(EXP/'failure.json',dict(error=str(e),updated=time.time()));atomic(EXP/'status.json',dict(phase='failed',error=str(e),updated=time.time()));raise
