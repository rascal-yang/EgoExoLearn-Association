import signal
import subprocess
import numpy as np
from rel_common import *

children=[]
def hardware():
    raw=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,temperature.gpu,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[list(map(int,line.split(','))) for line in raw.strip().splitlines()]
    with (EXP/'temperature.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpus=rows))+'\n')
    for gpu,temp,free,util in rows:
        if temp>83:
            atomic(EXP/'THERMAL_STOP.json',dict(gpu=gpu,temperature=temp,time=time.time()))
            raise RuntimeError('GPU temperature exceeded83C; no automatic restart')
    if (EXP/'STOP').exists() or (EXP/'THERMAL_STOP.json').exists():raise RuntimeError('Stop marker')
    return {r[0]:r for r in rows}

def status(phase,**kw):atomic(EXP/'status.json',dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw))

def freeze():
    assert read(EXP/'parse_complete.json')['passed'] and read(EXP/'text_complete.json')['passed']
    assert read(EXP/'preflight.json')['passed'] and read(EXP/'review_plan.json')['passed']
    assert read(EXP/'text_complete.json')['relations_sha256']==digest(EXP/'relations.json')
    if (EXP/'protocol.json').exists():verify_protocol();return
    files=list(CODE.glob('*.py'))+[EXP/n for n in ['relations.json','relation_evidence.json','relation_text.npy','selection_holdout.npy','preflight.json','data_audit.json','review_plan.json']]
    files += [BASE/n for n in ['records.json','phrases.json','text_features.npy','splits.npz','val_episodes.npz']]
    files += [ROOT/'scripts/egoexo_textalign'/n for n in ['common.py','backend.py','model.py']]
    files += list((ROOT/'repos/EgoExoLearn/association_benchmark/models').glob('*.py'))
    files += [MBASE/'heads/M/seed17/best.pt',ROOT/'scripts/time_origin_reeval/run.py']
    protocol=dict(sources={str(p):digest(p) for p in files},arms=list(ARMS),seeds_this_round=[17],
        batch=read(EXP/'preflight.json')['nominal_batch'],epochs=4,train_clips=32079,selection_holdout=len(np.load(EXP/'selection_holdout.npy')),
        frames=16,scoring='cosine,FP32 inference',slot_count=8,
        learning_rates=dict(trunk=1e-5,head=1e-4),weight_decay=.01,
        losses=dict(caption=1.,global_phrase=.1,local_mean_phrase=.1,relation=.1),
        objectives=dict(A_continue_M='Original M objective, continued16-frame training',B_slots='Same M objective + eight-slot architecture',
            C_bound='Correct weak relations, same-slot joint role matching',D_unbound='Identical role words/parameters; independent-slot matching per role',
            E_scrambled='C with controlled action-object reassignment where TRAIN supports all recombined vocabulary; role word marginals preserved'),
        labels_are_weak=True,human_phase_labels=False,human_grounding_labels=False,
        artificial_pair_ground_truth=False,TEST_access=False,
        selection='Minimum caption loss on fixed1171 internal clips; epoch0 allowed; locked_review videos excluded from new fit selection',
        initial_checkpoint_caveat='Existing M had historical exposure to original internal holdout for checkpoint selection',
        training_videos='Unchanged original train split, same shuffle and exposure for all arms',
        cache='Start from read-only original192GiB cache; switch each shard only after byte-verified SSD file is atomically published; originals unchanged',
        scope='Five full seed17 fits; report before expanding to seeds29/43; no automatic hyperparameter search',
        gpu_policy=dict(allowed=[2,3],allocator_fraction=.90,stop_above_celsius=83,no_automatic_thermal_restart=True))
    atomic(EXP/'protocol.json',protocol)

def launch(arm,gpu):
    path=EXP/f'{arm}_seed17.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    with path.open('a') as log:
        p=subprocess.Popen([str(PY),'-u',str(CODE/'fit_eval.py'),'--arm',arm,'--seed','17'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(p);return p

def main():
    if (EXP/'THERMAL_STOP.json').exists():raise RuntimeError('Previous thermal stop; inspect before restarting')
    freeze();status('launching_with_immutable_cache')
    hardware()
    queue=[a for a in ['C_bound','A_continue_M','D_unbound','B_slots','E_scrambled'] if not (EXP/f'fits/{a}/seed17/evaluation_complete.json').exists()]
    active=[]
    while queue or active:
        cards=hardware();remaining=[]
        for p,g,a in active:
            if p.poll() is None:remaining.append((p,g,a))
            elif p.returncode:raise RuntimeError(f'{a} failed with exit{p.returncode}; see its log')
        active=remaining
        for gpu in (2,3):
            if queue and not any(g==gpu for _,g,_ in active) and cards[gpu][2]>=22000:
                arm=queue.pop(0);active.append((launch(arm,gpu),gpu,arm))
        status('training_and_evaluating',active=[dict(pid=p.pid,gpu=g,arm=a) for p,g,a in active],queue=queue)
        time.sleep(3)
    status('summarizing_seed17')
    subprocess.run([str(PY),'-u',str(CODE/'report.py')],check=True)
    status('seed17_complete_pending_mechanism_review',human_review_pending=True,additional_seeds_not_started=True)

if __name__=='__main__':
    def interrupted(*_):raise InterruptedError('Coordinator stopped')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:main()
    except BaseException as exc:
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        deadline=time.monotonic()+12
        while any(p.poll() is None for p in children) and time.monotonic()<deadline:time.sleep(.5)
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
        status('stopped',error=repr(exc),no_automatic_restart=True);raise
