"""Persistent six-fit scheduler; read-only reuse of matching R0 baselines."""
import fcntl
import signal
from b10_common import *


def gpu_available(gpu):
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if int(row[0])<20000 or int(row[1])>83:return False
    rows=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
    return not any(r.strip().isdigit() and int(r.strip())>=1024 for r in rows)


def main():
    lock=(EXP/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify_protocol();signature=digest(PROTOCOL);pending=[];done=[];active={};stopping=[False]
    logs=EXP/'logs';logs.mkdir(exist_ok=True)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *a:stopping.__setitem__(0,True))
    for job in p['schedule']:
        folder=EXP/f"fits/{job['arm']}/seed{job['seed']}"
        stage='train'
        for candidate in ('train','final','best'):
            path=folder/('complete.json' if candidate=='train' else f'val_{candidate}/complete.json')
            if not path.exists():stage=candidate;break
            ck=read(path);assert ck['passed'] and ck['protocol_sha256']==signature
        else:done.append(job);continue
        pending.append(dict(job=job,stage=stage))
    atomic(EXP/'pipeline_pid.json',dict(pid=os.getpid(),started=time.time(),protocol_sha256=signature))
    try:
        while pending or active:
            if stopping[0] or any((EXP/f).exists() for f in ('STOP','THERMAL_STOP.json','RUN_FAILED.json')):
                raise StudyStop('Controller stop marker or signal; no automatic restart')
            for gpu,item in list(active.items()):
                rc=item['process'].poll()
                if rc is None:continue
                item['log'].close();del active[gpu]
                if rc:
                    failure=dict(job=item['job'],stage=item['stage'],returncode=rc,time=time.time())
                    atomic(EXP/'RUN_FAILED.json',failure);raise RuntimeError(str(failure))
                next_stage={'train':'final','final':'best','best':None}[item['stage']]
                if next_stage is None:done.append(item['job'])
                else:pending.insert(0,dict(job=item['job'],stage=next_stage))
            for gpu in (2,3):
                if gpu in active or not pending or not gpu_available(gpu):continue
                task=pending.pop(0);job=task['job'];stage=task['stage']
                program='b10_train.py' if stage=='train' else 'b10_evaluate.py'
                command=[str(PY),'-u',str(CODE/program),'--arm',job['arm'],'--seed',str(job['seed'])]
                if stage!='train':command+=['--endpoint',stage]
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
                log=(logs/f"{job['arm']}_seed{job['seed']}_{stage}.log").open('a')
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(job=job,stage=stage,process=child,log=log)
            live=[]
            for gpu,item in active.items():
                job=item['job'];folder=EXP/f"fits/{job['arm']}/seed{job['seed']}"
                path=folder/('progress.json' if item['stage']=='train' else f"val_{item['stage']}/progress.json")
                live.append(dict(gpu=gpu,**job,stage=item['stage'],pid=item['process'].pid,progress=read(path) if path.exists() else None))
            atomic(EXP/'pipeline_status.json',dict(phase='running' if active else 'waiting_for_GPU2_3',pid=os.getpid(),
                   completed=len(done),total=6,active=live,finished_jobs=done,pending=pending,updated=time.time(),protocol_sha256=signature))
            time.sleep(.5)
        with (logs/'summary.log').open('a') as log:
            subprocess.run([str(PY),'-u',str(CODE/'b10_summary.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic(EXP/'pipeline_status.json',dict(phase='training_and_VAL_complete',completed=6,total=6,updated=time.time(),
               protocol_sha256=signature,baseline='read_only_reused_R0',TEST_evaluated=False))
    except BaseException as exc:
        for item in active.values():
            if item['process'].poll() is None:item['process'].terminate()
        deadline=time.time()+180
        while any(item['process'].poll() is None for item in active.values()) and time.time()<deadline:time.sleep(1)
        atomic(EXP/'pipeline_status.json',dict(phase='stopped',reason=repr(exc),completed=len(done),total=6,updated=time.time(),
               active_owned_pids_still_exiting=[v['process'].pid for v in active.values() if v['process'].poll() is None],no_automatic_restart=True))
        raise


if __name__=='__main__':main()
