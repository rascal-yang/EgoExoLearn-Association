"""Own-process-only controller for eighteen frozen fits and full VAL."""
import fcntl
import signal
import time
import subprocess
from sp_common import *


def gpu_available(gpu):
    info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if int(info[0])<20000 or int(info[1])>83:return False
    rows=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
    return not any(row.strip().isdigit() and int(row.strip())>=1024 for row in rows)


def main():
    lock=(EXP/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify_protocol();protocol_hash=digest(EXP/'protocol_v2_frozen.json')
    active={};done=[];stop=[False];failed=None
    for job in p['schedule']:
        folder=EXP/f"fits/{job['arm']}/seed{job['seed']}"
        terminal=[folder/f'val_{endpoint}/complete.json' for endpoint in ('final','best')]
        if all(path.exists() and read(path).get('passed') and read(path).get('protocol_sha256')==protocol_hash for path in terminal):
            done.append(job)
    schedule=[job for job in p['schedule'] if job not in done];next_job=0
    logs=EXP/'logs';logs.mkdir(exist_ok=True)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *args:stop.__setitem__(0,True))
    atomic(EXP/'pipeline_pid.json',dict(pid=os.getpid(),protocol_sha256=protocol_hash,started=time.time()))

    def launch(gpu,job,stage):
        arm,seed=job['arm'],job['seed']
        program='train.py' if stage=='train' else 'evaluate.py'
        command=[str(PY),'-u',str(CODE/program),'--arm',arm,'--seed',str(seed)]
        if stage!='train':command+=['--endpoint',stage]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        log=(logs/f'{arm}_seed{seed}_{stage}.log').open('a')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[gpu]=dict(job=job,stage=stage,process=child,log=log,started=time.time())

    try:
        while next_job<len(schedule) or active:
            if stop[0] or any((EXP/name).exists() for name in ('STOP','THERMAL_STOP.json','RUN_FAILED.json')):
                raise StudyStop('Controller stop marker or signal; no automatic restart')
            for gpu,item in list(active.items()):
                status=item['process'].poll()
                if status is None:continue
                item['log'].close();job=item['job'];stage=item['stage'];del active[gpu]
                if status:
                    failed=dict(job=job,stage=stage,gpu=gpu,returncode=status,time=time.time())
                    atomic(EXP/'RUN_FAILED.json',failed)
                    raise RuntimeError(f'Owned child failed: {failed}')
                if stage=='train':launch(gpu,job,'final')
                elif stage=='final':launch(gpu,job,'best')
                else:done.append(job)
            for gpu in (2,3):
                if gpu not in active and next_job<len(schedule) and gpu_available(gpu):
                    job=schedule[next_job];folder=EXP/f"fits/{job['arm']}/seed{job['seed']}"
                    stage='train'
                    if (folder/'complete.json').exists():
                        stage='best' if (folder/'val_final/complete.json').exists() else 'final'
                    launch(gpu,job,stage);next_job+=1
            live=[]
            for gpu,item in active.items():
                job=item['job'];path=EXP/f"fits/{job['arm']}/seed{job['seed']}/progress.json"
                live.append(dict(gpu=gpu,**job,stage=item['stage'],pid=item['process'].pid,
                                 progress=read(path) if path.exists() else None))
            atomic(EXP/'pipeline_status.json',dict(phase='running' if active else 'waiting_for_GPU2_3',
                  pid=os.getpid(),completed=len(done),total=18,dispatched=len(p['schedule'])-len(schedule)+next_job,active=live,finished_jobs=done,updated=time.time(),
                  protocol_sha256=protocol_hash,runtime_amendment_sha256=p.get('runtime_amendment_sha256')))
            time.sleep(.5)
        with (logs/'summary.log').open('a') as log:
            subprocess.run([str(PY),'-u',str(CODE/'summarize.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic(EXP/'pipeline_status.json',dict(phase='training_and_VAL_complete',completed=18,total=18,
            protocol_sha256=protocol_hash,updated=time.time(),remaining=['internal_interventions','pixel_interventions','official_TEST_confirmation','human_review']))
    except BaseException as exc:
        for item in active.values():
            if item['process'].poll() is None:item['process'].terminate()
        deadline=time.time()+180
        while any(item['process'].poll() is None for item in active.values()) and time.time()<deadline:
            time.sleep(2)
        live_pids=[]
        for item in active.values():
            if item['process'].poll() is None:live_pids.append(item['process'].pid)
            item['log'].close()
        atomic(EXP/'pipeline_status.json',dict(phase='stopped',reason=repr(exc),completed=len(done),total=18,
              active_owned_pids_still_exiting=live_pids,failed=failed,updated=time.time(),no_automatic_restart=True))
        raise


if __name__=='__main__':main()
