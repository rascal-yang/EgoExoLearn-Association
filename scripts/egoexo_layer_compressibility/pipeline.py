"""Own-process-only controller; persistent, locked and thermally guarded."""
import fcntl
import signal
import subprocess
import time
from lc_common import *


def main():
    EXP.mkdir(parents=True,exist_ok=True)
    lock=(EXP/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify()
    for gpu in ('2','3'):
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip())
        assert free>=22000,(gpu,free,'GPU not sufficiently free')
    children=[]
    def launch(script,gpu,extra=()):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        env['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:128'
        env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='1'
        log=(EXP/f'{script}_gpu{gpu}.log').open('a')
        process=subprocess.Popen([str(PY),str(CODE/f'{script}.py'),*extra],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
        children.append(process)
        return process
    def stop(*_):
        atomic(EXP/'STOP',dict(reason='controller_signal',time=time.time()))
        raise RuntimeError('Controller interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        bench=launch('benchmark','3')
        scan0=launch('scan','2',('--shard','0'))
        scan1=None
        while True:
            for gpu in ('2','3'):
                os.environ['CUDA_VISIBLE_DEVICES']=gpu
                guard(True)
            for child in children:
                code=child.poll()
                assert code in (None,0),(child.pid,code,'Child failed; no automatic retry')
            if bench.poll()==0 and scan1 is None:
                scan1=launch('scan','3',('--shard','1'))
            atomic(EXP/'status.json',dict(state='running',pid=os.getpid(),updated_at=time.time(),
                benchmark_pid=bench.pid,benchmark_exit=bench.poll(),
                scan0_pid=scan0.pid,scan0_exit=scan0.poll(),
                scan1_pid=None if scan1 is None else scan1.pid,
                scan1_exit=None if scan1 is None else scan1.poll()))
            if scan0.poll()==0 and scan1 is not None and scan1.poll()==0:
                break
            time.sleep(5)
        log=(EXP/'report.log').open('a')
        check=subprocess.run([str(PY),str(CODE/'report.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert check.returncode==0,'Report failed'
        atomic(EXP/'complete.json',dict(passed=True,protocol_sha256=digest(EXP/'protocol_frozen.json'),
            completed_at=time.time(),training=False,TEST_access=False,GPUs_released=True))
        atomic(EXP/'status.json',dict(state='complete',updated_at=time.time(),GPUs_released=True))
    except BaseException as error:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        atomic(EXP/'failure.json',dict(error=repr(error),time=time.time(),auto_retry=False))
        raise


if __name__=='__main__':main()
