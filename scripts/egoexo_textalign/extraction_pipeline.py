import os,signal,subprocess,time
from pathlib import Path
from common import ROOT,EXP,read,atomic
children=[];CODE=Path(__file__).parent;PY=ROOT/'.venv-sampling-fixed/bin/python'
def launch(script,gpu,args,name):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    with (EXP/(name+'.log')).open('w') as f:p=subprocess.Popen([str(PY),'-u',str(CODE/script),*args],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(p)
def main():
    assert read(EXP/'smoke_gpu2/complete.json')['passed']
    launch('extract.py','2',['--gpu','2'],'extract_gpu2');launch('extract.py','3',['--gpu','3'],'extract_gpu3');launch('text_teacher.py','2',[],'text_teacher')
    while any(p.poll() is None for p in children):
        assert all(p.poll() in (None,0) for p in children),'Extraction child failed'
        atomic(EXP/'extraction_status.json',dict(phase='extracting',pids=[p.pid for p in children if p.poll() is None],updated=time.time()));time.sleep(5)
    assert all(p.returncode==0 for p in children)
    atomic(EXP/'extraction_status.json',dict(phase='complete',complete=True,updated=time.time()))
if __name__=='__main__':
    def stop(*_):raise InterruptedError('Stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main()
    except BaseException as e:
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        atomic(EXP/'extraction_failure.json',dict(error=str(e),updated=time.time()));raise
