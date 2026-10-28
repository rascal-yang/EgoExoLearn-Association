import os,signal,time,subprocess
from tm_common import *
CODE=Path(__file__).parent;PY=ROOT/'.venv-sampling-fixed/bin/python';children=[]
def status(phase,**kw):atomic(EXP/'status.json',dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw))
def launch(script,args,gpu,name):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
 with (EXP/f'{name}.log').open('a') as f:p=subprocess.Popen([str(PY),'-u',str(CODE/script),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 children.append(p);return p
def wait(jobs,phase):
 while any(p.poll() is None for p in jobs):
  for p in jobs:
   if p.poll() is not None:assert p.returncode==0,(p.pid,p.returncode)
  status(phase,active_pids=[p.pid for p in jobs if p.poll() is None]);time.sleep(5)
 for p in jobs:assert p.returncode==0,(p.pid,p.returncode)
def main():
 assert read(EXP/'preflight.json')['passed'] and read(EXP/'benchmark_complete.json')['passed'] and read(EXP/'exact_preflight.json')['passed']
 for p,h in read(EXP/'protocol.json')['sources'].items():assert digest(p)==h,p
 jobs=[launch('evaluate_worker.py',['--gpu',str(g),'--batch','32'],g,f'gpu{g}') for g in (2,3) if not (EXP/f'gpu{g}/complete.json').exists()]
 wait(jobs,'fixed_weight_validation')
 p=launch('report.py',[],'','report');wait([p],'reporting');status('complete',complete=True,TEST_access=False)
if __name__=='__main__':
 def stop(*_):raise InterruptedError('Stopped')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:main()
 except BaseException as e:
  for p in children:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  status('failed',error=str(e));raise
