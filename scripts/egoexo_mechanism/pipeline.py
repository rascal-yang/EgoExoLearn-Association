import os,time,subprocess,signal,sys
from mech_common import *
CODE=Path(__file__).parent;PY=ROOT/'.venv-sampling-fixed/bin/python';children=[]
def launch(script,args,gpu,name):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with (EXP/(name+'.log')).open('w') as f:p=subprocess.Popen([str(PY),'-u',str(CODE/script),*args],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 children.append(p);return p
def main():
 assert read(EXP/'preflight.json')['passed'];assert not (EXP/'protocol.json').exists()
 # Original protocol sources remain immutable; no TEST data.
 for p,h in read(BASE/'protocol.json')['sources'].items():assert digest(p)==h,p
 source=list(CODE.glob('*.py'))+[BASE/'protocol.json',EXP/'preflight.json']
 source_hashes=dict(read(BASE/'protocol.json')['sources']);source_hashes.update({str(p):digest(p) for p in source})
 atomic(EXP/'protocol.json',dict(batch=32,epochs=2,seeds=list(SEEDS),arms=list(ARMS),sources=source_hashes,TEST_access=False,local_weight=.1,remove_term_without_rescaling=True,N='B plus noun local term at original C weight',V='B plus verb local term at original C weight',M='B plus both local terms; uniform mean of normalized token similarities replacing logmeanexp, matched neutral slot',selection='original fixed internal caption holdout loss, epoch0 allowed',time_conditions=['normal','reverse','shuffle','repeat'],primary='raw dot, full original 20-choice benchmark'))
 temp=launch('temporal.py',['extract'],3,'temporal_extract');queue=[(a,s) for s in SEEDS for a in ARMS];active=[]
 while queue or active or temp.poll() is None:
  if temp.poll() is not None:assert temp.returncode==0,'temporal extraction failed'
  for gpu in (2,3):
   capacity=2 if gpu==2 or temp.poll() is not None else 1
   while queue and sum(g==gpu for p,g,a,s in active)<capacity:
    a,s=queue.pop(0);p=launch('train_ablation.py',['--arm',a,'--seed',str(s)],gpu,f'train_{a}_{s}');active.append((p,gpu,a,s))
  live=[]
  for p,g,a,s in active:
   if p.poll() is None:live.append((p,g,a,s))
   else:assert p.returncode==0,(a,s,p.returncode)
  active=live;atomic(EXP/'status.json',dict(phase='training_and_temporal_cache',pid=os.getpid(),active=[dict(pid=p.pid,gpu=g,arm=a,seed=s) for p,g,a,s in active],queued=queue,temporal_pid=temp.pid if temp.poll() is None else None,updated=time.time()));time.sleep(5)
 p=launch('temporal.py',['evaluate'],2,'evaluation')
 while p.poll() is None:atomic(EXP/'status.json',dict(phase='evaluation',pid=os.getpid(),eval_pid=p.pid,updated=time.time()));time.sleep(5)
 assert p.returncode==0,'evaluation failed'
 atomic(EXP/'status.json',dict(phase='evaluated_pending_analysis',updated=time.time()))
if __name__=='__main__':
 def stop(*_):raise InterruptedError('Stopped')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:main()
 except BaseException as e:
  for p in children:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  atomic(EXP/'status.json',dict(phase='failed',error=str(e),updated=time.time()));raise
