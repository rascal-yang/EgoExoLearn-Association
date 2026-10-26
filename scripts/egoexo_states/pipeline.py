import os,time,signal,subprocess
from state_common import *
import numpy as np
CODE=Path(__file__).parent;PY=ROOT/'.venv-sampling-fixed/bin/python';children=[]
def start(script,args,gpu,name):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with (EXP/(name+'.log')).open('a') as f:p=subprocess.Popen([str(PY),'-u',str(CODE/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 children.append(p);return p
def status(phase,**kw):atomic(EXP/'status.json',dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw))
def waitall(jobs,phase):
 while any(p.poll() is None for p in jobs):
  for p in jobs:
   if p.poll() is not None:assert p.returncode==0,(phase,p.pid,p.returncode)
  status(phase,active_pids=[p.pid for p in jobs if p.poll() is None]);time.sleep(5)
 for p in jobs:assert p.returncode==0,(phase,p.pid,p.returncode)
def main():
 assert read(EXP/'preflight.json')['passed'];assert read(EXP/'pair_source_audit.json')['passed']
 if not (EXP/'protocol.json').exists():
  files=list(CODE.glob('*.py'))+[EXP/'plan.json',EXP/'preflight.json',EXP/'direction_labels.json',EXP/'triplets.json',EXP/'cache_ids.npy',EXP/'splits.npz',BASE/'records.json',BASE/'phrases.json',BASE/'text_features.npy',BASE/'val_episodes.npz',BASE/'plan.json']
  original=Path(read(BASE/'plan.json')['checkpoint']);assert digest(original)==read(BASE/'plan.json')['checkpoint_sha256']
  files += [original]+list((ROOT/'repos/EgoExoLearn/association_benchmark/models').glob('*.py'))
  files += [MBASE/f'heads/M/seed{s}/best.pt' for s in SEEDS]
  files += [ROOT/'scripts/egoexo_textalign'/n for n in ['backend.py','model.py','common.py']]+[ROOT/'scripts/time_origin_reeval/run.py',ROOT/'scripts/sampling_corrected/bounded_decode.py']
  atomic(EXP/'protocol.json',dict(sources={str(p):digest(p) for p in files},plan_sha256=digest(EXP/'plan.json'),primary_metric='dot of normalized embeddings, original20choices',TEST_access=False,human_stage_labels=False,finite_fits=6,epochs=4))
 else:
  for p,h in read(EXP/'protocol.json')['sources'].items():assert digest(p)==h,p
 jobs=[]
 for gpu in (2,3):
  folder=EXP/f'cache{gpu}'
  if not (folder/'complete.json').exists():
   assert not folder.exists(),'Partial cache requires inspection; not automatically overwritten'
   jobs.append(start('cache.py',['--gpu',str(gpu)],gpu,f'cache{gpu}'))
 waitall(jobs,'extracting16frame_cache')
 expected=np.load(EXP/'cache_ids.npy')
 for gpu in (2,3):
  folder=EXP/f'cache{gpu}';assert read(folder/'complete.json')['passed'];assert np.array_equal(np.load(folder/'ids.npy'),expected[gpu-2::2]);assert np.load(folder/'prefix.npy',mmap_mode='r').shape==(len(expected[gpu-2::2]),3137,768)
 atomic(EXP/'cache_audit.json',dict(passed=True,rows=len(expected),source_plan_sha256=digest(EXP/'plan.json')))
 temporal=None
 if not (EXP/'interventions/complete.json').exists():
  assert not (EXP/'interventions').exists();temporal=start('interventions.py',[],3,'interventions')
 queue=[(arm,s) for s in SEEDS for arm in ARMS if not (EXP/f'fits/{arm}/seed{s}/complete.json').exists()];active=[]
 while queue or active:
  if temporal is not None and temporal.poll() is not None:assert temporal.returncode==0,'Interventions failed'
  for gpu in (2,3):
   if queue and not any(g==gpu for p,g,a,s in active):
    arm,s=queue.pop(0);p=start('train.py',['--arm',arm,'--seed',str(s)],gpu,f'train_{arm}_{s}');active.append((p,gpu,arm,s))
  live=[]
  for p,g,a,s in active:
   if p.poll() is None:live.append((p,g,a,s))
   else:assert p.returncode==0,(a,s,p.returncode)
  active=live;status('training',active=[dict(pid=p.pid,gpu=g,arm=a,seed=s) for p,g,a,s in active],queued=queue,intervention_pid=temporal.pid if temporal is not None and temporal.poll() is None else None);time.sleep(5)
 if temporal is not None:waitall([temporal],'finishing_interventions')
 p=start('evaluate.py',[],2,'evaluation');waitall([p],'validation_evaluation')
 p=start('report.py',[],2,'report');waitall([p],'analysis');status('complete',complete=True,TEST_access=False)
if __name__=='__main__':
 def stop(*_):raise InterruptedError('Stopped')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:main()
 except BaseException as e:
  for p in children:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  status('failed',error=str(e));raise
