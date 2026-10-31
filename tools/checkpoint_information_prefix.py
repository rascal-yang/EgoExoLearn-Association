"""Persist a completed, immutable prefix without copying active memmap tails."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def main(a):
 s=json.loads(Path(a.state).read_text());run=Path(s['run_dir']);record=json.loads((run/'run.json').read_text());progress=json.loads((run/'progress.json').read_text());cfg=json.loads((run/'config.json').read_text())
 n=int(progress['completed_clips']);ids=np.load(run/'clip_ids.npy');assert 0<n<=len(ids)
 parent=run.parent/(run.name+'-partial-checkpoints');parent.mkdir(exist_ok=True)
 path=parent/f'prefix-{n:08d}.npz';assert not path.exists(),path
 payload={'clip_ids':ids[:n].copy()}
 for name in cfg['conditions']:
  source=np.load(run/'features'/f'{name}.npy',mmap_mode='r')
  assert source.shape==(len(cfg['models']),len(ids),256)
  x=np.array(source[:,:n],copy=True)
  assert np.isfinite(x).all() and (np.linalg.norm(x,axis=-1)>0).all()
  payload[name]=x
 temp=path.with_suffix('.tmp')
 with temp.open('wb') as f:np.savez(f,**payload)
 temp.replace(path)
 meta={'run_id':s['run_id'],'completed_clips':n,'total_clips':len(ids),'models':cfg['models'],'conditions':cfg['conditions'],'code_commit':record['git_commit'],'config_sha256':digest(run/'config.json'),'prefix_sha256':digest(path),'bytes':path.stat().st_size,'script_sha256':digest(__file__),'progress_snapshot':progress,'created_at_unix':time.time(),'assumption':'Source writes each prefix once, flushes all conditions before publishing its cursor; only rows before that cursor are captured.'}
 metadata=path.with_suffix('.json');metadata.write_text(json.dumps(meta,indent=2)+'\n')
 shutil.copyfile(__file__,parent/'checkpoint_information_prefix.py')
 archive=Path(s['archive_dir'])/'partial-checkpoints';archive.mkdir(parents=True,exist_ok=True)
 for p in (path,metadata,parent/'checkpoint_information_prefix.py'):shutil.copyfile(p,archive/p.name)
 assert digest(archive/path.name)==meta['prefix_sha256']
 assert digest(archive/metadata.name)==digest(metadata)
 print(json.dumps({'completed_clips':n,'bytes':meta['bytes'],'sha256':meta['prefix_sha256'],'archive':str(archive/path.name),'readback_verified':True}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--state',required=True);main(p.parse_args())
