"""Run an already-frozen recipe, persist completed artifacts and preserve exit code."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from info_common import atomic,sha

def sync(run,archive,finished=False):
    cache={}
    cachefile=run/'sync_state.json'
    if cachefile.exists():cache=json.loads(cachefile.read_text())
    stable_arrays=(run/'inference_complete.json').exists() or finished
    copied=[]
    for p in run.rglob('*'):
        if not p.is_file() or p.name.endswith('.tmp') or p.name in ('sync_state.json','supervisor.log'):continue
        if p.suffix in ('.npy','.npz') and not stable_arrays:continue
        rel=str(p.relative_to(run));stat=p.stat();signature=[stat.st_size,stat.st_mtime_ns]
        if cache.get(rel)==signature:continue
        dest=archive/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest);cache[rel]=signature;copied.append(rel)
    atomic(cachefile,cache)
    atomic(archive/'persistence_status.json',dict(time=time.time(),finished=finished,copied=copied,
                                               mutable_features_deferred=not stable_arrays))

def main(a):
    state=json.loads(Path(a.state).read_text());run=Path(state['run_dir']);archive=Path(state['archive_dir'])
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=state['repo'],text=True).strip(),'Commit experiment code first'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=state['repo'],text=True).strip()
    assert (run/'regions_frozen.json').exists()
    for name in ('plan.txt','config.json','regions_frozen.json'):
        assert (run/name).is_file();shutil.copyfile(run/name,archive/name)
    py=state['python'];folder=Path(state['repo'])/'scripts/information_mechanism'
    arguments=[item for key in ('run_dir','snapshot','videos','annotations','metadata') for item in ('--'+key.replace('_','-'),state[key])]
    commands=[[py,'-u',str(folder/'experiment.py'),*arguments,'--reuse-calibration'],
              [py,'-u',str(folder/'analyze.py'),*[item for key in ('run_dir','snapshot','annotations','metadata') for item in ('--'+key.replace('_','-'),state[key])]]]
    import shlex
    script='#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=0\nexport PYTHONHASHSEED=17\nexport HF_HUB_OFFLINE=1\ncd '+shlex.quote(state['repo'])+'\n'+'\n'.join(shlex.join(c) for c in commands)+'\n'
    (run/'command.sh').write_text(script);(run/'command.sh').chmod(0o700)
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(run/'code.tar.gz'),commit],cwd=state['repo'],check=True)
    subprocess.run(['git','bundle','create',str(run/'code.bundle'),'HEAD'],cwd=state['repo'],check=True)
    lock=run/'requirements-lock.txt'
    with lock.open('w') as f:subprocess.run([py,'-c','import importlib.metadata as m;print("\\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions() if d.metadata["Name"])))'],stdout=f,check=True)
    record=dict(**state,status='running',git_commit=commit,worktree_clean=True,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),supervisor_pid=os.getpid(),training=False,TEST_access=False,
                config_sha256=sha(run/'config.json'),plan_sha256=sha(run/'plan.txt'),regions_sha256=sha(run/'regions_frozen.json'))
    atomic(run/'run.json',record);sync(run,archive)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONHASHSEED='17',HF_HUB_OFFLINE='1')
    with (run/'experiment.log').open('a') as log:
        proc=subprocess.Popen(['bash',str(run/'command.sh')],cwd=state['repo'],env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic(run/'process.json',dict(supervisor_pid=os.getpid(),experiment_pid=proc.pid,git_commit=commit))
        synced=time.monotonic()
        while proc.poll() is None:
            time.sleep(5)
            if time.monotonic()-synced>=300:
                sync(run,archive);synced=time.monotonic()
        code=proc.returncode
    record.update(status='complete' if code==0 else 'failed',exit_code=code,ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic(run/'run.json',record);sync(run,archive,finished=True)
    hashes={str(p.relative_to(run)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in run.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','sync_state.json','supervisor.log') and not p.name.endswith('.tmp')}
    atomic(run/'artifact_manifest.json',hashes);shutil.copyfile(run/'artifact_manifest.json',archive/'artifact_manifest.json')
    print(json.dumps(dict(status=record['status'],exit_code=code,archive=str(archive))),flush=True)
    return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);sys.exit(main(p.parse_args()))
