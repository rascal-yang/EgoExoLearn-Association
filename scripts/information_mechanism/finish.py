"""Watch a specific source job, then audit/render/persist immutable outputs.

Launch log/status must live outside the primary run until it is terminal.
This program never restarts the inference process.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from info_common import atomic,sha

def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def capture_environment(run,state,producer):
    """Supplement the original package manifest with driver/GPU/runtime detail."""
    if (run/'hardware.json').exists() and (run/'environment.txt').exists():return
    import torch
    import importlib.metadata
    assert Path(sys.executable).resolve()==Path(state['python']).resolve()
    fields=['index','uuid','name','driver_version','memory.total']
    text=subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader'],text=True)
    hardware=[dict(zip(fields,[v.strip() for v in row.split(',')])) for row in text.strip().splitlines()]
    initial=json.loads((run/'environment.json').read_text())
    assert any(r['uuid']==initial['device']['uuid'] for r in hardware)
    packages={n:importlib.metadata.version(n) for n in initial['environment']}
    assert packages==initial['environment']
    primary_record=json.loads((run/'run.json').read_text())
    stage='during primary inference' if primary_record['status']=='running' and alive(primary_record['supervisor_pid']) else 'post-run verification'
    value=dict(**producer,python=sys.version,packages=packages,cuda_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
               hardware=hardware,captured_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               capture_stage=stage,primary_precision='FP32, TF32 off')
    atomic(run/'hardware.json',value)
    (run/'environment.txt').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main(a):
    state_path=Path(a.state);state=json.loads(state_path.read_text());run=Path(state['run_dir'])
    archive=Path(state['archive_dir']);repo=Path(__file__).resolve().parents[2]
    status=run.parent/(run.name+'-audit-status.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    producer=dict(pid=os.getpid(),audit_commit=commit,state=str(state_path))
    capture_environment(run,state,producer)
    while True:
        current=json.loads((run/'run.json').read_text());pid=current['supervisor_pid']
        if current['status']=='complete' and (run/'artifact_manifest.json').exists():break
        if current['status']=='failed':raise RuntimeError('Primary experiment failed; no automatic restart')
        if not alive(pid):raise RuntimeError('Primary supervisor is missing before terminal artifacts')
        atomic(status,dict(**producer,phase='waiting_for_primary',primary_pid=pid,time=time.time()))
        time.sleep(10)
    out=run/'audit';out.mkdir(exist_ok=True)
    shutil.copyfile(repo/'docs/INFORMATION_MECHANISM_AUDIT_20261031.txt',out/'plan.txt')
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(out/'code.tar.gz'),commit],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(out/'code.bundle'),'HEAD'],cwd=repo,check=True)
    folder=repo/'scripts/information_mechanism'
    atomic(status,dict(**producer,phase='auditing',time=time.time()))
    subprocess.run([state['python'],'-u',str(folder/'audit.py'),'--state',str(state_path),'--output-dir',str(out)],cwd=repo,check=True)
    atomic(status,dict(**producer,phase='rendering',time=time.time()))
    subprocess.run([a.plot_python,'-u',str(folder/'figures.py'),'--run-dir',str(run),'--audit-dir',str(out),'--output-dir',str(out/'figures')],cwd=repo,check=True)
    checked=json.loads((out/'audit.json').read_text())
    primary=json.loads((run/'main_results.json').read_text())
    atomic(out/'provenance.json',dict(**producer,primary_commit=current['git_commit'],primary_artifact_manifest_sha256=sha(run/'artifact_manifest.json'),
                                    numerical_lineage_passed=primary['lineage_passed'],plot_python=a.plot_python))
    hashes={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name!='artifact_manifest.json'}
    atomic(out/'artifact_manifest.json',hashes)
    for p in out.rglob('*'):
        if not p.is_file() or p.name.endswith('.tmp'):continue
        dest=archive/'audit'/p.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for name,want in hashes.items():
        dest=archive/'audit'/name
        assert dest.stat().st_size==want['bytes'] and sha(dest)==want['sha256'],name
    assert sha(archive/'audit/artifact_manifest.json')==sha(out/'artifact_manifest.json')
    final=dict(**producer,phase='complete',passed=checked['passed'],numerical_lineage_passed=primary['lineage_passed'],archive=str(archive/'audit'),
               files_readback_verified=len(hashes),time=time.time())
    atomic(status,final);atomic(out/'completion.json',final);atomic(archive/'audit/completion.json',final)
    print(json.dumps(final),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--plot-python',required=True);a=p.parse_args()
    try:main(a)
    except BaseException as e:
        state=json.loads(Path(a.state).read_text());run=Path(state['run_dir'])
        atomic(run.parent/(run.name+'-audit-status.json'),dict(phase='failed',pid=os.getpid(),error=repr(e),time=time.time()))
        raise
