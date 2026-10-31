"""Verify existing video bytes against the migration manifest; never decode/evaluate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()

def main(a):
    state=json.loads(Path(a.state).read_text());manifest_path=Path(a.manifest)
    receipt=json.loads(Path(a.receipt).read_text());digest=sha(manifest_path)
    assert digest==receipt['video_manifest_sha256']
    manifest=json.loads(manifest_path.read_text());files=manifest['files'];videos=Path(state['videos'])
    names={Path(r['path']).name for r in files}
    assert len(names)==len(files)==manifest['total_files']
    assert names=={p.name for p in videos.glob('*.mp4')}
    started=time.monotonic()
    def check(r):
        p=videos/Path(r['path']).name;size=p.stat().st_size;actual=sha(p)
        return dict(path=r['path'],bytes=size,sha256=actual,passed=size==r['bytes'] and actual==r['sha256'])
    with ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(check,files))
    result=dict(passed=all(r['passed'] for r in rows),videos=len(rows),bytes=sum(r['bytes'] for r in rows),
                manifest_sha256=digest,source_videos=str(videos),script_sha256=sha(__file__),workers=a.workers,
                elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat(),
                scope='Byte-level integrity only, all dataset videos; no model inference or TEST evaluation',files=rows)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
    assert result['passed'],'Cache differs from the original manifest'

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('state','manifest','receipt','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args();assert 1<=a.workers<=32;main(a)
