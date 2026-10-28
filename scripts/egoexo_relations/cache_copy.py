"""Copy immutable historical prefixes to SSD, without modifying originals."""
import shutil
from rel_common import *

def main():
    prior=read(CACHE/'ssd_cache_integrity.json')['files']
    reserve=100*2**30
    if (EXP/'ssd_ready.json').exists():return
    if not SSD.exists():
        assert shutil.disk_usage('/var/tmp').free>sum(r['bytes'] for r in prior)+reserve
        SSD.mkdir(mode=0o700)
    start=time.time();completed=0;files=[]
    for row in prior:
        source=Path(row['source']);target=SSD/(source.parent.name+'_prefix.npy')
        if target.exists():
            assert target.stat().st_size==row['bytes'] and digest(target)==row['sha256']
            files.append(dict(source=str(source),target=str(target),sha256=row['sha256'],bytes=row['bytes']))
            completed+=row['bytes'];continue
        temp=target.with_suffix('.partial')
        assert not temp.exists(),'Interrupted copy requires explicit inspection'
        original=source.stat();h=hashlib.sha256();copied=0;last=0
        with source.open('rb') as inp,temp.open('xb') as out:
            for block in iter(lambda:inp.read(32*2**20),b''):
                if (EXP/'STOP').exists():raise RuntimeError('STOP')
                if shutil.disk_usage(SSD).free<reserve:raise RuntimeError('SSD reserve below100GiB')
                h.update(block);out.write(block);copied+=len(block)
                if time.time()-last>5:
                    atomic(EXP/'cache_progress.json',dict(phase='copying',pid=os.getpid(),bytes_done=completed+copied,total_bytes=sum(r['bytes'] for r in prior),elapsed_s=time.time()-start))
                    last=time.time()
            out.flush();os.fsync(out.fileno())
        now=source.stat()
        assert (original.st_size,original.st_mtime_ns)==(now.st_size,now.st_mtime_ns)
        assert h.hexdigest()==row['sha256'] and digest(temp)==row['sha256']
        temp.replace(target);completed+=copied
        files.append(dict(source=str(source),target=str(target),sha256=row['sha256'],bytes=copied))
    atomic(EXP/'ssd_ready.json',dict(passed=True,files=files,elapsed_s=time.time()-start,source_unchanged=True))
    atomic(EXP/'cache_progress.json',dict(phase='complete',bytes_done=completed,elapsed_s=time.time()-start))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        atomic(EXP/'cache_failed.json',dict(error=repr(e),time=time.time()));raise
