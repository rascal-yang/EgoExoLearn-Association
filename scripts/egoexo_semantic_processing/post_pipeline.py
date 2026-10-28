"""Resume-safe scheduler for separately frozen post-training evaluations."""
import fcntl
import signal
import subprocess
from posthoc_core import *


def gpu_available(gpu):
    info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free,temperature.gpu',
                                    '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if int(info[0]) < 20000 or int(info[1]) > 83:
        return False
    rows = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=used_memory',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    return not any(row.strip().isdigit() and int(row.strip()) >= 1024 for row in rows)


def main(stage):
    protocol = verify_stage(stage)
    folder = POST / stage
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = protocol['jobs']
    active, pending, done = {}, [], []
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *args: stopping.__setitem__(0, True))
    for job in jobs:
        path = folder/job['name']/'complete.json'
        if path.exists():
            result = read(path)
            assert result['passed'] and result['stage_protocol_sha256'] == digest(POST/f'{stage}_protocol.json')
            done.append(job['name'])
        else:
            pending.append(job)
    try:
        while pending or active:
            if stopping[0] or any((EXP/x).exists() for x in ('STOP', 'THERMAL_STOP.json')):
                raise StudyStop('Post-training stop marker/signal; no automatic restart')
            for gpu, item in list(active.items()):
                rc = item['process'].poll()
                if rc is None:
                    continue
                item['log'].close()
                del active[gpu]
                if rc:
                    raise RuntimeError(f"{item['job']['name']} failed: {rc}; see {item['log'].name}")
                result = read(folder/item['job']['name']/'complete.json')
                assert result['passed']
                done.append(item['job']['name'])
            for gpu in (2, 3):
                if gpu in active or not pending or not gpu_available(gpu):
                    continue
                job = pending.pop(0)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                log = (folder/(job['name']+'.log')).open('a')
                command = [str(PY), '-u', str(CODE/job['program']), *job['args']]
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = dict(process=child, job=job, log=log)
            atomic(folder/'status.json', dict(phase='running' if active else 'waiting_for_GPU2_3', completed=done,
                   total=len(jobs), pid=os.getpid(), active=[dict(gpu=gpu, name=v['job']['name'], pid=v['process'].pid) for gpu,v in active.items()], updated=time.time()))
            time.sleep(1)
        atomic(folder/'status.json', dict(phase='complete', completed=done, total=len(jobs), updated=time.time()))
    except BaseException as exc:
        for item in active.values():
            if item['process'].poll() is None:
                item['process'].terminate()
        atomic(folder/'status.json', dict(phase='stopped', reason=repr(exc), completed=done, no_automatic_restart=True, updated=time.time()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['internal', 'pixels', 'test'], required=True)
    main(p.parse_args().stage)
