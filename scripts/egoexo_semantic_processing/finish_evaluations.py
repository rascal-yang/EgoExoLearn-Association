"""Finish authorized fixed post-training stages, then release own GPU workers."""
import subprocess
import signal
from posthoc_core import *


def main():
    status=POST/'overall_status.json';current=[None];stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *a:stopping.__setitem__(0,True))
    def command(program,args,label):
        log=(POST/(label+'.log')).open('a')
        process=subprocess.Popen([str(PY),'-u',str(CODE/program),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        current[0]=process
        while process.poll() is None:
            if stopping[0] or any((EXP/x).exists() for x in ('STOP','THERMAL_STOP.json')):
                process.terminate();raise StudyStop('Finisher stopped; no automatic restart')
            atomic(status,dict(phase=label,pid=os.getpid(),child_pid=process.pid,updated=time.time()))
            time.sleep(2)
        log.close();current[0]=None
        if process.returncode:raise RuntimeError(f'{label} failed; see {log.name}')
    try:
        # Pixel worker can use a released card while the last internal seed ends.
        command('post_pipeline.py',['--stage','pixels'],'pixels_controller')
        while read(POST/'internal/status.json')['phase']!='complete':
            if read(POST/'internal/status.json')['phase']=='stopped':raise RuntimeError('Internal stage stopped')
            time.sleep(2)
        command('post_summary.py',[],'internal_summary')
        command('pixel_summary.py',[],'pixel_summary')
        command('post_pipeline.py',['--stage','test'],'test_controller')
        command('test_confirmation.py',['--summarize'],'test_summary')
        command('final_report.py',[],'final_report')
        atomic(status,dict(phase='all_automatic_evaluations_complete',updated=time.time(),remaining=['two_independent_human_reviews_and_agreement_analysis'],human_review_complete=False))
    except BaseException as exc:
        if current[0] is not None and current[0].poll() is None:current[0].terminate()
        atomic(status,dict(phase='stopped',reason=repr(exc),updated=time.time(),no_automatic_restart=True))
        raise


if __name__=='__main__':main()
