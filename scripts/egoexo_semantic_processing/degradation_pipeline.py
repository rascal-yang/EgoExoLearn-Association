"""Schedule the frozen diagnostic jobs and write one synthesis report."""
import subprocess
from degradation_core import *
from post_pipeline import main as schedule


def main():
    verify_degradation()
    schedule('degradation')
    with (DEG/'summary.log').open('a') as log:
        subprocess.run([str(PY),'-u',str(CODE/'degradation_summary.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic(DEG/'diagnosis_complete.json',dict(passed=True,protocol_sha256=digest(POST/'degradation_protocol.json'),
           report=str((REP/'退化原因诊断.txt').relative_to(ROOT)),training_launched=False,TEST_access=False))


if __name__=='__main__':main()
