import argparse
import subprocess
from rel_common import *

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    for name in ('train.py','evaluate.py'):
        guard();subprocess.run([sys.executable,'-u',str(CODE/name),'--arm',a.arm,'--seed',str(a.seed)],check=True)
