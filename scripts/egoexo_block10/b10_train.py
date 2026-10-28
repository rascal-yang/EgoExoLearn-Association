"""Run the audited parent trainer in an independent experiment namespace."""
from b10_common import *
from b10_model import Block10Model


def trainer():
    module=load_parent_script('train')
    module.SemanticModel=Block10Model
    return module


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--engineering-steps',type=int)
    a=p.parse_args();trainer().main(a.arm,a.seed,a.engineering_steps)
