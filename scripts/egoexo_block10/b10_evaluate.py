"""Unchanged FP32 full800 VAL evaluation, bound to the new block10 model."""
from b10_common import *
from b10_model import Block10Model


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--endpoint',choices=('final','best'),default='final');a=p.parse_args()
    module=load_parent_script('evaluate');module.SemanticModel=Block10Model
    module.evaluate(a.arm,a.seed,a.endpoint)
