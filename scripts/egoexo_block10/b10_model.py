"""Same router/objectives as R1/R2; compact spatial K/V in block10 only."""
from b10_common import *
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from sp_model import SemanticModel,make_noise
from memory_attention import run_block


class Block10Model(SemanticModel):
    def __init__(self,arm,seed,prototypes,types):
        assert arm in ('R0','S1','S2')
        super().__init__({'R0':'R0','S1':'R1','S2':'R2'}[arm],seed,prototypes,types)
        self.study_arm=arm

    def forward(self,prefix,valid,progress=1.,noise=None,use_checkpoint=True,dense=False,route_mode=None,route_override=None):
        route=None
        if self.arm!='R0':
            route=self.router(prefix,valid,self.arm,progress,noise,route_mode) if route_override is None else route_override
        assignment,mass=(None,None) if route is None or dense else (route['assignment'],route['mass'])
        x=prefix
        if self.training and torch.is_grad_enabled():x=prefix.detach().requires_grad_(True)
        for block_index,block in enumerate(self.visual.blocks[10:],10):
            # The sole mathematical change from the parent R1/R2 forward.
            a,m=(assignment,mass) if block_index==10 else (None,None)
            spatial_checkpoint=self.training and use_checkpoint and block_index in self.spatial_checkpoint_blocks
            def run(z,aa,mm,block=block,spatial_checkpoint=spatial_checkpoint):
                return run_block(block,z,aa,mm,checkpoint_spatial=spatial_checkpoint)
            if self.training and torch.is_grad_enabled() and use_checkpoint and block_index in self.checkpoint_blocks:
                x=checkpoint(run,x,a,m,use_reentrant=False,preserve_rng_state=True)
            else:x=run(x,a,m)
        x=self.visual.norm(x)
        g=F.normalize((x[:,0]@self.projection).float(),dim=-1)
        patches=(x[:,1:]@self.projection).reshape(-1,16,196,256)
        return dict(g=g,patches=patches,frames=patches.mean(2),route=route)
