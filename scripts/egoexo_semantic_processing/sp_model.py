import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from sp_common import *
from model import VideoTail
from router import Router
from memory_attention import run_block


class SemanticModel(VideoTail):
    def __init__(self, arm, seed, prototypes, types):
        # Model loading initializes unrelated modules; do not advance trunk RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            super().__init__()
            state = torch.load(MBASE / f'heads/M/seed{seed}/best.pt', map_location='cpu')['updates']
            expected = set(VideoTail.updates(self))
            assert expected == set(state), (expected - set(state), set(state) - expected)
            self.load_state_dict(state, strict=False)
            self.cuda()
            self.arm = arm
            # Runtime-only activation retention policy; no parameter/state change.
            self.checkpoint_blocks = (10, 11)
            self.spatial_checkpoint_blocks = ()
            if arm != 'R0':
                self.router = Router(self.visual.norm, self.projection, prototypes, types, seed)

    def forward(self, prefix, valid, progress=1., noise=None, use_checkpoint=True, dense=False, route_mode=None, route_override=None):
        route = None
        if self.arm != 'R0':
            route = self.router(prefix, valid, self.arm, progress, noise, route_mode) if route_override is None else route_override
        assignment, mass = (None, None) if route is None or dense else (route['assignment'], route['mass'])
        x = prefix
        if self.training and torch.is_grad_enabled():
            x = prefix.detach().requires_grad_(True)
        for block_index, block in enumerate(self.visual.blocks[10:], start=10):
            spatial_checkpoint=self.training and use_checkpoint and block_index in self.spatial_checkpoint_blocks
            def run(z, a, m, block=block, spatial_checkpoint=spatial_checkpoint):
                return run_block(block, z, a, m,checkpoint_spatial=spatial_checkpoint)
            if self.training and torch.is_grad_enabled() and use_checkpoint and block_index in self.checkpoint_blocks:
                x = checkpoint(run, x, assignment, mass, use_reentrant=False, preserve_rng_state=True)
            else:
                x = run(x, assignment, mass)
        x = self.visual.norm(x)
        g = F.normalize((x[:, 0] @ self.projection).float(), dim=-1)
        patches = (x[:, 1:] @ self.projection).reshape(-1, 16, 196, 256)
        return dict(g=g, patches=patches, frames=patches.mean(2), route=route)

    def updates(self):
        return {n: v.detach().cpu() for n, v in self.state_dict().items()
                if not n.startswith('visual.') or n.startswith(('visual.blocks.10.', 'visual.blocks.11.', 'visual.norm.'))}

    def parameter_groups(self):
        trunk, router = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (router if name.startswith('router.') else trunk).append(parameter)
        return trunk, router


def make_noise(model, batch, generator):
    if model.arm == 'R0':
        return None
    count = 48 if model.arm == 'R5' else 32
    uniform = torch.rand((48, batch, 16, 196), device='cuda', generator=generator).clamp_(1e-7, 1-1e-7)
    return -(-uniform[:count].log()).log()
