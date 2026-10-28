import copy
import torch
from torch import nn
from torch.nn import functional as F


def standardize(x):
    mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True, unbiased=False)
    return torch.where(std >= 1e-4, (x - mean) / (std + 1e-4), torch.zeros_like(x))


class Router(nn.Module):
    def __init__(self, norm, projection, prototypes, types, seed):
        super().__init__()
        self.norm = copy.deepcopy(norm)
        self.norm.requires_grad_(True)
        self.projection = nn.Parameter(projection.detach().clone())
        # Independent initialization does not advance the trunk RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(100000 + seed)
            self.content = nn.Linear(256, 1).to(projection.device)
            nn.init.normal_(self.content.weight, std=.02)
            nn.init.zeros_(self.content.bias)
        self.register_buffer('prototypes', prototypes.detach().clone())
        self.register_buffer('types', types.detach().clone())
        r = torch.arange(14, device=projection.device)
        cells = ((4 * r // 14)[:, None] * 4 + (4 * r // 14)[None, :]).flatten()
        self.register_buffer('cells', F.one_hot(cells, 16).T.float())

    def forward(self, prefix, valid, arm, progress=1., noise=None, mode=None):
        # Numerically sensitive routing is always FP32, including under AMP.
        with torch.cuda.amp.autocast(enabled=False):
            x = prefix.detach().float()
            u = F.normalize(self.norm(x) @ self.projection.float(), dim=-1)
            batch = len(x)
            patch = u[:, 1:].reshape(batch, 16, 196, 256)
            global_logits = u[:, 0] @ self.prototypes.T / .07
            semantic = []
            concepts = []
            for kind, top in ((0, 2), (1, 4)):
                ids = (self.types == kind).nonzero().flatten()
                logits = global_logits[:, ids]
                _, top_local = logits.topk(min(top, len(ids)), dim=-1)
                chosen = ids[top_local]
                probability = logits.gather(1, top_local).softmax(-1)
                prototypes = self.prototypes[chosen]
                affinity = torch.einsum('bfnd,bkd->bfnk', patch, prototypes)
                center = (affinity * valid[:, :, None, None]).sum((1, 2), keepdim=True) / (valid.sum(1)[:, None, None, None] * 196).clamp_min(1)
                score = .07 * torch.logsumexp((affinity - center) / .07 + probability.clamp_min(1e-20).log()[:, None, None], -1)
                semantic.append(score)
                concepts.append(chosen)
            content = self.content(patch).squeeze(-1)
            sem = (semantic[0] + semantic[1]) / 2
            score = standardize(content)
            if arm != 'R1' and mode != 'content':
                score = score + standardize(sem)
            if mode == 'random':
                if noise is None:
                    raise ValueError('Random intervention needs explicit fixed random scores')
                score = noise[:, 0].reshape(batch, 16, 196)
            evidence_count = 48 if arm == 'R5' else 32
            frames = batch * 16
            score = score.reshape(frames, 196)
            frozen = F.normalize(x[:, 1:].reshape(frames, 196, 768), dim=-1)
            penalty = torch.zeros_like(score)
            used = torch.zeros_like(score, dtype=torch.bool)
            hard, soft, indices = [], [], []
            gamma, tau = max(0., 1. - progress), 1. - .8 * min(1., progress)
            stochastic = self.training and torch.is_grad_enabled()
            if stochastic and noise is None:
                raise ValueError('Training requires explicit independent Gumbel noise')
            redundancy = 0. if arm == 'R4' else .25
            for k in range(evidence_count):
                z = score - redundancy * penalty
                if stochastic and gamma:
                    z = z + gamma * noise[k].reshape(frames, 196)
                z = z.masked_fill(used, -1e9)
                index = z.argmax(-1)  # First max gives the smaller native index.
                h = F.one_hot(index, 196).float()
                hard.append(h)
                if stochastic:
                    soft.append((z / tau).softmax(-1))
                indices.append(index)
                used = used | h.bool()
                chosen_feature = frozen.gather(1, index[:, None, None].expand(-1, 1, 768))
                similarity = (frozen * chosen_feature).sum(-1).clamp_min(0)
                penalty = torch.maximum(penalty, similarity)
            evidence = torch.stack(hard, 1)
            hard_assignment = evidence
            if arm != 'R5':
                context = self.cells[None] * (~used)[:, None].float()
                hard_assignment = torch.cat([evidence, context], 1)
            mass = hard_assignment.sum(-1).detach()
            assignment = hard_assignment
            if stochastic:
                p = torch.stack(soft, 1)
                p = p / p.sum(1, keepdim=True).clamp_min(1)
                relaxed = p
                if arm != 'R5':
                    context = self.cells[None] * (1 - p.sum(1, keepdim=True))
                    relaxed = torch.cat([p, context], 1)
                assignment = hard_assignment + (relaxed - relaxed.detach())
            # Auxiliary evidence count is32 in all processing arms.
            evidence_features = torch.bmm(assignment[:, :32], patch.reshape(frames, 196, 256)).reshape(batch, 16, 32, 256)
            return dict(assignment=assignment, mass=mass, indices=torch.stack(indices, 1).reshape(batch, 16, evidence_count),
                        global_logits=global_logits, evidence_features=evidence_features,
                        concepts=torch.cat(concepts, 1), content=content, semantic=sem)
