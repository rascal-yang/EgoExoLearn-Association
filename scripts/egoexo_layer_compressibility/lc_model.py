"""Full RGB encoder; only spatial K/V are replaced in one chosen block."""
import torch
from torch.nn import functional as F
from lc_common import *
from memory_attention import run_block, compact_spatial


def stem(visual, video):
    video = video.permute(0, 2, 1, 3, 4).contiguous()
    b, t = video.shape[:2]
    p = visual.patch_embed(video).flatten(2).transpose(1, 2).reshape(b, t, 196, 768)
    x = torch.cat([visual.cls_token.expand(b, -1, -1), p.flatten(1, 2)], 1)
    pos = visual.pos_embed[:, 1:].repeat(1, t, 1)
    pos = pos + visual.inflate_temporal_embeds(t).repeat_interleave(196, 1)
    x = x + torch.cat([visual.pos_embed[:, :1], pos], 1)
    if visual.ln_pre is not None:
        x = visual.ln_pre(x)
    return visual.pos_drop(x)


def random_assignment(ids, budget):
    """Stateless per-clip/frame hash ranking, reused at every layer and seed."""
    b = len(ids)
    frame = torch.arange(16, device=ids.device)[None, :, None]
    patch = torch.arange(196, device=ids.device)[None, None, :]
    value = (ids[:, None, None] * 2654435761 + frame * 2246822519 + patch + 20261028) & 0xffffffff
    for _ in range(2):
        value = ((value ^ (value >> 16)) * 73244475) & 0xffffffff
    value = value ^ (value >> 16)
    selected = value.argsort(dim=-1, stable=True)[..., :budget-16]
    evidence = F.one_hot(selected, 196).float().reshape(b*16, budget-16, 196)
    used = evidence.sum(1).bool()
    cells = ((patch[0, 0] // 14 * 4 // 14) * 4 + (patch[0, 0] % 14 * 4 // 14))
    grid = F.one_hot(cells, 16).T.float()
    assignment = torch.cat([evidence, grid[None] * (~used)[:, None]], 1)
    return assignment, assignment.sum(-1)


def similarity_assignment(normalized, budget):
    """Deterministic even/odd bipartite cosine merging within each frame.

    This is an independently specified K/V grouping, not an official ToMe port.
    Features are current pre-spatial normalized tokens; no extra K/V projection.
    Each pass removes the best matching sources, allowing multiple sources per
    destination, and uses exact original-position counts for all later passes.
    """
    patches = normalized[:, 1:].reshape(-1, 196, normalized.shape[-1]).float()
    bf = len(patches)
    assignment = torch.eye(196, device=patches.device).expand(bf, -1, -1)
    feature = patches
    while assignment.shape[1] > budget:
        z = F.normalize(feature, dim=-1)
        sim = z[:, ::2] @ z[:, 1::2].transpose(1, 2)
        best, target = sim.max(-1)
        order = best.argsort(dim=-1, descending=True, stable=True)
        remove = min(assignment.shape[1]-budget, z[:, ::2].shape[1])
        chosen, keep = order[:, :remove], order[:, remove:]
        source, destination = assignment[:, ::2], assignment[:, 1::2]
        merged = source.gather(1, chosen[..., None].expand(-1, -1, 196))
        targets = target.gather(1, chosen)
        destination = destination.scatter_add(1, targets[..., None].expand(-1, -1, 196), merged)
        kept = source.gather(1, keep[..., None].expand(-1, -1, 196))
        assignment = torch.cat([kept, destination], 1)
        if assignment.shape[1] > budget:
            feature = torch.bmm(assignment, patches) / assignment.sum(-1)[..., None]
    return assignment, assignment.sum(-1)


def compressed_block(block, x, ids, operator, budget):
    temporal = block.timeattn(block.norm3(x), 'b (f n) d', '(b n) f d', {'n': 196})
    if hasattr(block, 'alpha_timeattn'):
        temporal = temporal * torch.tanh(block.alpha_timeattn)
    normalized = block.norm1(x + temporal)
    if operator == 'random':
        assignment, mass = random_assignment(ids, budget)
    elif operator == 'similarity':
        assignment, mass = similarity_assignment(normalized, budget)
    elif operator == 'identity':
        assignment = torch.eye(196, device=x.device).expand(len(x)*16, -1, -1)
        mass = assignment.sum(-1)
    else:
        raise ValueError(operator)
    spatial = compact_spatial(block.attn, normalized, assignment, mass)
    assert block.attention_style == 'frozen-in-time'
    residual = x + block.drop_path(spatial)
    return residual + block.drop_path(block.mlp(block.norm2(residual)))


def readout(model, x):
    return F.normalize((model.visual.norm(x)[:, 0] @ model.projection).float(), dim=-1)


def encode(model, image, ids, condition):
    x = stem(model.visual, image)
    for layer, block in enumerate(model.visual.blocks):
        if layer == condition['layer']:
            x = compressed_block(block, x, ids, condition['operator'], condition['budget'])
        else:
            x = run_block(block, x)
    return readout(model, x)


def shared_scan(models, image, ids, configs):
    """Recompute every changed suffix; share only mathematically clean prefixes."""
    clean = [stem(models[0].visual, image)]
    for block in models[0].visual.blocks[:10]:
        clean.append(run_block(block, clean[-1]))
    tails = []
    for model in models:
        x10 = run_block(model.visual.blocks[10], clean[10])
        x11 = run_block(model.visual.blocks[11], x10)
        tails.append((x10, x11))
    yield 'dense', torch.stack([readout(model, tails[s][1]) for s, model in enumerate(models)])
    for c in configs:
        j = c['layer']
        if j < 0:
            continue
        if j < 10:
            x = compressed_block(models[0].visual.blocks[j], clean[j], ids, c['operator'], c['budget'])
            for block in models[0].visual.blocks[j+1:10]:
                x = run_block(block, x)
            values = []
            for model in models:
                y = run_block(model.visual.blocks[10], x)
                y = run_block(model.visual.blocks[11], y)
                values.append(readout(model, y))
        else:
            values = []
            for s, model in enumerate(models):
                x = clean[10] if j == 10 else tails[s][0]
                x = compressed_block(model.visual.blocks[j], x, ids, c['operator'], c['budget'])
                if j == 10:
                    x = run_block(model.visual.blocks[11], x)
                values.append(readout(model, x))
        yield c['name'], torch.stack(values)
