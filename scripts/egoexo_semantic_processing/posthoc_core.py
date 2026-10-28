"""Predeclared evaluation stages; never modifies the frozen training modules."""
import argparse
import collections
import time
import numpy as np
import torch
from torch.nn import functional as F
from sp_common import *
from sp_data import Data
from sp_model import SemanticModel
from memory_attention import run_block, compact_spatial

POST = EXP / 'post_training'
CONDITIONS = ('native', 'semantic_off', 'random_32_plus_context', 'donor_video_concepts',
              'restore_dense_block10', 'restore_dense_block11', 'restore_dense_both',
              'mask_context', 'common_context_value_within_frame', 'permute_context_values_across_frames')


def setup():
    guard(True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(19 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verify_protocol()


def load_model(data, arm, seed):
    folder = EXP / f'fits/{arm}/seed{seed}'
    completion = read(folder / 'complete.json')
    assert completion['passed'] and completion['protocol_sha256'] == digest(EXP / 'protocol_v2_frozen.json')
    assert digest(folder / 'final.pt') == completion['final_sha256']
    ck = torch.load(folder / 'final.pt', map_location='cpu')
    assert ck['epoch'] == 4
    model = SemanticModel(arm, seed, data.prototypes, data.types).eval().requires_grad_(False)
    assert set(ck['updates']) == set(model.updates())
    model.load_state_dict(ck['updates'], strict=False)
    return model


def diagnostic():
    queue = read(REP / 'v2_固定256题诊断队列.json')
    ep = np.load(BASE / 'val_episodes.npz')
    lookup = {u: i for i, u in enumerate(ep['uids'].astype(str))}
    rows = np.array([lookup[u] for u in queue['uids']])
    return ep, rows


def context_spatial(attention, normalized, assignment, mass, mode):
    """Only projected context V changes; K, Q, positions and hard masses stay fixed."""
    b, length, dim = normalized.shape
    f, n, slots = 16, 196, assignment.shape[1]
    h, d = attention.num_heads, dim // attention.num_heads
    patches = normalized[:, 1:].reshape(b*f, n, dim)
    memory = torch.bmm(assignment.float(), patches.float()) / mass.clamp_min(1)[..., None]
    memory = torch.cat([normalized[:, :1], memory.reshape(b, f*slots, dim)], 1)
    weight, bias = attention.qkv.weight, attention.qkv.bias
    q = F.linear(normalized, weight[:dim], None if bias is None else bias[:dim])
    k, v = F.linear(memory, weight[dim:], None if bias is None else bias[dim:]).chunk(2, -1)
    if mode != 'native':
        vf = v[:, 1:].reshape(b, f, slots, dim).clone()
        cm = mass.reshape(b, f, slots)[:, :, 32:]
        if mode == 'common_context_value_within_frame':
            common = (vf[:, :, 32:] * cm[..., None]).sum(2, keepdim=True) / cm.sum(2, keepdim=True).clamp_min(1)[..., None]
            vf[:, :, 32:] = common.expand(-1, -1, slots-32, -1)
        elif mode == 'permute_context_values_across_frames':
            # Fixed one-frame circular shift, including empty source cells. Destination
            # mass/key metadata is retained; empty source projections are not hidden.
            vf[:, :, 32:] = vf[:, :, 32:].roll(1, dims=1)
        else:
            raise ValueError(mode)
        v = torch.cat([v[:, :1], vf.reshape(b, f*slots, dim)], 1)
    q = q.reshape(b, length, h, d).transpose(1, 2) * attention.scale
    k = k.reshape(b, 1+f*slots, h, d).transpose(1, 2)
    v = v.reshape(b, 1+f*slots, h, d).transpose(1, 2)
    def attend(query, keys, values, counts):
        lm = counts.clamp_min(1).log().masked_fill(counts == 0, -1e9)
        return ((query @ keys.transpose(-2, -1)).float() + lm).softmax(-1).to(values.dtype) @ values
    counts = torch.cat([mass.new_ones(b, 1), mass.reshape(b, f*slots)], 1)
    cls = attend(q[:, :, :1], k, v, counts[:, None, None])
    qf = q[:, :, 1:].reshape(b, h, f, n, d).permute(0, 2, 1, 3, 4).reshape(b*f, h, n, d)
    kf = k[:, :, 1:].reshape(b, h, f, slots, d).permute(0, 2, 1, 3, 4).reshape(b*f, h, slots, d)
    vf = v[:, :, 1:].reshape(b, h, f, slots, d).permute(0, 2, 1, 3, 4).reshape(b*f, h, slots, d)
    ck = k[:, :, :1][:, None].expand(b, f, h, 1, d).reshape(b*f, h, 1, d)
    cv = v[:, :, :1][:, None].expand(b, f, h, 1, d).reshape(b*f, h, 1, d)
    fc = torch.cat([mass.new_ones(b*f, 1), mass], 1)
    out = attend(qf, torch.cat([ck, kf], 2), torch.cat([cv, vf], 2), fc[:, None, None])
    out = out.reshape(b, f, h, n, d).permute(0, 2, 1, 3, 4).reshape(b, h, f*n, d)
    out = torch.cat([cls, out], 2).transpose(1, 2).reshape(b, length, dim)
    return attention.proj_drop(attention.proj(out))


def suffix(model, prefix, valid, route=None, condition='native'):
    if model.arm == 'R0':
        route = None
    elif route is None:
        route = model.router(prefix, valid, model.arm)
    a, m = (None, None) if route is None else (route['assignment'], route['mass'])
    if condition == 'mask_context':
        a, m = a.clone(), m.clone()
        a[:, 32:] = 0
        m[:, 32:] = 0
    x = prefix
    for j, block in enumerate(model.visual.blocks[10:], 10):
        dense = condition == 'restore_dense_both' or condition == f'restore_dense_block{j}'
        if condition in ('common_context_value_within_frame', 'permute_context_values_across_frames'):
            temporal = block.timeattn(block.norm3(x), 'b (f n) d', '(b n) f d', {'n': 196})
            if hasattr(block, 'alpha_timeattn'):
                temporal = torch.tanh(block.alpha_timeattn) * temporal
            spatial = context_spatial(block.attn, block.norm1(x + temporal), a, m, condition)
            residual = x + block.drop_path(spatial)
            x = residual + block.drop_path(block.mlp(block.norm2(residual)))
        else:
            x = run_block(block, x, None if dense else a, None if dense else m)
    return F.normalize((model.visual.norm(x)[:, 0] @ model.projection).float(), dim=-1)


def make_random_route(router, ids):
    # Explicit per-record stream; no batch-size/order dependence, no answer access.
    selections = []
    for i in ids:
        rng = np.random.default_rng(20261027 + int(i))
        selections.append(np.stack([rng.choice(196, 32, replace=False) for _ in range(16)]))
    idx = torch.as_tensor(np.stack(selections), device='cuda')
    a = F.one_hot(idx, 196).float().reshape(-1, 32, 196)
    used = a.sum(1).bool()
    a = torch.cat([a, router.cells[None] * (~used)[:, None]], 1)
    return dict(assignment=a, mass=a.sum(-1), indices=idx)


def donor_mapping(ids, keys, metadata):
    bins = collections.defaultdict(list)
    def stratum(i):
        v, a, b = keys[int(i)]
        return metadata[v]['view'], 0 if b-a <= 5 else (1 if b-a <= 15 else 2)
    for i in ids:
        bins[stratum(i)].append(int(i))
    rng = np.random.default_rng(20261027)
    mapping = {}
    for i in ids:
        pool = [j for j in bins[stratum(i)] if keys[j][0] != keys[int(i)][0]]
        assert pool, ('No same-view/duration donor from another video', i)
        mapping[int(i)] = int(rng.choice(pool))
    return mapping


def prefix_from_rgb(visual, image):
    # Stop exactly after native block9. Using its forward hook avoids changing
    # native embedding, temporal interpolation, or residual code.
    result = []
    class PrefixReady(Exception):
        pass
    def capture(module, inputs, output):
        result.append(output)
        raise PrefixReady()
    hook = visual.blocks[9].register_forward_hook(capture)
    try:
        try:
            visual(image)
        except PrefixReady:
            pass
    finally:
        hook.remove()
    assert len(result) == 1
    return result[0]


def mask_from_pts(pts):
    pts = np.asarray(pts)
    mask = np.ones(pts.shape, bool)
    for t in range(1, 16):
        mask[:, t] = (pts[:, t, None] != pts[:, :t]).all(1)
    return torch.as_tensor(mask, device='cuda')


def decode16(key):
    import backend as b
    start, end = key[1:]
    times = tuple(start + (t+.5)*(end-start)/16 for t in range(16))
    images, pts, seq = b.decode_views(ROOT/'data/EgoExoLearn/videos'/(key[0]+'.mp4'), [(times, (start, end))])
    return b.ref.transform(torch.from_numpy(images[list(seq[0])].copy()).float()).numpy(), np.array([pts[t] for t in seq[0]])


def verify_stage(name):
    p = read(POST / f'{name}_protocol.json')
    assert p['base_protocol_sha256'] == digest(EXP/'protocol_v2_frozen.json')
    assert p['runtime_amendment_sha256'] == digest(EXP/'runtime_amendment.json')
    for path, sha in p['sources'].items():
        assert digest(ROOT/path) == sha, path
    return p
