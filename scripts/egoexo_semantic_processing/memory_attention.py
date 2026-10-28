"""Native VarAttention spatial topology with full Q and compact K/V."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def compact_spatial(attention, normalized, assignment, mass):
    b, length, dim = normalized.shape
    f, n = 16, 196
    h, d = attention.num_heads, dim // attention.num_heads
    slots = assignment.shape[1]
    patches = normalized[:, 1:].reshape(b * f, n, dim)
    # FP32 reduction, then the original projection under the caller's AMP mode.
    with torch.cuda.amp.autocast(enabled=False):
        memory = torch.bmm(assignment.float(), patches.float()) / mass.clamp_min(1)[..., None]
    memory = memory.to(normalized.dtype).reshape(b, f * slots, dim)
    memory = torch.cat([normalized[:, :1], memory], 1)
    weight, bias = attention.qkv.weight, attention.qkv.bias
    q = F.linear(normalized, weight[:dim], None if bias is None else bias[:dim])
    kv = F.linear(memory, weight[dim:], None if bias is None else bias[dim:])
    k, v = kv.chunk(2, -1)
    q = q.reshape(b, length, h, d).transpose(1, 2) * attention.scale
    k = k.reshape(b, 1 + f * slots, h, d).transpose(1, 2)
    v = v.reshape(b, 1 + f * slots, h, d).transpose(1, 2)
    counts = torch.cat([mass.new_ones(b, 1), mass.reshape(b, f * slots)], 1)
    logmass = counts.clamp_min(1).log().masked_fill(counts == 0, -1e9)

    def attend(query, keys, values, lm):
        logits = query @ keys.transpose(-2, -1)
        # FP32 softmax prevents a half overflow from the empty-cell sentinel.
        probability = (logits.float() + lm).softmax(-1).to(values.dtype)
        return probability @ values

    cls = attend(q[:, :, :1], k, v, logmass[:, None, None])
    q_frame = q[:, :, 1:].reshape(b, h, f, n, d).permute(0, 2, 1, 3, 4).reshape(b * f, h, n, d)
    k_frame = k[:, :, 1:].reshape(b, h, f, slots, d).permute(0, 2, 1, 3, 4).reshape(b * f, h, slots, d)
    v_frame = v[:, :, 1:].reshape(b, h, f, slots, d).permute(0, 2, 1, 3, 4).reshape(b * f, h, slots, d)
    cls_k = k[:, :, :1, :][:, None].expand(b, f, h, 1, d).reshape(b * f, h, 1, d)
    cls_v = v[:, :, :1, :][:, None].expand(b, f, h, 1, d).reshape(b * f, h, 1, d)
    frame_mass = torch.cat([mass.new_ones(b * f, 1), mass], 1)
    frame_logmass = frame_mass.clamp_min(1).log().masked_fill(frame_mass == 0, -1e9)
    out = attend(q_frame, torch.cat([cls_k, k_frame], 2), torch.cat([cls_v, v_frame], 2), frame_logmass[:, None, None])
    out = out.reshape(b, f, h, n, d).permute(0, 2, 1, 3, 4).reshape(b, h, f * n, d)
    out = torch.cat([cls, out], 2).transpose(1, 2).reshape(b, length, dim)
    return attention.proj_drop(attention.proj(out))


def run_block(block, x, assignment=None, mass=None, checkpoint_spatial=False):
    if assignment is None and not checkpoint_spatial:
        return block(x, 'b (f n) d', '(b f) n d', 'b (f n) d', '(b n) f d',
                     time_n=196, space_f=16, use_checkpoint=False)
    time_output = block.timeattn(block.norm3(x), 'b (f n) d', '(b n) f d', {'n': 196})
    if hasattr(block, 'alpha_timeattn'):
        time_output = torch.tanh(block.alpha_timeattn) * time_output
    normalized = block.norm1(x + time_output)
    def spatial_read(z, a, m):
        if a is None:
            return block.attn(z, 'b (f n) d', '(b f) n d', {'f':16})
        return compact_spatial(block.attn, z, a, m)
    if checkpoint_spatial and torch.is_grad_enabled():
        spatial=checkpoint(spatial_read,normalized,assignment,mass,use_reentrant=False,preserve_rng_state=True)
    else:
        spatial=spatial_read(normalized,assignment,mass)
    assert block.attention_style == 'frozen-in-time'
    residual = x + block.drop_path(spatial)
    return residual + block.drop_path(block.mlp(block.norm2(residual)))
