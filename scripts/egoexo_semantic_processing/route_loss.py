import torch
from torch.nn import functional as F
from sp_common import *
from model import caption_loss, phrase_ce


def m_losses(out, data, scale):
    # Identical to the16-frame continued-M implementation, including +1.
    g, p, frames = out['g'], out['patches'], out['frames']
    base = caption_loss(g, data['caption'], data['cid'], scale)
    gl, local, groups = g.sum() * 0, g.sum() * 0, 0
    for kind in (0, 1):
        take = data['types'] == kind
        if not take.any():
            continue
        groups += 1
        text, labels = data['phrase'][take].float(), data['pos'][:, take]
        gl = gl + phrase_ce(scale.float() * (F.normalize(g.float(), dim=-1) @ text.T), labels)
        features = F.normalize(frames.float(), dim=-1) if kind == 0 else F.normalize(p.flatten(1, 2).float(), dim=-1)
        valid = data['mask'] if kind == 0 else data['mask'][:, :, None].expand(-1, -1, 196).flatten(1)
        logits = torch.einsum('bkd,pd->bkp', features, text)
        average = (logits * valid[:, :, None]).sum(1) / (valid.sum(1).float() + 1)[:, None]
        local = local + phrase_ce(scale.float() * average, labels)
    return base, gl / max(groups, 1), local / max(groups, 1)


def route_losses(model, out, data):
    route = out['route']
    if route is None:
        zero = out['g'].sum() * 0
        return zero, zero
    with torch.cuda.amp.autocast(enabled=False):
        e = route['evidence_features'].float()
        noun = (e * data['mask'][:, :, None, None]).sum((1, 2)) / (data['mask'].sum(1)[:, None] * 32).clamp_min(1)
        verb_frames = F.normalize(e.mean(2), dim=-1)
        verb = (verb_frames * data['mask'][:, :, None]).sum(1) / data['mask'].sum(1)[:, None].clamp_min(1)
        prefix, evidence, groups = e.sum() * 0, e.sum() * 0, 0
        for kind in (0, 1):
            take = model.router.types == kind
            labels = data['route_labels'][:, take]
            if not labels.any():
                continue
            groups += 1
            prefix = prefix + phrase_ce(route['global_logits'][:, take], labels)
            vector = verb if kind == 0 else noun
            logits = vector @ model.router.prototypes[take].T / .07
            evidence = evidence + phrase_ce(logits, labels)
        return prefix / max(groups, 1), evidence / max(groups, 1)


def objective(model, out, data):
    base, gl, local = m_losses(out, data, model.scale)
    prefix, evidence = route_losses(model, out, data)
    loss = base + .1 * gl + .1 * local + .05 * prefix + .05 * evidence
    return loss, dict(caption=base, global_phrase=gl, local_mean_phrase=local,
                      route_prefix=prefix, route_evidence=evidence)
