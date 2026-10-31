"""Read-only archived adapters; distinct module name avoids legacy common.py."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def load_data(snapshot, annotations, metadata):
    root = Path(snapshot) / 'workspace/EgoExoLearn-Association'
    ep = np.load(root / 'experiments/egoexo_textalign_20261025/val_episodes.npz')
    plan = json.loads((root / 'experiments/egoexo_textalign_20261025/plan.json').read_text())
    ann = json.loads(Path(annotations).read_text())
    meta = {r['video_uid']: r for r in json.loads(Path(metadata).read_text())}
    return root, ep, plan, ann, meta

def decode(key, videos, snapshot):
    """Decode all conditions jointly, preserving real PTS and exact four anchors."""
    import torch
    import torchvision.transforms as tr
    root = Path(snapshot) / 'workspace/EgoExoLearn-Association'
    source = str(root / 'scripts/sampling_corrected')
    if source not in sys.path:
        sys.path.insert(0, source)
    from bounded_decode import decode_views
    start, end = key[1:]
    four = tuple(start + (i + .5) * (end - start) / 4 for i in range(4))
    dense = tuple(start + (i + .5) * (end - start) / 16 for i in range(16))
    remove = {min(range(4*q, 4*q+4), key=lambda j: abs(dense[j]-four[q])) for q in range(4)}
    nested = tuple(sorted(four + tuple(t for j,t in enumerate(dense) if j not in remove)))
    frames, pts, seq = decode_views(Path(videos)/(key[0]+'.mp4'),
                                    [(t, (start,end)) for t in (four,dense,nested)])
    rgb = tr.Resize((224,224))(torch.from_numpy(frames.copy()).float().permute(3,0,1,2))
    x = (rgb - torch.tensor([122.7709393,116.7460125,104.09373615])[:,None,None,None])
    x = x / torch.tensor([68.5005327,66.6321579,70.32316305])[:,None,None,None]
    p4 = np.array([pts[k] for k in seq[0]])
    pn = np.array([pts[k] for k in seq[2]])
    assert all(np.min(abs(pn-t)) < 1e-7 for t in p4)
    arrays = dict(dense16=x[:,seq[1]].numpy(), repeat4=x[:,np.repeat(seq[0],4)].numpy(),
                  nested16=x[:,seq[2]].numpy())
    times = dict(four=p4.tolist(), dense16=[pts[k] for k in seq[1]],
                 repeat4=np.repeat(p4,4).tolist(), nested16=pn.tolist())
    return arrays, times

def raw_rgb(normalized):
    mean = np.array([122.7709393,116.7460125,104.09373615])[:,None,None]
    std = np.array([68.5005327,66.6321579,70.32316305])[:,None,None]
    return np.clip((normalized * std + mean).transpose(1,2,0),0,255).astype(np.uint8)

def roi_mask(box, edge=2):
    x0,y0,x1,y1 = map(int,box)
    assert 0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224
    y,x = np.mgrid[:224,:224]
    distance = np.minimum.reduce((x-x0+.5, x1-x-.5, y-y0+.5, y1-y-.5))
    return np.clip(distance/max(edge,1),0,1).astype('float32')

def perturb(image, box, sigma):
    """Same global low-pass field and same mask shape for every region."""
    import cv2
    if sigma == 0:
        return image.copy()
    smoothed = cv2.GaussianBlur(image.transpose(1,2,0), (0,0), sigmaX=sigma,
                              sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101).transpose(2,0,1)
    mask = roi_mask(box)[None]
    return (image * (1-mask) + smoothed * mask).astype('float32')
