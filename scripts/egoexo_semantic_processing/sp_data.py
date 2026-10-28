"""Read-only existing prefix cache with two bounded staging buffers."""
import os
import numpy as np
import torch
from sp_common import *


class Data:
    def __init__(self, capacity=48):
        ready = read(REL / 'ssd_ready.json')
        assert ready['passed'] and ready['source_unchanged']
        self.arrays, self.pts, self.locations, self.fds = [], [], {}, {}
        for k, gpu in enumerate((2, 3)):
            p = SSD / f'cache{gpu}_prefix.npy'
            expected = next(row for row in ready['files'] if row['target'] == str(p))
            assert p.stat().st_size == expected['bytes']
            array = np.load(p, mmap_mode='r')
            assert array.shape[1:] == (3137, 768) and array.dtype == np.float16
            self.arrays.append(array)
            self.pts.append(np.load(CACHE / f'cache{gpu}/pts.npy', mmap_mode='r'))
            self.locations.update({int(i): (k, j) for j, i in enumerate(np.load(CACHE / f'cache{gpu}/ids.npy'))})
        self.capacity = capacity
        self.buffers = [torch.empty((capacity, 3137, 768), dtype=torch.float16, pin_memory=True).numpy() for _ in range(2)]
        self.buffer_cursor = 0
        self.records, self.phrases = read(BASE / 'records.json'), read(BASE / 'phrases.json')
        self.text = torch.as_tensor(np.load(BASE / 'text_features.npy'), device='cuda')
        self.labels = np.load(EXP / 'route_labels.npy')
        self.donor = np.load(EXP / 'route_donors.npy')
        self.prototypes = torch.as_tensor(np.load(EXP / 'route_prototypes.npy'), device='cuda')
        self.vocabulary = read(EXP / 'route_vocabulary.json')
        self.types = torch.tensor([0 if r['type'] == 'verb' else 1 for r in self.vocabulary], device='cuda')
        self.phrase_labels = np.array([p['label'] + (1000 if p['type'] == 'noun' else 0) for p in self.phrases])

    def cpu_batch(self, ids):
        ids = list(map(int, ids))
        assert len(ids) <= self.capacity
        x = self.buffers[self.buffer_cursor][:len(ids)]
        self.buffer_cursor = 1 - self.buffer_cursor
        positions = [self.locations[i] for i in ids]
        for b, (k, j) in enumerate(positions):
            array = self.arrays[k]
            path = str(array.filename)
            if path not in self.fds:
                self.fds[path] = os.open(path, os.O_RDONLY)
            view = memoryview(x[b]).cast('B')
            offset = array.offset + j * array.strides[0]
            done = 0
            while done < len(view):
                n = os.preadv(self.fds[path], [view[done:]], offset + done)
                if n <= 0:
                    raise IOError(f'Unexpected end of immutable cache {path}')
                done += n
        pts = np.stack([self.pts[k][j] for k, j in positions])
        mask = np.ones((len(ids), 16), bool)
        for t in range(1, 16):
            mask[:, t] = (pts[:, t, None] != pts[:, :t]).all(1)
        return ids, x, mask

    def batch(self, ids=None, prepared=None, training=True, arm='R2',transfer_non_blocking=False):
        ids, x, mask = self.cpu_batch(ids) if prepared is None else prepared
        result = dict(ids=ids, x=torch.from_numpy(x).cuda(non_blocking=transfer_non_blocking), mask=torch.tensor(mask, device='cuda'))
        if not training:
            return result
        records = [self.records[i] for i in ids]
        cid = torch.tensor([r['caption_id'] for r in records], device='cuda')
        phrases = sorted({j for r in records for j in r['phrases']})
        index = torch.tensor([self.phrases[j]['text_id'] for j in phrases], device='cuda', dtype=torch.long)
        positive = np.stack([np.isin(self.phrase_labels[phrases], self.phrase_labels[r['phrases']]) for r in records])
        route_ids = self.donor[ids] if arm == 'R3' else ids
        result.update(cid=cid, caption=self.text[cid], phrase=self.text[index],
                      pos=torch.tensor(positive, device='cuda'),
                      types=torch.tensor([0 if self.phrases[j]['type'] == 'verb' else 1 for j in phrases], device='cuda'),
                      route_labels=torch.tensor(self.labels[route_ids], device='cuda'))
        return result
