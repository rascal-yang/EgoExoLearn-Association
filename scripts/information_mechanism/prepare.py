"""Select region cohort before new inference; render source images without scores."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from info_common import atomic, decode, load_data, raw_rgb, sha

def main(a):
    import torch
    torch.set_num_threads(2)
    out = Path(a.run_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    root, ep, plan, ann, meta = load_data(a.snapshot,a.annotations,a.metadata)
    rng = np.random.default_rng(cfg['roi_sample_seed'])
    selected=[]
    for view in ('ego','exo'):
        seen=set()
        for row in rng.permutation(len(ep['uids'])):
            uid=str(ep['uids'][row]); key=plan['keys'][int(ep['indices'][row,0])]
            if meta[key[0]]['view'] != view or key[0] in seen:
                continue
            selected.append(dict(row=int(row),uid=uid,clip_id=int(ep['indices'][row,0]),
                                 key=key,view=view,cohort='random_query',
                                 caption=ann[uid]['query'].get('narration_en','')))
            seen.add(key[0])
            if len(seen)==cfg['roi_questions_per_view']: break
    assert len(selected)==2*cfg['roi_questions_per_view']
    known=cfg['known_clip_id']
    uses=[]
    for row,indices in enumerate(ep['indices']):
        truth=int(ep['truth'][row])
        for role,slot in [('query',0),('positive',truth+1)]:
            if int(indices[slot])==known:
                uses.append(dict(row=row,uid=str(ep['uids'][row]),role=role))
    selected.append(dict(clip_id=known,key=plan['keys'][known],view=meta[plan['keys'][known][0]]['view'],
                         cohort='known_case',uses=uses,caption='Historical chopsticks example; outcome-selected'))
    folder=out/'region_sources';folder.mkdir(exist_ok=True)
    for i,item in enumerate(selected):
        arrays,times=decode(item['key'],a.videos,a.snapshot)
        p4=np.array(times['four']);pn=np.array(times['nested16'])
        extra=[t for t in np.unique(pn) if np.min(abs(p4-t))>1e-7]
        item['record_id']=i;item['times']=times
        if not extra:
            item['status']='no_new_actual_observation';continue
        target=item['key'][1]+(cfg['known_relative_time'] if item['cohort']=='known_case' else .5*(item['key'][2]-item['key'][1]))
        t=min(extra,key=lambda v:abs(v-target)); positions=np.flatnonzero(abs(pn-t)<1e-7).tolist()
        item.update(target_pts=float(t),positions=positions,relative_time=float(t-item['key'][1]),status='needs_visual_annotation')
        frame=raw_rgb(arrays['nested16'][:,positions[0]])
        Image.fromarray(frame).save(folder/f'{i:02d}.png')
        item['frame_sha256']=sha(folder/f'{i:02d}.png')
        nearest=int(abs(p4-t).argmin())
        Image.fromarray(raw_rgb(arrays['repeat4'][:,4*nearest])).save(folder/f'{i:02d}_old.png')
    atomic(out/'region_selection.json',dict(seed=cfg['roi_sample_seed'],rule='view balanced, unique query source, no new scores',records=selected))
    for start in range(0,len(selected),8):
        sheet=Image.new('RGB',(4*336,2*292),'#151515');draw=ImageDraw.Draw(sheet)
        for offset,item in enumerate(selected[start:start+8]):
            x=(offset%4)*336;y=(offset//4)*292
            file=folder/f"{item['record_id']:02d}.png"
            if file.exists(): sheet.paste(Image.open(file).resize((224,224)),(x,y+52))
            draw.text((x+3,y+2),f"{item['record_id']:02d} {item['view']} {item.get('uid','known')} t={item.get('relative_time',0):.2f}",fill='white')
            caption=item['caption']
            for k in range(2):draw.text((x+3,y+18+15*k),caption[k*47:(k+1)*47],fill='white')
        sheet.save(folder/f'sheet_{start//8}.png')
    print(json.dumps(dict(selected=len(selected),region_selection_sha256=sha(out/'region_selection.json'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run-dir','config','snapshot','videos','annotations','metadata'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
