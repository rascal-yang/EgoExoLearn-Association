"""Deterministic low-level matched controls; outputs require visual review."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from info_common import atomic,sha

def intersects(a,b,gap=0):
    return a[0]<b[2]+gap and a[2]+gap>b[0] and a[1]<b[3]+gap and a[3]+gap>b[1]

def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)

def main(a):
    run=Path(a.run_dir);cfg=json.loads((run/'config.json').read_text())
    selection=json.loads((run/'region_selection.json').read_text())
    annotations=json.loads(Path(a.labels).read_text());labels={r['record_id']:r for r in annotations['regions']}
    rows=[];folder=run/'region_sources'
    for item in selection['records']:
        row=dict(item,**labels[item['record_id']]);box=row['box'];controls=[]
        if item['status']=='no_new_actual_observation':rows.append(row);continue
        rgb=np.array(Image.open(folder/f"{row['record_id']:02d}.png"))/255.
        gray=cv2.cvtColor(rgb.astype('float32'),cv2.COLOR_RGB2GRAY)
        gradient=np.hypot(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))
        def values(b):
            x0,y0,x1,y1=b
            return float(gray[y0:y1,x0:x1].mean()),float(gradient[y0:y1,x0:x1].mean())
        mu,grad=values(box);w,h=box[2]-box[0],box[3]-box[1]
        eligible=[]
        for y in range(0,225-h,2):
            for x in range(0,225-w,2):
                b=[x,y,x+w,y+h]
                if any(intersects(b,e,cfg['background_min_gap_pixels']) for e in [box]+row['exclude']):continue
                m,g=values(b);ratio=g/max(grad,1e-8)
                if abs(m-mu)>cfg['background_max_brightness_difference'] or not .5<=ratio<=2:continue
                score=abs(m-mu)/.15+abs(np.log(max(ratio,1e-8)))/np.log(2)
                eligible.append((score,b,m,g))
        for score,b,m,g in sorted(eligible):
            if any(iou(b,c['box'])>.25 for c in controls):continue
            controls.append(dict(box=b,matching_score=score,brightness=m,gradient=g))
            if len(controls)==cfg['background_controls']:break
        row.update(controls=controls,roi_brightness=mu,roi_gradient=grad,
                   area_fraction=w*h/(224*224),status='controls_generated_needs_review')
        row['matched_analysis_eligible']=len(controls)>=cfg['background_min_controls'] and row['area_fraction']<=cfg['max_roi_fraction'] and row['type'] not in ('uncertain','state_context')
        im=Image.fromarray((rgb*255).round().astype('uint8'));d=ImageDraw.Draw(im)
        d.rectangle(box,outline='red',width=2)
        for k,c in enumerate(controls):d.rectangle(c['box'],outline=['lime','cyan','yellow'][k],width=2)
        for b in row['exclude']:d.rectangle(b,outline='magenta',width=1)
        im.save(folder/f"{row['record_id']:02d}_regions.png")
        rows.append(row)
    for start in range(0,len(rows),8):
        sheet=Image.new('RGB',(4*280,2*254),'#151515');d=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows[start:start+8]):
            x=i%4*280;y=i//4*254
            file=folder/f"{row['record_id']:02d}_regions.png"
            if file.exists():sheet.paste(Image.open(file),(x,y+25))
            d.text((x,y+3),f"{row['record_id']:02d} {len(row.get('controls',[]))} BG {row['type']}",fill='white')
        sheet.save(folder/f'controls_{start//8}.png')
    atomic(run/'regions_proposed.json',dict(annotation_author=annotations['annotation_author'],independent_human_GT=False,selection_sha256=sha(run/'region_selection.json'),labels_sha256=sha(a.labels),records=rows))
    print(json.dumps(dict(records=len(rows),matched_eligible=sum(r.get('matched_analysis_eligible',False) for r in rows),control_counts=[len(r.get('controls',[])) for r in rows])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--labels',required=True);main(p.parse_args())
