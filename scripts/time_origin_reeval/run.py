"""Fixed existing weights, unchanged nominal sampling, corrected playback decoding."""
import os,sys,json,time,argparse,hashlib,sqlite3,fcntl,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'experiments/time_origin_reeval_20261020'
SOURCE=ROOT/'repos/EgoExoLearn/association_benchmark'
sys.path.insert(0,str(ROOT/'scripts/sampling_study'))
from policies import fixed_plans
from decoder import load_times

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def key(c):return (c.get('video_uid',c.get('video_id')),float(c['start_sec']),float(c['end_sec']))
def atomic(path,data):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False));temp.replace(path)
def config(name):
 if name=='egoexo':return Path(os.environ.get('EGOEXO_BASE_CHECKPOINT',str(ROOT/'checkpoints/egoexo_base.pt'))),None,4
 stem={'egovideo':'egovideo_egoexo','internvideo2':'internvideo2_b14_egoexo'}[name]
 return SOURCE/'exps'/stem/'checkpoint_best.pt',SOURCE/'exps'/stem/'config.json',4 if name=='egovideo' else 8

def init_worker(name):
 global transform
 import torch
 torch.set_num_threads(1)
 if name=='egoexo':
  import torchvision.transforms as tr
  mean=torch.tensor([122.7709393,116.7460125,104.09373615])[:,None,None,None]
  std=torch.tensor([68.5005327,66.6321579,70.32316305])[:,None,None,None]
  resize=tr.Resize((224,224))
  transform=lambda x:(resize(x.permute(3,0,1,2))-mean)/std
 else:
  sys.path.insert(0,str(SOURCE))
  from omegaconf import OmegaConf
  from function.utils import build_val_transform
  transform=build_val_transform(None,OmegaConf.load(config(name)[1]).model.name)
def decode(task):
 import torch
 k,targets=task
 images,pts,fallback=load_times(ROOT/'data/EgoExoLearn/videos'/(k[0]+'.mp4'),targets)
 return transform(torch.from_numpy(images).float()).numpy(),pts,fallback

def bounded_map(worker,tasks,workers=2,initializer=None,initargs=(),timeout=180):
 pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initializer,initargs=initargs)
 pending=deque();iterator=iter(tasks)
 def fill():
  while len(pending)<workers*2:
   try:task=next(iterator)
   except StopIteration:break
   pending.append(pool.submit(worker,task))
 try:
  fill()
  while pending:
   result=pending.popleft().result(timeout=timeout);fill();yield result
 except BaseException:
  for process in list(pool._processes.values()):process.terminate()
  pool.shutdown(wait=False,cancel_futures=True);raise
 else:pool.shutdown(wait=True)

def video_metadata(vid):
 import decord
 vr=decord.VideoReader(str(ROOT/'data/EgoExoLearn/videos'/(vid+'.mp4')),num_threads=1)
 return vr.get_avg_fps(),len(vr)

def model_load(name,cp,cfgpath):
 import torch
 w=torch.load(cp,map_location='cpu');state={k.removeprefix('module.'):v for k,v in w['state_dict'].items()}
 if name=='egoexo':
  sys.path.insert(0,str(ROOT/'repos/EgoExoLearn/association_benchmark'))
  from models.retriever import CLIP,SpaceTimeTransformer,QuickGELU
  visual=SpaceTimeTransformer(num_frames=state['visual.temporal_embed'].shape[1],time_init='zeros',attention_style='frozen-in-time',ln_pre=True,act_layer=QuickGELU,is_tanh_gating=False,drop_path_rate=0)
  visual.head=visual.pre_logits=visual.fc=torch.nn.Identity()
  model=CLIP(embed_dim=state['image_projection'].shape[1],vision_width=768,vision_model=visual,context_length=77,vocab_size=49408,transformer_width=512,transformer_heads=8,transformer_layers=12)
 else:
  sys.path.insert(0,str(SOURCE))
  from omegaconf import OmegaConf
  from models.builder import build_model
  model=build_model(OmegaConf.load(cfgpath).model)
 model.load_state_dict(state,strict=True)
 return model.eval().cuda(),w.get('epoch')

def scores_and_predictions(out,q,features):
 scores=[];cosines=[];rows=[]
 for uid,r in q.items():
  f=np.stack([features[key(c)] for c in [r['query']]+[r['choices'][str(i)] for i in range(20)]])
  sc=f[1:]@f[0];norm=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12);cos=norm[1:]@norm[0]
  assert np.isfinite(sc).all() and np.isfinite(cos).all()
  scores.append(sc);cosines.append(cos)
  rows.append(dict(id=uid,type=r['types'],answer=r['answer'],pred=int(sc.argmax()),scores=sc.tolist(),cosine_pred=int(cos.argmax()),cosine_scores=cos.tolist()))
 (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 np.savez_compressed(out/'scores.npz',uids=np.array(list(q)),dot=np.stack(scores),cosine=np.stack(cosines))
 metrics={}
 for score in ['dot','cosine']:
  pred='pred' if score=='dot' else 'cosine_pred'
  metrics[score]={}
  for group in ['all',1,2]:
   subset=[r for r in rows if group=='all' or r['type']==group]
   metrics[score][str(group)]=dict(n=len(subset),correct=sum(r[pred]==r['answer'] for r in subset),accuracy=100*sum(r[pred]==r['answer'] for r in subset)/len(subset) if subset else None)
 return metrics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('model',choices=['egovideo','internvideo2','egoexo']);ap.add_argument('split',choices=['val','test']);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('1','3')
 import torch,decord,av
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);torch.cuda.set_per_process_memory_fraction(.33)
 if not a.limit:
  from preflight import snapshot
  assert snapshot()==json.loads((BASE/'manifest.json').read_text()),'Frozen inputs/source changed'
 cp,cfg,n=config(a.model);ann=ROOT/f'data/association_annotations/association_{a.split}_hard.json';q=json.loads(ann.read_text());assert len(q)==(800 if a.split=='val' else 2200)
 if a.limit:q=dict(list(q.items())[:a.limit])
 out=BASE/(f'smoke_{a.limit}' if a.limit else 'full')/a.model/a.split;out.mkdir(parents=True,exist_ok=True)
 lock=(out/'writer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sources=[HERE/'decoder.py',HERE/'run.py',ROOT/'scripts/sampling_study/policies.py']
 if cfg:sources += [cfg,SOURCE/'function/utils.py']
 meta=dict(model=a.model,split=a.split,checkpoint=str(cp.resolve()),checkpoint_sha256=digest(cp),annotations_sha256=digest(ann),sources={str(p):digest(p) for p in sources},frames=n,limit=a.limit,torch=torch.__version__,av=av.__version__,dtype='float32',manifest_sha256=digest(BASE/'manifest.json') if (BASE/'manifest.json').exists() else None,protocol='old-trained weights + corrected playback PTS, original nominal target grid, minimum1s context retained, first frame >= target, explicit EOF last-frame fallback',training=False)
 if (out/'run.json').exists():assert json.loads((out/'run.json').read_text())==meta,'Provenance changed: use a fresh output'
 else:atomic(out/'run.json',meta)
 if (out/'metrics.json').exists():print('Already complete',flush=True);return
 db=sqlite3.connect(out/'features.sqlite');db.execute('CREATE TABLE IF NOT EXISTS features(key TEXT PRIMARY KEY,value BLOB NOT NULL,targets TEXT,pts TEXT,fallback TEXT)')
 keys=list(dict.fromkeys(key(c) for r in q.values() for c in [r['query']]+list(r['choices'].values())))
 known={k for k, in db.execute('SELECT key FROM features')};todo=[];video={};started=time.time()
 missing=[k for k in keys if json.dumps(k) not in known]
 vids=list(dict.fromkeys(k[0] for k in missing))
 for vid,metadata in zip(vids,bounded_map(video_metadata,vids,workers=a.workers)):
  video[vid]=metadata
 for k in missing:
  fps,length=video[k[0]];ids=fixed_plans(k[1],k[2],fps,length,n)['original'][0]
  todo.append((k,tuple(i/fps for i in ids)))
 print(json.dumps(dict(model=a.model,split=a.split,pending=len(todo),total=len(keys))),flush=True)
 model,epoch=model_load(a.model,cp,cfg)
 if a.model=='egoexo':assert epoch==json.loads((ROOT/'experiments/egoexo_colleague_test/run.json').read_text())['checkpoint_epoch']
 pos=0
 results=bounded_map(decode,todo,workers=a.workers,initializer=init_worker,initargs=(a.model,))
 try:
  with torch.inference_mode():
   for task in todo:
    atomic(out/'current_clip.json',dict(clip=task[0],targets=task[1],started=time.time(),completed=pos))
    pixels,pts,fallback=next(results)
    feat=model.encode_image(torch.from_numpy(pixels)[None].cuda()).float().cpu().numpy()[0]
    if not np.isfinite(feat).all():raise ValueError('Nonfinite model features')
    db.execute('INSERT INTO features VALUES (?,?,?,?,?)',(json.dumps(task[0]),feat.tobytes(),json.dumps(task[1]),json.dumps(pts),json.dumps(fallback)))
    pos+=1
    if pos%25==0:db.commit()
    if pos<=3 or pos%50==0:
     progress=dict(completed=len(known)+pos,total=len(keys),elapsed_s=time.time()-started,updated=time.time());atomic(out/'progress.json',progress);print(json.dumps(progress),flush=True)
 finally:
  results.close();db.commit()
 db.commit();features={tuple(json.loads(k)):np.frombuffer(v,np.float32) for k,v in db.execute('SELECT key,value FROM features')}
 assert set(keys)<=set(features)
 metrics=scores_and_predictions(out,q,features);metrics.update(epoch=epoch,questions=len(q),clips=len(keys),elapsed_s=time.time()-started,complete=True)
 atomic(out/'metrics.json',metrics);atomic(out/'progress.json',dict(completed=len(keys),total=len(keys),complete=True,updated=time.time()));print(json.dumps(metrics),flush=True);db.close()
if __name__=='__main__':main()
