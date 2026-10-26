import os,argparse,time
from state_common import *
import numpy as np,torch
import backend as b

def decode16(k):
 times=tuple(k[1]+(i+.5)*(k[2]-k[1])/16 for i in range(16));images,pts,seq=b.decode_views(ROOT/'data/EgoExoLearn/videos'/(k[0]+'.mp4'),[(times,(k[1],k[2]))]);return b.ref.transform(torch.from_numpy(images[list(seq[0])].copy()).float()).numpy(),[pts[i] for i in seq[0]]
def main(gpu,smoke=False):
 assert gpu in ('2','3') and os.environ['CUDA_VISIBLE_DEVICES']==gpu;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8);b.ref.init_worker('egoexo')
 ids=np.load(EXP/'cache_ids.npy')[int(gpu)-2::2]
 if smoke:ids=ids[:4]
 folder=EXP/(('smoke' if smoke else 'cache')+gpu);folder.mkdir(exist_ok=False);np.save(folder/'ids.npy',ids);keys=read(BASE/'plan.json')['keys'];m=b.load_model().eval();m.requires_grad_(False);captured=[];hook=m.visual.blocks[9].register_forward_hook(lambda mod,i,o:captured.append(o))
 prefix=np.lib.format.open_memmap(folder/'prefix.npy',mode='w+',dtype='float16',shape=(len(ids),3137,768));times=np.lib.format.open_memmap(folder/'pts.npy',mode='w+',dtype='float64',shape=(len(ids),16));start=time.time();errors=[]
 with torch.inference_mode():
  decoded=b.ref.bounded_map(decode16,[keys[int(i)] for i in ids],workers=4,initializer=b.ref.init_worker,initargs=('egoexo',))
  for begin in range(0,len(ids),2):
   cool();items=[next(decoded) for _ in ids[begin:begin+2]];x=torch.tensor(np.stack([v[0] for v in items]),device='cuda');captured.clear();g=m.encode_image(x);p=captured.pop();assert p.shape[1:]==(3137,768)
   if smoke:
    z=p.half().float()
    for block in m.visual.blocks[10:]:z=block(z,m.visual.einops_from_space,m.visual.einops_to_space,m.visual.einops_from_time,m.visual.einops_to_time,time_n=196,space_f=16,use_checkpoint=False)
    recon=m.visual.norm(z)[:,0]@m.image_projection;err=float((recon-g).norm()/g.norm());assert err<.002;errors.append(err)
   prefix[begin:begin+len(x)]=p.cpu().numpy().astype('float16');times[begin:begin+len(x)]=[v[1] for v in items]
   if begin%32==0:atomic(folder/'progress.json',dict(completed=begin+len(x),total=len(ids),elapsed_s=time.time()-start,pid=os.getpid(),gpu=gpu))
  decoded.close()
 hook.remove();prefix.flush();times.flush();atomic(folder/'complete.json',dict(passed=True,rows=len(ids),elapsed_s=time.time()-start,quantization_errors=errors,prefix_bytes=(folder/'prefix.npy').stat().st_size));print('cache complete',gpu,len(ids),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.gpu,a.smoke)
