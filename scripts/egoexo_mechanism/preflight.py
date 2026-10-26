import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','2')
from mech_common import *
import numpy as np,torch
from model import VideoTail,auxiliary
from data import Prefixes
from mech_loss import auxiliary_variant

def main():
 torch.set_num_threads(2);a=Prefixes();m=VideoTail();ids=np.load(BASE/'splits.npz')['train'][:16];x,mask,txt,cid,phr,pos,typ,_=a.batch(ids);perm=torch.arange(len(x),device='cuda')
 with torch.cuda.amp.autocast():
  g,p,f=m(x);gc,lc=auxiliary(g,p,f,phr,pos,typ,mask,m.scale,perm)
  gn,ln=auxiliary_variant('N',g,p,f,phr,pos,typ,mask,m.scale,perm)
  gv,lv=auxiliary_variant('V',g,p,f,phr,pos,typ,mask,m.scale,perm)
  gm,lm=auxiliary_variant('M',g,p,f,phr,pos,typ,mask,m.scale,perm)
 torch.testing.assert_close(gc,gn);torch.testing.assert_close(gc,gv);torch.testing.assert_close(gc,gm);torch.testing.assert_close(lc,ln+lv)
 for loss in [ln,lv,lm]:
  m.zero_grad();loss.backward(retain_graph=True);assert torch.isfinite(loss);assert m.projection.grad.abs().sum()>0;assert all(v.grad is None for v in m.visual.blocks[0].parameters())
 temporal=m.visual.temporal_embed.detach().float();atomic(EXP/'preflight.json',dict(passed=True,local_C=float(lc),local_N=float(ln),local_V=float(lv),local_M=float(lm),N_plus_V_equals_C=True,global_identical=True,temporal_embedding_norm=float(temporal.norm()),temporal_frame_difference_norm=float((temporal-temporal[:,:1]).norm())))
 print(read(EXP/'preflight.json'))
if __name__=='__main__':main()
