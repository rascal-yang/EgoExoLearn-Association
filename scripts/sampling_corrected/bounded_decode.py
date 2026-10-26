"""Single-thread playback decoding with explicit ACTUAL-frame window checks."""
import math
import numpy as np

def decode_views(path,views):
 import av
 requests=set()
 for times,w in views:
  if not times or any(not math.isfinite(t) or t<0 for t in times):raise ValueError('Invalid target')
  if w is not None and not 0<=w[0]<w[1]:raise ValueError('Invalid window')
  for t in times:requests.add((max(t,w[0]) if w else t,w))
 requests=sorted(requests,key=lambda r:(r[0],str(r[1])));windows={w for _,w in requests if w};chosen={};saved={};lastinside={};last=None;pos=0
 def save(f):
  t=float(f.pts*f.time_base)
  if t not in saved:saved[t]=f.to_ndarray(format='rgb24')
  return t
 with av.open(str(path)) as c:
  stream=c.streams.video[0];stream.thread_count=1
  begin=min([t for t,w in requests]+[w[0] for w in windows]);c.seek(int(max(0,begin-.1)/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
  for f in c.decode(video=0):
   if f.pts is None:continue
   t=float(f.pts*f.time_base)
   if t<0:continue
   last=f
   for w in windows:
    if w[0]<=t<w[1]:lastinside[w]=f
   while pos<len(requests) and t>=requests[pos][0]-1e-6:
    req=requests[pos];w=req[1];picked=f if w is None or w[0]<=t<w[1] else lastinside.get(w)
    if picked is None:raise ValueError(f'No displayable frame in {w}: {path}')
    chosen[req]=save(picked);pos+=1
   if pos==len(requests):break
  for req in requests[pos:]:
   picked=last if req[1] is None else lastinside.get(req[1])
   if picked is None:raise ValueError(f'No displayable frame in {req[1]}: {path}')
   chosen[req]=save(picked)
 pts=sorted(saved);index={t:i for i,t in enumerate(pts)};sequences=[]
 for times,w in views:
  seq=tuple(index[chosen[(max(t,w[0]) if w else t,w)]] for t in times)
  if w:assert all(w[0]<=pts[i]<w[1] for i in seq)
  sequences.append(seq)
 return np.stack([saved[t] for t in pts]),pts,sequences
