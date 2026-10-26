"""Read native RGB pixels on the displayed MP4 timeline (edit lists enabled)."""
import math
import numpy as np

def load_times(path,times):
 import av
 times=[float(t) for t in times]
 if not times or not all(math.isfinite(t) and t>=0 for t in times):
  raise ValueError('Expected nonempty finite nonnegative playback times')
 targets=sorted(set(times));picked={};last=None
 def scan(seek):
  nonlocal last
  with av.open(str(path)) as c:
   stream=c.streams.video[0];stream.thread_count=1
   if seek:c.seek(int(max(0,min(targets)-.1)/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
   pos=0
   for f in c.decode(video=0):
    if f.pts is None:continue
    t=float(f.pts*f.time_base)
    if t < -1e-6:continue
    last=f
    while pos<len(targets) and t>=targets[pos]-1e-6:
     picked[targets[pos]]=(t,f.to_ndarray(format='rgb24'),False);pos+=1
    if pos==len(targets):break
 # Fresh decoder for each request group: no seek-history dependence.
 scan(True)
 if last is None:scan(False)
 if last is None:raise ValueError(f'No displayable frames: {path}')
 for t in targets:
  if t not in picked:picked[t]=(float(last.pts*last.time_base),last.to_ndarray(format='rgb24'),True)
 return np.stack([picked[t][1] for t in times]),[picked[t][0] for t in times],[picked[t][2] for t in times]
