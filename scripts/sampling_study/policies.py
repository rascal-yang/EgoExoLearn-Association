"""Predeclared, annotation-only sampling policies. No label/score dependent sampling."""
import math
COMMON=['original','strict','uniform','context0125','context025','context05','early','late','multi2','multi4']
EXTRA=['fps2','fps8']
COST={k:(2 if k=='multi2' else 4 if k=='multi4' else 1) for k in COMMON+EXTRA}

def bins(lo,hi,n,phase=.5):
 if hi<=lo:raise ValueError('No frame inside annotation')
 seg=(hi-lo-1)/n
 return tuple(min(hi-1,int(min(round(seg*i)+lo,hi-1)+(min(round(seg*(i+1))+lo,hi)-min(round(seg*i)+lo,hi-1))*phase)) for i in range(n))

def fixed_plans(start,end,fps,length,n):
 if not 0<=start<end:raise ValueError('Invalid annotation')
 s=min(start,length/fps);e=max(s+1,min(end,length/fps));off=round(s*fps)
 original=bins(off,min(off+max(int((e-s)*fps),n),length),n)
 def window(a,b,phase=.5):return bins(max(0,math.ceil(a*fps-1e-7)),min(length,math.ceil(b*fps-1e-7)),n,phase)
 uniform=window(start,end)
 strict=uniform if end-start<1 or any(i/fps<start-1e-7 or i/fps>=end-1e-7 for i in original) else original
 p={'original':[original],'strict':[strict],'uniform':[uniform]}
 for name,pad in [('context0125',.125),('context025',.25),('context05',.5)]:p[name]=[window(max(0,start-pad),end+pad)]
 p['early']=[window(start,end,.25)];p['late']=[window(start,end,.75)]
 p['multi2']=p['early']+p['late'];p['multi4']=[window(start,end,x) for x in [.125,.375,.625,.875]]
 return p

def prism_plans(start,end):
 if not 0<=start<end:raise ValueError('Invalid annotation')
 dur=end-start;n=min(128,max(1,math.ceil(dur*4)))
 def view(a,b,n,phase=.5,clamp=True):return dict(times=tuple(a+(i+phase)*(b-a)/n for i in range(n)),window=(a,b) if clamp else None)
 orig=view(start,end,n,clamp=False) if n/4>=dur else dict(times=tuple(min(end-.001,start+(i+.5)/4) for i in range(n)),window=None)
 strict=dict(times=orig['times'],window=(start,end))
 p={'original':[orig],'strict':[strict],'uniform':[view(start,end,n)]}
 for name,pad in [('context0125',.125),('context025',.25),('context05',.5)]:p[name]=[view(max(0,start-pad),end+pad,n)]
 p['early']=[view(start,end,n,.25)];p['late']=[view(start,end,n,.75)]
 p['multi2']=p['early']+p['late'];p['multi4']=[view(start,end,n,x) for x in [.125,.375,.625,.875]]
 for name,fps in [('fps2',2),('fps8',8)]:p[name]=[view(start,end,min(128,max(1,math.ceil(dur*fps))))]
 return p

def choose_best(scores,split='val',allowed=None):
 if split!='val':raise ValueError('Selection is validation-only')
 names=[k for k in COMMON+EXTRA if k in scores and (allowed is None or k in allowed)]
 return min(names,key=lambda k:(-scores[k],COST[k],(COMMON+EXTRA).index(k)))
