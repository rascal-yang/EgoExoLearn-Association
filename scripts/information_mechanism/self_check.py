"""Small independent arithmetic/mask controls for the scientific intervention."""
import numpy as np
from analyze import score_metrics,intervals
from info_common import roi_mask,perturb
from freeze_regions import intersects,iou

def main():
    result=score_metrics(np.array([[.8,.1,.5],[.2,.9,.3]]),np.array([0,1]))
    assert np.allclose(result['margin'],[.3,.6]) and result['correct'].sum()==2
    # Increasing all similarities equally does not improve discrimination.
    shifted=score_metrics(np.array([[.9,.2,.6],[.3,1.,.4]]),np.array([0,1]))
    assert np.allclose(shifted['margin'],result['margin'])
    ci=intervals(np.ones(8),np.repeat(['a','b'],4),np.array(list('aabbccdd')),17,100)
    assert ci['ci95']==[1.,1.] and ci['ci9875']==[1.,1.]
    x=np.random.default_rng(1).normal(size=(3,224,224)).astype('float32')
    a=[10,20,60,80];b=[110,120,160,180]
    assert roi_mask(a).sum()==roi_mask(b).sum()
    y=perturb(x,a,4)
    assert np.array_equal(y[:,roi_mask(a)==0],x[:,roi_mask(a)==0])
    assert np.array_equal(perturb(x,a,0),x)
    assert not intersects(a,b,8) and iou(a,a)==1
    print('PASS: margins, candidate-wide score shifts, clustered constant effect, equal-area soft masks, pixel boundary/identity')

if __name__=='__main__':main()
