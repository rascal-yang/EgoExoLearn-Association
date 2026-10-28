"""Numerical and geometry gates before RGB/TEST workers produce scores."""
from pixel_interventions import *


def main():
    setup();data=Data(capacity=8)
    models={arm:load_model(data,arm,17) for arm in ('R0','R1','R2')}
    ep,rows=diagnostic();i=int(ep['indices'][rows[0],0]);key=read(BASE/'plan.json')['keys'][i]
    import backend as b
    b.ref.init_worker('egoexo')
    image,pts=decode16(key)
    with torch.inference_mode():
        x=torch.as_tensor(image[None],device='cuda');valid=mask_from_pts(pts[None])
        fresh=prefix_from_rgb(models['R0'].visual,x)
        native=models['R0'].visual(x) @ models['R0'].projection
        expected=F.normalize(native.float(),dim=-1)
        g=suffix(models['R0'],fresh,valid)
        torch.testing.assert_close(g,expected,atol=1e-6,rtol=1e-5)
        cached=data.batch([i],training=False)['x'].float()
        relative=float((fresh.half().float()-cached).norm()/cached.norm())
        assert relative<.002,relative
        k,j=data.locations[i];assert np.array_equal(pts,data.pts[k][j])
        clean_route=models['R2'].router(fresh,valid,'R2')
        windows,details=choose_windows(image,pts,clean_route,i,17)
        assert len(details)<=4
        for f in details:
            assert all(iou(f['centers_patch_rc'][0],c)<=.2 for c in f['centers_patch_rc'][1:])
            assert all(1<=r<=12 and 1<=c<=12 for r,c in f['centers_patch_rc'])
        variants=blurred_variants(image,windows)
        expected_change=torch.zeros_like(variants,dtype=torch.bool)
        for positions,centers in windows:
            for vi,(r,c) in enumerate(centers,1):
                for t in positions:expected_change[vi,:,t,(r-1)*16:(r+2)*16,(c-1)*16:(c+2)*16]=True
        assert not bool(((variants-variants[:1]).ne(0)&~expected_change).any())
        prefix=torch.cat([fresh]+[prefix_from_rgb(models['R0'].visual,variants[v:v+1]) for v in range(1,5)],0)
        checks={}
        for arm,m in models.items():
            adaptive=None
            if arm!='R0':
                routes=[m.router(prefix[v:v+1],valid,arm) for v in range(5)]
                adaptive={name:torch.cat([r[name] for r in routes],0) for name in ('assignment','mass')}
                fixed=dict(assignment=routes[0]['assignment'].reshape(1,16,48,196).expand(5,-1,-1,-1).reshape(80,48,196),
                           mass=routes[0]['mass'].reshape(1,16,48).expand(5,-1,-1).reshape(80,48))
                fixed_output=suffix(m,prefix,valid.expand(5,-1),fixed)
            output=suffix(m,prefix,valid.expand(5,-1),adaptive);assert bool(torch.isfinite(output).all())
            if arm!='R0':torch.testing.assert_close(output[0],fixed_output[0],atol=0,rtol=0)
            checks[arm]=float((output[1:]-output[:1]).abs().max())
        # Verify donor changes concept input while retaining actual patch input.
        d=data.batch([int(ep['indices'][rows[-1],0])],training=False)['x'].float()
        altered=fresh.clone();altered[:,0]=d[:,0]
        donor=models['R2'].router(altered,valid,'R2')
        assert torch.equal(altered[:,1:],fresh[:,1:])
        donor_effect=float((donor['global_logits']-clean_route['global_logits']).abs().max())
        assert donor_effect>0
        result=dict(passed=True,full_encoder_R0_replay_max_abs=float((g-expected).abs().max()),
                    native_prefix_cache_relative_L2=relative,PTS_exact=True,window_area_pixels=2304,
                    control_iou_max=.2,changed_pixels_confined_to_windows=True,finite_perturbed_all_arms=True,
                    clean_fixed_adaptive_exact=True,RGB_and_router_batch_size=1,
                    donor_global_logits_max_change=donor_effect,perturbed_feature_max_change=checks,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    atomic(POST/'RGB_TEST_preflight.json',result);print(result,flush=True)


if __name__=='__main__':main()
