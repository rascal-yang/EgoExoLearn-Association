"""Exploratory failure diagnosis using fixed checkpoints and saved memberships."""
from posthoc_core import *

DEG=POST/'degradation'
POLICIES=('dense','r2_memory','random_memory','compact_block10_only','compact_block11_only')


def saved_route(selection,cells):
    indices=torch.as_tensor(selection,device='cuda',dtype=torch.long)
    b,f,k=indices.shape
    assert (f,k)==(16,32)
    evidence=F.one_hot(indices,196).float().reshape(b*f,k,196)
    used=evidence.sum(1)
    assert bool((used<=1).all())
    assignment=torch.cat([evidence,cells[None]*(1-used)[:,None]],1)
    mass=assignment.sum(-1)
    assert bool((assignment.sum(1)==1).all()) and bool((mass.sum(1)==196).all())
    return dict(assignment=assignment,mass=mass,indices=indices)


def readout(model,x):
    return F.normalize((model.visual.norm(x)[:,0]@model.projection).float(),dim=-1)


def crossed_forward(block_model,readout_models,prefix,route=None,compact_blocks=(),capture=False):
    # R0's production forward intentionally ignores routing. Explicitly call
    # the unchanged native block primitive for this crossed intervention.
    x=prefix;states={}
    for j,block in enumerate(block_model.visual.blocks[10:],10):
        compact=j in compact_blocks
        a,m=(route['assignment'],route['mass']) if compact else (None,None)
        x=run_block(block,x,a,m)
        if capture:states[j]=x
    return {name:readout(model,x) for name,model in readout_models.items()},states


def cls_approximation(attention,normalized,route,dense_spatial=None):
    """Exact CLS grouping decomposition on one common normalized layer input.

    Exact grouped probability + exact within-group V reconstruct dense CLS.
    Replacing either quantity by the compact-memory approximation separately
    diagnoses the softmax-mass and V-averaging errors. No ground-truth labels.
    """
    b,length,dim=normalized.shape;h=attention.num_heads;d=dim//h
    a=route['assignment'].reshape(b,16,48,196);mass=route['mass'].reshape(b,16,48)
    weight,bias=attention.qkv.weight,attention.qkv.bias
    q=F.linear(normalized[:,:1],weight[:dim],None if bias is None else bias[:dim])
    k=F.linear(normalized,weight[dim:2*dim],None if bias is None else bias[dim:2*dim])
    v=F.linear(normalized,weight[2*dim:],None if bias is None else bias[2*dim:])
    q=q.reshape(b,1,h,d).transpose(1,2)*attention.scale
    k=k.reshape(b,length,h,d).transpose(1,2)
    v=v.reshape(b,length,h,d).transpose(1,2)
    logits=(q@k.transpose(-2,-1)).squeeze(2).float()
    prob=logits.softmax(-1)
    pl=logits[:,:,1:].reshape(b,h,16,196)
    pp=prob[:,:,1:].reshape(b,h,16,196)
    vv=v[:,:,1:].reshape(b,h,16,196,d)
    grouped_prob=torch.einsum('bhfn,bfkn->bhfk',pp,a)
    mean_logit=torch.einsum('bhfn,bfkn->bhfk',pl,a)/mass[:,None].clamp_min(1)
    compact_logits=(mean_logit+mass[:,None].clamp_min(1).log()).masked_fill(mass[:,None]==0,-1e9)
    compressed_prob=torch.cat([logits[:,:,:1],compact_logits.flatten(2)],2).softmax(-1)
    exact_prob=torch.cat([prob[:,:,:1],grouped_prob.flatten(2)],2)
    mean_v=torch.einsum('bhfnd,bfkn->bhfkd',vv,a)/mass[:,None,:,:,None].clamp_min(1)
    weighted_v=torch.einsum('bhfn,bhfnd,bfkn->bhfkd',pp,vv,a)/grouped_prob[...,None].clamp_min(1e-30)
    mean_v=torch.cat([v[:,:,:1],mean_v.flatten(2,3)],2)
    weighted_v=torch.cat([v[:,:,:1],weighted_v.flatten(2,3)],2)
    def output(p,value):
        z=(p[...,None]*value).sum(2).reshape(b,dim)
        return attention.proj_drop(attention.proj(z))
    native=output(prob,v)
    approximate=output(compressed_prob,mean_v)
    exact_mass=output(exact_prob,mean_v)
    exact_values=output(compressed_prob,weighted_v)
    reconstruction=output(exact_prob,weighted_v)
    reconstruction_error=(reconstruction-native).abs().max()
    assert float(reconstruction_error)<2e-5,float(reconstruction_error)
    if dense_spatial is not None:
        assert float((native-dense_spatial[:,0]).abs().max())<2e-5
    logz=logits.logsumexp(-1)
    true_group_logits=grouped_prob.clamp_min(1e-30).log()+logz[:,:,None,None]
    gap=(true_group_logits-compact_logits)[:,:,:,32:]
    valid_context=mass[:,:,32:]>0
    expanded=valid_context[:,None].expand_as(gap)
    assert float(gap[expanded].min())>-2e-4,float(gap[expanded].min())
    def relative(z):return (z-native).norm(dim=-1)/native.norm(dim=-1).clamp_min(1e-12)
    cp=compressed_prob[:,:,1:].reshape(b,h,16,48)
    dense_context=grouped_prob[:,:,:,32:].sum((2,3))
    compact_context=cp[:,:,:,32:].sum((2,3))
    assert float((compact_context-dense_context).max())<2e-5
    dense_evidence=grouped_prob[:,:,:,:32].sum((2,3))
    compact_evidence=cp[:,:,:,:32].sum((2,3))
    return dict(CLS_relative_error=relative(approximate),
                exact_group_mass_CLS_relative_error=relative(exact_mass),
                exact_group_values_CLS_relative_error=relative(exact_values),
                exact_both_CLS_relative_error=relative(reconstruction),
                dense_context_attention_mass=dense_context.mean(1),
                compact_context_attention_mass=compact_context.mean(1),
                dense_evidence_attention_mass=dense_evidence.mean(1),
                compact_evidence_attention_mass=compact_evidence.mean(1),
                context_Jensen_gap_weighted=(gap*grouped_prob[:,:,:,32:]).sum((1,2,3))/grouped_prob[:,:,:,32:].sum((1,2,3)).clamp_min(1e-12),
                exact_reconstruction_max_abs=reconstruction_error.expand(b))


def local_error_diagnostics(model,prefix,route):
    """Each compact error measured against native attention on identical input."""
    x=prefix;result={}
    for j,block in enumerate(model.visual.blocks[10:],10):
        temporal=block.timeattn(block.norm3(x),'b (f n) d','(b n) f d',{'n':196})
        if hasattr(block,'alpha_timeattn'):temporal=torch.tanh(block.alpha_timeattn)*temporal
        normalized=block.norm1(x+temporal)
        dense=block.attn(normalized,'b (f n) d','(b f) n d',{'f':16})
        compact=compact_spatial(block.attn,normalized,route['assignment'],route['mass'])
        values=cls_approximation(block.attn,normalized,route,dense)
        for part,take in (('all',slice(None)),('patch',slice(1,None)),('CLS',slice(0,1))):
            left,right=dense[:,take].flatten(1),compact[:,take].flatten(1)
            values[f'{part}_spatial_relative_L2']=(right-left).norm(dim=1)/left.norm(dim=1).clamp_min(1e-12)
            values[f'{part}_spatial_cosine']=F.cosine_similarity(left,right,dim=1)
        assert float((values['CLS_relative_error']-values['CLS_spatial_relative_L2']).abs().max())<3e-5
        result[j]={k:v.cpu().numpy() for k,v in values.items()}
        residual=x+block.drop_path(dense)
        x=residual+block.drop_path(block.mlp(block.norm2(residual)))
    return result


def verify_degradation():
    p=read(POST/'degradation_protocol.json')
    assert p['base_protocol_sha256']==digest(EXP/'protocol_v2_frozen.json')
    assert p['runtime_amendment_sha256']==digest(EXP/'runtime_amendment.json')
    for relative,expected in p['sources'].items():assert digest(ROOT/relative)==expected,relative
    return p
