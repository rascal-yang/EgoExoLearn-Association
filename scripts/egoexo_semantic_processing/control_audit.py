"""Dense bypass, exact M objective and auxiliary isolation on native weights."""
import torch
import numpy as np
import sp_common as common
from sp_common import *
from sp_data import Data
from sp_model import SemanticModel,make_noise
from route_loss import m_losses,route_losses


def main():
    guard(True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Data();train=np.load(BASE/'splits.npz')['train'];d=data.batch(train[:2])
    dense=SemanticModel('R0',17,data.prototypes,data.types).train()
    auxiliary=SemanticModel('R2',17,data.prototypes,data.types).train()
    # Load historical reference only after the new model/backend is established.
    sys.path.insert(0,str(ROOT/'scripts/egoexo_relations'))
    from rel_model import m_losses as historical_m_losses
    gen=torch.Generator(device='cuda').manual_seed(70017)
    a=dense(d['x'].float(),d['mask'],use_checkpoint=False)
    b=auxiliary(d['x'].float(),d['mask'],.1,make_noise(auxiliary,2,gen),use_checkpoint=False,dense=True)
    assert torch.equal(a['g'],b['g'])
    ma=m_losses(a,d,dense.scale);mb=m_losses(b,d,auxiliary.scale);old=historical_m_losses(a,d,dense.scale)
    assert all(torch.equal(x,y) for x,y in zip(ma,old))
    assert all(torch.equal(x,y) for x,y in zip(ma,mb))
    prefix,evidence=route_losses(auxiliary,b,d)
    trunk_b,head_b=auxiliary.parameter_groups()
    grads=torch.autograd.grad(prefix+evidence,trunk_b,allow_unused=True,retain_graph=True)
    assert all(g is None for g in grads)
    loss_a=ma[0]+.1*ma[1]+.1*ma[2]
    loss_b=mb[0]+.1*mb[1]+.1*mb[2]+.05*(prefix+evidence)
    loss_a.backward();loss_b.backward()
    trunk_a,_=dense.parameter_groups()
    grad_error=max(float((x.grad-y.grad).abs().max()) for x,y in zip(trunk_a,trunk_b))
    assert grad_error==0.,grad_error
    for group in (trunk_a,trunk_b,head_b):torch.nn.utils.clip_grad_norm_(group,1.,error_if_nonfinite=True)
    opt_a=torch.optim.AdamW(trunk_a,lr=1e-5,weight_decay=.01)
    opt_b=torch.optim.AdamW([dict(params=trunk_b,lr=1e-5),dict(params=head_b,lr=1e-4)],weight_decay=.01)
    opt_a.step();opt_b.step()
    error=max(float((x-y).abs().max()) for x,y in zip(trunk_a,trunk_b))
    assert error==0.,error
    # A mocked84C reading must write an isolated stop marker and raise.
    old_exp,old_call=common.EXP,common.subprocess.check_output
    thermal_folder=EXP/'engineering_thermal_guard'
    thermal_folder.mkdir(exist_ok=True)
    stopped=False
    try:
        common.EXP=thermal_folder
        common.subprocess.check_output=lambda *args,**kwargs:'84\n'
        try:common.guard(True)
        except StudyStop:stopped=True
        assert stopped and (thermal_folder/'THERMAL_STOP.json').exists()
    finally:
        common.EXP=old_exp;common.subprocess.check_output=old_call
    atomic(EXP/'control_audit.json',dict(passed=True,dense_descriptor_exact=True,
           original_M_loss_exact=True,bypass_auxiliary_trunk_gradient_path=False,
           trunk_gradient_max_error=grad_error,trunk_update_max_error=error,
           simulated84C_stop_marker_and_exception=True,actual_GPU_thermal_stop=False,TEST_access=False))
    print('CONTROL_AUDIT_PASSED',flush=True)


if __name__=='__main__':main()
