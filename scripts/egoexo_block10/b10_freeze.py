"""Freeze six fits only after engineering and read-only R0 reuse checks pass."""
import copy
import numpy as np
from b10_common import *


def main():
    assert not PROTOCOL.exists()
    parent=old_common.verify_protocol()
    for name in ('algebra_gate.json','memory_S1.json','memory_S2.json','prefetch_S1.json','prefetch_S2.json','resume_gate.json'):
        assert read(EXP/name)['passed'],name
    for arm in ARMS:
        gate=read(EXP/f'prefetch_{arm}.json')
        assert gate['batch']==48 and gate['depth']==6 and gate['steps']==32
        assert gate['input_bytes_exact'] and gate['sample_order_exact'] and gate['finite_loss_and_gradients']
        policy=read(EXP/f'memory_{arm}.json')['policies'][-1]
        assert policy['policy']==[10] and policy['output_error']==0 and policy['gradient_error']==0 and policy['update_error']==0
    resume=read(EXP/'resume_gate.json')
    assert resume['resumed_from_steps']==3 and resume['resumed_to_steps']==4 and resume['exposure_delta_exact_next48']
    runtime=read(EXP/'runtime_choices.json')
    inputs=[OLD_EXP/'protocol_v2_frozen.json',OLD_EXP/'runtime_amendment.json',
            BASE/'val_episodes.npz',BASE/'splits.npz',BASE/'records.json',BASE/'phrases.json',
            REL/'selection_holdout.npy',REL/'ssd_ready.json',
            OLD_EXP/'route_prototypes.npy',OLD_EXP/'route_labels.npy',OLD_EXP/'route_donors.npy',OLD_EXP/'route_vocabulary.json',
            ROOT/'data/association_annotations/association_val_hard.json',ROOT/'repos/EgoExoLearn/video_meta_downsampled.json']
    baseline={};exposure_sha=None
    for seed in SEEDS:
        folder=OLD_EXP/f'fits/R0/seed{seed}';ck=read(folder/'complete.json');manifest=read(folder/'manifest.json')
        assert ck['passed'] and ck['steps']==2676 and ck['frozen_prefix_exact'] and ck['numerical_retries']==0
        assert ck['protocol_sha256']==digest(OLD_EXP/'protocol_v2_frozen.json')
        assert manifest['batch']==48 and manifest['epochs']==4 and manifest['train_clips']==32079 and manifest['holdout_clips']==1171
        initial=MBASE/f'heads/M/seed{seed}/best.pt'
        assert manifest['initial_M_sha256']==digest(initial)
        assert digest(folder/'final.pt')==ck['final_sha256']
        assert digest(folder/'exposures.npy')==ck['exposure_sha256']
        if exposure_sha is None:exposure_sha=ck['exposure_sha256']
        assert ck['exposure_sha256']==exposure_sha
        for endpoint in ('final','best'):
            ev=read(folder/f'val_{endpoint}/complete.json')
            assert ev['passed'] and ev['questions']==800 and ev['precision']=='FP32'
            assert digest(folder/f'val_{endpoint}/scores.npz')==ev['scores_sha256']
            inputs.extend([folder/f'val_{endpoint}/complete.json',folder/f'val_{endpoint}/scores.npz'])
        baseline[str(seed)]=dict(folder=str(folder.relative_to(ROOT)),initial_M_sha256=digest(initial),
                final_sha256=ck['final_sha256'],accuracy=read(folder/'val_final/complete.json')['accuracy'])
        inputs.extend([initial,folder/'complete.json',folder/'manifest.json',folder/'final.pt',folder/'exposures.npy'])
    files=list(CODE.glob('*.py'))+inputs+[REP/'实验方案与执行范围.txt']
    files.extend(EXP/name for name in ('algebra_gate.json','memory_S1.json','memory_S2.json','prefetch_S1.json',
                  'prefetch_S2.json','resume_gate.json','worst_training_batch.json','runtime_choices.json'))
    signatures=[]
    for gpu in (2,3):
        path=SSD/f'cache{gpu}_prefix.npy';st=path.stat()
        signatures.append(dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
    training=copy.deepcopy(parent['training']);training['fits']=6
    training['batch_decision']='Preserve48 from the parent R0; passed both arms worst natural batch and32-step prefetch gates.'
    p=dict(design_version='3.0_block10_only',status='frozen_for_execution',created_date='2026-10-28',
        parent_protocol_sha256=digest(OLD_EXP/'protocol_v2_frozen.json'),
        question='With native dense final-block reading preserved, does semantic routing in block10 add value beyond the matched content router and continued-M baseline?',
        exploratory_after_parent_results=True,modified_spatial_blocks=[10],block11='native_dense',
        groups=[dict(id='S1',parent_math='R1',route_loss='correct',semantic_affinity=False),dict(id='S2',parent_math='R2',route_loss='correct',semantic_affinity=True)],
        training=training,loss=copy.deepcopy(parent['loss']),router=copy.deepcopy(parent['router']),
        initialization='Original matching-seed M best, never R0/R2 final; frozen block9 prefix reused.',
        baseline_reuse=dict(arm='R0',read_only=True,exposure_sha256=exposure_sha,models=baseline,
                           dense_bypass_descriptor_gradient_update_exact=True,common_training_conditions_verified=True),
        evaluation=dict(annotation='data/association_annotations/association_val_hard.json',questions=800,
             annotation_sha256=digest(ROOT/'data/association_annotations/association_val_hard.json'),precision='FP32',score='cosine20',
             primary_endpoint='epoch4_final',secondary_endpoint='minimum internal caption loss epoch0_to4 earliest tie',
             primary_comparisons=[['S2','R0'],['S2','S1']],secondary_comparison=['S1','R0'],
             bootstrap=dict(iterations=20000,seed=20261028,cluster='query_source_video',retain_seeds_jointly=True,primary_interval=.975,secondary_interval=.95),
             sensitivity=['macro_video','macro_metadata_task','correct_candidate_video_clusters','shared_physical_video_Poisson_product',
                          'delete_largest_source','leave_one_source_out','task_stratified_source','internal_best_secondary'],
             TEST='No additional TEST evaluation or tuning in this run'),
        success_target=dict(S2_R0_mean_gain_pp_min=1.,S2_R0_all_seed_gains_positive=True,both_primary_interval_lower_bounds_positive=True,
             S1_assessment='Report effect and interval; a nonsignificant loss does not prove noninferiority.'),
        runtime_amendment=runtime,runtime_amendment_sha256=digest(EXP/'runtime_choices.json'),
        runtime_compatibility_note='Legacy runtime_amendment keys describe choices fixed before this new protocol; original protocol/runtime files remain unchanged.',
        cache_file_signatures=signatures,
        frozen_sources={str(path.relative_to(ROOT)):digest(path) for path in files},
        schedule=[dict(arm=a,seed=s) for s,pair in ((17,('S2','S1')),(29,('S1','S2')),(43,('S2','S1'))) for a in pair],
        resources=dict(GPUs=[2,3],target_device_memory_gib=20,temperature_stop_above_celsius=83,automatic_thermal_restart=False,
                       combined_artifact_limit_gib=12,new_prefix_copy=False),
        authorized_by_user='2026-10-28: 可以，开始实验吧',
        limits=['Known VAL used for development; exploratory follow-up, not pristine confirmation',
                'Historical TEST exposure; no new TEST used for tuning',
                'TRAIN/VAL long-video sharing and initialization holdout selection remain',
                'Fixed candidate pool, three seeds, incomplete candidate reuse dependence treatment',
                'Weak narration labels, no human grounding/action claims'],
        scope='Six complete fits; full800 final/best VAL, paired statistics and automatic release. No result-driven architecture or hyperparameter branches.')
    atomic(PROTOCOL,p)
    atomic(EXP/'execution_ready.json',dict(passed=True,protocol_sha256=digest(PROTOCOL),fits=6,baseline_reused_read_only=True,
           batch=48,epochs=4,prefetch_depth=6,all_engineering_gates_passed=True,combined_storage_gib=storage_bytes()/2**30))
    print('BLOCK10_PROTOCOL_FROZEN',digest(PROTOCOL),flush=True)


if __name__=='__main__':main()
