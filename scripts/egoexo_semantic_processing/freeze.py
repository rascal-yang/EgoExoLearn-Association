"""Freeze the reviewed v2 training/VAL protocol after engineering passes."""
import copy
import random
import time
from sp_common import *


def main():
    target=EXP/'protocol_v2_frozen.json'
    assert not target.exists(),'Do not overwrite a frozen experiment protocol'
    main_check,dense_check=read(EXP/'preflight_main.json'),read(EXP/'preflight_dense.json')
    assert main_check['passed'] and dense_check['passed']
    for name in ('preparation','cache_replay','control_audit','trainer_engineering'):
        assert read(EXP/f'{name}.json')['passed'],name
    trainer=read(EXP/'trainer_engineering.json')
    assert trainer['actual_training_steps']>=3 and trainer['stop_exception_checkpoint_saved']
    protocol=copy.deepcopy(read(EXP/'design_protocol_v2_draft.json'))
    protocol.update(status='frozen_for_execution',frozen_time=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                    implementation_version='v2_native_memory_1',freeze_gates_pending=[])
    protocol['training']['batch']=min(main_check['supported_batch'],dense_check['supported_batch'])
    protocol['execution_scope']={'automatic_pipeline':'18 full fits plus final and internal-best full800 VAL and paired statistics; evaluate each fit when ready, scheduling never reads accuracy',
        'remaining_stages':'Internal/pixel interventions, official TEST confirmation and human review remain explicitly tracked; no fabricated completion.',
        'future_stage_code':'Separately hash before that stage, keep frozen method/hyperparameters/endpoints/diagnostic IDs; disclose implementation amendments.'}
    protocol['resources']['training_launched_by_design_task']=False
    protocol['resources']['training_authorized_by_user']='2026-10-27: 可以开始实验了'
    protocol['runtime_choices']={'AMP':'FP16 autocast with FP32 routing and reductions',
        'TF32':False,'route_noise_stream_seed':'70000+seed; draw48 slots per batch for all processing arms',
        'trunk_RNG_seed':'seed, model initialization isolated in fork_rng',
        'foreign_GPU_process_threshold_mib':1024,'required_free_mib_at_launch':20000,
        'staging':'Reuse existing hash-verified immutable SSD prefix copies via exact read-only row reads; no new copy'}
    protocol['preparation_result']=read(EXP/'preparation.json')
    rng=random.Random(20261027);schedule=[]
    for seed in SEEDS:
        arms=list(ARMS);rng.shuffle(arms)
        schedule.extend(dict(arm=arm,seed=seed) for arm in arms)
    protocol['schedule']=schedule
    sources=list(CODE.glob('*.py'))
    sources += [EXP/'design_protocol_v2_draft.json',EXP/'route_prototypes.npy',EXP/'route_vocabulary.json',EXP/'route_labels.npy',EXP/'route_donors.npy',
                EXP/'preparation.json',EXP/'preflight_main.json',EXP/'preflight_dense.json',EXP/'cache_replay.json',EXP/'control_audit.json',EXP/'trainer_engineering.json',
                BASE/'records.json',BASE/'phrases.json',BASE/'splits.npz',BASE/'text_features.npy',BASE/'val_episodes.npz',BASE/'plan.json',REL/'selection_holdout.npy',REL/'ssd_ready.json',
                ROOT/'data/association_annotations/association_val_hard.json',ROOT/'repos/EgoExoLearn/video_meta_downsampled.json',
                REP/'深入实验方案_v2.txt',REP/'v2_固定256题诊断队列.json',
                ROOT/'scripts/egoexo_textalign/model.py',ROOT/'scripts/egoexo_textalign/backend.py',ROOT/'scripts/egoexo_textalign/common.py',
                ROOT/'scripts/time_origin_reeval/run.py',ROOT/'repos/EgoExoLearn/association_benchmark/models/timesformer.py']
    sources += [MBASE/f'heads/M/seed{seed}/best.pt' for seed in SEEDS]
    sources += [CACHE/f'cache{gpu}/{name}.npy' for gpu in (2,3) for name in ('ids','pts')]
    sources += list((ROOT/'repos/EgoExoLearn/association_benchmark/models').glob('*.py'))
    protocol['frozen_sources']={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(sources))}
    base_checkpoint=Path('/data2_8t/user/wrt/projects/association_benchmark/exps/baseline/checkpoint_best.pt')
    stat=base_checkpoint.stat()
    protocol['external_base_checkpoint']=dict(path=str(base_checkpoint),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest(base_checkpoint),
        subsequent_verification='Read-only source stat identity checked before every worker; whole-file hash recorded at freeze')
    import torch,numpy
    protocol['runtime_versions']=dict(python=sys.version,torch=torch.__version__,CUDA=torch.version.cuda,numpy=numpy.__version__)
    protocol['prefix_cache_source_signatures']=[]
    for gpu in (2,3):
        p=SSD/f'cache{gpu}_prefix.npy';stat=p.stat()
        protocol['prefix_cache_source_signatures'].append(dict(path=str(p),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,prior_whole_file_sha256=next(r['sha256'] for r in read(REL/'ssd_ready.json')['files'] if r['target']==str(p))))
    atomic(target,protocol)
    atomic(EXP/'execution_ready.json',dict(passed=True,protocol_sha256=digest(target),batch=protocol['training']['batch'],fits=18,
           primary_endpoint='epoch4_final',stage='training_and_VAL_ready',time=time.time()))
    print(json.dumps(dict(protocol=str(target),sha256=digest(target),batch=protocol['training']['batch'],schedule=schedule),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
