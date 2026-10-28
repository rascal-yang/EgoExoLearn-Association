"""Freeze the initial finite layer scan after numerical gates, before scores."""
import time
from lc_common import *


def main():
    path=EXP/'protocol_frozen.json'
    assert not path.exists(),'Never overwrite an executed protocol'
    gate=read(EXP/'preflight_gpu2_b32.json')
    assert gate['passed'] and not gate['task_scores_computed'] and gate['conditions']==25
    assert gate['dense_descriptor_max_abs']==0
    assert max(gate['shared_vs_full_max_abs'].values())<=2e-6
    assert gate['peak_allocated_GiB']<19
    _,rows,ids=cohort()
    sources={str(f.relative_to(ROOT)):digest(f) for f in sorted(CODE.glob('*.py'))}
    for name in ('posthoc_core.py','rgb_gpu_prefetch.py','sp_common.py','sp_model.py','memory_attention.py','summarize.py'):
        f=OLD_CODE/name;sources[str(f.relative_to(ROOT))]=digest(f)
    for name in ('model.py','backend.py','common.py'):
        f=ROOT/'scripts/egoexo_textalign'/name;sources[str(f.relative_to(ROOT))]=digest(f)
    files=[old.EXP/'protocol_v2_frozen.json',old.EXP/'runtime_amendment.json',
        old.REP/'v2_固定256题诊断队列.json',BASE/'plan.json',BASE/'val_episodes.npz',
        EXP/'preflight_gpu2_b32.json']
    for seed in SEEDS:
        files.extend([old.EXP/f'fits/R0/seed{seed}/final.pt',old.EXP/f'fits/R0/seed{seed}/complete.json'])
    inputs={str(f.relative_to(ROOT)):digest(f) for f in files}
    p=dict(version='1.0',status='frozen_before_task_scores',frozen_at=time.time(),
        scope='Stage1: all12 single-layer interventions at96 spatial patch KV slots, plus exact RGB FP32 dense baseline',
        conditions=conditions(),seeds=list(SEEDS),questions=len(rows),unique_clips=len(ids),
        all_query_and20_candidates=True,training=False,TEST_access=False,
        precision='FP32; TF32 disabled; dense and compressed use identical newly decoded RGB path',
        random='Per-global-clip/frame stateless32bit integer hash rank;80 singleton evidence+16 same-frame4x4 cells',
        similarity='Even/odd bipartite cosine grouping on current normalized pre-spatial features; stable sort; multiple passes to96; exact count conservation',
        invariants='All3137 queries,16 temporal frames,residuals,MLPs,CLS retained; only spatial KV memory changes',
        scan_batch=32,decode_workers_per_gpu=6,sharding='sorted unique global clip IDs sliced shard::2;25 configs×3 seeds each',
        scan_acceleration='Shared clean first10 weights/prefixes verified equal; every modified suffix recomputed; no block9 FP16 cache used',
        timing=dict(batches=[1,8],warmups_per_config=3,paired_rounds=12,repeats_per_measurement=2,
            order_seed=20261028,balanced_order='Random permutation each round and random baseline/candidate pair order',
            input='first8 sorted cohort clip IDs; identical per condition; no prefetch or writes in timed interval',
            includes='stem+all12 blocks+selector/matching+aggregation+mass correction+norm/projection/normalization',
            extra='Native temporal/spatial/MLP CUDA profiling; batch8 decode+transfer+encoder representative dense/random_b05/similarity_b05;3 warm-cache repeats'),
        screen=dict(min_gain_pp=-1.0,min_per_seed_gain_pp=-2.0,min_margin_change=-.005,
            speed='batch8 paired mean candidate/baseline latency<1 AND descriptive within-run95% bootstrap upper<1',
            rank='Descending mean accuracy gain, descending margin change, ascending batch8 paired median ratio, layer, operator',
            maximum_distinct_layers=3,interpretation='Exploratory screening, not noninferiority proof'),
        later_stages='Only if joint quality/speed screen passes: freeze a separate finite budget-curve/combination protocol before those new task scores; block11 retained as sensitive control. No automatic training or TEST.',
        statistics='Per-seed accuracy, correct-minus-hardest-wrong cosine margin, paired source-cluster20000 bootstrap; seeds joint; all scan intervals descriptive, no multiplicity-adjusted significance claim',
        resources=dict(gpus=[2,3],memory_cap_GiB=20,thermal_stop_above_C=83,auto_thermal_restart=False,
            combined_old_semantic_block10_new_artifact_cap_GiB=12),sources=sources,inputs=inputs,
        preflight_summary=gate)
    atomic(path,p)
    text=['逐层K/V可压缩性与实际速度：本轮冻结执行范围',
        '25个条件：新RGB FP32稠密基线；block0–11各自随机/相似性96槽。三种子，固定256题及全部候选。',
        '新完整RGB路径与原模型输出一致；共享前缀与完整重跑一致；196槽单位分组逐层相对误差<1e-5。',
        'GPU2先跑clip分片0；GPU3先做完整编码器batch1/8配对计时，再跑clip分片1。',
        '固定batch32，不训练、不读取TEST；温度>83停止自己的任务、不自动重启；结束释放显卡。',
        '包含分组/聚合开销，记录CUDA和同步墙钟延迟；峰值显存单列；解码+传输+编码另测。',
        '候选筛选：平均准确率变化>=-1pp，各种子>=-2pp，margin变化>=-.005，batch8配对耗时比均值和其本次运行95%区间上界均<1。',
        '区间和筛选仅为探索性，不证明无损；没有联合可行条件时报告否定结果，不自动扩方法或开训练。',
        '若出现候选，再单独冻结少量预算曲线/组合的执行清单；本轮不声称未测预算或组合。',
        f'protocol SHA256: {digest(path)}',f'当前三项研究合计磁盘用量 {storage_bytes()/2**30:.3f} GiB（上限12）。']
    (REP/'本轮冻结执行范围.txt').write_text('\n'.join(text)+'\n')
    print('\n'.join(text),flush=True)


if __name__=='__main__':main()
