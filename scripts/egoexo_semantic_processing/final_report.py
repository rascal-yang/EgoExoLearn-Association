"""Integrate all frozen automatic stages without redefining the main method."""
from posthoc_core import *


def main():
    protocol=verify_protocol()
    val=read(REP/'v2_VAL结果.json');internal=read(REP/'v2_内部干预结果.json')
    pixels=read(REP/'v2_像素干预结果.json');test=read(REP/'v2_TEST确认结果.json')
    audit=read(REP/'v2_完整运行核验.json');materials=read(REP/'人工核验材料/materials_complete.json')
    for stage in ('internal','pixels','test'):
        verify_stage(stage)
        stage_protocol=read(POST/f'{stage}_protocol.json')
        for job in stage_protocol['jobs']:
            folder=POST/stage/job['name'];done=read(folder/'complete.json')
            assert done['passed'] and done['stage_protocol_sha256']==digest(POST/f'{stage}_protocol.json')
            filename='features.npy' if stage=='test' else 'scores.npz'
            hashkey='features_sha256' if stage=='test' else 'scores_sha256'
            assert digest(folder/filename)==done[hashkey]
    histories={};train_seconds=0
    for arm in ARMS:
        histories[arm]={}
        for seed in SEEDS:
            folder=EXP/f'fits/{arm}/seed{seed}'
            histories[arm][str(seed)]=read(folder/'history.json')
            train_seconds+=read(folder/'complete.json')['elapsed_s']
    reports=['v2_VAL结果.json','v2_内部干预结果.json','v2_像素干预结果.json','v2_TEST确认结果.json','v2_完整运行核验.json']
    routing_path=REP/'v2_路由诊断.json'
    if routing_path.exists():reports.append(routing_path.name)
    integrated=dict(main_method='R2',primary_endpoint='epoch4_final',seeds=list(SEEDS),
         automatic_stages_complete=True,human_review_complete=False,project_target_met=val['project_target_met'],
         VAL={arm:val['models'][arm]['mean_accuracy'] for arm in ARMS},TEST=test['models'],
         primary_VAL_comparisons={k:val['comparisons'][k] for k in ('R2-R0','R2-R1')},
         TEST_comparisons=test['comparisons'],training_audit=audit,
         runtime=dict(sum_of_fit_reported_elapsed_gpu_hours=train_seconds/3600,
                      elapsed_definition='Includes per-fit training/holdout time; excludes VAL, RGB/TEST decoding, diagnostics and engineering probes.',
                      final_artifact_storage_gib=storage_bytes()/2**30,whole_encoder_efficiency_claim_established=False),
         sources={str((REP/r).relative_to(ROOT)):digest(REP/r) for r in reports},
         report_generator_sha256=digest(Path(__file__)),
         base_protocol_sha256=digest(EXP/'protocol_v2_frozen.json'),
         stage_protocol_sha256={stage:digest(POST/f'{stage}_protocol.json') for stage in ('internal','pixels','test')},
         test_memory20g_runtime_amendment_sha256=read(POST/'test_protocol.json').get('runtime_memory20g_amendment_sha256'),
         caveats=['Historical M holdout selection and VAL/TEST exposure',
                  'All547 TEST long videos also occur in TRAIN; no positive-duration interval overlap by annotation times',
                  'Fixed candidate pool and only three fixed seeds; source clustering incompletely treats candidate reuse',
                  'Narration-derived weak concepts; no completed human regions or fine-action adjudication',
                  'R5 changes evidence count and represented patch mass as well as allocation',
                  'Internal and pixel interventions are distribution shifts and descriptive mechanism evidence'],
         next_action='Two independent human reviews; no automatic architecture expansion or hyperparameter sweep')
    atomic(REP/'v2最终实验报告.json',integrated)
    lines=['EgoExoLearn 语义路由实验 v2：自动实验最终报告',
           '主方案固定为R2；主终点固定第4轮；种子17/29/43整体报告。',
           '18次训练、800题VAL、固定256题内部/RGB干预、9模型2200题TEST确认已完成。',
           '预声明提升目标未达到。人工核验材料已准备，人工标注尚未完成。','',
           '一、完整VAL主结果（%）','组别    seed17    seed29    seed43    三种子均值']
    for arm in ARMS:
        r=val['models'][arm]
        lines.append(f"{arm}      {r['17']['accuracy']:.3f}    {r['29']['accuracy']:.3f}    {r['43']['accuracy']:.3f}    {r['mean_accuracy']:.3f}")
    for key in ('R2-R0','R2-R1'):
        r=val['comparisons'][key]
        lines.append(f"{key}: {r['gain_pp']:+.3f}个百分点；97.5%源视频聚类区间 {r['source_cluster_interval']}。")
    lines.extend(['R2三个种子均低于R0；删除最大视频、替代聚类和内部best次终点未改变R2−R0负向结论。',
                  'R2−R3为+0.042个百分点；正确额外语义对应没有表现出可辨认的整体优势。',
                  'R5均值51.500%，高于R2但仍低于R0；该比较同时改变32→48证据数和196→48代表质量，不能归因于单独移除上下文。','',
                  '二、TEST确认（%）'])
    for arm,r in test['models'].items():lines.append(f"{arm}: 三种子 {r['per_seed_accuracy']}；均值 {r['mean_accuracy']:.3f}")
    for key,r in test['comparisons'].items():lines.append(f"{key}: {r['gain_pp']:+.3f}个百分点；97.5%源视频聚类区间 {r['source_cluster_interval97_5']}")
    lines.extend(['TEST未用于选epoch、模型或参数。项目存在历史TEST接触，不能称为首次盲测。',
                  f"TEST数据审计: {test['interval_audit']}",
                  '547个TEST长视频均见于TRAIN；标注区间无正时长重叠，不等于全新视频泛化，也不是逐解码帧重叠审计。','',
                  '三、同权重内部诊断（固定256题，不能替代完整800题主结果）'])
    for c,r in internal['results'].items():lines.append(f"{c}: {r['mean_accuracy']:.3f}%，较原生 {r['gain_vs_native_pp']:+.3f}pp，margin变化 {r['margin_change_vs_native']:+.5f}")
    lines.extend(['原生回放与原FP32描述符逐值一致。显式语义项会改变路由和预测，但没有显示出稳定任务收益。',
                  '屏蔽/同质化上下文显著改变该已训练模型的表现，说明上下文信息被使用；不能据此推翻稠密基线更高的主结果。',
                  '恢复稠密注意力的变化属于固定权重下的分布替换，不等于重新训练稠密基线的组件因果效应。','',
                  '四、原视频像素诊断'])
    if routing_path.exists():
        routing=read(routing_path)['metrics']
        def recall(arm,kind):
            return float(np.mean([routing[arm][str(s)]['weak_concept_recall'][kind]['micro_recall'] for s in SEEDS])*100)
        lines[-1:-1]=[f"弱叙述概念诊断：R2名词Recall@4 {recall('R2','noun'):.2f}%，R3 {recall('R3','noun'):.2f}%；R2动词Recall@2 {recall('R2','verb'):.2f}%，R3 {recall('R3','verb'):.2f}%。",
                      '标签打乱明显改变概念预测，但主任务未同步退化；概念命中不等于有效的跨视角决策证据。详见v2_路由诊断.json，包含缺失/词表外分母、频次、方向、覆盖、有效秩和跨种子稳定性。']
    for arm in ('R0','R1','R2'):
        for mode,r in pixels['results'][arm].items():lines.append(f"{arm}/{mode}: 清洁 {r['clean_accuracy']:.3f}%，处理窗 {r['selected_accuracy']:.3f}%，处理窗相对对照的超额margin下降 {r['selected_minus_control_margin_drop']:+.6f}；95%描述性区间 {r['descriptive_source_cluster_ci95']}")
    lines.extend([f"帧×对照窗口匹配层级: {pixels['control_frame_matching_counts']}",
                  '所有条件均重算完整前缀；候选不变；处理窗48×48、Gaussian49/sigma8。每种子相同物理窗口用于R0/R1/R2。',
                  '批大小敏感的硬排名由清洁一致性门禁发现后停止；修订为逐片段编码/路由，共享清洁分组，修订及原协议已归档。',
                  '局部模糊只测试画面敏感性；无人工区域真值，不能宣称物体定位或精细动作理解。','',
                  '五、运行核验、人工材料和本轮处置',
                  '18/18次各2676步，相同4轮样本曝光；冻结前缀逐值不变；总数值重试0。',
                  'TEST按用户要求增加真实后续RGB批次的GPU预取；保留batch16、FP32、原block9半精度边界及片段顺序。预取输入/PTS逐值一致，9模型描述符最大差0；两分片均从已提交游标续跑。',
                  f"各fit报告时长相加约 {train_seconds/3600:.2f} GPU小时；不含VAL、RGB/TEST、解码和工程探测。无完整编码器效率优势结论。",
                  f"人工材料：{materials['clips']}片段、{materials['frames']}帧、{materials['action_triplets']}动作三元组、2份独立标注表，人工字段全部留空。",
                  '区域可见性/框及动作三元组需真人独立核验，保留原始分歧并计算一致性后才能报告。',
                  '按预声明规则结束本轮自动训练和评测，不根据失败结果追加架构、扫参数或将消融最高分改名为主方法。',
                  '后续仅剩人工核验及基于真实人工结果的一致性/区域分析。'])
    (REP/'v2最终实验报告.txt').write_text('\n'.join(lines)+'\n')
    original=read(EXP/'pipeline_status.json')
    atomic(EXP/'pipeline_status.json',dict(original,phase='all_automatic_experiments_complete',remaining=['human_review_and_agreement_analysis'],
           post_training_report=str((REP/'v2最终实验报告.txt').relative_to(ROOT)),updated=time.time()))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
