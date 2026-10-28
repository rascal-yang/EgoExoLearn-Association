# 近期实验：路由、block10与逐层压缩

2026-10-28导出。本版本保存实际执行的77个Python研究源文件、历史协议与汇总结果。新源码与原工作区逐字节一致；此前发布的便携baseline路径适配保留，差异见来源清单。

## 执行范围

| 目录 | 实验 | 已完成范围 |
|---|---|---|
| `scripts/egoexo_relations/` | 关系弱标签、操作槽位、绑定/打乱 | 5组、seed17、800题VAL |
| `scripts/egoexo_tokenmerge/` | 共享静态token、动态token、近似误差 | 历史推理与机制实验 |
| `scripts/egoexo_semantic_processing/` | R0–R5、内容/语义路由、退化诊断、像素干预 | 18次训练；800题VAL；R0/R1/R2的2200题TEST |
| `scripts/egoexo_block10/` | S1/S2只压缩block10，block11稠密 | 6次训练；800题VAL；复用匹配R0 |
| `scripts/egoexo_layer_compressibility/` | 从RGB重算的12层随机/相似性96槽扫描与整网测速 | 25条件×3种子，固定256题及全部候选；无训练 |

block从0编号。近期论文讨论的联合新方法仍处于设计阶段，不是本版本已验证的方法。

## 主要结果

800题VAL、固定epoch4、种子17/29/43：

| 实验 | 条件 | 准确率均值 |
|---|---|---:|
| 两层K/V | R0 稠密继续训练 | 54.292% |
| 两层K/V | R1 内容路由 | 48.167% |
| 两层K/V | R2 显式语义路由 | 47.417% |
| block10-only | S1 内容路由 | 54.625% |
| block10-only | S2 显式语义路由 | 54.042% |

R0/R1/R2的2200题TEST均值分别55.242%、51.091%、50.606%。两层方案退化；block10-only未达到预设提升目标，差值区间跨0。R1/S1本身也接受语义监督，不能将它们理解为“无语义训练”。

逐层96槽使用RGB FP32和固定256题，稠密均值54.948%。随机＋上下文的block2/8/10分别56.120%、55.339%、55.990%，准确率增量区间均跨0，仅作筛选。batch8完整编码器延迟下降约0.42%–0.45%，相似性方案0.13%–0.24%；含分组/聚合开销。batch1不稳定，没有建立可靠端到端加速或整网峰值显存下降。

256题RGB与800题缓存边界的结果不能混用。层扫描的random不是训练后的R1/S1路由；similarity是独立规定的同帧分组，不是官方ToMe复现。

## 阅读与发布检查

`router.py`实现视觉评分、视频预测概念、语义亲和度与分组；`memory_attention.py`保留全部Q、时间分支、残差和MLP，只压缩空间K/V并按源token数量加log-mass；`sp_model.py`/`train.py`保存模型、损失、AMP、恢复与曝光审计。

`degradation_core.py`/`degradation_run.py`用于权重/记忆交叉和局部概率/值误差；`pixel_interventions.py`用于原RGB干预。`b10_model.py`限定block10压缩；其包装器复用父实验训练/评测。

`lc_model.py`实现完整RGB编码、随机/相似分组和共享清洁前缀，每个受修改的后缀均重新计算。`benchmark.py`实现FP32、关闭TF32、batch1/8配对计时、随机顺序、显存及有限端到端测量。

```bash
python tools/verify_recent_release.py
python tools/reproduce_recent_results.py
```

第一条校验导出文件SHA256，第二条从归档计数和原始计时样本重算主表算术。两者不需要数据或GPU，不等于重新训练/执行模型。

结果在`results/recent_20261028/`。来源清单、24个最终checkpoint哈希、原协议分别在`provenance/recent_20261028/{source_manifest.json,checkpoints.json,protocols/}`。权重另行准备。

## 环境与外部资产

沿用Python3.9、PyTorch1.13.1+cu117及原baseline依赖；绘图增量依赖可执行`pip install -r requirements-recent.txt`。仅关系弱标签的`prepare_relations.py`需要spaCy与`en_core_web_sm`，历史使用独立NLP环境；主路由/block10/层扫描不依赖运行此解析器。

主要外部资产布局：

```text
data/EgoExoLearn/videos/*.mp4
data/association_annotations/{ego_train.csv,exo_train.csv,association_val_hard.json,...}
repos/EgoExoLearn/video_meta_downsampled.json
experiments/egoexo_textalign_20261025/       # plan、records、phrases、文本特征、划分、题索引
experiments/egoexo_mechanism_20261025/heads/M/seed{17,29,43}/best.pt
experiments/egoexo_states_20261025/cache{2,3}/{ids.npy,pts.npy,prefix.npy,...}
experiments/egoexo_relations_20261026/{selection_holdout.npy,ssd_ready.json,...}
experiments/egoexo_semantic_processing_20261027/
  protocol_v2_frozen.json runtime_amendment.json
  route_prototypes.npy route_labels.npy route_donors.npy route_vocabulary.json
  fits/{R0,R1,R2,R3,R4,R5}/seed{17,29,43}/{final.pt,complete.json,...}
  post_training/                           # 阶段协议与预检
experiments/egoexo_block10_20261028/
  protocol_v2_frozen.json runtime_choices.json
  fits/{S1,S2}/seed{17,29,43}/{final.pt,complete.json,...}
reports/egoexo_semantic_processing_20261027/v2_固定256题诊断队列.json
```

资产不随本版本导出。16帧block9后的FP16前缀缓存约192GiB；`sp_data.py`使用只读SSD副本，历史位置`/var/tmp/yzj-egoexo-relations-20261026`。新层扫描从RGB开始，不能用此缓存测试前面block。

基础权重沿用`EGOEXO_BASE_CHECKPOINT`适配。历史解释器路径、GPU2/3编排、83°C停止和12GiB产物预算保留。原始入口为：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/egoexo_semantic_processing/train.py --arm R0 --seed 17
CUDA_VISIBLE_DEVICES=2 python scripts/egoexo_semantic_processing/evaluate.py --arm R0 --seed 17 --endpoint final
CUDA_VISIBLE_DEVICES=2 python scripts/egoexo_block10/b10_train.py --arm S1 --seed 17
CUDA_VISIBLE_DEVICES=2 python scripts/egoexo_block10/b10_evaluate.py --arm S1 --seed 17 --endpoint final
CUDA_VISIBLE_DEVICES=3 python scripts/egoexo_layer_compressibility/benchmark.py
CUDA_VISIBLE_DEVICES=2 python scripts/egoexo_layer_compressibility/scan.py --shard 0
CUDA_VISIBLE_DEVICES=3 python scripts/egoexo_layer_compressibility/scan.py --shard 1
```

这些命令要求完整外部资产、预检、协议和匹配权重，不是一键复现命令。各目录的prepare/preflight/freeze脚本保存相应过程；pipeline是历史调度器，发布检查不启动它。

`provenance/.../protocols/`是历史记录，不会自动变成本次`experiments/`的有效运行协议。原记录包含绝对路径、缓存签名和原源码哈希；便携baseline已有路径适配，导出环境不直接等价于原冻结环境。迁移后需真实预检并另行冻结协议，复制历史`passed:true`文件不能替代验证。

本次发布不重训、不重跑完整benchmark。人工区域、关系、动作方向核验仍有未完成部分；TEST已有历史使用，不称为首次盲测。此前baseline发布说明继续适用。
