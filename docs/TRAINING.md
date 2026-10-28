# 训练实现与外部资产

## 三种方法

- **M**：EgoExo视觉编码器，冻结前10个block，只更新后2个block、最终norm和投影。完整文字对齐 + 0.1全局短语损失 + 0.1局部平均短语损失；局部动作使用帧特征，物体使用patch特征。训练4帧、2轮，batch32。
- **temporal_mean**：从同种子M权重初始化，增加两层时间Transformer（128维、4头、FFN512），时间均值池化形成状态表示，与全局表示门控融合。
- **ordered_states**：相同时间Transformer，预测正的进度增量，累积成单调进度；围绕0、0.5、1软聚合三个隐阶段，拼接阶段与相邻差分后投影。其结构和参数量并非与均值对照完全相同。

两种新模型训练16帧、4轮，anchor batch16，按规则加入正负例后最多48个片段。原视觉部分学习率1e-5，新模块1e-4。训练目标为文字匹配 + 0.1全局短语 + 0.1局部均值短语 + 0.2跨视角弱三元组 + 0.1方向/反转代理监督。方向标签、反转标签都不能视为人工动作阶段真值。

最佳轮次使用内部留出文字损失选择，包含epoch0。原结果的模型哈希和轮次见 `provenance/checkpoints.json`。

## 原实现所需布局

以下是相对本仓库的路径，运行时生成的所有内容均被 `.gitignore` 排除：

```text
checkpoints/egoexo_base.pt
data/EgoExoLearn/videos/*.mp4
data/association_annotations/{ego_train.csv,exo_train.csv,association_val_hard.json,...}
repos/EgoExoLearn/annotations/narration_{verb,noun}_taxonomy.csv
experiments/complementary_fusion_20261021/eligible_training_clips.json
experiments/complementary_fusion_20261021/weak_episodes/records.json
experiments/egoexo_textalign_20261025/
  plan.json records.json phrases.json texts.json splits.npz val_episodes.npz
  text_features.npy protocol.json preflight/audit JSON
  gpu{2,3}/{prefix.npy,pts.npy,indices.npy,manifest.json,complete.json}
experiments/egoexo_mechanism_20261025/
  protocol.json preflight.json heads/M/seed{17,29,43}/best.pt
experiments/egoexo_states_20261025/
  plan.json direction_labels.json triplets.json cache_ids.npy splits.npz
  protocol.json preflight.json pair_source_audit.json
  cache{2,3}/{ids.npy,prefix.npy,pts.npy,complete.json}
  fits/{temporal_mean,ordered_states}/seed{17,29,43}/best.pt
```

基础checkpoint也可通过环境变量 `EGOEXO_BASE_CHECKPOINT` 指定。历史训练脚本仍按GPU2/3及双分片缓存编排；**不要在不了解占用情况时直接启动历史pipeline**。可指定任意单卡的入口是 `tools/evaluate.py`。

## 数据准备与训练依赖顺序

1. `scripts/complementary_fusion/training_inventory.py` 根据官方训练标注建立可用片段清单，使用验证/测试时间区间排除重叠；不以测试答案构造标签。
2. `scripts/complementary_fusion/weak_episodes.py` 去重并按视频哈希划分内部训练/留出集合。脚本也生成弱配对，不能把它们解释为人工真值。
3. `scripts/egoexo_textalign/prepare.py` 对齐文字、短语、划分和验证题索引；`text_teacher.py` 提取冻结文本塔特征；`extract.py` 生成4帧前10层缓存。需要从数据来源单独准备taxonomy CSV。
4. 进行真实预检和缓存审计，再固定本机当前源码、数据、权重的协议哈希。不能复制历史 `passed:true` 文件伪装本次验证。历史pipeline对校验文件的要求见其源码；迁移机器后不要复用带旧绝对路径的protocol。
5. 使用 `scripts/egoexo_mechanism/train_ablation.py --arm M --seed 17` 训练M，依次重复种子29、43。该入口要求匹配的M协议与预检，不会自动生成它们。
6. `scripts/egoexo_states/prepare.py` 构建16帧清单和弱方向标签；`cache.py` 生成16帧缓存；`preflight.py` 检查前向/梯度。方向样本需核对数据来源并生成 `pair_source_audit.json`，这不是人工阶段标注。
7. 在经过审计并冻结的状态模型协议下运行：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/egoexo_states/train.py --arm temporal_mean --seed 17
CUDA_VISIBLE_DEVICES=3 python scripts/egoexo_states/train.py --arm ordered_states --seed 17
```

重复29、43种子后评测。新 `tools/evaluate.py` 可跳过验证缓存和历史时间干预文件，直接读取原视频；历史 `evaluate.py` 另外需要 `interventions.py` 产生的reverse/shuffle/repeat缓存。历史 `report.py` 的方向样本分析需要之前生成的人工待核验清单，不是重现主表的必需步骤。

16帧全量prefix缓存约192GiB，另需4帧缓存、模型和临时文件空间。原数据和权重不随仓库发布；这个仓库保存可检查的训练实现，不把缺失外部资产描述为“一键复现”。

## 导出时的修改

`provenance/source_manifest.json`保存原工作区文件SHA256和导出文件SHA256。原训练/模型数学实现未重写；仅基础checkpoint路径改成环境变量/相对路径、上游路径改成仓库相对路径，以及StateModel增加可选M checkpoint参数。上游已有的checkpoint兼容补丁单独保存在 `provenance/upstream-local.patch`。

新增的 `tools/evaluate.py` 是便携入口；`tools/reproduce_table.py` 从逐种子计数重算结果。上传不包括数据标注内容、权重、原视频或GPU日志。2026-10-28已补充后续路由和压缩实验的原始实现与汇总结果，依赖与协议边界见[近期实验说明](RECENT_EXPERIMENTS.md)。
