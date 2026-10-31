# EgoExoLearn Association: baselines, routing and compression experiments

用于保存 EgoExoLearn Cross-view Association 上 **M baseline、时序/状态模型、内容/语义路由、block10单层K/V与逐层压缩测速** 的研究代码和已完成结果。

## 2026-10-31：信息机制实验

新增[事前实验计划](docs/INFORMATION_MECHANISM_20261031.txt)及 `scripts/information_mechanism/`。
固定已有 P_base/P_rich 权重，比较完整800题中重复4帧与补充实拍画面（均16位置），再对盲选区域队列进行手物与匹配背景扰动。
计划、实现和运行状态与已完成研究结果分开记录；这项新实验的结果须以对应运行归档为准。
区域为助手视觉标注，不是独立人工真值；单种子开发集分析不能作为独立泛化或唯一因果机制证明。

CPU检查：`python scripts/information_mechanism/self_check.py`。运行入口、参数和产物规则见事前计划。

## 2026-10-28：近期实验代码

新增77个原始Python源文件，覆盖关系槽位、早期token合并、R0–R5及退化诊断、block10-only和12层96槽扫描；附原协议、24个checkpoint哈希、汇总结果与原始配对计时样本。详见[近期实验与复现边界](docs/RECENT_EXPERIMENTS.md)。

```bash
python tools/verify_recent_release.py
python tools/reproduce_recent_results.py
```

两条命令不需要GPU，用于校验文件和重算主表算术，不会重新训练或评测模型。R0–R5两层方案发生退化，block10-only未建立稳定提升；逐层扫描仅在固定256题上筛选，单层完整编码器延迟节省不足0.5%。所有条件及边界均保留，不将负结果描述为方法成功。

新代码保留历史执行约束；原数据、权重、缓存和机器相关协议另行准备。新论文联合方案仍在设计中。

## 已记录结果

同一验证集800题、每题20个候选，三个种子（17、29、43）的准确率均值，使用余弦评分。

| 方法 | 4帧推理 | 16帧推理 | 增加帧数的变化 |
|---|---:|---:|---:|
| 原有 M | 50.750% | 53.292% | +2.542个百分点 |
| 普通时序聚合 `temporal_mean` | 51.708% | 53.208% | +1.500个百分点 |
| 新状态模型 `ordered_states` | 51.583% | 53.250% | +1.667个百分点 |

直接核验表格，无需模型或GPU：

```bash
python tools/reproduce_table.py
```

**比较口径：** M使用既有4帧训练权重；两种新模型均用16帧训练，其4帧结果来自同一组权重直接改为4帧推理，位置编码由16线性插值为4。不是六组分别训练的模型。16帧结果中，新状态模型没有超过M，不应宣称它在该比较中取得最佳结果。

这里的M是本项目的后训练基线，不是原论文官方数字。阶段是学习出的有序隐变量，没有人工阶段真值；方向标签和跨视角三元组是文字规则弱监督。未报告测试集准确率。历史训练清单构造会读取验证/测试片段的时间区间以排除重叠，但不使用其答案训练；这与“测试集完全未被读取”不同。

## 代码组织

| 路径 | 内容 |
|---|---|
| `repos/EgoExoLearn/association_benchmark/` | 上游原始 EgoExo baseline 的训练、模型、配置代码；保留许可证与已有兼容补丁 |
| `scripts/egoexo_textalign/` | 公共视频/文本模型、文字特征、4帧缓存与基础后训练 |
| `scripts/egoexo_mechanism/` | M 的损失、训练代码与时间干预评测；N/V是其历史消融 |
| `scripts/egoexo_states/` | 普通时序聚合、新状态模型及16帧训练/评测 |
| `scripts/egoexo_relations/`、`scripts/egoexo_tokenmerge/` | 关系绑定/槽位与早期共享静态token实验 |
| `scripts/egoexo_semantic_processing/` | R0–R5、内容/语义路由、K/V算子、退化与像素干预 |
| `scripts/egoexo_block10/` | 仅block10压缩的匹配训练/评测 |
| `scripts/egoexo_layer_compressibility/` | 从RGB开始的12层扫描、完整编码器计时和汇总 |
| `scripts/evaluate_states_fourframes.py` | 原始4帧适配与对照评测脚本 |
| `scripts/sampling_corrected/bounded_decode.py` | 根据实际播放PTS取帧，并限制在标注时间窗口内 |
| `tools/evaluate.py` | 可指定路径、GPU、帧数的原视频评测入口，不要求预先生成特征缓存 |
| `results/`、`provenance/` | 逐种子汇总、权重哈希、代码来源及上游补丁 |

上面的4/16帧主表属于2026-10-26的早期发布；新路由和压缩结果在`results/recent_20261028/`单独归档，评测范围与数值路径不同，不混入该表。多模型融合等其他研究仍不属于此版本。

## 环境

历史环境为 Python 3.9、PyTorch 1.13.1+cu117、torchvision 0.14.1+cu117。其余记录版本见 `requirements.txt`。

```bash
python3.9 -m venv .venv-sampling-fixed
source .venv-sampling-fixed/bin/activate
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

需要能够运行 `nvidia-smi`。这里记录历史环境，不保证其他PyTorch版本的数值或checkpoint行为一致。

## 权重、数据与推理

数据集、原始/微调权重、特征缓存不在 Git 中。请自行准备有权使用的数据和匹配的权重；本仓库不会自动下载这些文件。用于核对实验权重的SHA256见 `provenance/checkpoints.json`。

M baseline 的例子：

```bash
python tools/evaluate.py \
  --method M --frames 16 --seed 17 --gpu 0 \
  --base-checkpoint /path/to/egoexo_base.pt \
  --m-checkpoint /path/to/M/seed17/best.pt \
  --annotations /path/to/association_val_hard.json \
  --video-root /path/to/videos \
  --output outputs/M_seed17_16frames.npz
```

新状态模型的例子：

```bash
python tools/evaluate.py \
  --method ordered_states --frames 4 --seed 17 --gpu 0 \
  --base-checkpoint /path/to/egoexo_base.pt \
  --m-checkpoint /path/to/M/seed17/best.pt \
  --state-checkpoint /path/to/ordered_states/seed17/best.pt \
  --annotations /path/to/association_val_hard.json \
  --video-root /path/to/videos \
  --output outputs/ordered_states_seed17_4frames.npz
```

`temporal_mean`使用对应权重，命令形式相同；`--method egoexo`只加载原始checkpoint。基准表使用 `--score cosine --prefix-precision float16`（默认值），模拟历史第10个block后缓存的量化。不要将raw-dot结果和余弦结果混在一起。

原视频入口默认batch=2、PyTorch显存分配器上限16%，超过83°C停止。它是复现实验的评测入口，不是训练调度器。用户需自行协调共享GPU。

发布检查使用一个真实片段、seed17，覆盖三个方法的4/16帧：新入口前10层输出与相同输入下原始编码器一致。相对历史缓存存在小幅数值差异，最终embedding最大绝对差约3.90e-4，不能据此宣称完整800题预测逐位一致；详细记录见 `provenance/release_validation.json`。重算历史表格与重新执行模型评测是两项不同的检查。

## 训练与复现边界

原训练实现完整保存，包含冻结前10层、M的文字/短语目标、状态模型的方向与三元组弱监督、内部留出选择、AMP异常重试和断点恢复。训练所需文件布局及命令见 [训练说明](docs/TRAINING.md)。

历史流水线包含已完成实验的来源校验和预检要求，不能在空目录下直接运行就得到表中结果。原始数据、精确初始化权重、清单/缓存和预检仍须准备。发布时只做源码/入口验证和局部推理等价检查，不宣称从空环境重新训练过全部模型。

## 来源与许可

上游：[OpenGVLab/EgoExoLearn](https://github.com/OpenGVLab/EgoExoLearn)，固定版本见 `provenance/upstream.json`。上游代码许可见 [MIT LICENSE](repos/EgoExoLearn/LICENSE)；文件原有作者和来源声明保留。额外研究代码暂未单独授予开源许可证，不应把上游许可证自动解释为所有新增代码的许可证。

原始上游 `data/data_utils.py` 保留历史实现，可能含本项目已诊断的时间定位问题；本表及新评测入口使用 `bounded_decode.py`，不是该旧读取路径。
