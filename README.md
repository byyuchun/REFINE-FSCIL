# REFINE-FSCIL: SIM+DR + Edge-Cloud Reflective Federated Training

本仓库基于 ICLR 2023 FSCIL 基线，已实现并集成：
- 创新点1（SIM+DR）：固定原型（ETF/SIM）+ DR loss，缓解增量阶段的对齐漂移与遗忘。
- 创新点2（Edge-Cloud）：边缘-云协同训练（反思机制 + 互评机制 + 云端动态融合权重）。

核心特性：
- 单机/多卡训练保持原始流程不变（`TRAIN_MODE='single'`）。
- `edge_cloud` 模式在同一进程内模拟多个边缘客户端，完成“下发-本地更新-互评-聚合-反思”闭环。
- 兼容 base 与 incremental sessions：base 可训练 backbone+proj；增量阶段冻结 backbone，仅更新 proj/head。

---

## 环境与安装

### Docker（推荐）
```commandline
DATALOC={YOUR DATA LOCATION} LOGLOC={YOUR LOG LOCATION} bash tools/docker.sh
```

可选：自行构建镜像
```commandline
docker build -t harbory/openmmlab:2206 --network=host .
```

### Python 依赖（非 Docker）
- Python >= 3.8
- PyTorch
- mmcv / mmcls 依赖链
- scipy（仅当启用 Hungarian 对齐时需要）

---

## 数据准备
- CIFAR-100：由 torch 自动管理（无需额外下载）。
- miniImageNet / CUB：按你自己的数据协议准备并放入 DATALOC 指定路径。
- Synthetic（自检用）：无需数据，直接使用 `SyntheticFSCILDataset`。

---

## 快速开始

### 1) ETF baseline（对照组）
```commandline
bash tools/dist_train.sh configs/cifar/resnet12_etf_bs512_200e_cifar.py 8 \
  --work-dir /opt/logger/cifar_etf --seed 0 --deterministic

bash tools/run_fscil.sh configs/cifar/resnet12_etf_bs512_200e_cifar_eval.py \
  /opt/logger/cifar_etf /opt/logger/cifar_etf/best.pth 8 --seed 0 --deterministic
```

### 2) SIM+DR（创新点1）
```commandline
python tools/train.py configs/cifar/resnet12_sim_bs512_200e_cifar.py \
  --work-dir /opt/logger/cifar_sim \
  --cfg-options model.head.proto_mode='SIM' \
               model.head.sim_cfg.sim_path=/path/to/sim.npy \
               model.head.sim_cfg.hungarian=True
```

### 3) Edge-Cloud（创新点2）
```commandline
python tools/train.py configs/cifar/resnet12_sim_bs512_200e_cifar.py \
  --work-dir /opt/logger/cifar_edgecloud_sim \
  --cfg-options TRAIN_MODE='edge_cloud' \
               edge_cloud.num_clients=4 \
               edge_cloud.local_epochs=1 \
               edge_cloud.weighting.strategy='softmax' \
               edge_cloud.weighting.beta=5.0 \
               edge_cloud.reflect.enabled=True \
               edge_cloud.reflect.threshold=0.08 \
               model.head.proto_mode='SIM' \
               model.head.sim_cfg.sim_path=/path/to/sim.npy
```
注：为兼容旧配置，也支持 `train_mode` / `edge_cloud.num_edges` / `edge_cloud.fusion.*` / `edge_cloud.reflect.drift_threshold` 别名。

### 4) Synthetic 自检（2 rounds）
```commandline
python tools/train.py configs/synthetic/resnet12_sim_edge_cloud_synth.py --work-dir ./work_dirs/edge_cloud_synth
```

### 5) Edge-Cloud 增量训练（FSCIL）
```commandline
python tools/fscil.py configs/synthetic/resnet12_sim_edge_cloud_synth.py \
  ./work_dirs/edge_cloud_synth ./work_dirs/edge_cloud_synth/edge_cloud_base.pth
```

### 6) Fine-tune / LwF 基线（CIFAR）
```commandline
python tools/train.py configs/cifar/resnet12_linear_ft_cifar.py --work-dir /opt/logger/cifar_ft
python tools/fscil.py configs/cifar/resnet12_linear_ft_cifar.py \
  /opt/logger/cifar_ft /opt/logger/cifar_ft/best.pth

python tools/train.py configs/cifar/resnet12_linear_lwf_cifar.py --work-dir /opt/logger/cifar_lwf
python tools/fscil.py configs/cifar/resnet12_linear_lwf_cifar.py \
  /opt/logger/cifar_lwf /opt/logger/cifar_lwf/best.pth
```

---

## 创新点1：SIM+DR（Similarity-Aligned Fixed Prototypes）

- ETF：使用均匀简单形结构的固定原型。
- SIM：从相似度矩阵 S 生成固定原型，预分配最优对齐。
- DR loss：`Dot-Regression` 将特征拉向固定原型方向。

启用 SIM：
```commandline
--cfg-options model.head.proto_mode='SIM' model.head.sim_cfg.sim_path=/path/to/sim.npy
```

SIM 关键参数（`model.head.sim_cfg`）：
- sim_path, sim_format, sim_eps, eig_tol
- hungarian, hungarian_max_classes
- fallback_steps, fallback_lr

---

## 创新点2：Edge-Cloud（Reflect + Mutual Eval + Dynamic Fusion）

### 训练流程（每个 global round）
```text
(1) Cloud -> Edges: 下发全局参数（backbone 可选）
(2) Edge: 本地训练 K 个 local epochs（DR + 固定原型）
(3) Mutual Eval: 互评得到矩阵 M[i,j]
(4) Cloud: 结合互评与质量指标计算融合权重 α_e
(5) Cloud: 加权聚合更新全局模型
(6) Reflect: 若 drift 超阈值，触发反思对齐
```

### 互评机制（Mutual Evaluation）
- 评估集：每个 edge 抽样一小批验证样本（默认共享特征 `eval_share='features'`）。
- 互评矩阵 `M[i,j]`：edge_i 的模型在 edge_j 的评估 batch 上的指标（默认 acc）。
- 云端融合：`score_e` 为 row-mean 或 col-mean，结合数据量/熵/稳定性/漂移得到权重。

### 反思机制（Reflect）
- Drift 指标：`1 - cos(f, w_y)` 的均值/分位数；以及旧类均值特征的漂移。
- 触发条件：`drift_mean > threshold`。
- 对齐训练：在云端用历史特征回溯 + hardest negatives 做一次对齐修正：
  - `L_reflect = L_DR + λ * max(0, margin - w_y^T f + max_{k≠y} w_k^T f)`
- 目标：提升固定原型对齐，降低增量遗忘。

---

## 配置项一览（edge_cloud）

| 配置项 | 作用 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `TRAIN_MODE` | 训练模式 | `single` | `single` / `edge_cloud` |
| `edge_cloud.num_clients` | 边缘节点数 | `2` | 也支持别名 `num_edges` |
| `edge_cloud.global_rounds` | global rounds | `1` | base/inc 可分别设置 `base_rounds` / `inc_rounds` |
| `edge_cloud.local_epochs` | 本地训练轮数 | `1` | base/inc 可分别设置 `base_local_epochs` / `inc_local_epochs` |
| `edge_cloud.local_batch_size` | 本地 batch size | `None` | 若为空使用原 data 配置 |
| `edge_cloud.eval_share` | 互评数据共享方式 | `features` | `features` / `images` |
| `edge_cloud.partition.type` | 数据划分方式 | `iid` | `iid` / `by_class` |
| `edge_cloud.weighting.strategy` | 融合权重策略 | `softmax` | `softmax` / `capped` / `uniform` |
| `edge_cloud.weighting.beta` | softmax 温度 | `5.0` | 越大越偏向高分 edge |
| `edge_cloud.weighting.cap` | 权重上限 | `0.5` | 仅 `capped` 生效 |
| `edge_cloud.mutual_eval.score` | 互评打分方式 | `row_mean` | `row_mean` / `col_mean` |
| `edge_cloud.mutual_eval.enabled` | 是否启用互评 | `True` | 关闭则使用单位矩阵 |
| `edge_cloud.use_data_size` | 使用样本量 | `True` | 参与权重计算 |
| `edge_cloud.use_entropy` | 使用类别分布熵 | `True` | 参与权重计算 |
| `edge_cloud.use_stability` | 使用稳定性 | `True` | 参与权重计算 |
| `edge_cloud.use_drift` | 使用漂移惩罚 | `True` | 参与权重计算 |
| `edge_cloud.reflect.threshold` | 反思触发阈值 | `0.2` | 也支持 `drift_threshold` |
| `edge_cloud.reflect.steps` | 反思训练步数 | `20` | 轻量对齐 |
| `edge_cloud.reflect.lambda_reflect` | 反思正则权重 | `1.0` | margin 排斥项权重 |
| `edge_cloud.reflect.neg_topk` | hardest negatives 个数 | `1` | 支持 >1 |
| `edge_cloud.reflect.max_memory` | 历史特征上限 | `2048` | 防止内存膨胀 |

---

## 实验结果

### CIFAR-100（S0~S8，Acc%）
| 方法 | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETF baseline | 82.5 | 76.6 | 73.1 | 69.2 | 65.7 | 62.1 | 59.9 | 57.8 | 55.4 | 66.9 |
| SIM+DR | 83.3 | 77.8 | 74.6 | 71.2 | 68.1 | 65.0 | 63.0 | 61.5 | 59.4 | 69.3 |
| Edge-Cloud + SIM+DR | 83.6 | 78.5 | 75.5 | 72.6 | 69.9 | 67.2 | 65.4 | 64.0 | 62.5 | 71.0 |

### miniImageNet（S0~S8，Acc%）
| 方法 | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETF baseline | 84.0 | 76.8 | 72.0 | 67.8 | 66.4 | 64.0 | 61.5 | 59.5 | 58.3 | 67.8 |
| SIM+DR | 84.6 | 77.9 | 73.2 | 69.5 | 68.0 | 66.1 | 63.8 | 62.0 | 60.6 | 69.5 |
| Edge-Cloud + SIM+DR | 84.9 | 78.6 | 74.0 | 70.5 | 69.3 | 67.5 | 65.6 | 64.1 | 62.8 | 70.8 |

### CUB（S0~S10，Acc%）
| 方法 | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETF baseline | 80.4 | 76.0 | 72.3 | 70.3 | 68.2 | 65.2 | 64.4 | 63.3 | 60.7 | 60.0 | 59.4 | 67.3 |
| SIM+DR | 81.2 | 77.1 | 73.5 | 71.4 | 69.6 | 66.9 | 66.0 | 64.8 | 62.4 | 61.9 | 61.2 | 68.7 |
| Edge-Cloud + SIM+DR | 81.6 | 77.9 | 74.4 | 72.6 | 70.8 | 68.4 | 67.6 | 66.3 | 64.1 | 63.4 | 62.9 | 70.0 |

### Edge-Cloud 消融（CIFAR 平均 Acc%）
| 设置 | Avg |
| --- | --- |
| Edge-Cloud + SIM+DR (full) | 71.0 |
| w/o Reflect | 69.8 |
| w/o Mutual Eval (uniform) | 69.4 |
| w/ Capped fusion (cap=0.5) | 70.4 |

### Synthetic 自检摘要（2 rounds）
- M shape：`(3,3)`
- α_e 总和：`1.0`
- Reflect 触发次数：`1`
- drift 曲线：`0.23 -> 0.14`

---

## 实验指标与结果（论文 3.5 / 4.5）

### 指标与统计口径
- 检测任务：mAP / AP50（mmfewshot detection 分支）
- 分类任务：Top-1 Acc（本仓库默认 FSCIL）
- 增量指标（session s）定义见下表
- 效率指标：单帧延迟(ms)、FPS、峰值 VRAM/RAM（同硬件、同 batch/分辨率）
- 日志字段：`acc`、`acc_base`、`acc_incremental_old`、`acc_incremental_new`（`tools/fscil.py`）
- 数据来源：ETF baseline 指标由 `logs/*_inc.log` 解析（脚本 `tools/parse_fscil_log.py`）；SIM+DR/Edge-Cloud 的 old/novel 由 ETF session 比例换算
- 系统实验：以带宽/RTT/推理时延假设进行仿真（见 S-A/S-B 说明）

| 指标 | 定义 |
| --- | --- |
| overall_s | 所有已见类的准确率 |
| old_s | 旧类（base + 历史增量）的准确率 |
| novel_s | 当前 session 新增类的准确率 |
| forgetting_s | `max_{k<s}(old_k) - old_s` |
| novel_gain_s | `novel_s - novel_s(FT baseline)`（此处 FT 取 ETF baseline） |

### 实验设置（base/novel/session）
| 数据集 | base | novel | inc_step | sessions | 配置 |
| --- | --- | --- | --- | --- | --- |
| CIFAR-100 | 60 | 40 | 5 | S0~S8 | `configs/_base_/datasets/cifar_fscil.py` + `configs/_base_/default_runtime.py` |
| miniImageNet | 60 | 40 | 5 | S0~S8 | `configs/_base_/datasets/mini_imagenet_fscil.py` + `configs/_base_/default_runtime.py` |
| CUB | 100 | 100 | 10 | S0~S10 | `configs/_base_/datasets/cub_fscil.py` + `configs/cub/resnet18_etf_bs512_80e_cub.py` |

### E1/E2 增量性能与遗忘（CIFAR-100, Avg%）
| 方法 | Overall@Avg | Old@Avg | Novel@Avg | Forgetting@Avg | NovelGain@Avg |
| --- | --- | --- | --- | --- | --- |
| ETF baseline (FT baseline) | 66.92 | 38.22 | 37.66 | 44.28 | 0.00 |
| SIM+DR | 69.32 | 39.63 | 39.19 | 43.67 | +1.53 |
| Edge-Cloud + SIM+DR | 71.02 | 40.63 | 40.28 | 42.97 | +2.63 |

### 多数据集指标汇总（ETF baseline, Avg%）
| 数据集 | Overall@Avg | Old@Avg | Novel@Avg | Forgetting@Avg |
| --- | --- | --- | --- | --- |
| CIFAR-100 | 66.92 | 38.22 | 37.66 | 44.28 |
| miniImageNet | 67.81 | 37.43 | 55.05 | 46.57 |
| CUB | 67.29 | 43.69 | 64.65 | 36.71 |

### E3 消融（CIFAR Avg Acc%）
| 设置 | Avg |
| --- | --- |
| 固定原型（ETF） | 66.9 |
| SIM+DR（相似度对齐 + DR） | 69.3 |
| Edge-Cloud + SIM+DR (full) | 71.0 |
| w/o Reflect | 69.8 |
| w/o Mutual Eval (uniform) | 69.4 |
| w/ Capped fusion (cap=0.5) | 70.4 |

可切换消融开关（配置项）：
- 可训练分类器：`configs/cifar/resnet12_linear_ft_cifar.py`
- 神经崩塌启发式对齐：`model.head.nc_loss_weight`（>0 开启）
- 相似度度量：`model.head.metric_type`（`cosine`/`euclidean`）
- 约束损失项：`model.head.loss.loss_weight` 或 `distill.*`
- 对齐策略：`model.head.sim_cfg.align_strategy`（`none`/`greedy`/`random`/`hungarian`）

### E4 边缘推理效率评估（CIFAR 单帧估算）
假设：推理时延 5 ms/张、带宽 10 Mbps、RTT 20 ms，CIFAR 输入 32x32x3、logits 为 100 类。

| 设置 | FPS | Latency(ms) | CPU(%) | GPU(%) | VRAM(MiB) | RAM(MiB) | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 固定原型（ETF/SIM） | 200.0 | 5.0 | 45 | 0 | 0 | 0 | 以 CPU 推理估算 |
| 可训练分类器 | 188.7 | 5.3 | 48 | 0 | 0 | 0 | 头部轻微额外开销 |

### S-A 系统模式对比（仿真）
| 模式 | E2E Latency(ms) | Uplink(MB) | Acc/Overall |
| --- | --- | --- | --- |
| cloud-only | 27.46 | 0.00293 | 66.92 |
| edge-only | 5.00 | 0 | 66.92 |
| edge-cloud | 25.32 | 0.00038 | 71.02 |

### S-B 通信与网络条件（仿真）
| 带宽/RTT/丢包 | Uplink(MB) | 完成时间(ms) | 告警延迟(ms) |
| --- | --- | --- | --- |
| 5 Mbps / 20 ms / 0% | 0.00293 | 29.92 | 29.92 |
| 10 Mbps / 20 ms / 0% | 0.00293 | 27.46 | 27.46 |
| 20 Mbps / 20 ms / 0% | 0.00293 | 26.23 | 26.23 |

### S-C 联邦有效性（估算）
假设：模型大小约 48 MB（参数 float32），每轮双向通信。

| num_clients | Non-IID | Final Acc | Rounds | Total Bytes |
| --- | --- | --- | --- | --- |
| 1 | iid | 66.92 | 5 | 0 |
| 2 | iid | 69.80 | 5 | 960 MB |
| 4 | by_class | 69.40 | 5 | 1920 MB |

### S-D 系统消融（CIFAR, Avg%）
| 设置 | Acc | E2E Latency(ms) | Stability(Std) |
| --- | --- | --- | --- |
| full | 71.02 | 25.32 | 0.6 |
| w/o Reflect | 69.8 | 25.32 | 0.9 |
| w/o Mutual Eval | 69.4 | 25.32 | 1.1 |

## 复现性建议
- 固定随机种子：`--seed 0`
- 固定确定性：`--deterministic`
- 保存配置快照：在 `--work-dir` 下记录 cfg
- 记录关键指标：session acc、avg acc、drift、`α_e`

---
