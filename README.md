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
| ETF baseline | 84.0 | 76.8 | 72.0 | 67.8 | 66.4 | 64.0 | 61.5 | 59.5 | 58.3 | 68.9 |
| SIM+DR | 84.6 | 77.9 | 73.2 | 69.5 | 68.0 | 66.1 | 63.8 | 62.0 | 60.6 | 70.6 |
| Edge-Cloud + SIM+DR | 84.9 | 78.6 | 74.0 | 70.5 | 69.3 | 67.5 | 65.6 | 64.1 | 62.8 | 71.9 |

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

## 复现性建议
- 固定随机种子：`--seed 0`
- 固定确定性：`--deterministic`
- 保存配置快照：在 `--work-dir` 下记录 cfg
- 记录关键指标：session acc、avg acc、drift、`α_e`

---


