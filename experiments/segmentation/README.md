# v2.0：融合入口 Mul + Add 对照实验

本分支 `seg_rgbdvs_640x440_v2.0` 基于 `main/c573491`。唯一模型计算变化是在四个stage的原融合入口增加逐元素乘加（不是矩阵乘法）：

```text
P = R ⊙ D                # 原始同尺度 [B,C,H,W] 特征，共用同一个 P
R_enh = R + P
D_enh = D + P
(R_next, D_next, O) = 原融合模块(R_enh, D_enh)
```

保留前置IRB＋Concat更新及全部原注意力归一化、参数和后续连接；不额外添加论文图中的卷积或其他模块。参数量不变。数据、预训练、seed42、150epoch、batch32、accumulation1、workers8和学习率日程均与来源版本相同。

原训练入口如下（**当前FP16 AMP冒烟失败，暂勿启动**；BF16诊断通过，但训练精度变更尚待确认）：

```bash
CUDA_VISIBLE_DEVICES=2 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py --single --amp --seed 42
```

新输出为 `logs/segmentation/efficientvit_b1_cross_v20/`，`fusion_mode=cross_stage_muladd`。仅允许从本实验自己的检查点resume；旧版本参数键和形状虽相同，计算语义已改变，不能作为本实验续训或评估。训练契约会拒绝旧fusion_mode；评估/导出仍需明确选择本版本训练所得权重，单靠strict加载不能区分此无参数变更。

新增乘加的结构/梯度/BN回归通过；真实440×640 FP32小批次更新通过，FP16 batch32出现前向溢出。BF16诊断两步batch32更新、FP32 batch32推理及严格保存恢复通过，但这不代表原FP16命令可用。证据保存在本地 `logs/segmentation/muladd_validation/`。尚未正式训练、评估精度或验收部署。

下文保留来源版本说明、历史验证与命令供参照；其中旧输出路径、显存/精度/ONNX结论均属于来源版本，不代表本乘加实验已验收。当前实验以本节路径与命令为准。

---

# EfficientViT-B1 分割：方案 C

当前RGB+DVS使用**标准归一化LiteMLA双分支阶段交互**。主配置为 [efficientvit_b1.py](config/efficientvit_b1.py)。旧相加融合实现和旧cooldown配置已移除；历史模型请在备份工程中使用。`efficientvit_b1_cross.py` 仅是主配置的别名，保留给已经启动的方案C任务，二者模型和参数完全相同。

输出仍为 `logs/segmentation/efficientvit_b1_cross/`，已有方案C训练可正常恢复。三任务独立训练，不共享参数；检测和深度继续使用单模态结构，数据准备见各自README。当前没有HWAware、S-FIFO或时序模块。

下面命令均从工程根目录运行。

## 1. 数据集准备

已完成共享缓存，无需重新处理：

```text
/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/
├── dsec_b1/          train=7295，dev=787（zurich_city_08_a）
├── dsec_b1_test/     官方test=2809
└── dsec_b1_assets/   标定资产
```

三组均使用同一RGB已配准的缓存及其有效样本索引，避免模态对照混入样本差异。RGB使用ImageNet标准化；DVS使用50ms、10bins、20通道RVT Histogram。仅训练时共同水平翻转，不新增随机时间或尺寸增强。翻转由seed、数据轮次和样本索引确定，三组及续训采用相同增强；每轮重建加载worker以更新轮次。数据读取器仍可读取缓存中的两种输入，但模型仅使用所选模态，未选模态不送入GPU骨干。

在新机器上准备时（已有manifest会拒绝覆盖）：

```bash
./scripts/hmnet-python scripts/prepare_dsec_b1.py \
  --source /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/source \
  --assets /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_assets \
  --download-assets
```

增加 `--split test` 会生成独立的 `dsec_b1_test`。预训练文件为本工程 `pretrained/efficientvit_b1_r224.pth`，权重、缓存不进入Git。

## 方案 C：骨干阶段内双向 LiteMLA 交互

配置：[efficientvit_b1.py](config/efficientvit_b1.py)。当前工程RGB+DVS仅实现方案C，旧相加融合已删除；单模态仍供检测、深度和对照使用。它从官方 ImageNet B1 初始化，需要新建训练，不恢复旧相加融合的任务检查点。

### 结构

各阶段使用原生通道 `32/64/128/256`，空间尺寸为 `110×160、55×80、28×40、14×20`。每路先经过一个 stride=1、expansion=2 的倒残差块（1×1扩展、3×3深度卷积、1×1投影，BN+ReLU，末端无激活），再投影产生 Q/K/V。RGB/DVS 参数独立。采用每头16通道，基础尺度和5×5深度卷积聚合尺度（后接按头分组1×1卷积），两尺度分别做注意力后拼接通道。

设 `phi=ReLU`，`N=H×W`，下面按 `[B,heads,N,d]` 记法描述每个尺度：

```text
S_D = phi(K_D)^T V_D
z_D = sum_N phi(K_D)
A_R = phi(Q_R) S_D / (phi(Q_R) z_D + 1e-15)
A_D 对称地使用 Q_D 与 K_R/V_R
R′ = R + alpha_R * ConvBN(Concat(R, A_R))
D′ = D + alpha_D * ConvBN(Concat(D, A_D))
O  = ReLU(ConvBN(Concat(R′, D′)))
```

`alpha_R/alpha_D` 是各阶段独立的可学习标量，初始化0.1。实现内部采用 `[B,heads,d,N]` 布局，以给 V 添加常数1通道的方式一次计算分子/分母；Q/K使用ReLU，V保留符号。两次矩阵乘法及除法在FP32执行，结果转回输入dtype，再进行后续投影；保留标准LiteMLA除法归一化，没有采用HWAware近似。

**R′/D′ 分别进入两路下一阶段；O 不回灌任何分支。** 每个 O 单独通过1×1Conv+BN+ReLU投影到256通道供原Pyramid使用，辅助头读取融合后的 `/4` 特征。两路更新采用相同阶段的原输入同时计算，不先更新一侧再给另一侧使用。该模块是借鉴LiteMLA和CMX交互思想的新实验结构，不声称复现CMX。

### 数据准备、Train、Test

**数据准备**：直接复用第1节的共享 DSEC 缓存和官方 B1 权重，无需重新生成 Histogram 或配准。

**Train**（用户手动选择空闲GPU启动）：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py --single --amp --seed 42
```

训练超参数为：150 epoch、batch=32、accumulation=1、workers=8、AdamW、5轮预热、lr从2e-4余弦降到2e-6；标签、增强、损失和开发序列不变。输出为 `logs/segmentation/efficientvit_b1_cross/`，与原训练、cooldown平级。TensorBoard使用本README已有命令。

恢复本实验（必须是方案 C 检查点）：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py --single --amp --seed 42 \
  --resume logs/segmentation/efficientvit_b1_cross/checkpoint.pth
```

**Test**（开发集；官方test命令见下方）：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/test.py \
  experiments/segmentation/config/efficientvit_b1.py \
  --pretrained logs/segmentation/efficientvit_b1_cross/best_checkpoint.pth
```

`fusion_mode` 记录到训练契约和settings中；恢复时禁止把相加结构检查点当成方案 C。测试同样严格加载模型权重。旧相加检查点不再支持，请在备份工程中使用。

### 验证与显存

交互模块训练时默认使用非重入激活重计算，以维持 batch=32、accumulation=1。反向重计算时使用临时 BN 缓冲，避免运行均值、方差和计数被更新两次；测试已对照无重计算实现的输出、梯度和 BN 缓冲。eval/ONNX 不执行重计算。

在 RTX 4090 24GB 上，440×640、真实32样本的两步AMP更新通过，峰值已分配约18.52 GiB、预留约19.47 GiB；FP32 eval batch=32约2.70 GiB（该测量仍保留优化器状态）。这是小样本工程验证，不是完整训练吞吐或精度结论。未启用重计算的本结构 batch=32 曾发生OOM，因此不要直接删除该保护。

已验证：归一化线性注意力与显式注意力矩阵的输出/梯度一致、四尺度形状、实际阶段连接、双模态梯度、预训练首层适配、无状态推理、FP32/AMP两步真实样本更新、保存及恢复模型/AdamW/scaler、三任务有效样本处理。恢复结果按 `atol=1e-6, rtol=1e-5` 核对，CUDA归约不保证逐位一致。这些检查不代表正式训练收敛，也未声称精度优于基线。

冒烟日志、权重和可复运行的验证脚本在 `logs/segmentation/efficientvit_b1_cross_smoke/`，均受Git忽略规则保护。其中 `onnx/` 的完整ONNX与简化图使用两步冒烟权重，仅用于查看结构和检查导出；真实样本预测与PyTorch一致，零输入预测一致率约99.995%，不是训练完成的模型。

### 当前可检查的 ONNX

已导出当前方案C第1次更新的冻结检查点：

- [完整模型（onnxsim）](../../logs/segmentation/efficientvit_b1_cross/onnx/step_1/segmentation.sim.onnx)
- [骨干与阶段交互（onnxsim）](../../logs/segmentation/efficientvit_b1_cross/onnx/step_1/segmentation_backbone.sim.onnx)
- 同目录保留原始ONNX、源权重快照、来源说明和数值对照报告。文件仅保存在本地logs，不进入Git。

输入固定batch=1、440×640；简化图去除了Cast，保留必要的归一化Div。该权重仅训练1次更新，适合查看网络结构，不代表最终精度。

### ONNX 与硬件边界

训练完成后导出完整模型及onnxsim版本：

```bash
./scripts/hmnet-python scripts/export_b1_onnx.py --task segmentation \
  --checkpoint logs/segmentation/efficientvit_b1_cross/best_checkpoint.pth \
  --output logs/segmentation/efficientvit_b1_cross/onnx
```

增加 `--backbone-only` 只导出四尺度骨干输出。导出脚本执行PyTorch/ONNX Runtime/onnxsim数值对照。图外仍是Histogram和几何配准，图内无历史状态。FP32归一化和多尺度交互的Dremi算子映射、量化精度尚需后续验证；本实验没有移除除法，也没有替换原官方B1的激活/注意力。

### TrainSettings 参数

| 参数 | 设置与含义 |
|---|---|
| `modality` | 唯一的模态变量：`rgbdvs` / `rgb` / `dvs`。 |
| `epochs` | 150个数据轮次；由实际DataLoader长度自动换算。 |
| `updates` | `None`；指定正整数时改用累计优化器更新预算。正式对照不覆盖。 |
| `batch_size` | 32；每次前向/反向32个样本，最后不足一批保留。 |
| `accumulation` | 1；每个有效小批次更新一次，有效batch通常32。 |
| `learning_rate` | AdamW峰值学习率 `2e-4`。 |
| `lr_schedule` | `warmup_cosine`：线性预热后余弦下降。 |
| `warmup_epochs` | 5轮，对应1140次更新。 |
| `warmup_start_factor` | 0.1，首次更新lr=`2e-5`，预热末尾到`2e-4`。 |
| `min_learning_rate` | `2e-6`，150轮最后一次更新时到达。 |
| `weight_decay` | 0.01。 |
| `workers` | 8个数据加载进程，训练和验证共用。 |
| `prefetch_factor` | 1；每个worker预取一批，限制三实验并行时的主机内存。 |
| `eval_every_epochs` | 1；每轮验证，另在第1次更新和结束时验证。 |
| `eval_batch_size` | 32；控制训练中验证的显存，不影响训练batch。 |
| `overfit` | 0表示全训练集；正整数限制前N个样本并禁用翻转，仅供调试。 |
| `pretrained` | 官方ImageNet初始化路径，不是任务resume检查点。 |
| `resume` | 默认空；新实验从头初始化。 |
| `cache` | 上述共享训练/dev缓存。 |
| `output` | 本工程的独立logs目录。 |

7295样本、batch=32得到每轮228批，150轮共 **34200次更新**。所有LR按成功优化器更新推进；AMP溢出重试不推进LR，会略增加实际数据遍历量。固定seed=42、11类、ignore=255、主/辅助CE权重1.0/0.4。

### 恢复与 TensorBoard

恢复时保持原训练的epoch预算、batch、seed和学习率设置，使用上方方案C的 `--resume` 命令。`checkpoint.pth`为最新状态，`best_checkpoint.pth`按开发集mIoU选取。不要用旧相加检查点恢复当前模型。

```bash
./scripts/hmnet-python -m tensorboard.main --logdir logs/segmentation --host 127.0.0.1 --port 6006
```

正式测试集评估（训练方案确定后使用）：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/test.py \
  experiments/segmentation/config/efficientvit_b1.py test \
  /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_test \
  --pretrained logs/segmentation/efficientvit_b1_cross/best_checkpoint.pth
```

# 原 HMNet 使用说明

以下是原 HMNet 的历史说明。其 B1/B3 命名、论文指标、训练策略及相对路径不属于上方 EfficientViT-B1 基线。当前 B1 请使用上方方案 C 命令；历史命令需按原实验目录布局执行，不能默认从仓库根目录照抄。

# Dataset Preparation

## [DSEC-Semantic](https://dsec.ifi.uzh.ch/dsec-semantic/)

Download events, images, and labels and place them in `./data/dsec/source/`

```bash
mkdir ./data/dsec/source/
cd ./data/dsec/source/
wget https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_events.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_images.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/semantic/train_semantic_segmentation.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/test_coarse/test_events.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/test_coarse/test_images.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/semantic/test_semantic_segmentation.zip
```

Unzip the files and run the following script.

```bash
cd ./data/dsec/
bash ./scripts/prepair.sh
```

Download and unzip metadata.

```bash
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/dsec_meta.tar
tar -xvf dsec_meta.tar
```

If you want to generate metadata from scratch, run the following command:

```bash
bash ./scripts/prepair.sh -a
```

# Reproduce our results

To reproduce the results of HMNet-B3:

(1) Download pretrained weights.

```bash
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_B3.pth
```
Put the weights in `./pretrained/`

(2) Run inference with the following commands.

```bash
python ./scripts/test.py ./config/hmnet_B3.py ./data/dsec/list/test/ ./data/dsec/ --speed_test --fast --pretrained ./pretrained/dsec_hmnet_B3.pth
```

(3) Evaluate the results.

```bash
sh ./scripts/run_eval.sh ./config/hmnet_B3.py
```

# Training & Inference

## Step1. Training

Single node training:

```bash
python ./scripts/train.py ./config/hmnet_B3.py --amp --distributed
```

Multi-node training:

```bash
# Run the following command at the first node.
python ./scripts/train.py ./config/hmnet_B3.py --amp --distributed --master ${master} --node 1/2
# Run the following command at the second node.
python ./scripts/train.py ./config/hmnet_B3.py --amp --distributed --master ${master} --node 2/2
```

`${master}`is the IP address of the first node.

## Step2. Inference using the trained model

```bash
python ./scripts/test.py ./config/hmnet_B3.py ./data/dsec/list/test/ ./data/dsec/ --fast --speed_test
```

# Evaluation

```bash
sh ./scripts/run_eval.sh ./config/hmnet_B3.py
```

# Training Details

The pre-trained weights are released under the Creative Commons BY-SA 4.0 License.

|  | GPU | Training Time [hr] | Loss | Weights | Log |
| --- | --- | --- | --- | --- | --- |
| hmnet_B1 | A100 (40GB) x 16 | 31.0 | 0.2817 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_B1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/dsec_hmnet_B1.csv) |
| hmnet_L1 | A100 (40GB) x 16 | 42.5 | 0.1881 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_L1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/dsec_hmnet_L1.csv) |
| hmnet_B3 | A100 (40GB) x 16 | 46.0 | 0.1685 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_B3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/dsec_hmnet_B3.csv) |
| hmnet_L3 | A100 (40GB) x 16 | 63.1 | 0.1410 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_L3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/dsec_hmnet_L3.csv) |

数据加载优化：三工程统一 workers=8、prefetch_factor=1、训练/验证 batch=32。验证保持 FP32；更改 worker 和验证 batch 后需重启或从 checkpoint 恢复，运行中的进程不会自动加载新配置。训练 batch、梯度累积和学习率曲线保持原值。

TensorBoard 兼容性：使用 `tensorboard==2.17.1`，修复 protobuf 5 下 HParams 接口的 `including_default_value_fields` 报错。升级后仅需在各 TensorBoard 终端 Ctrl+C 并重新运行原启动命令（端口保持原值），无需重启训练或删除事件文件。未安装 TensorFlow 的提示在 PyTorch 工程中可忽略。
