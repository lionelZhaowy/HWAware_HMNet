# EfficientViT-B1 分割：300 epoch 模态对照实验

本工程默认 **RGB-only**，分支 `seg_rgb_640x440_v1`。统一模型代码支持 `rgb`、`dvs`、`rgbdvs`；单模态只创建自己的编码器和投影层，不以置零输入冒充单模态训练。配置见 [efficientvit_b1.py](config/efficientvit_b1.py)。

| 工程 | 分支 | 用途 |
|---|---|---|
| HWAware_HMNet | main | 新一轮 RGB+DVS |
| HWAware_HMNet_Seg_RGB_640x440_v1 | seg_rgb_640x440_v1 | RGB-only |
| HWAware_HMNet_Seg_DVS_640x440_v1 | seg_dvs_640x440_v1 | DVS-only |
| HWAware_HMNet_Seg_RGBDVS_640x440_v1 | seg_rgbdvs_640x440_v1 | 上一轮固定学习率实验的存档，不应用新超参数 |

以下命令从**各自工程根目录**运行。三种新实验分别从同一官方ImageNet B1预训练权重初始化，不续训上一轮任务权重。对照实验共用样本、划分、增强、任务头和训练设置，模型参数不共享。

## 0. 当前实现、验证范围与后续路线

当前 B1 基线使用 **50 ms / 10 bins / 20 通道 RVT Histogram → 官方 EfficientViT-B1 → 四尺度 Pyramid → 各任务头**。它是无循环状态模型；尚未加入 HWAware B1、S-FIFO、自回归或 RENet/FRN 等复杂融合。目录名称含 HWAware 不代表当前骨干已经替换为硬件版本。

三任务复用同一骨干实现，分别实例化、训练和保存权重，**不是共享参数的多任务联合训练**。当前分割支持 RGB-only、DVS-only、RGB+DVS；融合为双独立分支四尺度投影相加。检测及 Eventscape/MVSEC 深度当前使用 DVS-only，尚未接入 RGB 融合实验。

| 任务 | 数据表示如何得到 | 当前验证范围 |
|---|---|---|
| DSEC 分割 | 离线共享缓存：Histogram、配准 RGB、标签及有效样本索引 | 已有真实数据冒烟、短训和训练/评估；正在开展三模态对照。保留当前缓存方案。 |
| GEN1 检测 | 必须先转换事件/标注并生成时间索引；Histogram 在 CPU DataLoader worker 在线构建 | 已接入任务头并做真实样本冒烟；未完成正式全量收敛实验。 |
| Eventscape / MVSEC 深度 | 必须先整理文件和生成时间索引；Histogram 在 CPU DataLoader worker 在线构建 | 已接入任务头并做真实样本冒烟；未完成正式全量收敛实验。 |

**在线 Histogram 不等于 GPU 预处理，也不等于下载后即可训练。** 一次性格式转换/索引与逐窗口稠密缓存是不同步骤；共享准备结果可被多个工程复用，不需要每个工程重新生成。

后续先在分割上验证事件表示、骨干、融合和时序自回归的方案，再将确认的公共结构迁移到检测/深度，适配任务头与数据协议并重新验证。下方命令服务于当前 B1 基线，不代表这些后续模块已实现或检测/深度已完成正式实验。分割三模态实验的训练设置应一致；不同任务目前的超参数并不相同，检测/深度默认值是待验证的基线设置。

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

## 2. Train

选择空闲GPU；在三个工程分别运行以下命令即可，默认模态由各工程配置决定：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py --single --amp --seed 42
```

正式输出为各工程独立的 `logs/segmentation/efficientvit_b1/`，不会写入共享缓存或artifacts。初始输出目录为空时直接开始；新实验使用其他 `--output`，不要误覆盖仍需保留的结果。

### 三组统一 TrainSettings

| 参数 | 设置与含义 |
|---|---|
| `modality` | 唯一的模态变量：`rgbdvs` / `rgb` / `dvs`。 |
| `epochs` | 300个数据轮次；由实际DataLoader长度自动换算。 |
| `updates` | `None`；指定正整数时改用累计优化器更新预算。正式对照不覆盖。 |
| `batch_size` | 32；每次前向/反向32个样本，最后不足一批保留。 |
| `accumulation` | 1；每个有效小批次更新一次，有效batch通常32。 |
| `learning_rate` | AdamW峰值学习率 `2e-4`。 |
| `lr_schedule` | `warmup_cosine`：线性预热后余弦下降。 |
| `warmup_epochs` | 5轮，对应1140次更新。 |
| `warmup_start_factor` | 0.1，首次更新lr=`2e-5`，预热末尾到`2e-4`。 |
| `min_learning_rate` | `2e-6`，300轮最后一次更新时到达。 |
| `weight_decay` | 0.01。 |
| `workers` | 16个数据加载进程，训练和验证共用。 |
| `prefetch_factor` | 1；每个worker预取一批，限制三实验并行时的主机内存。 |
| `eval_every_epochs` | 1；每轮验证，另在第1次更新和结束时验证。 |
| `eval_batch_size` | 32；控制训练中验证的显存，不影响训练batch。 |
| `overfit` | 0表示全训练集；正整数限制前N个样本并禁用翻转，仅供调试。 |
| `pretrained` | 官方ImageNet初始化路径，不是任务resume检查点。 |
| `resume` | 默认空；新三组必须从头初始化。 |
| `cache` | 上述共享训练/dev缓存。 |
| `output` | 本工程的独立logs目录。 |

7295样本、batch=32得到每轮228批，300轮共 **68400次更新**。所有LR按成功优化器更新推进；AMP溢出重试不推进LR，会略增加实际数据遍历量。三组固定相同seed=42、11类、ignore=255、主/辅助CE权重1.0/0.4。

旧版本没有任何LR调度器；1600轮误设不是lr恒定的直接原因。本轮是在加入调度器的同时，把旧accumulation=8改为1，因此与旧实验相比并非只改变了LR；新三组之间的超参数保持一致。

### 检查点与续训

- `checkpoint.pth`：最新完整训练状态，每轮及新best时保存。
- `best_checkpoint.pth`：开发集mIoU最高的完整状态，只由dev选取。
- `settings.json`记录实际训练预算、模态与调度器设置；TensorBoard和metrics记录实际使用的lr、loss、epoch、验证IoU等。

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py --single --amp --seed 42 \
  --resume logs/segmentation/efficientvit_b1/checkpoint.pth
```

续训保留原300轮目标，不再指定“追加300轮”。恢复时校验模态、样本数、数据路径、seed、batch、accumulation以及完整LR曲线；改变这些条件或使用旧恒定LR检查点会明确报错。这样可避免学习率重置或衰减周期悄悄变化。

```bash
./scripts/hmnet-python -m tensorboard.main --logdir logs --host 127.0.0.1 --port 6006
```

三终端比较可在一个工程中将logdir改为三个工程logs路径的共同父目录。`B1_*`旧环境变量不再生效；batch等参数编辑配置，常用输出/路径可用 `--output` / `--data-root` 覆盖。

## 3. Test

开发集评估最佳检查点：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/test.py \
  experiments/segmentation/config/efficientvit_b1.py \
  --pretrained logs/segmentation/efficientvit_b1/best_checkpoint.pth
```

固定训练方案后，最终评估官方测试集；不要用test选超参数：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/test.py \
  experiments/segmentation/config/efficientvit_b1.py test \
  /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_test \
  --pretrained logs/segmentation/efficientvit_b1/best_checkpoint.pth
```

结果为实验目录的 `evaluation_dev.json` / `evaluation_test.json`。不指定 `--pretrained` 时仍读取最新 `checkpoint.pth`；测试 `TestSettings.batch_size=32`。三种模态均使用同一指标实现。

## 4. 提前停止后的低学习率追加实验

此阶段用于检查平台期是否受学习率影响，不修改原始300轮计划的历史记录。RGB+DVS、RGB 从约120个epoch-equivalents的完整检查点开始；DVS 因第120轮未保留检查点，经确认从约123轮开始，结果须保留该差异。

统一使用 `efficientvit_b1_cooldown.py`：**追加20轮**（4560次成功更新），lr 从 `2e-5` 余弦下降到 `2e-6`、无预热；batch=32、accumulation=1及其他设置继承同工程基线。恢复模型、AdamW动量、AMP scaler、数据游标和随机状态，仅重置阶段内更新计数与阶段最佳指标。新阶段输出必须是独立空目录；普通 `--resume` 的契约校验仍然保留。

```bash
# 在实验管理器未启动该阶段时手动使用；不要重复启动同一输出目录。
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1_cooldown.py --single --amp --seed 42 \
  --resume logs/segmentation/efficientvit_b1/parent_epoch120.pth
```

DVS 的输入文件为 `parent_epoch123.pth`。阶段自身续训将 `--resume` 改为 `logs/segmentation/efficientvit_b1/cooldown_20ep/checkpoint.pth`，仍使用 cooldown 配置，不会再追加另一个20轮。TensorBoard step 为本阶段计数，`data_epochs` 保留从原训练累计的数据轮次；控制脚本保存的 `baseline.json` 标明真实起点。

判断仅使用开发集：任一模态的低LR最后5轮平均mIoU比原阶段最后10次验证平均值增加至少 **0.3个百分点**，且阶段最佳mIoU超过原最佳至少 **0.1个百分点**，则将三组新DSEC实验的默认epoch统一设为150，否则设为120。该阈值用于减少单次波动造成的误判，不是统计显著性检验。异常退出或结果不完整时不自动修改默认值。当前阶段实验不是对“从头训练150轮”的直接精度验证；新默认轮数仍需后续实验确认。

训练、检查点和结果在 `logs/segmentation/efficientvit_b1/cooldown_20ep/`，原始最新/最佳权重及阶段起点快照保留。其他任务的训练预算不随此实验修改。

---

# 原 HMNet 使用说明

以下是原 HMNet 的历史说明。其 B1/B3 命名、论文指标、训练策略及相对路径不属于上方 EfficientViT-B1 基线。当前 B1 请使用上方第 1–3 节；历史命令需按原实验目录布局执行，不能默认从仓库根目录照抄。

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

数据加载优化：三工程统一 workers=16、prefetch_factor=1、训练/验证 batch=32。验证保持 FP32；更改 worker 和验证 batch 后需重启或从 checkpoint 恢复，运行中的进程不会自动加载新配置。训练 batch、梯度累积和学习率曲线保持原值。

TensorBoard 兼容性：使用 `tensorboard==2.17.1`，修复 protobuf 5 下 HParams 接口的 `including_default_value_fields` 报错。升级后仅需在各 TensorBoard 终端 Ctrl+C 并重新运行原启动命令（端口保持原值），无需重启训练或删除事件文件。未安装 TensorFlow 的提示在 PyTorch 工程中可忽略。
