# EfficientViT-B1 使用说明

本节是无状态的官方 EfficientViT-B1 RGB+DVS双分支分割模型，配置见 [efficientvit_b1.py](config/efficientvit_b1.py)。它与原 `hmnet_B1.py` 不同。以下命令均从**仓库根目录**运行，使用 `scripts/hmnet-python` 对应的 Conda pytorch 环境。

## 1. 数据集准备（共享缓存，只需一次）

本机全量缓存已准备并迁移至：

```text
/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/
├── source/                         原始事件、RGB、语义标签
└── preprocessed/
    ├── dsec_b1/                    8082个样本，manifest及配准检查图
    └── dsec_b1_assets/             已下载的标定与rectify_map
```

**已有完整缓存直接进入 train，无需重新转换。** 新机器或新数据的准备命令：

```bash
./scripts/hmnet-python scripts/prepare_dsec_b1.py \
  --source /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/source \
  --assets /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_assets \
  --download-assets
```

默认输出到 `source` 的父目录下 `preprocessed/dsec_b1`，默认使用全部标签；`--per-sequence 32` 仅用于有界检查，应配合不同 `--output`。已有 manifest 时准备器拒绝覆盖。HDD 上同样遵循 `<数据集>/preprocessed/` 布局。多个工程直接读取同一缓存，不跨工程访问 logs。

Histogram 为50ms、10bins、20通道，复用RVT；官方标定将左RGB映射至原始左事件坐标，再保留顶部440×640。RGB仅选择目标之前50ms内的最近帧；映射不补偿平移造成的深度相关视差。`zurich_city_08_a` 的787个样本用于开发验证，其余7295个训练；不用官方测试集调参。

原工程 `logs/datasets/dsec_b1` 和标定路径仅保留为软链接，供迁移前已启动的训练继续读取。新配置 `cache` 直接指向上述共享路径。

## 2. Train

在 [TrainSettings](config/efficientvit_b1.py) 设置 `epochs=100, updates=None` 后直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py --single --amp
```

先用 `nvidia-smi` 选择空闲GPU。当前保留用户设置的 `batch_size=32, accumulation=8`，有效batch通常256；7295个样本对应每轮228批，100轮自动换算为2850次更新。默认输出 `logs/segmentation/efficientvit_b1/`。独立实验可加 `--output logs/segmentation/run02`。

续训示例（累计目标必须超过已完成步数）：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py --single --amp \
  --resume logs/segmentation/efficientvit_b1/checkpoint.pth --epochs 200
```

**2026-09-19核查：已有 `full_100ep` 实验实际 batch=32，但沿用了按batch=2计算的45600步预算，等价约1600轮。目录名不能作为epoch依据。** 本轮未停止或修改该进程的预算。要维持其原目标，用新版 `--updates 45600` 并指定原 `--resume`、`--output`；若改用epochs，先确认检查点实际轮数，100轮目标已经不足以继续当前实验。

分割专用设置：`overfit=0` 使用全部训练样本且共同水平翻转；正整数N只用前N个样本、关闭翻转，用于过拟合检查，不是训练轮数。小样本检查可把 `workers=0`。`cache` 是共享数据路径。测试的 `TestSettings.batch_size=2` 独立于训练batch，避免大批量评估占用过多显存。

### TrainSettings 参数

| 参数 | 含义与设置 |
|---|---|
| `epochs` | 目标数据轮数，默认100；`updates=None` 时生效。 |
| `updates` | 显式的**累计优化器更新次数**，默认 `None`。正整数优先于 epochs，用于短训/精确步数实验。 |
| `batch_size` | 每个 microbatch 的样本数；单次前向/反向的批量。最后不足一批也保留。 |
| `accumulation` | 累积多少个有效 microbatch 才更新参数；默认8。通常有效 batch=`batch_size × accumulation`。 |
| `learning_rate` | AdamW 学习率，默认 `2e-4`，当前固定不衰减。 |
| `weight_decay` | AdamW 权重衰减，默认 `0.01`。 |
| `workers` | DataLoader 子进程数，默认2；0在主进程加载。 |
| `output` | 本工程的训练产物目录，默认 `logs/<任务>/efficientvit_b1/...`，包含权重、指标、TensorBoard及评估结果。 |
| `resume` | 同一实验的完整任务检查点路径；空字符串从初始化开始。 |
| `pretrained` | 官方 ImageNet B1 分类权重路径，仅用于初始化，不是任务续训权重。 |
| `cache` / `data_root` | 共享预处理数据目录；分割使用 cache，检测/深度使用 data_root。 |
| `task` / `frame_training` | 内部任务标识和帧式分派开关，通常无需修改。 |

训练器自动计算 `ceil(len(train_loader) × epochs / accumulation)`，无需终端计算 step。累积可跨轮，最后可能多读不足一个累积窗口的批次；无效批次和AMP溢出重试也可能增加实际遍历量，因此 epochs 是换算预算，最终按成功更新次数停止。启动日志和 `settings.json` 记录实际样本数、批次数、epoch预算及最终更新预算；逐步记录 `epoch`（从1开始）和 `data_epochs`（已遍历轮数，可有小数）。

可选覆盖参数只有常用项：`--epochs N`（清除配置中的显式 updates）、`--updates N`、`--resume 路径`、`--output 路径`、`--data-root 路径`。`--epochs` 与 `--updates` 不可同时指定。batch/累积次数/学习率等直接编辑配置。新版不再读取 `B1_UPDATES`、`B1_OUTPUT`、`B1_RESUME` 等旧环境变量；运行中的旧进程不受源码修改影响。

续训目标仍为**累计目标**，不是追加轮数或步数。目标已经小于或等于检查点步数会报错提示，不会清空历史。续训应保持同一数据拆分、batch和accumulation；跨数据集权重迁移不能使用带采样游标的 resume。

### 日志与 TensorBoard

```bash
./scripts/hmnet-python -m tensorboard.main --logdir logs --host 127.0.0.1 --port 6006
```

事件文件位于实验目录 `tensorboard/`，记录总损失、任务头分项损失、实际学习率、已遍历轮数、AMP缩放/跳过次数、显存、耗时和吞吐。分割额外记录开发集mIoU/各类IoU（0–1）。固定学习率的曲线为水平线。

`metrics.jsonl` 每次更新写入；`checkpoint.pth` 每50次更新和结束时保存最新状态，没有单独的最佳模型文件。恢复时清除超过检查点的旧曲线点。已有实验默认拒绝覆盖，新实验用不同 `--output`；显式 `--overwrite` 会重建本实验历史。远程查看时在本地使用 `ssh -N -L 6006:127.0.0.1:6006 用户名@服务器`，浏览器打开 `http://localhost:6006`。

数据在数据集的共享 `preprocessed/` 目录，训练产物在本工程 `logs/`，诊断历史在 `artifacts/`。`logs/` 已忽略，无新增 `!` 例外。

## 3. Test

默认使用配置 cache 的 dev 划分及 output/checkpoint.pth：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/test.py \
  experiments/segmentation/config/efficientvit_b1.py
```

评估本次正在训练的实验的最近检查点（评估时选择另一张空闲GPU）：

```bash
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python experiments/segmentation/scripts/test.py \
  experiments/segmentation/config/efficientvit_b1.py \
  --output logs/segmentation/efficientvit_b1/full_100ep
```

可用 `--pretrained 路径` 显式选择任务权重；覆盖划分/缓存时使用 `test.py 配置 train或dev 共享缓存目录`。直接输出mIoU、各类IoU和混淆矩阵至实验目录 `evaluation_dev.json`，无需运行旧 `run_eval.sh`，不传 `--fast`。训练第1步、每50步及最后一步也自动验证；总损失含主/辅助CE系数1.0/0.4，TensorBoard分项为头部提供的未加权CE。


### 官方测试集与单序列演示

2026-09-19：`full_100ep` 已按要求停止，使用step=8500检查点完成评估。官方test 2809帧mIoU为59.49%，本机HMNet-B3纯事件基线为53.97%；输入模态不同。完整[评估报告](../../logs/segmentation/efficientvit_b1/full_100ep/assessment/评估与能力分析.md)与[视频页面](../../logs/segmentation/efficientvit_b1/full_100ep/assessment/index.html)保存在实验logs目录。

训练缓存中的 `dev` 是留出的训练序列，不是官方测试集。需要与 HMNet 测试精度比较时，单独准备测试缓存（仅一次）：

```bash
./scripts/hmnet-python scripts/prepare_dsec_b1.py \
  --source /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/source \
  --split test \
  --assets /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_assets \
  --download-assets
```

默认写入共享 `preprocessed/dsec_b1_test`，不会混入训练缓存。完整指标可用原 test 入口：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/segmentation/scripts/test.py \
  experiments/segmentation/config/efficientvit_b1.py test \
  /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_test \
  --output logs/segmentation/efficientvit_b1/full_100ep
```

同时导出逐序列指标、预测和演示视频：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/evaluate_dsec_b1.py \
  --checkpoint logs/segmentation/efficientvit_b1/full_100ep/checkpoint.pth \
  --cache /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_test \
  --output logs/segmentation/efficientvit_b1/full_100ep/assessment \
  --demo-sequence zurich_city_13_a \
  --hmnet-dir artifacts/full-eval/seg
```

`--hmnet-dir` 可省略；本机该目录是历史 HMNet-B3 纯事件官方权重预测，不是新产物目录。新增产物均位于当前实验 `assessment/`：`assessment.json`、`sequence_demo.mp4`、示例截图、逐帧预测和指标。视频使用环境中的 ffmpeg/libx264。比较按同一语义帧索引及11类标签进行；HMNet预热缺失帧同时报告“缺失计错”和“共同有预测帧”两个口径。RGB+DVS与纯事件的输入信息不同，不能据此单独归因骨干优劣。

---

# 原 HMNet 使用说明

以下保留原 HMNet 的数据准备、配置及复现实验说明。

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
