# EfficientViT-B1 使用说明

配置 [efficientvit_b1.py](config/efficientvit_b1.py)：50ms RVT Histogram → DVS单分支官方B1 → 四尺度Pyramid → YOLOX（strides 8/16/32）。无循环状态，不需要TBPTT。以下均从仓库根目录运行。

## 1. 数据集准备（共享目录）

将原HMNet预处理结果放在 `/data/lab_dataset/RGB_DVS_Fusion/GEN1/preprocessed/hmnet/`（或SSD数据集目录下同名层级），配置 `data_root` 指向这里。要求 `train_evt/train_lbl/val_evt/val_lbl/test_evt/test_lbl`，以及 `list/<split>/{events.txt,labels.txt,meta.pkl,gt_interval.csv}`。原始DAT目录不能直接用于训练。复用下方原预处理流程时，在共享目录布置 `source` 和 `scripts` 链接后运行脚本，输出仍留在共享目录；已有完整结果直接复用。

DSEC缓存已迁移；本轮未对GEN1原数据或其他预处理目录进行移动、未执行全量预处理。GEN1 Histogram通过现有EventFrame在线构建，不需要每个工程生成一套稠密缓存。

## 2. Train

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/detection/scripts/train.py \
  experiments/detection/config/efficientvit_b1.py --single --amp
```

默认epochs=100、updates=None、batch=2、accumulation=8；训练长度自动计算。该预算尚未通过全量检测收敛实验验证。配置数据根目录不同可加 `--data-root /共享目录`，独立实验加 `--output logs/detection/run02`。

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/detection/scripts/train.py \
  experiments/detection/config/efficientvit_b1.py --single --amp \
  --resume logs/detection/efficientvit_b1/checkpoint.pth --epochs 200
```

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

## 3. Test（推理及mAP）

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/detection/scripts/test.py \
  experiments/detection/config/efficientvit_b1.py \
  /data/lab_dataset/RGB_DVS_Fusion/GEN1/preprocessed/hmnet/list/test \
  /data/lab_dataset/RGB_DVS_Fusion/GEN1/preprocessed/hmnet

mkdir -p logs/detection/efficientvit_b1/evaluation/test
./scripts/hmnet-python experiments/detection/scripts/psee_evaluator.py \
  /data/lab_dataset/RGB_DVS_Fusion/GEN1/preprocessed/hmnet/test_lbl \
  logs/detection/efficientvit_b1/result/pred_test --camera GEN1 \
  > logs/detection/efficientvit_b1/evaluation/test/result.txt
```

默认读output/checkpoint.pth；可通过 `--output` 选择实验、`--pretrained` 选择权重。验证集改为list/val、val_lbl、pred_val。GT使用含invalid字段的预处理标注，保持原GEN1时间容差与框尺寸过滤。检测训练目前不自动计算验证mAP，需要单独执行上述评估。

---

# 原 HMNet 使用说明

以下保留原 HMNet 的数据准备、配置及复现实验说明。

# Dataset Preparation

## [GEN1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)

Clone and copy toolbox for gen1 dataset.

```bash
git clone https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox.git
mv prophesee-automotive-dataset-toolbox/src ${HMNet}/hmnet/utils/psee_toolbox
rm -rf ./prophesee-automotive-dataset-toolbox
```

Download events and bbox annotations at [dataset page](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) and place the files in `./data/gen1/source/`

```bash
./data/gen1/source/
├── test_a.7z
├── test_b.7z
├── train_a.7z
├── train_b.7z
├── train_c.7z
├── train_d.7z
├── train_e.7z
├── train_f.7z
├── val_a.7z
└── val_b.7z
```

Unzip the files and run the following script.

```bash
cd ./data/gen1/
bash ./scripts/prepair.sh
```

Download and unzip metadata.

```bash
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_train_meta.tar
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_valtest_meta.tar
tar -xvf gen1_train_meta.tar
tar -xvf gen1_valtest_meta.tar
```

If you want to generate metadata from scratch, run the following command:

```bash
bash ./scripts/prepair.sh -a
```

# Reproduce our results

To reproduce the results of HMNet-B3:

(1) Download pretrained weights.

```bash
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_B3_tbptt.pth
```
Put the weights in `./pretrained/`

(2) Run inference with the following commands.

```bash
python ./scripts/test.py ./config/hmnet_B3_yolox_tbptt.py ./data/gen1/list/test/ ./data/gen1/ --pretrained ./pretrained/gen1_hmnet_B3_tbptt.pth --fast --speed_test
```

(3) Evaluate the results.

```bash
sh ./scripts/run_eval.sh ./config/hmnet_B3_yolox_tbptt.py
```

# Training & Inference

## Step1. Training with short sequences

Single node training:

```bash
python ./scripts/train.py ./config/hmnet_B3_yolox.py --amp --distributed
```

Multi-node training:

```bash
# Run the following command at the first node.
python ./scripts/train.py ./config/hmnet_B3_yolox.py --amp --distributed --master ${master} --node 1/2
# Run the following command at the second node.
python ./scripts/train.py ./config/hmnet_B3_yolox.py --amp --distributed --master ${master} --node 2/2
```

`${master}`is the IP address of the first node.

## Step2. Training with long sequences (TBPTT)

```bash
python ./scripts/train.py ./config/hmnet_B3_yolox_tbptt.py --amp --distributed
```

## Step3. Inference using the trained model

```bash
python ./scripts/test.py ./config/hmnet_B3_yolox_tbptt.py ./data/gen1/list/test/ ./data/gen1/ --fast --speed_test
```

# Evaluation

```bash
sh ./scripts/run_eval.sh ./config/hmnet_B3_yolox_tbptt.py
```

# Training Details

The pre-trained weights are released under the Creative Commons BY-SA 4.0 License.

|  | GPU | Training Time [hr] | Loss | Weights | Log |
| --- | --- | --- | --- | --- | --- |
| hmnet_B1 | A100 (40GB) x 8 | 28.8 | 2.3797 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_B1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_hmnet_B1.csv) |
| hmnet_B1_tbptt | A100 (40GB) x 8 | 12.3 | 2.8653 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_B1_tbptt.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_hmnet_B1_tbptt.csv) |
| hmnet_L1 | A100 (40GB) x 8 | 42.0 | 2.0687 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_L1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_hmnet_L1.csv) |
| hmnet_L1_tbptt | A100 (40GB) x 16 | 10.5 | 2.8095 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_L1_tbptt.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_hmnet_L1_tbptt.csv) |
| hmnet_B3 | A100 (40GB) x 8 | 47.9 | 2.1555 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_B3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_hmnet_B3.csv) |
| hmnet_B3_tbptt | A100 (40GB) x 16 | 18.7 | 2.5021 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_B3_tbptt.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_hmnet_B3_tbptt.csv) |
| hmnet_L3 | A100 (40GB) x 8 | 64.5 | 1.9462 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_L3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_hmnet_L3.csv) |
| hmnet_L3_tbptt | A100 (40GB) x 16 | 19.0 | 2.4505 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_L3_tbptt.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/gen1_hmnet_L3_tbptt.csv) |
