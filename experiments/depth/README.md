# EfficientViT-B1 使用说明

配置 [efficientvit_b1.py](config/efficientvit_b1.py)：50ms RVT Histogram → DVS单分支官方B1 → 四尺度Pyramid → 原深度头。Eventscape和MVSEC独立实例化及训练。以下均从仓库根目录运行。

## 1. 数据集准备（共享目录）

- Eventscape：`/data/lab_dataset/RGB_DVS_Fusion/Eventscape/preprocessed/hmnet/`，包含原HMNet的事件NPY、图像/深度索引、`list/<split>/{events.txt,images.txt,labels.txt,meta.pkl,video_duration.csv}`。将配置 `data_root` 指向此目录。
- MVSEC：`/data/lab_dataset/RGB_DVS_Fusion/MVSEC/preprocessed/hmnet/`，包含 `outdoor_day2_data.hdf5`、`outdoor_day2_gt.hdf5`、`outdoor_day2_meta.npy` 和day1/night1测试对应文件。大型原始HDF5可使用指向共享source的链接，生成的meta放在preprocessed内。

具体预处理沿用下文原脚本，输出布局放在共享目录。已有结果可复用，通过配置或 `--data-root` 指向实际路径；本轮未移动/重新转换这两套数据。Histogram在线构建。首版不加载RGB，但Eventscape仍使用原图像索引接口。

## 2. Train

Eventscape：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/depth/scripts/train.py \
  experiments/depth/config/efficientvit_b1.py --single --amp
```

MVSEC day2：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/depth/scripts/train.py \
  experiments/depth/config/efficientvit_b1.py --single --amp --finetune
```

`--finetune` 在当前B1实现中**仅选择MVSEC数据配置和深度范围，不会自动加载Eventscape任务权重**。默认仍从ImageNet B1初始化。两者默认100轮，batch=2、累积8次；当前只有真实样本冒烟验证，未验证全量收敛。

MVSEC续训示例；Eventscape去掉 `--finetune` 并改用eventscape权重目录：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/depth/scripts/train.py \
  experiments/depth/config/efficientvit_b1.py --single --amp --finetune \
  --resume logs/depth/efficientvit_b1/mvsec/checkpoint.pth --epochs 200
```

`dataset` 选择数据类型，`FTSettings` 继承TrainSettings并覆盖MVSEC的data_root/output。两种数据集均不在训练中自动计算验证指标，需执行独立评估。

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

## 3. Test（推理及深度指标）

Eventscape：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/depth/scripts/test.py \
  experiments/depth/config/efficientvit_b1.py \
  /data/lab_dataset/RGB_DVS_Fusion/Eventscape/preprocessed/hmnet/list/test \
  /data/lab_dataset/RGB_DVS_Fusion/Eventscape/preprocessed/hmnet

mkdir -p logs/depth/efficientvit_b1/eventscape/evaluation/test
./scripts/hmnet-python experiments/depth/scripts/eval_depth.py \
  logs/depth/efficientvit_b1/eventscape/result/pred_test \
  /data/lab_dataset/RGB_DVS_Fusion/Eventscape/preprocessed/hmnet/list/test/labels.txt \
  logs/depth/efficientvit_b1/eventscape/evaluation/test \
  --max_depth 1000 --min_depth 3.346 --cutoff_depth 30 20 10 \
  --input_type info --skip_ts 199.9 \
  --gt_root /data/lab_dataset/RGB_DVS_Fusion/Eventscape/preprocessed/hmnet
```

`--skip_ts 199.9` 保留原评估过滤，不是模型预热。预测为米制深度，不传 `--nlog_pred`。

MVSEC：

```bash
for sequence in day1 night1; do
  CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python experiments/depth/scripts/test_mvsec.py \
    experiments/depth/config/efficientvit_b1.py "$sequence"
  mkdir -p "logs/depth/efficientvit_b1/mvsec/evaluation/$sequence"
  ./scripts/hmnet-python experiments/depth/scripts/eval_depth.py \
    "logs/depth/efficientvit_b1/mvsec/result/pred_$sequence/outdoor_${sequence}_data.npy" \
    "/data/lab_dataset/RGB_DVS_Fusion/MVSEC/preprocessed/hmnet/outdoor_${sequence}_gt.hdf5" \
    "logs/depth/efficientvit_b1/mvsec/evaluation/$sequence" \
    --max_depth 80 --min_depth 1.978 --cutoff_depth 30 20 10 --input_type mvsec
done
```

默认读取对应output/checkpoint.pth。用 `--output` 选择其他实验、`--pretrained` 指定权重；MVSEC测试新增 `--data-root` 可指定其他共享数据目录。day2训练、day1/night1测试，不以测试集调参。

---

# 原 HMNet 使用说明

以下保留原 HMNet 的数据准备、配置及复现实验说明。

# Dataset Preparation

## [Eventscape](https://github.com/uzh-rpg/rpg_ramnet)

Download files and place them in `./data/eventscape/source/`

```bash
mkdir ./data/eventscape/source/
cd ./data/eventscape/source/
wget http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town01-03_train.zip
wget http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town05_val.zip
wget http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town05_test.zip
unzip Town01-03_train.zip
unzip Town05_val.zip
unzip Town05_test.zip
```

Run the following script.

```bash
cd ./data/eventscape/
bash ./scripts/prepair.sh
```

Download and unzip metadata.

```bash
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_meta.tar
tar -xvf eventscape_meta.tar
```

If you want to generate metadata from scratch, run the following command:

```bash
bash ./scripts/prepair.sh -a
```

## [MVSEC](https://daniilidis-group.github.io/mvsec/)

Download files at [this page](https://daniilidis-group.github.io/mvsec/download/) and place them in `./data/mvsec/source/`

```bash
./data/mvsec/source/
├── outdoor_day1_data.hdf5
├── outdoor_day1_gt.hdf5
├── outdoor_day2_data.hdf5
├── outdoor_day2_gt.hdf5
├── outdoor_night1_data.hdf5
└── outdoor_night1_gt.hdf5
```

Download metadata.

```bash
cd ./data/mvsec/source/
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_meta.tar
tar -xvf mvsec_meta.tar
```

If you want to generate metadata from scratch, run the following command:

```bash
cd ./data/mvsec/
bash ./scripts/prepair.sh
```

# Reproduce our results

To reproduce the results of HMNet-B3:

(1) Download pretrained weights.

```bash
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3.pth
```
Put the weights in `./pretrained/`

(2) Run inference with the following commands.

```bash
# inference on outdoor day1
python ./scripts/test_mvsec.py ./config/hmnet_B3.py day1 --fast --speed_test --pretrained ./pretrained/hmnet_B3_mvsec.pth
# inference on outdoor night1
python ./scripts/test_mvsec.py ./config/hmnet_B3.py night1 --fast --speed_test --pretrained ./pretrained/hmnet_B3_mvsec.pth
```

(3) Evaluate the results.

```bash
sh ./scripts/run_eval_mvsec.sh ./config/hmnet_B3.py
```

# Training & Inference

## Step1. Pre-training on Eventscape

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

## Step2. Fine-tuning on MVSEC outdoor day2

```bash
python ./scripts/train.py ./config/hmnet_B3.py --amp --distributed --finetune --overwrite
```

## Step3. Inference using the trained model

Eventscape dataset:

```bash
python ./scripts/test.py ./config/hmnet_B3.py ./data/eventscape/list/test/ ./data/eventscape/ --fast --speed_test
```

MVSEC outdoor day1/night1:

```bash
python ./scripts/test_mvsec.py ./config/hmnet_B3.py day1 --fast --speed_test
python ./scripts/test_mvsec.py ./config/hmnet_B3.py night1 --fast --speed_test
```

# Evaluation

```bash
# Eval on Eventscape
sh ./scripts/run_eval_eventscape.sh ./config/hmnet_B3.py
# Eval on MVSEC outdoor day1/night1
sh ./scripts/run_eval_mvsec.sh ./config/hmnet_B3.py
```

# Training Details

The pre-trained weights are released under the Creative Commons BY-SA 4.0 License.

Pre-training on Eventscape

|  | GPU | Training Time [hr] | Loss | Weights | Log |
| --- | --- | --- | --- | --- | --- |
| hmnet_B1 | A100 (40GB) x 8 | 11.3 | 0.0317 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_B1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_B1.csv) |
| hmnet_L1 | A100 (40GB) x 8 | 20.2 | 0.0293 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_L1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_L1.csv) |
| hmnet_B3 | A100 (40GB) x 8 | 20.6 | 0.0304 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_B3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_B3.csv) |
| hmnet_L3 | A100 (40GB) x 8 | 30.1 | 0.0294 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_L3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_L3.csv) |
| hmnet_B3_fuse_rgb | A100 (40GB) x 8 | 25.9 | 0.0277 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_B3_fuse_rgb.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_B3_fuse_rgb.csv) |
| hmnet_L3_fuse_rgb | A100 (40GB) x 8 | 30.2 | 0.0273 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_L3_fuse_rgb.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_L3_fuse_rgb.csv) |

Fine-tuning on MVSEC

|  | GPU | Training Time [hr] | Loss | Weights | Log |
| --- | --- | --- | --- | --- | --- |
| hmnet_B1 | A100 (40GB) x 8 | 1.8 | 0.0319 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_B1.csv) |
| hmnet_L1 | A100 (40GB) x 8 | 1.2 | 0.0296 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_L1.csv) |
| hmnet_B3 | A100 (40GB) x 8 | 1.7 | 0.0286 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_B3.csv) |
| hmnet_L3 | A100 (40GB) x 8 | 1.3 | 0.0279 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_L3.csv) |
| hmnet_B3_fuse_rgb | A100 (40GB) x 8 | 2.7 | 0.0270 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3_fuse_rgb.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_B3_fuse_rgb.csv) |
| hmnet_L3_fuse_rgb | A100 (40GB) x 8 | 2.7 | 0.0264 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L3_fuse_rgb.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_L3_fuse_rgb.csv) |
