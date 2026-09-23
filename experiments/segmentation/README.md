> 本分支当前采用异步RGB–DVS实验，请使用 [ASYNC_RGB_DVS.md](ASYNC_RGB_DVS.md) 的数据、两阶段训练和评估命令。下面保留历史HMNet/同步基线说明。

# v1.2_T时序实验入口

本工程为v1.2的独立M=2 DVS时序实验，融合保持SimpleAdd。只在新_T工程修改，公共时序实现与另一_T工程一致。

以[本实验训练/恢复/评估/ONNX说明](TRAINING_PRECISION.md)为准。该文包含连续采样、FP32状态、BF16训练、TBPTT=1和已知ONNX数值限制。下文为继承的历史说明，旧输出路径/无状态描述不代表本实验默认配置。

---

# HWAware_HMNet_Seg_RGBDVS_640x440_v1.2：DSEC融合消融

分支 `seg_rgbdvs_640x440_v1.2`；从 `HWAware_HMNet_Seg_RGBDVS_640x440_v1.1` 的 `04f01df` 创建源码副本。当前配置：`fusion_mode=add`，BF16、150 epoch、batch32。

**[当前训练/恢复命令、模型公式及验证限制](TRAINING_PRECISION.md)**。正式输出为 `logs/segmentation/efficientvit_b1_add_v12_bf16/`。请从官方B1预训练初始化，不resume父工程任务检查点或artifacts内的诊断权重。

公共训练代码、骨干与任务头在v1.2/v3/v2.1.1一致；主配置只选择融合模式和输出路径。旧FP16、300轮、低LR追加实验命令保留于父工程历史版本，不作为本轮默认日程。

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

## 评估与导出

正式训练后，在本工程使用 `scripts/evaluate_dsec_b1.py --help` 选择固定检查点和独立artifacts输出；开发集选模后再评估官方test。评估脚本拒绝结构契约不一致的检查点。当前尚未获得正式收敛模型。

此次结构图在 `artifacts/onnx/step3/`，包含整网/骨干、原图/简化图和数值报告。v3有边界输入超差，详见TRAINING_PRECISION.md，不能当成部署验收通过。

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

数据加载优化：三工程统一 workers=8、prefetch_factor=1、训练/验证 batch=32。验证保持 FP32；更改 worker 和验证 batch 后需重启或从 checkpoint 恢复，运行中的进程不会自动加载新配置。训练 batch、梯度累积和学习率曲线保持原值。

TensorBoard 兼容性：使用 `tensorboard==2.17.1`，修复 protobuf 5 下 HParams 接口的 `including_default_value_fields` 报错。升级后仅需在各 TensorBoard 终端 Ctrl+C 并重新运行原启动命令（端口保持原值），无需重启训练或删除事件文件。未安装 TensorFlow 的提示在 PyTorch 工程中可忽略。
