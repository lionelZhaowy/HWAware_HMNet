# Hierarchical Neural Memory Network

This repo is a PyTorch implementation of HMNet proposed in our paper: [Hierarchical Neural Network for Low Latency Event Processing](https://hamarh.github.io/hmnet/).

## 当前工程入口与实验阶段

本仓库保留原 HMNet，并增加无状态官方 EfficientViT-B1 基线。**下方原论文 HMNet-B1/B3 的结果不是新 EfficientViT-B1 的实测结果**，名称中的 B1 也不表示相同模型。

- [分割：DSEC，数据准备 / train / test](experiments/segmentation/README.md)：当前先验证 RGB、DVS、RGB+DVS，保留共享离线缓存。
- [检测：GEN1，数据准备 / train / test](experiments/detection/README.md)：已接入 B1 并做真实样本冒烟，正式全量实验待开展。
- [深度：Eventscape / MVSEC，数据准备 / train / test](experiments/depth/README.md)：已接入 B1 并做真实样本冒烟，正式全量实验待开展。

三任务复用骨干结构、独立实例化训练。当前工程为 **HWAware_HMNet_Seg_RGBDVS_640x440_v1.2**，分支 `seg_rgbdvs_640x440_v1.2`，默认融合 `add`，BF16、150 epoch。结构说明、验证范围与训练/恢复命令见[融合实验说明](experiments/segmentation/TRAINING_PRECISION.md)。保留原HMNet、检测/深度入口；当前模型尚未加入HWAware、S-FIFO或自回归。

训练产物放本工程 `logs/`；Agent诊断与ONNX放 `artifacts/`；数据预处理放共享数据集 `preprocessed/`。新工程未复制父工程历史日志或训练权重。

## Results and models

The pre-trained weights are released under the Creative Commons BY-SA 4.0 License.

**DSEC-Semantic (Semantic Segmentation)**

| Model | size | mIoU [%] | latency V100 [ms] | latency V100 x 3 [ms] | weights |
| --- | --- | --- | --- | --- | --- |
| HMNet-B1 | 640 x 440 | 51.2 | 7.0  | -    | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_B1.pth) |
| HMNet-L1 | 640 x 440 | 55.0 | 10.5 | -    | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_L1.pth) |
| HMNet-B3 | 640 x 440 | 53.9 | 9.7  | 8.0  | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_B3.pth) |
| HMNet-L3 | 640 x 440 | 57.1 | 13.9 | 11.9 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_L3.pth) |

**GEN1 (Object Detection)**

| Model | size | mAP [%] | latency V100 [ms] | latency V100 x 3 [ms] | weights |
| --- | --- | --- | --- | --- | --- |
| HMNet-B1 | 304 x 240 | 45.5 | 4.6 | -   | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_B1_tbptt.pth) |
| HMNet-L1 | 304 x 240 | 47.0 | 5.6 | -   | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_L1_tbptt.pth) |
| HMNet-B3 | 304 x 240 | 45.2 | 7.0 | 5.9 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_B3_tbptt.pth) |
| HMNet-L3 | 304 x 240 | 47.1 | 7.9 | 7.0 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_L3_tbptt.pth) |

**MVSEC day1 (Monocular Depth Estimation)**

| Model | size | AbsRel | RMS | RMSElog | latency V100 [ms] | latency V100 x 3 [ms] | weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HMNet-B1 | 346 x 260 | 0.385 | 9.088 | 0.509 | 2.4 | -   | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B1.pth) |
| HMNet-L1 | 346 x 260 | 0.310 | 8.383 | 0.393 | 4.1 | -   | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L1.pth) |
| HMNet-B3 | 346 x 260 | 0.270 | 7.101 | 0.332 | 5.0 | 4.1 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3.pth) |
| HMNet-L3 | 346 x 260 | 0.254 | 6.890 | 0.319 | 6.9 | 5.4 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L3.pth) |
| HMNet-B3 w/ RGB | 346 x 260 | 0.252 | 6.972 | 0.318 | 5.4 | 4.1 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3_fuse_rgb.pth) |
| HMNet-L3 w/ RGB | 346 x 260 | 0.230 | 6.922 | 0.310 | 7.1 | 5.4 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L3_fuse_rgb.pth) |

# Environment (maintained server configuration)

当前工程已适配服务器上的 `/opt/miniconda3/envs/pytorch`：Python 3.12.2、
PyTorch 2.5.0 / torchvision 0.20.0、Torch CUDA 12.1、NumPy 1.26.4、timm 1.0.15。
原始 Python 3.7 / PyTorch 1.12 安装步骤不再作为当前 checkout 的默认方案。
完整检查、安装记录、运行方法和限制见
[环境配置与兼容性记录](docs/03_pytorch环境配置与兼容性记录.md)。
第1–3项补充修复已完成，CPU/CUDA严格警告检查通过。数据、元数据与权重链接见
[下载清单](docs/04_数据集与权重下载清单.md)；尚未下载或开展真实数据冒烟测试。

## Run with the existing environment

```bash
# At repository root; sets PYTHONPATH and defaults MKL_THREADING_LAYER to GNU.
./scripts/hmnet-python -c 'import torch, torch_scatter; print(torch.__version__, torch.version.cuda)'

# Compatibility checks only; no dataset or HMNet forward/training.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./scripts/hmnet-python -B -m pytest tests/test_environment_compatibility.py -q

# From a task directory, use the same launcher:
cd experiments/detection
../../scripts/hmnet-python scripts/train.py --help
```

## Dependencies

`requirements.txt` lists the maintained runtime stack. The shared server environment
already has these packages; no reinstall is needed. To fill missing packages in the
same base environment, preserve its existing versions:

```bash
./scripts/hmnet-python -m pip install --only-binary=:all: -r requirements.txt -c requirements/pytorch-current-constraints.txt
./scripts/hmnet-python scripts/setup_psee_toolbox.py
```

The constraint file is a snapshot of this server's existing Conda/Pip packages,
not a portable recipe for an empty environment. Install matching PyTorch/CUDA
first on another machine; the pinned `torch-scatter` wheel targets Torch 2.5 and
CUDA 12.1. `setup_psee_toolbox.py` installs the pinned official GEN1 tools, including
the license, into the ignored `hmnet/utils/psee_toolbox/` directory. No dataset or
weights are downloaded by these commands. Pytest is a development dependency
(already installed here as 7.4.4).

# Experiments

Please see the instructions on each task page.

- [Semantic Segmentation](https://github.com/hamarh/HMNet_pth/blob/main/experiments/segmentation/)
- [Object Detection](https://github.com/hamarh/HMNet_pth/blob/main/experiments/detection/)
- [Monocular Depth Estimation](https://github.com/hamarh/HMNet_pth/blob/main/experiments/depth/)

# License

The majority of this project is licensed under BSD 3-clause License. However, some code ([psee_evaluator.py](https://github.com/hamarh/HMNet_pth/blob/main/experiments/detection/scripts/psee_evaluator.py), [coco_eval.py](https://github.com/hamarh/HMNet_pth/blob/main/experiments/detection/scripts/coco_eval.py), [det_head_yolox.py](https://github.com/hamarh/HMNet_pth/blob/main/hmnet/models/base/head/task_head/det_head_yolox.py)) is available under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) license.
The pre-trained weights are released under the Creative Commons BY-SA 4.0 License.

# Acknowledgments

This work is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
