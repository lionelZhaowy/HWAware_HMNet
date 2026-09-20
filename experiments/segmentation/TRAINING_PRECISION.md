# RGB-DVS 融合实验：BF16 单卡 / FP32 多卡

v2.0 和 v2.2 共用相同的训练入口、训练器和DDP辅助代码。模型分别保留前置IRB＋Concat更新和后置MBConv结构；两者都有入口 `P=R⊙D; R_enh=R+P; D_enh=D+P`。通过命令选择精度，不再复制出额外代码工程。

## 可执行命令

以下是四个独立实验的启动命令。**选择需要的方式启动，不要把使用同一GPU的命令同时执行。** 先用 `nvidia-smi` 核对空闲卡。命令不带 `--resume`，从官方B1预训练初始化；不复用main/v2.1的任务检查点。

### v2.0：BF16 单卡

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v2.0
CUDA_VISIBLE_DEVICES=2 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py \
  --single --precision bf16 --seed 42 \
  --output logs/segmentation/efficientvit_b1_cross_v20_bf16
```

### v2.2：BF16 单卡

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v2.2
CUDA_VISIBLE_DEVICES=3 ./scripts/hmnet-python experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py \
  --single --precision bf16 --seed 42 \
  --output logs/segmentation/efficientvit_b1_cross_v22_bf16
```

### v2.0：FP32 双卡

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v2.0
CUDA_VISIBLE_DEVICES=2,3 ./scripts/hmnet-python -m torch.distributed.run \
  --standalone --nproc_per_node=2 experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py \
  --distributed --precision fp32 --seed 42 \
  --output logs/segmentation/efficientvit_b1_cross_v20_fp32_ddp2
```

### v2.2：FP32 双卡

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v2.2
CUDA_VISIBLE_DEVICES=4,5 ./scripts/hmnet-python -m torch.distributed.run \
  --standalone --nproc_per_node=2 experiments/segmentation/scripts/train.py \
  experiments/segmentation/config/efficientvit_b1.py \
  --distributed --precision fp32 --seed 42 \
  --output logs/segmentation/efficientvit_b1_cross_v22_fp32_ddp2
```

不需要再添加 `--amp`。历史 `--amp` 单独使用仍代表FP16；本次新增乘加在FP16真实输入上发生前向溢出，**不使用旧的FP16命令**。BF16已经通过验证，并不是此前溢出的精度。

## 实验条件和多卡语义

- 两种模式都保持150epoch、全局batch32、accumulation1、总workers8、全局eval batch32、seed42、AdamW、原学习率和数据增强。多卡不按GPU数放大学习率。
- 双卡各16个样本、各4个worker。每轮7295样本仍是228批，150轮34200次更新；最后31个样本分为16＋15，没有补样或丢弃。全局shuffle顺序与原单卡DataLoader一致。
- DDP使用一进程一卡和SyncBatchNorm，BN统计覆盖全局batch；融合激活重计算时保留BN临时缓冲，避免反向再次更新持久统计。
- 主/辅助损失都是全分辨率无类别权重的mean CE。各rank按有效像素数加权后再做DDP梯度平均；不能直接将不等有效像素的局部均值平均。
- 验证仍用FP32，各rank处理不重复的样本，最终相加混淆矩阵；只由rank0写日志、TensorBoard和检查点。保存的模型键不带 `module.`，普通单卡FP32评估可严格加载。
- `--precision fp32` 关闭autocast及TF32；`--precision bf16` 使用BF16 autocast，权重和AdamW状态仍为FP32，注意力归一化沿用已有FP32计算。BF16不启用GradScaler；FP32/BF16遇到非有限loss或梯度直接报错。
- 多卡当前支持此分割CE训练。若任一rank没有有效标签而其他rank有标签，显式报错，避免SyncBN死锁；全局无监督batch共同跳过。末批样本数须不小于卡数，全局batch须能整除卡数。DSEC默认设置满足这些条件。

## 保存、续训与评估

保留原命令中的精度、卡数、batch、seed、完整训练日程和输出路径，追加：

```bash
--resume <对应输出目录>/checkpoint.pth
```

例如v2.0 BF16追加 `--resume logs/segmentation/efficientvit_b1_cross_v20_bf16/checkpoint.pth`。

检查点保存模型、优化器、精度、world_size、SyncBN、数据游标和每个rank的Python/NumPy/PyTorch/CUDA RNG。禁止将BF16检查点当作FP32实验续训，或中途切换卡数/结构；旧训练器契约缺少这些字段时也拒绝作为本实验续训。四组输出独立，不使用 `--overwrite` 覆盖其他实验。

评估时使用对应新实验权重，默认FP32，不加 `--fp16`：

```bash
./scripts/hmnet-python experiments/segmentation/scripts/test.py \
  experiments/segmentation/config/efficientvit_b1.py \
  --pretrained <对应输出目录>/best_checkpoint.pth \
  --output <对应输出目录>/evaluation
```

`--stop-after N` 仅用于短时诊断：在第N次更新保存并退出，完整LR预算保持不变。正式训练不要添加。它不是新的epoch预算，续训仍需同一契约。

## BF16 与 FP32 的最终精度

两者**不保证得到相同权重或相同mIoU**。BF16指数范围接近FP32，因此本次避免了FP16的范围溢出；但BF16尾数只有7位，FP32为23位，BF16舍入误差更大。autocast让合适的算子采用低精度，并保留部分计算为FP32，不能理解成整个训练都用16位。

实际精度差异取决于模型、数据和优化过程；不能仅凭两步loss接近就断言最终无损。本工程尚无两种模式完整150epoch的同协议结果。单卡与DDP的归约顺序、每rank dropout随机流也会产生差异，即使seed相同也不保证逐位一致。比较模型结构时应让v2.0、v2.2使用同一种精度/卡数；研究精度影响时最好固定卡数（此训练器也支持BF16 DDP），并对多个seed报告结果。

参考：[PyTorch混合精度说明](https://docs.pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)、[数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。

## 验证范围

两个新版本均验证真实440×640输入的BF16单卡和FP32双卡更新，包含完整batch32、末批31、FP32开发样本评估和保存。RTX4090 24GiB首次更新测得BF16单卡峰值allocated约18.84GiB，FP32双卡每卡约22.19GiB；续训冒烟峰值分别约18.96GiB、22.31GiB；两者FP32单卡batch32均OOM。显存是短时实测值，不是任意未来改动的保证。

验证证据及可复运行脚本位于各工程 `logs/segmentation/precision_ddp_validation/`，仅本地保留。正式150epoch训练、最终精度比较及新结构ONNX部署验收尚未执行。旧v2.1全零输入ONNX超差结论仍保留。
