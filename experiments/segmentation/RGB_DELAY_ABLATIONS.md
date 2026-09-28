# RGB delay augmentation ablations

这两组实验补齐训练时序 × RGB 延迟增强的缺失组合。只做 Stage1；不使用伪标签，不继承父版已训练的分割权重。正式训练由用户手动启动。

|工程|父版|训练步长 / TBPTT|额外延迟概率|默认输出|
|---|---|---|---:|---|
|v1.2_T_delay1|v1.2_T / 76f1a8c|50ms / 1|0.5|logs/segmentation/efficientvit_b1_add_v12_T_delay1_bf16|
|v1.2_T_1_wo_delay1|异步B v1.2_T_1 / 982b763|25ms / 2|0|logs/segmentation/efficientvit_b1_v1.2_T_1_wo_delay1_stage1|

两组均保留父版50ms/10bin/20通道RVT、SimpleAdd、DVS LiteMLA M=2、FP32摘要、BF16训练、seed42、batch32、accumulation1、workers8、150epoch（34200更新），同一官方EfficientViT预训练。各自父版的训练器、采样、loss、学习率和验证规则保留。

`delay1` 工程名表示启用与旧B相同的训练增强，不是把全部训练样本都延迟：按seed/epoch/lane/sequence固定Bernoulli(0.5)，选中流的RGB落后一张。两新工程共享逐字节一致的 `hmnet/utils/rgb_delay.py`，hash策略与旧B一致。`wo_delay1` 仍然在正常中间25ms时刻使用旧RGB；只取消额外延迟，不取消异步读出或状态更新。

## 数据与冷启动

无需重新预处理，也不复制已有训练日志：

- 同步组事件、GT和正常RGB仍来自 `DSEC_Semantic/preprocessed/dsec_b1`，保持原事件窗口边界；延迟RGB复用 `preprocessed/async_v1/dsec_async_B/manifest.json` 中的已配准RGB资产。校验父manifest SHA、全量GT成员及当前RGB时间对应，不将事件/GT一起错移。
- 异步组仍使用 `preprocessed/async_v1/dsec_async_B`；25ms时间网格、两步训练均保留。
- 延迟后序列开始没有历史RGB时，同步组跳过该样本的RGB编码，使用零RGB特征；不编码一张假黑图，也不取未来图像。此路径不增加参数；正常输入前向与父版一致。
- 同步增强概率、策略和RGB目录manifest SHA进入新训练契约；异步概率已由原训练契约记录。父版或其他增强概率的checkpoint不能当resume使用。改路径/概率应另建实验，不能绕过契约。
- normal-dev不施加训练延迟增强。父同步dev沿用FP32，父异步dev沿用BF16逐步预测；这是既有差异。最终四组对照须统一推理精度/调度后评估，不能直接混用训练dev分数。

## 首次训练

先用 `nvidia-smi` 确认GPU空闲。以下GPU编号仅示例，可换成空闲卡；正式batch不随显存调整。两个终端分别运行：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_delay1
CUDA_VISIBLE_DEVICES=2 ./scripts/hmnet-python experiments/segmentation/scripts/train.py experiments/segmentation/config/efficientvit_b1.py --single --precision bf16 --seed 42
```

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_1_wo_delay1
CUDA_VISIBLE_DEVICES=3 ./scripts/hmnet-python scripts/train_async.py --single --stage 1 --precision bf16 --seed 42
```

## 同一实验断点恢复

迁移到另一张GPU只改变 `CUDA_VISIBLE_DEVICES`。以下两条必须在各自工程根目录执行；首次训练不要带resume。

```bash
# 同步delay1：父训练器要求显式checkpoint路径
CUDA_VISIBLE_DEVICES=2 ./scripts/hmnet-python experiments/segmentation/scripts/train.py experiments/segmentation/config/efficientvit_b1.py --single --precision bf16 --seed 42 --resume logs/segmentation/efficientvit_b1_add_v12_T_delay1_bf16/checkpoint.pth
```

```bash
# 异步无额外delay：自动选择本工程Stage1的last
CUDA_VISIBLE_DEVICES=3 ./scripts/hmnet-python scripts/train_async.py --single --stage 1 --precision bf16 --seed 42 --resume
```

不要用 `--overwrite` 覆盖正式输出；不要用父版best启动resume。验证产生的3步权重在 `artifacts/delay_ablation/`，不用于正式训练或精度比较。

## 评估口径与限制

每组使用normal-dev选择best，同时保留last。训练后与原Sync及原Async B组成四组，统一BF16、normal/delay1/keep2和40Hz推理调度，仅在真实20Hz标签时刻累计mIoU。无中间真值，不能声称测得完整40Hz精度。

对照首先回答“各自训练方式内，增加/去掉额外RGB延迟增强的影响”。原同步与异步还存在TBPTT、采样、窗口边界及BN计算等差异；单seed、旧B曾提前结束，不应声称是所有因素都严格匹配的因果实验。新两组按完整150轮预算执行，不根据test调整训练。

本次不启动正式训练、不重建缓存、不修改原Sync/B/C。实现、冒烟、恢复、ONNX及分支证据见本地 `artifacts/delay_ablation/README.md`。
