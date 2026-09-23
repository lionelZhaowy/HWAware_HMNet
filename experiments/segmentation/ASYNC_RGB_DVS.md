# v1.2_T_1 / v1.2_T_2 异步 RGB–DVS 分割

父版本：`seg_rgbdvs_640x440_v1.2_T` / `76f1a8c`。两个新工程独立运行，共同源码一致；仅 `async_variant.py` 与版本说明不同。

|版本/分支后缀|方案|事件窗口|bin/通道|输出时间网格|
|---|---|---|---|---|
|`v1.2_T_1`|B|50ms|10 / 20|约25ms一步，窗口重叠|
|`v1.2_T_2`|C|25ms|5 / 10|相同|

保留 SimpleAdd、DVS Stage3/4 七个 LiteMLA 的 M=2 FP32摘要，其余BF16。新RGB只更新四尺度特征缓存；每个事件步都输出分割。40Hz指请求的时间网格，不是未经测量的实时吞吐承诺。部署调用见下文。

## 运行前与数据准备

在服务器现有环境运行，无需升级Conda。先检查 `nvidia-smi`，下文GPU0/1只是两个终端的选择示例。正式训练统一单GPU、batch32、accumulation1；不支持异步局部RGB更新配合SyncBN/DDP，避免部分流无RGB导致集体通信不匹配。

新数据目录为 `/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/async_v1/`。B/C共同RGB资产只保存一份；旧 `dsec_b1` 不修改。若完整的两个manifest已经存在，直接复用，不重复准备。换服务器/首次准备时仅执行一次：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_1
./scripts/hmnet-python scripts/prepare_dsec_async.py
```

可选 `--workers 8` 按独立序列并行准备（默认4）；只在全部序列完成后发布完整manifest。脚本从原始事件生成两个版本，继承父缓存的train/dev成员，保留真实GT时间并插入中间事件步；非正常时间间隔明确重置。窗口为 `(t-W,t]`，RGB只取过去数据。无传输延迟标注，按零传输延迟处理。直方图沿用冻结RVT：极性优先、首末事件分桶、uint8 fastmode及cutoff10。

可用 `--source`、`--base-cache`、`--assets`、`--output-root` 更换根目录。`--sequences ... --per-sequence 96` 仅供独立小样本诊断，**不能以此启动正式150轮比较**。未完成缓存无manifest；成功写完才发布manifest。完整缓存禁止覆盖。

官方预训练放每个工程 `pretrained/efficientvit_b1_r224.pth`。两组共享层初始化一致，C仅在完整初始化之后将事件首层适配成10通道；不从训练完的B权重初始化C。

## 阶段一：真实GT训练

两个终端可同时执行。默认150epoch、batch32、workers8、seed42、AdamW 2e-4、5轮预热、余弦到2e-6。

终端B：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_1
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/train_async.py --single --stage 1
```

终端C：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_2
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python scripts/train_async.py --single --stage 1
```

训练单元是一个GT终点及此前事件步，通常TBPTT=2；batch32指32个GT终点。GT每轮只出现一次，约一半流使用落后一张RGB的输入。B/C监督次数和更新预算一致；相对同步父版多了事件前向与不同BN统计，不能称纯粹只改变异步缓存的一项消融。中间步通过状态链获得梯度，GT时刻的旧RGB融合获得直接监督。GT-only中间步不运行无损失Head来改变BN。RGB缓存只在一个片段内保留梯度，每次参数更新后重建；DVS状态在更新边界detach。

输出分别为 `logs/segmentation/efficientvit_b1_v1.2_T_1_stage1/` 与 `...v1.2_T_2_stage1/`。默认每轮验证、保存last/best，TensorBoard在相应 `tensorboard/`。

B/C续训：在各自工程根目录运行，`--resume`不带路径时自动加载本版本、所选阶段输出目录中的`checkpoint.pth`：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/train_async.py --single --stage 1 --resume
```

自定义训练输出目录需指定`--output`；也保留显式检查点路径：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/train_async.py --single --stage 1 --output /path/to/run --resume
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/train_async.py --single --stage 1 \
  --resume logs/segmentation/efficientvit_b1_v1.2_T_1_stage1/checkpoint.pth
```

显式路径示例为B，C改成`...v1.2_T_2_stage1/checkpoint.pth`。自动恢复只选择最终输出目录的`checkpoint.pth`，不搜索其他实验、不回退到best；缺失时直接报错，不开始新训练。`--resume`与`--init-from`互斥。

换物理GPU只需改变`CUDA_VISIBLE_DEVICES`，保持原数据、seed、精度、阶段和训练预算。`CUDA_VISIBLE_DEVICES=0`让程序的逻辑`cuda:0`对应物理GPU0；两个独立训练进程可各自在逻辑`cuda:0`运行并共享物理卡。换卡前结束该实验的旧训练进程，避免两个进程写同一输出目录；不需要停止同卡的另一实验。恢复点是最后一次已保存的检查点，当前训练器没有Ctrl+C自动保存功能，未保存的更新会重做。

将C移到GPU0（与B合卡）：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_2
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/train_async.py --single --stage 1 --resume
```

合卡训练保持各自batch32和原超参数；单卡运行的显存实测不能保证两进程所有阶段同时达到峰值时一定足够，启动后观察`nvidia-smi`和日志。

`--resume`恢复同阶段的优化器、LR日程、随机状态、连续流游标和DVS状态；RGB来源可重建。不能跨B/C、数据manifest、batch、seed、精度或阶段恢复。`--stop-after N`仅限诊断：在第N次更新保存退出，保留原总预算。

## 阶段二前置：审核并生成共享伪标签

必须等待两组阶段一有效权重；不可拿本次冒烟权重生成正式监督。以下整段在B工程终端执行，先设路径：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_1
ASYNC_B=/home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_1
ASYNC_C=/home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_2
ASYNC_DATA=/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/async_v1
ASYNC_B_CKPT="$ASYNC_B/logs/segmentation/efficientvit_b1_v1.2_T_1_stage1/best_checkpoint.pth"
ASYNC_C_CKPT="$ASYNC_C/logs/segmentation/efficientvit_b1_v1.2_T_2_stage1/best_checkpoint.pth"

CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/pseudo_dsec_async.py audit \
  --b-checkpoint "$ASYNC_B_CKPT" --c-checkpoint "$ASYNC_C_CKPT" \
  --b-data "$ASYNC_DATA/dsec_async_B" --c-data "$ASYNC_DATA/dsec_async_C" \
  --confidence 0.95 --min-precision 0.90 --min-coverage 0.10 \
  --min-class-precision 0.70 --min-class-pixels 1000 \
  --output artifacts/async_pseudo_audit
```

审核在dev真实GT时刻隐藏标签、输入旧RGB，统计精度、覆盖、各类及边界；dev是校准集，不是最终独立测试。阈值是本实验的预设质量门槛，不是文献保证。审核失败会非零退出并保留报告；不自动放宽门槛。不直接插值相邻分割mask，不使用检测框/track监督。

仅审核通过后执行：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/pseudo_dsec_async.py generate \
  --b-checkpoint "$ASYNC_B_CKPT" --c-checkpoint "$ASYNC_C_CKPT" \
  --b-data "$ASYNC_DATA/dsec_async_B" --c-data "$ASYNC_DATA/dsec_async_C" \
  --confidence 0.95 --audit artifacts/async_pseudo_audit/audit.json \
  --output "$ASYNC_DATA/shared_pseudo_stage1"
```

教师采用因果双模型同类一致＋置信度筛选，生成同一份train中间时刻标签，两组学生共用。教师、audit、网格和伪标签manifest有hash溯源；质量不合格像素为255。教师的一致错误仍可能保留。低频GT审核不证明真正25ms中间时刻精度，最终仍需要对应标注。

## 阶段二：额外微调

下列给出**额外20轮**的共同预算，需在上述审核合格后固定使用；不计入阶段一150轮。学习率2e-5、1轮预热、最低2e-6；pseudo权重0.2，在5轮内渐增。GT/pseudo分别按有效像素归一化，空pseudo掩码返回零损失。

终端B：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_1
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/train_async.py --single --stage 2 --epochs 20 \
  --init-from logs/segmentation/efficientvit_b1_v1.2_T_1_stage1/best_checkpoint.pth \
  --pseudo-root /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/async_v1/shared_pseudo_stage1 \
  --pseudo-weight 0.2
```

终端C：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_2
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python scripts/train_async.py --single --stage 2 --epochs 20 \
  --init-from logs/segmentation/efficientvit_b1_v1.2_T_2_stage1/best_checkpoint.pth \
  --pseudo-root /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/async_v1/shared_pseudo_stage1 \
  --pseudo-weight 0.2
```

`--init-from`仅加载对应阶段一权重，重新建立优化器、LR及运行状态。输出为各版本 `..._stage2/`。阶段二续训保持所有参数一致，删除 `--init-from`，改用无路径 `--resume`（自动选择stage2输出）或 `--resume ..._stage2/checkpoint.pth`；保留 `--stage 2 --epochs 20 --pseudo-root ... --pseudo-weight 0.2`。

## 评估、旧RGB诊断与导出

B例子（C改工程、检查点后缀以及 `dsec_async_C`；阶段二改检查点路径）：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T_1
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/evaluate_dsec_async.py \
  --checkpoint logs/segmentation/efficientvit_b1_v1.2_T_1_stage1/best_checkpoint.pth \
  --data-root /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/async_v1/dsec_async_B \
  --output artifacts/evaluation/stage1_dev
```

旧RGB诊断分别另加 `--rgb-delay-frames 1`（保持频率但延迟一帧）或 `--rgb-keep-every 2`（RGB20→10Hz），并改成独立 `--output`。输出每个DVS步的时间记录、GT-only mIoU、RGB年龄分组、RGB编码次数及吞吐（包括冷启动，另列单步延迟中位数/p95）。`--save-predictions`可保存每步预测mask；无GT时不计算伪mIoU。

```bash
./scripts/hmnet-python scripts/export_async_onnx.py \
  --checkpoint logs/segmentation/efficientvit_b1_v1.2_T_1_stage1/best_checkpoint.pth \
  --data-root /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/async_v1/dsec_async_B \
  --output artifacts/onnx/stage1_best
```

导出固定batch1/440×640的 `rgb_update.onnx`、`event_prediction.onnx`、`event_backbone.onnx`及简化图。RGB更新输出四尺度256通道缓存；事件图接收20/10通道直方图、4个RGB缓存和7个FP32摘要，输出11类logits及7个新摘要。无新RGB时不调用RGB图；序列重置时清零全部缓存和状态。

原始/简化图分别保持独立状态轨迹。使用原阈值atol1e-3/rtol1e-4，包含真实序列、全零、空事件和reset；失败仍保留全部图和 `report.json`，非零退出。结构通过不代表数值或部署验收通过。当前导出只是FP32诊断，不是BF16部署。

Python流式入口为 `hmnet.models.async_frame.AsyncPredictor(model.eval())`：`step(events, meta, rgb=None)`；每个流独立实例。meta含真实时间、序列和RGB来源；无新RGB传None。可处理非整数RGB/DVS频率比，例如30Hz/200Hz；不提供相机硬件接入或调度服务。

## 复用来源与验证范围

[算法源码来源](ASYNC_REFERENCES.md)。本地诊断产物在 `artifacts/async_implementation/`，不提交Git。正式训练及高频精度待用户运行；阶段二合成fixture冒烟只验证软件路径，正式CLI会拒绝该fixture。
