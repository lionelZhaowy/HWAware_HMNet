# PEOD 四组公平检测实验

已完成代码实现、四组真实数据冒烟和严格保存恢复验证。全量 DAT 索引已完成（121/121）；共享缓存仍在续建，最终吞吐验收尚未完成，正式训练未启动。当前证据见 RVT 融合工程的 `artifacts/peod/README.md`；不能将冒烟结果当作正式训练结果或部署验收。

## 工程与结构

四工程共同派生自 GEN1 RVT 提交 `7803484`，公共源码一致，由各自 `experiment.toml` 选择模态和事件表示。每组独立训练；B1 分支规格、256 通道投影、Pyramid neck、六类 YOLOX head 相同，单双流总参数量不同，不属于严格等参数量消融。

下表目录均位于 `/home/zhaowenyao24/Conda_prj/Detection_DVS/`。

|工程目录|模态与表示|参数量|
|---|---|---:|
|`HWAware_HMNet_Det_PEOD_RGB_v1.2_T`|真实 RGB，无事件分支|15,403,393|
|`HWAware_HMNet_Det_PEOD_DVS_v1.2_T_RVT`|DVS，RVT20|15,405,841|
|`HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_RVT`|RGB + RVT20，同步 SimpleAdd|20,174,705|
|`HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_Binary`|RGB + Binary2，同步 SimpleAdd|20,172,113|

含 DVS 的三组保留 Stage3/4 的 M=2 记忆：当前与前一个标签步的摘要，FP32 累积和状态、其余 BF16、TBPTT=1。RGB 每步重新编码，不缓存跨帧 RGB 特征；RGB-only 无状态，但采用完全相同的序列采样和翻转规则。损失、检测头和注意力数学沿用已验证实现。

## 数据、时间与坐标

原始数据位于 `/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig`，四组共享其 `preprocessed/hmnet_v12t_240x304_v1/` 下的索引和输入缓存。常规预处理不修改原始 DAT、PNG、CSV、JSON；本次经用户授权的源文件修复单独留有报告和备份。原始文件仍用于来源核验、无缓存时读取和审计，不能删除。

发布的 RGB 为真实彩色 1280×720 图像，采用发布数据已有的事件坐标对齐结果；缩放本身不等于跨相机标定。COCO 文件名与排序后的 RGB 文件对应，CSV 第二列为整数微秒时间戳，作为标签及 RGB 的共同时间锚点，发布时刻下 RGB 年龄为 0。未找到独立曝光结束时间元数据，不能进一步声称曝光结束时刻的严格因果性。

四组使用完全相同的样本清单，保留合法空框和零事件样本。官方 test 保持独立；官方训练数据按条件分别排序，每第 10、20……个完整序列划为验证，避免逐帧切分泄漏。元数据计数为训练 88 序列/54,808 帧、验证 9 序列/4,820 帧、test-challenge 12 序列/6,403 帧、test-normal 12 序列/5,444 帧。验证序列为 010、020、030、043、053、063、073、083、093。这些样本计数已由完整事件索引验收确认。

训练 024 原先有 118 个标签超出事件文件末尾，025 的全部 557 个标签超出末尾，现已用官方 `sequence_024-5.zip` 完整恢复，CRC、已有完整事件前缀和标签覆盖核验通过。测试 normal019 仅删除所有标签之后的 7 字节不完整记录，监督窗口不变，没有补造末尾事件。三份损坏原件及修复哈希均保留。

RGB 采用 PIL 双线性插值，将 1280×720 缩放至 304×171，上补黑色 34 行、下补 35 行，得到 **H=240、W=304**，再进行 ImageNet 标准化。事件先按 `x*19//80`、`y*19//80+34` 映射并取整，再构建目标网格表示。训练框先裁到源图边界，删除无正面积交集的框，保留浮点坐标，乘 0.2375 后加 y 偏移 34。评估保留原始浮点 GT，预测框逆变换到 1280×720、裁边并删除零面积预测。不应用 GEN1 的启动时间过滤或小框过滤。

事件窗口固定为闭区间 `[t-50000,t]`。全文件物理扫描确定包围范围；读取时准确过滤成员并稳定排序，可处理时间戳回退。超过 2^31 的回退会明确报错，不猜测回绕。RVT 沿用冻结的 10 时间 bin、极性优先通道顺序、`count_cutoff=10`、`fastmode=True`，包括 uint8 溢出语义。Binary 直接由原事件构建占用，不能由 RVT 反推。可选压缩缓存须逐值一致，写入受总容量和剩余空间限制。

录制开始后前 50 ms 内的 35 个发布标签在四组中一致保留，仅使用窗口内实际记录的事件；这些帧没有完整的录制前 50 ms 历史。reset gap 取训练源标签间隔 p99 的两倍并向上取整到毫秒，最终值记录在 manifest；M=2 指标签步，不保证固定 100 ms。输入不再额外补齐，卷积向上取整得到 /4 为 60×76、/8 为 30×38、/16 为 15×19、/32 为 8×10。

来源：[官方数据发布](https://github.com/bupt-ai-cz/PEOD)、[论文](https://arxiv.org/html/2511.08140v1)。采用 COCO bbox mAP/AP50/AP75，并报告逐类 AP。论文的分辨率、窗口和训练预算与本实验指定的 240×304、50 ms、100 轮不同，本实验是内部公平对照，不是论文排行榜设置的复现。

## 初始化与共同训练合同

官方 B1 权重 SHA256 为 `bf8798aa03ed2bba21fc8d58f43ca9a92096f3f03bd700e1df1dcde11559f499`。各工程引用持久保存的官方 `pretrained/efficientvit_b1_r224.pth`，不以 GEN1 已训练权重初始化 PEOD。按模块名称稳定设种子，确保对应 RGB/DVS 非 stem 参数和公共 neck/head 逐位一致；Binary 在 canonical20 初始化后适配首层，相当于官方 RGB 通道均值乘 3/2。

共同配置：seed42、100 轮、AdamW（默认 betas/epsilon）、lr2e-4、weight_decay0.01、预热 5 轮且起始系数 0.1、余弦下降至 2e-6、BF16、累积 1、单 GPU、普通 BN、确定性算法、关闭 TF32、不裁剪梯度、prefetch1、pin_memory、持久 workers。按每轮进行验证，继承训练器另含第 1 步验证；eval batch32、分数阈值 0.01、NMS IoU0.65。主表使用末轮权重，best 仅由 val mAP 选择并另表列出，不根据 test 择优。总更新数根据真实训练样本和采样器重新计算。

候选 **batch128/workers2** 已通过融合组短时容量测试，约 19.4 GiB allocated；显存 reserved 随测试阶段变化，以原始报告为准。融合组 workers2/4/8 的小缓存持续测试分别约为 54.2/51.0/46.8 samples/s，均去掉预热后计量 110 步且超过 2 分钟。workers2 是当前候选，不代表全量磁盘吞吐已确认。**共同 batch/workers 尚未最终冻结；四组不得分别调大单模态 batch。**

## 数据准备命令

共享缓存上限现采用32 GiB，磁盘剩余空间保护仍为40 GiB。此前20 GiB总量上限在sequence_094处触发，117个完整缓存已占19.796 GiB；四个剩余序列共2540帧，正在有界续跑。错误信息现会明确区分总量上限与剩余空间保护。

当前已有后台准备与验收队列，不要重复启动同一共享目录的写入任务。需要独立重跑时，在确认旧任务已退出后执行：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_RVT
./scripts/hmnet-python scripts/prepare_peod.py
./scripts/hmnet-python scripts/cache_peod.py --limit-sequences 0 --max-cache-gib 32 --reserve-gib 40
```

## 四组 GPU 正式训练命令

**以下命令采用候选 batch128/workers2，待全量数据和最终配置验收后执行。** 当前 `COMPLETE.json` 已生成，但完整输入缓存及最终性能验收尚未完成。共享缓存和并发吞吐尚未全部验收，建议先按单 GPU 顺序运行；下列 GPU2 是物理卡号，执行前用 `nvidia-smi` 核对空闲状态，需要时仅替换卡号。进程内部设备统一为 `cuda:0`。这些命令只供用户手动执行，本任务未自动启动正式训练。

先在服务器终端设置公共路径：

```bash
export PEOD_BASE=/home/zhaowenyao24/Conda_prj/Detection_DVS
export PEOD_DATA=/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_240x304_v1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

RGB-only：

```bash
(cd "$PEOD_BASE/HWAware_HMNet_Det_PEOD_RGB_v1.2_T" && \
 CUDA_VISIBLE_DEVICES=2 ./scripts/hmnet-python scripts/task_temporal.py train \
 --modality rgb --data-root "$PEOD_DATA" --device cuda:0 \
 --epochs 100 --batch 128 --workers 2 --eval-batch 32 --seed 42 --precision bf16 \
 --output logs/detection/peod_rgb_none)
```

DVS-only RVT：

```bash
(cd "$PEOD_BASE/HWAware_HMNet_Det_PEOD_DVS_v1.2_T_RVT" && \
 CUDA_VISIBLE_DEVICES=2 ./scripts/hmnet-python scripts/task_temporal.py train \
 --modality dvs --representation rvt_histogram --data-root "$PEOD_DATA" --device cuda:0 \
 --epochs 100 --batch 128 --workers 2 --eval-batch 32 --seed 42 --precision bf16 \
 --output logs/detection/peod_dvs_rvt_histogram)
```

RGB-DVS RVT：

```bash
(cd "$PEOD_BASE/HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_RVT" && \
 CUDA_VISIBLE_DEVICES=2 ./scripts/hmnet-python scripts/task_temporal.py train \
 --modality rgbdvs --representation rvt_histogram --data-root "$PEOD_DATA" --device cuda:0 \
 --epochs 100 --batch 128 --workers 2 --eval-batch 32 --seed 42 --precision bf16 \
 --output logs/detection/peod_rgbdvs_rvt_histogram)
```

RGB-DVS Binary：

```bash
(cd "$PEOD_BASE/HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_Binary" && \
 CUDA_VISIBLE_DEVICES=2 ./scripts/hmnet-python scripts/task_temporal.py train \
 --modality rgbdvs --representation polarity_binary --data-root "$PEOD_DATA" --device cuda:0 \
 --epochs 100 --batch 128 --workers 2 --eval-batch 32 --seed 42 --precision bf16 \
 --output logs/detection/peod_rgbdvs_polarity_binary)
```

四条命令各自在子 shell 中切换到独立工程，输出位于该工程的 `logs/detection/`。需要恢复时，保持全部训练参数及输出目录一致，在对应命令末尾加 `--resume <该输出目录>/checkpoint.pth`。例如融合 RVT 使用 `--resume logs/detection/peod_rgbdvs_rvt_histogram/checkpoint.pth`。检查点包含模型、优化器、RNG、流游标、记忆状态和完整训练合同；模态、表示、数据、几何、初始化或预算不一致会拒绝恢复。不能把诊断检查点用于正式训练。

需要手动冒烟时，将 `--output` 改为新的 `artifacts/peod/smoke_new`，并加 `--stop-after 4`，避免占用正式训练输出。

## 评估、导出和可视化

以下示例在融合 RVT 工程执行；其他组使用对应工程及其输出目录。`PEOD_DATA` 沿用前面的公共路径。

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_RVT
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 ./scripts/hmnet-python scripts/task_temporal.py eval --split test \
 --data-root "$PEOD_DATA" --checkpoint logs/detection/peod_rgbdvs_rvt_histogram/checkpoint.pth \
 --eval-batch 32 --workers 2 --precision bf16 --device cuda:0 \
 --output artifacts/evaluation/peod_test
./scripts/hmnet-python scripts/export_peod.py \
 --checkpoint logs/detection/peod_rgbdvs_rvt_histogram/checkpoint.pth \
 --data-root "$PEOD_DATA" --output artifacts/onnx/final
# 导出四尺度骨干时另加 --backbone-only。
```

评估保存完整的原始坐标预测 JSONL，计分和画框复用同一份预测。test normal/challenge 由同一 dump 分组计分。对比前固定四组检查点；下面四个尖括号路径需替换为实际完整预测文件：

```bash
./scripts/hmnet-python scripts/render_peod_comparison.py \
 --data-root "$PEOD_DATA" --split test \
 --dumps <rgb.jsonl> <dvs.jsonl> <fusion_rvt.jsonl> <fusion_binary.jsonl> \
 --output artifacts/comparison --videos
```

静态图选择 8 个均匀分布的观测；视频每秒播放 10 个观测并显示真实采集时间戳，不代表原始时间播放或实时部署。显示阈值 0.25 仅影响画框，AP 使用已保存的完整后处理预测。

## 验证范围与 ONNX 限制

四组均通过连续 4 步与分段 2+2 步的真实 GPU 严格恢复比较，覆盖参数、BN、优化器、RNG、游标和 DVS 状态。边界测试覆盖空事件、单事件、双极性、闭区间端点、时间回退、缩放后像素碰撞、越界拒绝和 RVT 溢出。RGB-only 无伪造事件张量或时序状态，公共初始化已逐项核对。

四份整网和四份骨干 ONNX 位于各工程 `artifacts/peod/onnx/`，权重来源为 4 步冒烟。结构检查全部通过，**严格数值检查均有失败，未通过部署验收**。输入为 `[1,C,240,304]`，整网 NMS 前输出为 `[1,1505,11]`。含事件的模型显式输入/输出 7 个 FP32 状态：3×`[1,16,17,16]`、4×`[1,32,17,16]`。报告分别维护 PyTorch/ORT 状态反馈轨迹，容差固定 atol1e-3、rtol1e-4。全零测试指标准化后输入全零，不等于原始黑色 RGB。

尚无完整 PEOD 正式训练或最终任务精度结果；GPU 短测和冒烟权重的 COCO 输出不能作为精度证据。

## 独立stage2追加100轮

四组共同追加100轮，batch128/workers2/eval batch32/BF16/seed42不变，峰值LR5e-5，2轮从2e-6预热（factor0.04），再余弦下降到2e-6。使用各自stage1完成100轮的末检查点，恢复模型、优化器、RNG、游标和DVS状态，仅重启阶段日程与best记录。低训练loss不保证val AP提高，该设置是统一续训实验而非已验证的最优参数。

首次stage2需要独立空目录，禁止覆盖stage1或写入其子目录，未完成的父阶段会拒绝启动。请等待每组stage1自然结束，再在独立终端执行：

### GPU2：RGB-only

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Det_PEOD_RGB_v1.2_T
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
./scripts/hmnet-python scripts/task_temporal.py train \
  --modality rgb --representation rvt_histogram \
  --data-root /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_240x304_v1 \
  --device cuda:0 --epochs 100 --batch 128 --workers 2 --eval-batch 32 \
  --seed 42 --precision bf16 \
  --learning-rate 5e-5 --min-learning-rate 2e-6 \
  --warmup-epochs 2 --warmup-start-factor 0.04 \
  --start-new-stage --resume logs/detection/peod_rgb_none/checkpoint.pth \
  --output logs/detection/peod_rgb_none_stage2_100ep
```

### GPU3：DVS-only RVT

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Det_PEOD_DVS_v1.2_T_RVT
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
./scripts/hmnet-python scripts/task_temporal.py train \
  --modality dvs --representation rvt_histogram \
  --data-root /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_240x304_v1 \
  --device cuda:0 --epochs 100 --batch 128 --workers 2 --eval-batch 32 \
  --seed 42 --precision bf16 \
  --learning-rate 5e-5 --min-learning-rate 2e-6 \
  --warmup-epochs 2 --warmup-start-factor 0.04 \
  --start-new-stage --resume logs/detection/peod_dvs_rvt_histogram/checkpoint.pth \
  --output logs/detection/peod_dvs_rvt_histogram_stage2_100ep
```

### GPU4：RGB-DVS RVT

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_RVT
CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
./scripts/hmnet-python scripts/task_temporal.py train \
  --modality rgbdvs --representation rvt_histogram \
  --data-root /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_240x304_v1 \
  --device cuda:0 --epochs 100 --batch 128 --workers 2 --eval-batch 32 \
  --seed 42 --precision bf16 \
  --learning-rate 5e-5 --min-learning-rate 2e-6 \
  --warmup-epochs 2 --warmup-start-factor 0.04 \
  --start-new-stage --resume logs/detection/peod_rgbdvs_rvt_histogram/checkpoint.pth \
  --output logs/detection/peod_rgbdvs_rvt_histogram_stage2_100ep
```

### GPU5：RGB-DVS Binary

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_Binary
CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
./scripts/hmnet-python scripts/task_temporal.py train \
  --modality rgbdvs --representation polarity_binary \
  --data-root /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_240x304_v1 \
  --device cuda:0 --epochs 100 --batch 128 --workers 2 --eval-batch 32 \
  --seed 42 --precision bf16 \
  --learning-rate 5e-5 --min-learning-rate 2e-6 \
  --warmup-epochs 2 --warmup-start-factor 0.04 \
  --start-new-stage --resume logs/detection/peod_rgbdvs_polarity_binary/checkpoint.pth \
  --output logs/detection/peod_rgbdvs_polarity_binary_stage2_100ep
```


再次恢复stage2时，重复同组命令，保持全部参数和输出目录，仅把 `--resume` 改为stage2输出目录内的 `checkpoint.pth`。可以保留 `--start-new-stage`；已有stage_parent时按普通恢复处理，不再清零局部step或重启LR。两阶段的TensorBoard、metrics、settings、checkpoint和best均独立保存；stage2累计轮数需加父阶段100轮。
