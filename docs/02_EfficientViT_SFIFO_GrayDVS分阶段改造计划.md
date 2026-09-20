# EfficientViT-B1 首版实现与后续实验边界

> 更新：2026-09-17。本页取代原先的三层记忆适配候选方案，记录本轮**已实现、已运行**的帧式路线。原 HMNet 网络结构和 B3 ONNX 说明继续保留在 [05 文档](05_HMNet三任务B3详细网络结构与ONNX导出.md)。

## 方案 C 更新（2026-09-20）

当前工程主配置为 `experiments/segmentation/config/efficientvit_b1.py`：在官方B1四个阶段原生32/64/128/256通道上使用带除法归一化的双向LiteMLA交互，两路交互输出分别输入下一阶段，融合输出投影为256通道送入Pyramid。训练设置继承150轮相加基线，输出独立保存到 `logs/segmentation/efficientvit_b1_cross/`。

具体结构、数据准备、训练、恢复、评估和ONNX命令见[分割README方案C](../experiments/segmentation/README.md#方案-c骨干阶段内双向-litemla-交互)。本次不实现B/D方案、HWAware、S-FIFO或时序模块；当前工程已移除旧相加实现和cooldown配置，检测/深度单模态及其他备份工程保留。下面各节仅记录历史首版基线，当前结构以本节及分割README为准。

## 1. 已实现的网络

```text
50 ms 原始事件 → RVT StackedHistogram [B,20,H,W] → 官方 B1 事件分支
                                                        ↓ 四尺度 Conv1×1 + BN
DSEC 左 RGB → 官方几何映射 → ImageNet 标准化 → 独立官方 B1 RGB 分支
                                                        ↓ 四尺度 Conv1×1 + BN
                              单分支 ReLU，或两分支相加后 ReLU
                                               ↓
                              原 Pyramid：bottom-up + top-down
                                               ↓
                          原 YOLOX / SegHead + AuxHead / DepthRegHead
```

三个任务使用同一个 [EfficientViTB1 实现](../hmnet/models/base/backbone/efficientvit_b1.py)，分别实例化和训练；没有共享训练参数。检测和深度不创建 RGB 分支。分割两分支参数独立，两侧梯度均已验证。

| 输出 | 原生通道 | 投影通道 | DSEC 440×640 空间尺寸 |
|---|---:|---:|---|
| stage1 /4 | 32 | 256 | 110×160 |
| stage2 /8 | 64 | 256 | 55×80 |
| stage3 /16 | 128 | 256 | 28×40 |
| stage4 /32 | 256 | 256 | 14×20 |

核心接口为 `forward(event_hist, rgb=None) -> (f4,f8,f16,f32)`。不存在历史状态输入、warmup、TBPTT、三层设备调度或 RGB 特征缓存。检测在四尺度 Pyramid 融合后选择 `/8,/16,/32`，YOLOX strides 同步设为 `[8,16,32]`。分割、深度主输出使用 `/4`；分割辅助头使用 Pyramid 之前融合后的 `/4` 特征，主/辅助 CE 权重为 1.0/0.4。

官方 B1 保留 LiteMLA、HardSwish、BN，以及注意力归一化中的除法。本轮没有施加 HWAware 算子替换。

### 初始化

默认权重：[作者发布的 ImageNet B1-r224](https://huggingface.co/han-cai/efficientvit/resolve/main/efficientvit_b1_r224.pt)，本地为 `pretrained/efficientvit_b1_r224.pth`。RGB 骨干严格加载；DVS 首层为 `RGB_weight.mean(dim=1).repeat(20) × 3/20`。只删除分类 `head.*`，新增投影、Pyramid 和任务头独立初始化。骨干权重在任务通用初始化之后加载，其余骨干参数严格检查，没有宽松的 missing-key 加载。

## 2. 数据表示与配准

### RVT Histogram

[薄适配器](../hmnet/models/base/event_repr/rvt_histogram.py) 调用仓库内冻结的上游 `StackedHistogram`，没有重写计数或时间分桶算法。原始文件 SHA256 为 `fdae1cd82c1f6b69a85d2ceac3d59ced9b625dbd28c45a4da1db9b638faaa759`；来源与许可证位于其 `vendor/` 目录。

- 时间戳为整数微秒，极性 `{0,1}`；窗口包含 `[target−50000,target]` 两端事件。
- 10 bins，20 通道，**先 p=0 的 10 个时间 bin，再 p=1 的 10 个 bin**。
- 时间分桶以窗口内实际首末事件归一化，沿用 RVT 行为。空窗口全零，同时间戳进入第一个 bin。
- `count_cutoff=10, fastmode=True`。uint8 先累加再截断，因此同位置/bin 的 256 次累加回绕到 0，257 次到 1；测试已覆盖。不得把它静默改成饱和累加。
- 网络输入 float32 计数，不做 RGB 标准化。DSEC 先筛选顶部有标签区域的事件，再调用 RVT。

[GEN1/Eventscape/MVSEC 适配](../hmnet/dataset/frame_datasets.py) 继承现有 EventFrame，复用标签读取、框过滤和稠密批次约定；仅修改独立监督窗口选择，避免旧毫秒采样引入目标之后的事件。MVSEC 秒时间戳先转换、舍入至整数微秒。GEN1 极性先转有符号整数，避免 uint8 的 `−1` 回绕。

### DSEC RGB

[缓存准备入口](../scripts/prepare_dsec_b1.py) 使用 [官方 dsec-det remapping](https://github.com/uzh-rpg/dsec-det/blob/master/src/dsec_det/remapping.py) 的冻结副本，原始 SHA256 为 `4d33159541d0453f8227da2325012cbb8e3ee7707c7d35af10b1ed101512d635`。

1. 输入使用 DSEC 官方左 RGB 的 `images/left/rectified`，不读取旧 HMNet resize/crop RGB 缓存。
2. 用本序列 `cam_to_cam.yaml` 和左事件 `rectify_map.h5`，将 RGB 采样至 **640×480 原始、未去畸变的左事件坐标**；`cv2.remap(..., INTER_CUBIC)` 后保留顶部 440×640。
3. 检查标定分辨率、rectify_map 数值及其与相机内外参的一致性；缺失或不匹配报错。缺少资产时可通过官方链接下载，小 rectify_map 用 HTTP Range 从事件 ZIP 提取。
4. `events.h5` 内事件时间加 `t_offset` 才对应 RGB/语义全局时间。选择 `RGB_time <= target` 的最新图像，最大年龄 50 ms；无合法图像剔除并统计。
5. RGB 按 ImageNet mean/std 标准化。训练仅对 Histogram、已配准 RGB 和标签共同随机水平翻转。

**限制：官方映射只使用旋转和内参，没有利用深度补偿相机平移引起的视差。** 近处物体可能仍有边缘偏移；叠加图不能被解释为逐像素完全对齐的证明。

八序列叠加样例已目视检查，汇总见 [all_sequences.jpg](../artifacts/b1/dsec/all_sequences.jpg)。

缓存根目录 `artifacts/b1/dsec/`：每样本独立 NPZ，内含 uint8 Histogram、配准 RGB、原标签；manifest 记录序列、监督/RGB 时间、路径、标定 SHA256、剔除计数。原始数据未覆盖。

## 3. 本轮实测结果

### 数据和正确性

- SSD 上 8 个训练序列各均匀选取 32 个标签窗口，总计 256 个；本批无超龄/未来 RGB、无全 ignore 样本。
- `zurich_city_08_a` 的 32 个窗口全部作为开发验证；其余 7 序列共 224 个窗口训练。没有使用官方测试集调参，也没有全量转换数据。
- Histogram 测试覆盖空事件、同时间戳、两种极性、边界坐标、截断、uint8 溢出，逐元素对照冻结 RVT 实现。
- 八序列的首个缓存样本均对照官方映射、原始 RGB、标签、事件时间和 RVT，逐元素一致；全部 256 个样本检查同步与拆分。
- 骨干四尺度尺寸、预训练首层、参数独立性、两分支有限非零梯度、eval 调用顺序无关性通过。
- 全 ignore、无监督、无有效深度跳过；检测空框按背景计算合法损失。GEN1、Eventscape、MVSEC、DSEC 均完成真实样本 FP32 前向、损失、反向、优化器更新和严格保存/恢复。
- 原 HMNet-B3 分割真实样本前向、反向通过，损失约 0.14352；原配置未替换。

证据：[契约测试](../artifacts/b1/contracts.log)、[配准与时间检查](../artifacts/b1/data-validation.log)、[真实样本冒烟](../artifacts/b1/smoke.json)、[无效标签检查](../artifacts/b1/validity.log)、[原模型回归](../artifacts/b1/legacy.log)。测试脚本在既有忽略目录 `tests/`，没有添加 `.gitignore` 例外。

### DSEC 小样本与 200 步短训

采用空闲 RTX 4090；seed=42，batch=2，累积 8 个有效 microbatch 后更新一次；AdamW，固定 lr=2e-4、weight decay=0.01，11 类、ignore=255。先验证 FP32，再使用 AMP；检查点包含模型、优化器、GradScaler、更新步数、采样游标及随机状态。

| 实验 | 结果 |
|---|---|
| 固定 2 样本、40 次更新 | loss 3.79274 → 0.36085；固定样本 mIoU 64.55%，证明可学习，未宣称各类完全记忆 |
| 224 训练窗口、200 次更新 | loss 3.82215 → 0.46880 |
| 开发集 mIoU：step 50 / 100 / 150 / 200 | 45.67% / 49.51% / 51.97% / **53.56%** |
| 稳态吞吐中位数 | 约 16.51 样本/秒，包含取批、前后向和更新，不含开发集评估/存盘 |
| 峰值 PyTorch allocated 显存 | 8687 MiB，约 8.48 GiB |
| 保存与恢复 | 200 步权重通过原测试入口复算 mIoU 一致；在独立目录恢复并完成第 201 次更新 |

最后一轮各类 IoU（按 DSEC 类 ID 0–10）：90.67、80.93、16.40、0.00、8.71、94.63、82.32、88.42、80.53、15.40、31.17%。稀有类仍有明显不足。

[200 步权重](../artifacts/b1/segmentation/checkpoint.pth)、[逐步指标](../artifacts/b1/segmentation/metrics.jsonl)、[复算指标](../artifacts/b1/segmentation/evaluation_dev.json)、[配置记录](../artifacts/b1/segmentation/settings.json)。这是小规模工程验证结果，不能与原 HMNet 全量训练/官方测试精度直接比较。恢复已验证可继续更新；多 worker 随机增强状态没有逐 worker 序列化，不承诺恢复后位级复现连续运行的增强序列。

## 4. 正式训练接口与共享数据（2026-09-19更新）

三任务README按“数据集准备、Train、Test”组织，包含完整TrainSettings参数表：[分割](../experiments/segmentation/README.md)、[检测](../experiments/detection/README.md)、[深度](../experiments/depth/README.md)。正常训练设置epochs=100、updates=None，由脚本自动计算更新数；可用--epochs/--updates/--resume/--output/--data-root覆盖，不再读取旧B1环境变量。原进程已加载的配置保持不变。

全量DSEC缓存和标定已从本工程logs/datasets迁移至 `/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/{dsec_b1,dsec_b1_assets}`，8082个样本；原路径保留软链接支持运行中的训练。新准备脚本默认输出source父目录下preprocessed/dsec_b1，使用全部标签，拒绝覆盖已有完整缓存。训练产物和TensorBoard仍存本工程logs，历史诊断仍在artifacts。

当前运行的full_100ep实验实际batch=32、累积8、每轮228批，却沿用45600次更新，相当于1600轮；100轮应为2850次。已告知用户，本轮未停止或改变其训练预算。新记录含epoch/data_epochs；已达到目标的续训会明确报错。

## 5. 帧式 ONNX

本轮历史导出：[artifacts/b1/onnx](../artifacts/b1/onnx)。脚本后续新导出的默认目录为 `logs/onnx/efficientvit_b1/`。每套有 `*.onnx`、`*.sim.onnx` 和 `*.report.json`。固定 batch=1 和空间尺寸；改变分辨率需重新导出。图外执行 Histogram、时间同步、几何配准和 RGB 标准化；没有任何记忆状态输入。

| 文件前缀 | 输入 | 输出语义 | 权重 |
|---|---|---|---|
| segmentation | event_hist [1,20,440,640]，rgb [1,3,440,640] | 11 类全尺寸 logits | 200 步短训 |
| detection | event_hist [1,20,240,304] | YOLOX 解码 xywh、objectness、类别概率，NMS 在图外 | 真实样本冒烟更新 |
| depth_eventscape | event_hist [1,20,256,512] | 米制全尺寸 depth | 真实样本冒烟更新 |
| depth_mvsec | event_hist [1,20,260,346] | 米制全尺寸 depth | 真实样本冒烟更新 |

化简后节点数：分割 1439→510、检测 1140→328、两套深度各 798→280；四套图的 Cast 均降至 0。完整统计见 [summary.json](../artifacts/b1/onnx/summary.json)。

执行 ONNX full checker、onnxsim，以及 PyTorch/ORT 与原图/简化图两组对照（非零输入、全零输入）。分割非零输入使用真实开发缓存。分割 PyTorch/ORT 最大绝对差：真实样本 `1.30e-4`，全零输入 `5.47e-4`；真实样本 281600 个像素中 1 个 argmax 不同（99.999645% 一致）。这是 FP32 计算与图折叠后的数值差异，不宣称位级等价。分割检查阈值 atol=1e-3、rtol=1e-4；其他任务 atol=1e-4、rtol=1e-4；原图/简化图单独使用 atol=1e-5、rtol=1e-4。

```bash
./scripts/hmnet-python scripts/export_b1_onnx.py --task segmentation \
  --checkpoint artifacts/b1/segmentation/checkpoint.pth \
  --sample artifacts/b1/dsec/zurich_city_08_a/000000.npz
./scripts/hmnet-python scripts/export_b1_onnx.py --task detection \
  --checkpoint artifacts/b1/gen1_smoke.pth
./scripts/hmnet-python scripts/export_b1_onnx.py --task depth_eventscape \
  --checkpoint artifacts/b1/eventscape_smoke.pth
./scripts/hmnet-python scripts/export_b1_onnx.py --task depth_mvsec \
  --checkpoint artifacts/b1/mvsec_smoke.pth
```

导出脚本也支持 `--backbone-only` 查看四尺度接口。部署仍需处理 LiteMLA 的 MatMul/Div、HardSwish、Resize 等算子；图变为无状态不代表这些算子已适配 Dremi。

## 6. 后续独立实验

本轮实现和短训已完成。下一阶段先扩大同一拆分下的数据规模与训练时长，评估稀有类和 RGB/DVS 消融，再决定是否接受官方 B1 精度。后续依次独立比较 HWAware B1、S-FIFO/自回归、RENet/FRN 等融合，以及必要时以官方 B1 为教师蒸馏；本轮未预建这些模块。

源码位于正常目录；上游源码保留许可证和快照记录。历史诊断产物保留在 `artifacts/` 等既有忽略目录，正式训练输出位于新增忽略目录 `logs/`；未新增 `!` 例外，也没有执行 Git 提交或推送。
