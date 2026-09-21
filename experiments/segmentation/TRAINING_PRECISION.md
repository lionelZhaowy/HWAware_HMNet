# 三组融合消融：150 epoch / BF16

2026-09-21 实现。各工程源码独立、Git分支独立；三工程的骨干、门控、交叉注意力、任务构建器、训练/评估/导出脚本完全相同，主配置只在 `fusion_mode` 与输出目录上不同。没有跨工程Python导入或软链接依赖。配置别名 `efficientvit_b1_cross.py` 指向本工程主配置，不是旧结构选择器。

|工程后缀|父工程|fusion_mode|实验变化|输出目录（logs/segmentation/下）|
|---|---|---|---|---|
|v1.2|v1.1 / 04f01df|add|保留原投影Add；150轮对照|efficientvit_b1_add_v12_bf16|
|v3|v1.1 / 04f01df|adaptive_add|Add等价初始化的空间加权|efficientvit_b1_adaptive_v3_bf16|
|v2.1.1|v2.1 / d8037a8|cross_stage_post_mbconv_no_feedback|保留融合模块，取消后续主干Stage反馈|efficientvit_b1_cross_v211_bf16|

分支分别为 `seg_rgbdvs_640x440_v1.2`、`seg_rgbdvs_640x440_v3`、`seg_rgbdvs_640x440_v2.1.1`，完整目录名均为 `HWAware_HMNet_Seg_RGBDVS_640x440_` 加后缀。

## 固定协议与公平性

三组：BF16 autocast（参数及优化器状态仍为FP32）、seed42、150 epoch、batch32、accumulation1、workers8、prefetch1、eval batch32；AdamW，lr2e-4，weight decay0.01，5轮warmup后cosine至2e-6，每轮评估。train7295/dev787；每轮228次更新，总34200次，warmup1140次。样本、水平翻转、官方B1 ImageNet预训练、Neck、主/辅助Head、loss均不因版本改变。

三组均从官方 `pretrained/efficientvit_b1_r224.pth` 重新初始化，不resume旧实验。该文件已本地准备，不提交权重。旧v1.1/v2.1使用FP16，新三组明确改为BF16，与v2.0/v2.2一致。因旧实验的精度、训练器版本及日程并非全部一致，不能将差异全部归因于结构；若要严格隔离v2.1.1的反馈因素，应再做相同公共训练器、BF16下的反馈参考组（本轮未启动额外实验）。

即使新Add与复杂融合最终相近，也只能说明该设置下未观察到稳定收益；不足以单独证明“融合模块或下一Stage没有学习融合信息”。

## 训练命令

进入要运行的工程，例如：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2
# v3、v2.1.1分别进入相应同级目录。
nvidia-smi
GPU_ID=0  # 改成启动时确实空闲的卡；不要覆盖其他训练进程。
CUDA_VISIBLE_DEVICES="$GPU_ID" ./scripts/hmnet-python experiments/segmentation/scripts/train.py   experiments/segmentation/config/efficientvit_b1.py --single --precision bf16 --seed 42
```

三工程均使用上面同一训练命令，配置自动选择对应结构和独立输出。不加 `--amp`：它单独出现时仍按旧接口选择FP16。不要加 `--stop-after`、`--updates`，也不要resume冒烟检查点。并发运行时每个终端分别选择空闲GPU；本次没有自动启动正式训练。

同结构、同精度的正式断点恢复：

```bash
RUN_DIR=logs/segmentation/efficientvit_b1_add_v12_bf16  # 按上表替换
CUDA_VISIBLE_DEVICES="$GPU_ID" ./scripts/hmnet-python experiments/segmentation/scripts/train.py   experiments/segmentation/config/efficientvit_b1.py --single --precision bf16 --seed 42   --resume "$RUN_DIR/checkpoint.pth"
```

恢复会检查结构、精度、world size、BN、数据、batch、seed及学习率日程。v2.1和v2.1.1虽然参数形状相同，语义不同，禁止直接resume混用；导出/评估也校验结构契约。

训练器保留 `--precision fp16`、`--precision fp32` 及torchrun多卡支持，公共实现来自v2.2。此次仅验证三新结构的BF16单卡；未重新验收其FP32多卡路径。现有四组训练及父工程代码没有升级或重启。

## 结构与公式

四级原生通道32/64/128/256，空间尺寸110×160、55×80、28×40、14×20，均投影至256通道进入原Pyramid。累计stride为4/8/16/32。三新版本均无主干反馈、无独立Fusion级联，跨尺度交互发生在Neck；不同点为每级的融合算子。

v1.2：`X_i = ReLU(ConvBN_D(D_i) + ConvBN_R(R_i))`。两编码器独立推进。

v3：令 `d=ConvBN_D(D_i), r=ConvBN_R(R_i)`，均为[B,256,H,W]。沿通道取mean/max并按D、R顺序拼接为[B,4,H,W]：

```text
s = Cat(mean_c(d), max_c(d), mean_c(r), max_c(r))
z = Conv7x7(s)                        # 4→1, padding3
δ = 2*HardSigmoid(z)-1                # [B,1,H,W]
X_i = ReLU((1-δ)*d + (1+δ)*r)
```

每个位置两权重在[0,2]内且和为2；共享于通道，四级各有独立门控。Conv权重及bias零初始化，δ=0，起点等价于Add。用functional Conv和零Parameter实现，以避免消耗随机数或被任务通用初始化覆盖；CPU回归确认全模型公共参数、RNG及初始输出与Add一致。每级197参数，合计788。门控可以学习梯度，不是被冻结的常数；ONNX为标准Conv/HardSigmoid/乘加。此次未同时加入通道门控、额外BN或新的注意力。

v2.1.1：每级 `R_i=S_i^R(R_{i-1}), D_i=S_i^D(D_{i-1})`。原v2.1融合模块产生 `R_i+,D_i+,F_i`，保留注意力投影残差、后置MBConv残差以及两路更新的合并。`X_i=ReLU(ConvBN(F_i))` 送Neck；下一Stage只读取原 `R_i,D_i`，不读取 `R_i+,D_i+`。不添加v2.2的入口Mul+Add。

## 验证与ONNX边界

- `./scripts/hmnet-python scripts/check_fusion_variants.py`：初始Add等价、门控非零梯度/极值、反馈开关/模态隔离、checkpoint重计算BN一次更新，4项通过。
- 父版对照：新Add与v1.1、新参考反馈模式与v2.1的初始参数和骨干输出一致。
- 三组均以真实440×640样本、BF16/batch32训练2步，保存后恢复第3步；无非有限参数/梯度、无跳步，BN计数为3。冒烟保留完整34200步日程；仅workers降为2、dev限制5帧。此次不据3步loss宣称精度改善。
- 峰值已分配显存约v1.2 19.1 GiB、v3 19.8 GiB、v2.1.1 18.8 GiB，不含其他进程，正式长训以实测为准。v1.2一次恢复遇到共享GPU显存不足，换空闲卡重试成功；原错误日志保留。
- 各工程 `artifacts/onnx/step3/` 下有 `segmentation.onnx`、`segmentation.sim.onnx`、`segmentation_backbone.onnx`、`segmentation_backbone.sim.onnx` 及report。均来自本工程BF16训练第3步的FP32参数快照，供结构检查，不是正式收敛模型。
- v1.2、v2.1.1整网/骨干通过本次真实样本、全零、空事件3种输入检查。v3真实样本通过；整网全零失败；骨干全零与空事件的f16/f32超差。全部图通过ONNX checker，原图与简化图一致；没有放宽阈值。所有简化图BN已折叠为0个BatchNormalization；v3仍有4个HardSigmoid门控。
- 输入RGB[1,3,440,640]、DVS[1,20,440,640]；整网输出[1,11,440,640]，骨干4个256通道多尺度输出。完整来源见各工程ONNX目录README；本地诊断、权重及协作文档不上传GitHub。

结构检查、短步训练、精度评估、部署验收是不同阶段；目前完成前两项，正式训练/最终精度尚未完成，v3仍有ONNX边界输入数值限制。
