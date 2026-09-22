# v1.2_T：DVS LiteMLA M=2训练与验证

本工程独立派生自v1.2，分支`seg_rgbdvs_640x440_v1.2_T`。融合保持SimpleAdd；仅DVS骨干Stage3/4的7个LiteMLA读取当前＋上一帧。RGB骨干、融合算子、Neck/Head、loss和优化器不改。父工程不修改，不复制或续训其活动日志。

## 计算和训练契约

- token-row公式为`S=ReLU(K)^T V`、`z=ReLU(K)^T 1`；工程channel-first缓存其转置，合并为`[B,heads*scales,17,16]`。当前Q读取两帧S/z的和，归一化后投影。仅返回本帧摘要，不能把两帧总和继续当成上一帧。
- Q/K/V投影及网络其余部分使用BF16 autocast；Q/K/V在第一次注意力矩阵乘法前显式升FP32，归约、历史状态、第二次矩阵乘法和除法全为FP32。保留原eps与BN。优化器主权重仍按AMP惯例为FP32。
- M=2含当前帧，单层直接覆盖约100ms事件；7层递归会间接传递更早信息，不是整网严格100ms截断。FP32持久状态187 KiB/流。
- 150epoch、全局当前帧batch32、accumulation1、workers8、seed42、AdamW lr2e-4→2e-6、5epoch warmup，7295帧/228步每轮，共34200步。不使用32段×多帧偷偷扩大batch/监督预算。
- 序列内排序；训练将打乱顺序的完整序列连接并分成32条均衡连续流，每帧每轮出现一次、无padding/重复。片段/序列边界、异常时间间隔和增强变化重置状态。同一流内增强一致。当前片段划分本身是时序训练的必要变化，不能把全部收益纯归因于注意力算子。
- **TBPTT=1**：每步监督当前帧，跨优化器更新detach历史；有前向记忆，没有跨更新的完整时间反传。checkpoint保存各rank的时序摘要/流身份，以及采样游标、RNG、模型/BN/优化器。不得直接resume父版无状态权重，或跨M/融合/精度/world_size恢复。
- dev/test按完整序列因果评估，不为了凑batch32把一条序列切成32份冷启动。当前dev只有1序列，实际每次前向1帧，因此验证吞吐与父版不同。独立验证状态不污染训练状态；全部冷启动帧纳入指标。

## 正式训练

先用`nvidia-smi`确认空闲GPU。下面的1是示例编号；两个实验同时运行时必须选择不同空闲卡。不要添加`--amp`（该兼容参数表示FP16），也不要resume artifacts里的3步诊断。

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python experiments/segmentation/scripts/train.py experiments/segmentation/config/efficientvit_b1.py --single --precision bf16 --seed 42
```

默认输出：`logs/segmentation/efficientvit_b1_add_v12_T_bf16/`。本轮没有启动正式训练，也没有创建该正式输出目录。

中断后按原契约恢复，目标仍为累计150轮：

```bash
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python experiments/segmentation/scripts/train.py experiments/segmentation/config/efficientvit_b1.py --single --precision bf16 --seed 42 --resume logs/segmentation/efficientvit_b1_add_v12_T_bf16/checkpoint.pth
```

继承的FP32/分布式接口保留；本轮正式候选验证为BF16单卡batch32，未对两个新时序结构做完整多卡冒烟。分布式采样单测已覆盖每流固定rank及完整序列评估，不能把它当完整DDP验证。

## 检查和导出

```bash
./scripts/hmnet-python scripts/check_temporal_m2.py
./scripts/hmnet-python scripts/check_temporal_resume.py --checkpoint artifacts/temporal_validation/continuous_gpu1/checkpoint.pth --output artifacts/temporal_validation/cpu_resume.json
./scripts/hmnet-python scripts/export_temporal_onnx.py --checkpoint logs/segmentation/efficientvit_b1_add_v12_T_bf16/best_checkpoint.pth --output artifacts/onnx/trained_temporal
./scripts/hmnet-python scripts/export_temporal_onnx.py --checkpoint logs/segmentation/efficientvit_b1_add_v12_T_bf16/best_checkpoint.pth --output artifacts/onnx/trained_temporal --backbone-only
```

单步ONNX显式输入RGB/DVS及7个上一帧状态，输出logits（或四尺度特征）和7个当前摘要。外部按流保存状态、在场景边界置零；不使用Python隐式模型状态。导出为FP32推理图，不表示混合BF16推理或Dremi量化验收。旧`export_b1_onnx.py`拒绝时序检查点，避免误导出无状态图。

独立评估使用原`experiments/segmentation/scripts/test.py`入口（本工程默认配置已选择时序）：

```bash
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python experiments/segmentation/scripts/test.py experiments/segmentation/config/efficientvit_b1.py dev --pretrained logs/segmentation/efficientvit_b1_add_v12_T_bf16/best_checkpoint.pth --output artifacts/evaluation/temporal_dev
```

旧`scripts/evaluate_dsec_b1.py`视频诊断入口对时序配置主动报错，防止把相邻batch误当连续帧。验证默认FP32，与父版默认验证精度一致。

## 本轮结果与限制

两版均完成真实BF16/batch32更新2步＋恢复第3步，保持34200更新预算；诊断仅将workers设2、dev缩为5帧。loss/梯度/FP32状态有限。CPU回归覆盖token拼接等价、零历史、梯度、RGB/DVS状态隔离、流重置、采样/增强/rank连续性；CPU固定小裁剪的保存恢复检查中，权重、BN、优化器、状态、loss逐位一致。

GPU两次全新运行已存在微小差异，不能承诺逐位恢复一致；当前PyTorch二维CE CUDA不支持严格确定性开关。该失败证据保留，不为通过检查更改正式loss。首次v1.2_T冒烟因GPU6被其他任务占用OOM，改用空闲GPU1后通过；不是降低batch绕过。

3步诊断权重的整网/骨干原始及简化ONNX已放本工程`artifacts/onnx/temporal_step3/`，checker通过、BN已折叠。**数值报告仍为失败**：多步整网logits在本次阈值内，但部分原始S/z状态越过逐元素阈值，v2.1.1_T骨干还涉及部分特征输出。报告按原atol=1e-3、rtol=1e-4记录，未放宽阈值。结构图可人工查看，不等于完整状态数值/长期流式/Dremi部署验收。3步结果不证明收敛或精度收益。
