# v1.2_T.1 / v1.2_T.2：50ms同步双极性输入实验

两分支均直接来自v1.2_T/76f1a8c，共同源码一致，主配置和根README区分版本。点号版本与异步下划线v1.2_T_1/_2不同。

|版本|表示|分支|默认输出|
|---|---|---|---|
|v1.2_T.1|polarity_binary：count>0，0/1|seg_rgbdvs_640x440_v1.2_T.1|logs/segmentation/efficientvit_b1_add_v12_T_dot1_binary_bf16|
|v1.2_T.2|polarity_count：原始事件数|seg_rgbdvs_640x440_v1.2_T.2|logs/segmentation/efficientvit_b1_add_v12_T_dot2_count_bf16|

## 实验定义

- 时间t的输入使用父版闭区间[t-50000us,t]，没有时间子桶；两个通道分别为p=0负极性与p=1正极性，均非负，不相互抵消。DVS输入[B,2,440,640]，RGB输入[B,3,440,640]。
- 计数从原始events.h5重建，int64累加、检查范围后保存int32，模型输入float32；原始计数不clamp、不log、不归一化。二值版加载同一计数缓存后取count>0。不能合并旧RVT缓存代替原始事件累加，因为旧表示截断且uint8可能回绕。
- 新缓存按父manifest逐项保留样本、时间、配对RGB/GT与划分。RGB/GT引用旧cache，不复制或重新配准；新cache依赖父cache，迁移机器需保留两者并更新manifest中的资产根路径，再重新审计。路径改变会影响数据契约，不能冒充原运行resume。
- 模型仅DVS首层20→2，其他骨干、Add、Neck/Head及loss保持父版。DVS Stage3/4七个LiteMLA保留M=2，状态和注意力归约FP32，其余BF16。RGB每帧编码，仍然同步输出/监督，不使用异步缓存或伪标签。
- 完成标准20通道整网初始化后仅替换首层，保护随机数状态；首层沿用官方RGB权重均值复制并乘3/2。其余参数与父版seed42初始化逐项一致，两新版本所有初始参数相同。正式训练不继承父版已训练权重。
- 保持150epoch/34200更新、batch32、accumulation1、workers8、seed42、AdamW lr2e-4、weight_decay0.01、warmup5epoch及cosine到2e-6。连续lane采样/流内翻转/reset/TBPTT=1不变；每步detach历史。dev单序列按整条因果流验证，实际batch1，保留冷启动帧。
- 同版本才能resume，checkpoint记录输入表示与数据摘要；相同2通道也不能跨二值/计数恢复、评估或导出。默认验证为FP32，与父版协议相同。

## 1. 共享缓存（两个实验只需生成一次）

在任一点号工程根目录执行，沿用./scripts/hmnet-python。已生成manifest时不重复build，可执行audit；若build中断且尚无manifest，重跑build会从原始数据重新计算并原子覆盖本新目录中的未完成计数文件。单写者文件锁防止并发生成。原dsec_b1保持只读。

```bash
./scripts/hmnet-python scripts/prepare_dsec_polarity.py
./scripts/hmnet-python scripts/prepare_dsec_polarity.py --audit
```

默认输出 `/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_polarity50ms`，包含train7295/dev787。audit逐项从原始事件重算，对比count内容与文件哈希、父RGB/GT资产哈希和manifest成员。

可显式指定 `--source`、`--parent-cache`、`--output`。小样本诊断使用 `--limit N`（每split N帧）及独立artifacts输出，不能用诊断缓存启动正式全量训练。

## 2. 实现检查

```bash
./scripts/hmnet-python scripts/check_polarity_inputs.py --parent-project ../HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T --cache /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_polarity50ms
./scripts/hmnet-python scripts/check_temporal_m2.py
```

覆盖计数、边界、极性、空事件、溢出保护、非首层初始化、原采样与真实RGB/GT一致性。验证证据存artifacts，不上传GitHub。

## 3. 正式训练：两个独立终端

先执行 `nvidia-smi` 选择空闲GPU，以下4和5只是示例编号。不要用`--amp`，该旧选项表示FP16；不要resume冒烟权重。两个终端分别进入自己的工程。

二值版：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T.1
POLARITY_OUTPUT=logs/segmentation/efficientvit_b1_add_v12_T_dot1_binary_bf16
CUDA_VISIBLE_DEVICES=4 ./scripts/hmnet-python experiments/segmentation/scripts/train.py experiments/segmentation/config/efficientvit_b1.py --single --precision bf16 --seed 42
```

计数版：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Seg_RGBDVS_640x440_v1.2_T.2
POLARITY_OUTPUT=logs/segmentation/efficientvit_b1_add_v12_T_dot2_count_bf16
CUDA_VISIBLE_DEVICES=5 ./scripts/hmnet-python experiments/segmentation/scripts/train.py experiments/segmentation/config/efficientvit_b1.py --single --precision bf16 --seed 42
```

如需在自己选定时间做两步诊断，在上述对应训练命令末尾添加 `--stop-after 2 --output artifacts/polarity_smoke`。它保留150轮日程但仅执行两次更新，首步仍执行完整dev验证；正式训练不从该诊断恢复。当前实现交付的快速冒烟将dev缩为3帧，单独记录，不能替代完整精度评估。

## 4. 中断恢复

进入对应工程并按上节设置其POLARITY_OUTPUT。下面4同样替换为所选空闲卡；恢复目标仍是累计150轮。

```bash
CUDA_VISIBLE_DEVICES=4 ./scripts/hmnet-python experiments/segmentation/scripts/train.py experiments/segmentation/config/efficientvit_b1.py --single --precision bf16 --seed 42 --resume "$POLARITY_OUTPUT/checkpoint.pth"
```

## 5. 独立评估

在对应工程、正确POLARITY_OUTPUT下执行。默认FP32，与父版验证一致。

```bash
CUDA_VISIBLE_DEVICES=4 ./scripts/hmnet-python experiments/segmentation/scripts/test.py experiments/segmentation/config/efficientvit_b1.py dev --pretrained "$POLARITY_OUTPUT/best_checkpoint.pth" --output artifacts/evaluation/polarity_dev
```

冻结模型选择后，官方test另建共享计数缓存（只生成一次），随后各版本分别评估。test不参与调参：

```bash
./scripts/hmnet-python scripts/prepare_dsec_polarity.py --parent-cache /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_test --output /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_polarity50ms_test
./scripts/hmnet-python scripts/prepare_dsec_polarity.py --parent-cache /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_test --output /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_polarity50ms_test --audit
CUDA_VISIBLE_DEVICES=4 ./scripts/hmnet-python experiments/segmentation/scripts/test.py experiments/segmentation/config/efficientvit_b1.py test /home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/dsec_b1_polarity50ms_test --pretrained "$POLARITY_OUTPUT/best_checkpoint.pth" --output artifacts/evaluation/polarity_test
```

旧scripts/evaluate_dsec_b1.py会拒绝时序配置；使用上面的因果序列评估入口。正式训练结束后比较父版与两新版本的mIoU、逐类IoU、best、末轮和末10轮统计；单seed不证明统计显著。

## 6. ONNX结构与严格数值检查

在对应工程、正确POLARITY_OUTPUT下执行；默认按本工程配置检查表示，不能对错版检查点导出。

```bash
./scripts/hmnet-python scripts/export_temporal_onnx.py --checkpoint "$POLARITY_OUTPUT/best_checkpoint.pth" --output artifacts/onnx/polarity_trained
./scripts/hmnet-python scripts/export_temporal_onnx.py --checkpoint "$POLARITY_OUTPUT/best_checkpoint.pth" --output artifacts/onnx/polarity_trained --backbone-only
```

图显式输入双通道DVS、3通道RGB及7个FP32历史状态；输出主头logits（或四尺度特征）和7个新状态。外部在序列边界清零，并按流保存状态。导出是FP32推理，不代表BF16/Dremi部署验证。

会生成原图、简化图和JSON报告。checker、真实序列、全零/空事件、多步独立状态轨迹分别检查，沿用atol=1e-3、rtol=1e-4及类别一致率要求。若数值超差，报告仍保存且脚本非零退出，不代表没有生成可供人工查看的结构图。交付时的诊断图在各工程artifacts/onnx/polarity_initial（官方骨干预训练＋随机任务头，未训练），权重来源及实际验证状态见artifacts/polarity_implementation/README.md。

## 解释边界

两组相对RVT同时改变时间桶信息、数值分布/截断和必要首层输入维度；只能归结为输入表示整体替换的影响。二值与计数的对比更直接反映事件次数信息的价值。输入通道减少10倍不等于整网加速10倍；共享参数保持不变，只有首层减少2592参数。
