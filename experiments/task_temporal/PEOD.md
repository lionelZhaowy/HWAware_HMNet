# PEOD 原生720P公平检测实验

当前工程从 `HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_RVT` 的 `2edb8c94e7a564436553ced50f6a3b9054f32940` 独立派生，分支 `det_peod_rgbdvs_v1.2_T_rvt_720p`。默认模态 `rgbdvs`、表示 `rvt_histogram`。四组公共Python源码相同，仅配置中的模态/表示和工程标识不同。正式训练由用户手动启动；未使用低分辨率100/200轮模型或旧优化器。

## 数据与几何

数据根 `/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig`；完整共享索引 `preprocessed/hmnet_v12t_720x1280_v1/` 已完成。它复用旧版已完整物理扫描的事件时间索引与相同样本，采用新的schema和来源指纹。旧240×304索引/缓存会被拒绝。新索引从已审计的旧索引生成，脚本要求其COMPLETE标记；当前数据链路已就绪，无待运行的必需预处理。

train/val/test为54808/4820/11847帧、88/9/24序列，六类为car/person/bus/truck/2-wheeler/3-wheeler。合法空框与空事件保留；reset_gap_us=101000，RGB配对和split沿用来源。原始PNG、DAT、标注JSON依然必需；输入缓存可选，目前共享原生输入缓存未全量生成。

RGB直接读取1280×720 PNG并ImageNet标准化；事件在原DAT坐标生成，窗口为闭区间[t-50000,t]。RVT为10bin×2极性、clip10、uint8累加回绕；Binary直接由同一窗口原始事件形成2通道存在性。无显示归一化、诊断倍率或旧输入插值。外部与模型内部输入均为H×W=720×1280；不增加图像padding，仅保留现有卷积padding。四级特征为180×320、90×160、45×80、23×40；neck取后三层，YOLOX NMS前输出为[B,18920,11]。最后一级向上取整，neck按实际特征尺寸插值。GT裁到原图有效区，预测在原坐标裁到1280×720并去除零面积框。评分使用原始浮点GT和原图面积的COCO small/medium/large，无GEN1过滤。

## 初始化与共同配置

官方B1文件`pretrained/efficientvit_b1_r224.pth`的SHA256为`bf8798aa03ed2bba21fc8d58f43ca9a92096f3f03bd700e1df1dcde11559f499`，本服务器为持久文件的只读引用。换机器需自行放置该同一官方文件，不上传权重。共有组件按名称稳定初始化并核对逐位权重；RVT首层RGB通道均值×3/20，Binary首层×3/2。骨干、SimpleAdd、256通道投影、Pyramid与原六类检测头保持来源实现。含DVS三组Stage3/4的M=2摘要为FP32，TBPTT=1；RGB-only无DVS状态，RGB每步重新编码。

|项目|四组共同值|
|---|---|
|训练/验证batch|8/8，每组单GPU|
|累积/有效batch|4/32；每epoch最后一次更新为24样本|
|epoch/微批/更新|200完整epoch；6851微批/epoch，1713更新/epoch，总342600更新|
|优化器|AdamW，betas=(0.9,0.999)，eps=1e-8，wd=0.01，所有参数沿用来源分组规则|
|LR|初始1e-5，峰值1e-4，余弦末端2e-6|
|warmup|5epoch=8565次更新，起始系数0.1|
|BN/裁剪|普通BN正常更新，梯度不裁剪，无激活重计算|
|精度/TF32|BF16，其时序摘要FP32；关闭TF32|
|seed/采样/增强|42，完整顺序流；lane内一致水平翻转，无额外强增强|
|workers/预取|2/1，spawn、persistent_workers、pin_memory|
|验证/保存/选权|每完整epoch验证全部val并保存last；best按val mAP；test不选权|
|后处理|score=0.01，NMS IoU=0.65|

累积在每epoch尾部flush，loss按组内实际样本数缩放，跨微批状态立即detach；累积不改变BN的真实batch8，也不延长TBPTT。检查点仅在优化器更新边界保存，包含模型、BN、优化器、LR预算、RNG、采样游标及DVS状态。不兼容恢复在建模前拒绝；不支持旧stage2开关。200轮是一段连续日程。峰值LR按有效batch32相对低分辨率128作保守下调，真实高目标有限步检查通过，不代表最优收敛已验证。四卡并发真实读取测量与范围见主工程交付报告；全量磁盘冷读取可能更慢。

## 正式训练与恢复

GPU4是物理编号；进程内为cuda:0。执行前用nvidia-smi核对空闲；已有训练输出会被拒绝。终端日志使用追加，避免误覆盖既有记录。不要添加--stop-after或--limit到正式命令。

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet_Det_PEOD_RGBDVS_v1.2_T_RVT_720P
mkdir -p logs/detection
set -o pipefail
CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
./scripts/hmnet-python scripts/peod.py train \
  --modality rgbdvs --representation rvt_histogram \
  --data-root /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_720x1280_v1 \
  --device cuda:0 --epochs 200 --batch 8 --accumulation 4 --eval-batch 8 \
  --workers 2 --seed 42 --precision bf16 \
  --learning-rate 1e-4 --min-learning-rate 2e-6 --warmup-epochs 5 --warmup-start-factor 0.1 \
  --output logs/detection/peod_720p_rgbdvs_rvt_histogram \
  2>&1 | tee -a logs/detection/peod_720p_rgbdvs_rvt_histogram.terminal.log
```

恢复时重复上述命令，加入`--resume logs/detection/peod_720p_rgbdvs_rvt_histogram/checkpoint.pth`，全部共同参数保持不变。该命令从同一预算继续，不重启日程。若要在有界冒烟后恢复正式训练，必须确认该检查点使用完整正式契约；67帧尾组诊断权重不能恢复正式训练。

## 索引、可选缓存与进度检查

```bash
./scripts/hmnet-python scripts/prepare_peod.py
# 可选全量输入缓存：容量估算约138GiB，误差与序列相关；最多250GiB且保留80GiB。
# 原子单序列发布，单写者锁，断点跳过；只有inputs/CACHE_COMPLETE.json才表示全量缓存完成。
./scripts/hmnet-python scripts/cache_peod.py --limit-sequences 0 --max-cache-gib 250 --reserve-gib 80
./scripts/hmnet-python -c 'import json; from pathlib import Path; p=Path("/data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_720x1280_v1"); c=json.loads((p/"COMPLETE.json").read_text()); print(c["schema"], c["splits"]); print("full input cache marker:",(p/"inputs/CACHE_COMPLETE.json").exists())'
du -sh /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_720x1280_v1
df -h /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig
tail -n 2 logs/detection/peod_720p_rgbdvs_rvt_histogram/metrics.jsonl
./scripts/hmnet-python -m tensorboard.main --logdir logs/detection/peod_720p_rgbdvs_rvt_histogram/tensorboard --host 127.0.0.1 --port 6006
```

索引生成仅核对并重用已完整扫描的时间窗口，不复制旧inputs；只有新schema输入缓存可被读取。共享全量原生输入缓存当前未生成，直接原始读取已做四卡并发实测。容量/锁/完成标记/逐元素验收使用独立3帧诊断缓存验证。

## 评估与结构导出

```bash
# 等正式训练结束后，以完整val选出的best评估；报告集中在主工程artifacts下。
CUDA_VISIBLE_DEVICES=4 ./scripts/hmnet-python scripts/peod.py eval --split val \
  --checkpoint logs/detection/peod_720p_rgbdvs_rvt_histogram/best_checkpoint.pth \
  --output /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet/artifacts/peod_detection/native_720p/evaluation/RGBDVS_v1.2_T_RVT_720P/val
./scripts/hmnet-python scripts/export_peod.py \
  --checkpoint logs/detection/peod_720p_rgbdvs_rvt_histogram/checkpoint.pth \
  --data-root /data/lab_dataset/RGB_DVS_Fusion/PEOD_orig/preprocessed/hmnet_v12t_720x1280_v1 \
  --output artifacts/peod_720p/onnx/final
# 骨干图另加--backbone-only；严格超差会保留报告并exit2。
```

本轮372个真实原始输入核验、四组高目标更新、完整val理想预测、共有初始化、4步严格恢复及67帧尾组恢复已验证。四组整网/骨干ONNX和逐案例报告位于`artifacts/peod_720p/onnx/`；RGB-only严格数值通过，含DVS三组仍有超差，部署未验收。权重来源为官方初始化后的4步真实冒烟，非正式权重。CPU FP32容差沿用atol=1e-3、rtol=1e-4；真实连续反馈、全零、重置和融合空事件均保留。

本服务器的集中证据与结构图：`/home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet/artifacts/peod_detection/native_720p/README.md`。正式训练及泛化精度尚未完成。
