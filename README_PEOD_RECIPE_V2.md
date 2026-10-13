# PEOD 原生720P RecipeV2：rvt

本工程从 `4b43531e026c9a2ceb82bf8e79ece29495871201` 独立派生，分支 `det_peod_rgbdvs_v1.2_T_rvt_720p_recipe_v2`，父工程未覆盖。新入口是 `scripts/peod.py`；本页取代本分支继承文档中的旧200轮启动配置。正式训练由用户手动启动。

## 首阶段入口

本分支已包含共有P0数值修复、EMA与开发划分，供后续配方迁移。当前首阶段只启动两组RGB，命令见[RGB RecipeV2分支说明](https://github.com/lionelZhaowy/HWAware_HMNet/blob/det_peod_rgb_v1.2_T_720p_recipe_v2/README_PEOD_RECIPE_V2.md)。不将四模态同时重跑作为本轮实验。

## 共同协议

- 原生720×1280，原EfficientViT-B1、neck、三尺度YOLOX、SimpleAdd及DVS M=2保持；从官方B1初始化，不能用旧PEOD权重续训进入新划分。
- 四工程统一修复：FP32浮点GT；移除匹配区域、IoU中的半宽/半高整数截断；卷积保持BF16，解码、匹配、IoU/回归及后处理FP32。
- 固定70条训练序列44,415帧、18条dev序列10,393帧；原val4,820帧和官方test11,847帧不变。共享原生缓存只读复用，不复制数据。
- `development_split.toml`记录序列清单与父manifest SHA；仅依据原train注释，固定seed42与场景配额选择。缺少跨序列录制session标识，不能宣称已排除所有场景相关性。
- 60完整epoch，batch8/accumulation4/eval8/workers2，AdamW全部参数wd0.01，BF16，TF32关闭，seed42。每轮5,552微批、1,388次更新，共83,280次；5轮预热6,940次，起始系数0.1，余弦降至5e-6。
- A0峰值LR5e-4，A1为1e-4。其他训练配方一致；原始权重与EMA来自同一优化轨迹。
- EMA每次成功optimizer更新一次，FP32参数及浮点buffer，整数buffer拷贝，`decay=0.9998*(1-exp(-updates/2000))`。EMA推理独立重置时序bank。
- 每轮完整dev分别评估raw/EMA、独立保存best；原固定val仅在20/40/60轮分别报告，不参与best选权。正式训练不访问test预测或指标。验证阶段保存/恢复训练RNG。

## 检查点和兼容性

- `checkpoint.pth`：最近保存轮次的完整raw、EMA、优化器、日程契约、RNG、采样游标和时序状态，**唯一的严格resume入口**。每轮覆盖保存，突发中断后从最近保存点恢复。
- `best_checkpoint.pth` / `best_ema_checkpoint.pth`：分别按dev mAP选择的raw/EMA，仅推理权重，不可resume。
- `epoch_020/040/060_raw/ema.pth`：预定比较轮次的小型推理权重，不逐轮堆积。
- resume自动回退metrics到已保存checkpoint的step。若最后一行因中断未写完，会先将原始字节备份到训练目录的`history_backups/`，再原子替换；中间损坏、带换行的损坏记录或非法step仍报错，不静默跳过。模型/优化器/EMA与训练日程不变。
- 新旧数值、划分、EMA、LR、seed、batch契约不一致会拒绝resume；不支持静默迁移旧优化器。原子保存预留512MiB空间，空间不足时保留上一份检查点。
- TensorBoard关注`dev/mAP`与`dev_ema/mAP`、逐类`AP_*`和`condition_*_mAP`；`fixed_val*`仅在预定轮次出现。开发集指标不得与旧val指标直接当作同协议增益。

## 验证和后续阶段

本分支的检查入口：`scripts/check_peod_recipe.py`（浮点几何/EMA/划分/预算），`scripts/check_peod_model.py`（原生真实高目标前后向），`scripts/check_peod_resume.py --tail-check`（真实保存恢复），加`--smoke-validation`可检验有界raw/EMA验证和选权；必须显式指定独立`--output`诊断目录，产物置于主工程artifacts。

ONNX导出入口为`scripts/export_peod.py`，支持raw/EMA并校验新数值/划分契约。结构检查、数值检查和部署验收分别报告；父版DVS时序状态的严格ONNX失败不能视为本轮自动通过。实施报告、实际检查结果、导出图路径和来源SHA统一保存在主工程`artifacts/peod_detection/native_720p/recipe_v2/20261011/`，不提交GitHub。

当前只实施P0与第一阶段入口。片段采样、外观增强、参数分组衰减和门控融合须按第一阶段结果逐项选择；未混入当前A0/A1。尚无新正式训练精度结论。最终四模态对比另行固定共同胜出配方、预算与多个seed。

日志恢复回归：`./scripts/hmnet-python scripts/check_resume_history.py -v`；GPU精确恢复对照可运行`scripts/check_peod_resume.py --recipe a1 --tail-check --smoke-validation --torn-history --output <主工程artifacts中的独立目录>`，不会写正式run。
