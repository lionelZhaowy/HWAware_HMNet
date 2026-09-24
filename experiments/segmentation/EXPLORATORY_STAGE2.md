# B+C双教师：探索性阶段二

用户已选择在保留弱类别审核失败记录的前提下开展阶段二。两个学生结构、输入表示和优化超参数保持原方案，使用同一份B+C中间步伪标签。`audit_passed=false`表示审核结果，`audit_override.enabled=true`表示显式的探索选择，两者不混淆。

## 当前服务器路径

每个新终端执行：

```bash
source /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet/artifacts/async_stage2_bc_exploratory/env.sh
```

变量指向：

- 固定教师/学生初始化：main `artifacts/async_binary_teacher/teachers/B_best.pth`、`C_best.pth`。
- 原审核：main `artifacts/async_binary_teacher/audit_bc/audit.json`，保持原文件及SHA。
- 共享伪标签：数据集 `preprocessed/async_v1/shared_pseudo_bc_exploratory_stage1/`。
- 本地运行与检查报告：main `artifacts/async_stage2_bc_exploratory/`。

学生B从自己65轮best初始化，C从自己29轮best初始化；不从末轮last或二值教师初始化。阶段一停止时的更新数与best步数是两个概念，运行报告分别记录。

## 一次性共享数据准备

现有B/C事件缓存复用，不重建。生成器仍要求冻结教师路径/SHA、置信度、模式、推理精度和时间网格与审核一致；探索选项只允许质量未过门槛，不允许身份不匹配。

```bash
cd "$ASYNC_B"
CUDA_VISIBLE_DEVICES=2 ./scripts/hmnet-python scripts/pseudo_dsec_async.py generate \
  --teacher-mode bc \
  --b-checkpoint "$ASYNC_B_CKPT" --c-checkpoint "$ASYNC_C_CKPT" \
  --b-data "$ASYNC_DATA/dsec_async_B" --c-data "$ASYNC_DATA/dsec_async_C" \
  --confidence 0.95 --audit "$ASYNC_AUDIT" \
  --allow-failed-audit \
  --exploration-reason 'User-authorized B+C exploratory stage 2 on 2026-09-24; retain failed classwise audit and true-GT supervision' \
  --workers 2 --output "$ASYNC_PSEUDO"
```

完整遍历14590个train事件步，保存7295个无GT中间步标签。两教师同类一致且概率均≥0.95的像素保留，其余255；不替换真实GT，不用dev/test作为训练标签。`--workers`只预取CPU输入，模型仍按原始时间顺序更新状态。

完成标志是manifest；目录内同时保存原始 `teacher_audit.json`。生成中断且尚无manifest可以重跑，但会重新推理；已经完成则复用，不能覆盖。启动学生前，检查完整7295文件、网格成员、形状440×640、uint8及类别0–10/255，并核对manifest和audit哈希。

## 首次训练

**若运行报告已经记录阶段二启动，不要再执行首次启动命令；查看现有进程或使用恢复命令。** GPU0/1为本机示例，运行前重新确认资源。两个终端分别执行：

```bash
source /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet/artifacts/async_stage2_bc_exploratory/env.sh
cd "$ASYNC_B"
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/train_async.py \
  --single --stage 2 --epochs 20 --init-from "$ASYNC_B_CKPT" \
  --pseudo-root "$ASYNC_PSEUDO" --pseudo-weight 0.2 --allow-failed-pseudo-audit
```

```bash
source /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet/artifacts/async_stage2_bc_exploratory/env.sh
cd "$ASYNC_C"
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python scripts/train_async.py \
  --single --stage 2 --epochs 20 --init-from "$ASYNC_C_CKPT" \
  --pseudo-root "$ASYNC_PSEUDO" --pseudo-weight 0.2 --allow-failed-pseudo-audit
```

训练侧的显式选项独立于生成选项。缺少它时，数据集默认拒绝失败审核的标签；即使带上它，原审核缺失、SHA不符、身份不一致、失败原因被篡改或合成诊断标签仍会被拒绝。

共同设置：额外20轮/4560更新、batch32、accumulation1、workers8、seed42、BF16；AdamW lr2e-5，1轮预热，余弦最低2e-6；GT保留，伪标签权重前5轮0.04/0.08/0.12/0.16/0.20。保持原两步TBPTT和RGB延迟策略。

输出：各工程 `logs/segmentation/efficientvit_b1_v1.2_T_1_stage2/`、`...v1.2_T_2_stage2/`。

## 断点恢复

在各自工程使用同一命令，物理GPU按需要选择：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/train_async.py \
  --single --stage 2 --epochs 20 --resume \
  --pseudo-root "$ASYNC_PSEUDO" --pseudo-weight 0.2 --allow-failed-pseudo-audit
```

不要同时传 `--init-from`。裸 `--resume`选择本版本stage2输出的last；也可指定完整检查点路径。探索选项在恢复时仍需保留。伪标签manifest的SHA已在既有训练契约中绑定，修改教师、标签来源、审核信息或探索理由会导致恢复契约不同；不在续训中换标签。

阶段一CLI及契约不变；阶段一不能带 `--allow-failed-pseudo-audit`。本次没有模型结构修改。

## 检查与评估

```bash
./scripts/hmnet-python scripts/test_async_pseudo_audit.py
./scripts/hmnet-python scripts/test_async_binary_teacher.py
```

训练开始后检查 `loss_components` 中GT和pseudo主/辅损失均有记录，`pseudo_weight`从0.04开始；总loss新增加了伪标签项，不能直接与阶段一总loss比较。重点比较阶段一/二best在正常RGB、延迟RGB及低频RGB条件下的真实GT逐类指标。保留弱类审核失败作为实验背景，不将进入训练改写成质量已通过；中间时刻仍无真实GT。
