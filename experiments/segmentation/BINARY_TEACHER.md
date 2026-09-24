# 用同步二值模型监督异步阶段二

支持以 `v1.2_T.1` 的同步50ms双极性二值模型提供分割伪标签。学生仍为原异步B/C，其结构、输入、训练与恢复契约均不变。两个工程此功能源码相同。

## 教师模式

`scripts/pseudo_dsec_async.py` 增加以下模式：

|`--teacher-mode`|教师|额外参数|
|---|---|---|
|`bc`（默认）|原异步B+C|原有B/C checkpoint/data参数|
|`binary`|同步二值模型单教师|binary-checkpoint、binary-data、b-data|
|`binary_c`|同步二值模型+C一致性|binary模式参数及c-checkpoint、c-data|

单教师按自身置信度筛选，双教师要求类别一致且两者均达阈值。相同阈值并不保证相同监督覆盖率，应先审核；不能因为某教师test mIoU较高就跳过审核。

二值教师输入从原始事件重建，复用点号工程 `polarity_counts` 和共同 `read_window`：50ms闭区间、两极性、不截断计数，最后取 `count>0`。不能对RVT缓存求和后取非零代替，因为uint8回绕和区间约定不同。所有8082个真实GT时刻逐一验证与原二值训练缓存相同，补齐另8082个中间时刻；RGB、GT、学生事件缓存不改。

二值教师原来约50ms更新时序状态。推理时分别维护GT相位与中间相位的历史，两条流各约50ms前进，合起来覆盖25ms网格；不会把原记忆间隔直接缩为25ms。每条流只访问过去信息，异常间隔/序列切换清空状态。RGB特征可由固定权重的Add模型缓存复用，不改变有效RGB输入下的同步计算结果；无RGB冷启动用零特征。

此适配不改变模型参数或结构，无需重新训练教师。同步教师在旧RGB/中间相位上存在分布变化，所以审核仍必需。

## 数据准备

在任一异步工程运行；只需准备一次，新目录与学生缓存分开：

```bash
./scripts/hmnet-python scripts/prepare_dsec_async_binary_teacher.py --workers 4
```

默认输出：`/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_Semantic/preprocessed/async_v1/binary_teacher50ms`。已有完整manifest时直接复用；脚本拒绝覆盖完整缓存。中断且尚无manifest可以重跑，重新计算所有样本。manifest绑定时间网格、原二值训练manifest、每个事件文件SHA；教师读入时校验文件内容。

## 当前服务器的固定权重与命令

本地快照/运行结果放在main工程 `artifacts/async_binary_teacher/`。每个新终端先执行：

```bash
source /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet/artifacts/async_binary_teacher/env.sh
cd "$ASYNC_B"
```

此env指向2026-09-24固定的B/C/binary dev-best。B当时尚未完成150轮，用户已同意不再把完成150轮作为准备阶段二的前提；固定best不能被后续训练覆盖。教师不需要B权重，B快照只用于原B+C审核对照及B学生初始化。正式启动训练前以本地结果报告中的`passed`为准。

审核二值+C（示例GPU1，目录已存在audit时复用结果或另选新目录）：

```bash
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python scripts/pseudo_dsec_async.py audit \
  --teacher-mode binary_c \
  --binary-checkpoint "$ASYNC_BINARY_CKPT" --binary-data "$ASYNC_BINARY_DATA" \
  --b-data "$ASYNC_DATA/dsec_async_B" \
  --c-checkpoint "$ASYNC_C_CKPT" --c-data "$ASYNC_DATA/dsec_async_C" \
  --confidence 0.95 --min-precision 0.90 --min-coverage 0.10 \
  --min-class-precision 0.70 --min-class-pixels 1000 \
  --output "$ASYNC_RUN/audit_binary_c"
```

单教师审核改为 `--teacher-mode binary`，输出换成 `$ASYNC_RUN/audit_binary`，可删除两个C参数。阈值保持相同。

审核在dev GT时刻将RGB延迟一张，指标与原审核完全同口径。`class_precision`按真实GT类别分组，至少1000个保留像素的类别须正确率≥70%；总体正确率≥90%、覆盖率≥10%。边界指标只报告。某些小类失败不能被较高的总体像素正确率掩盖；不自动降低门槛。

仅`passed=true`后生成（以下为binary_c示例）：

```bash
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python scripts/pseudo_dsec_async.py generate \
  --teacher-mode binary_c \
  --binary-checkpoint "$ASYNC_BINARY_CKPT" --binary-data "$ASYNC_BINARY_DATA" \
  --b-data "$ASYNC_DATA/dsec_async_B" \
  --c-checkpoint "$ASYNC_C_CKPT" --c-data "$ASYNC_DATA/dsec_async_C" \
  --confidence 0.95 --audit "$ASYNC_RUN/audit_binary_c/audit.json" \
  --output "$ASYNC_PSEUDO"
```

旧审核报告若缺少`teacher_protocol`需要重新审核；不能换教师模式、权重、二值缓存、推理精度、事件manifest或阈值后复用审核。新伪标签仍是已有 `dsec_async_pseudo_v1`，训练器可直接消费；两个学生共用这一份train中间步标签，真实GT不替换。

## 阶段二训练与恢复

必须先有审核通过并完整生成的伪标签manifest。分别在两个终端执行，B旧训练是否停止由当前实验安排决定；保持两个阶段不同输出目录。

```bash
source /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet/artifacts/async_binary_teacher/env.sh
cd "$ASYNC_B"
CUDA_VISIBLE_DEVICES=0 ./scripts/hmnet-python scripts/train_async.py \
  --single --stage 2 --epochs 20 --init-from "$ASYNC_B_CKPT" \
  --pseudo-root "$ASYNC_PSEUDO" --pseudo-weight 0.2
```

```bash
source /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_HMNet/artifacts/async_binary_teacher/env.sh
cd "$ASYNC_C"
CUDA_VISIBLE_DEVICES=1 ./scripts/hmnet-python scripts/train_async.py \
  --single --stage 2 --epochs 20 --init-from "$ASYNC_C_CKPT" \
  --pseudo-root "$ASYNC_PSEUDO" --pseudo-weight 0.2
```

二值教师只提供标签，不能把2通道二值权重当作20/10通道学生的`--init-from`。学习率/20轮预算/两步TBPTT/权重渐增与原阶段二相同。恢复同阶段删除`--init-from`，改`--resume`；使用原有伪标签manifest，不能续训途中换标签。既有阶段一恢复行为未改。

监督源改变后，这属于“更换教师后的异步实验”。若B提前结束阶段一，也应披露两版实际阶段一更新预算。阶段二需同时比较正常RGB、延迟RGB和低RGB频率；dev校准不是独立test，缺少真实中间步GT仍是结论边界。

## 回归

```bash
./scripts/hmnet-python scripts/test_async_binary_teacher.py
```

覆盖两条同步状态流等价性、gap/sequence重置、未来RGB和倒序拒绝、256事件不回绕、闭区间边界、置信一致性、教师契约和审核溯源错配拒绝。运行时必须位于异步工程根目录，不能使用main工程的Python包装器导入另一工程脚本。
