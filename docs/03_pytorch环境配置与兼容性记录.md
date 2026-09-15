# pytorch环境配置与兼容性记录

日期：2026-09-15。环境：`/opt/miniconda3/envs/pytorch`。本轮按用户新指令安装缺失依赖并维护过时API；上一轮仅文档的执行边界不再限制这些已明确授权的环境修改。

## 本次补充处理摘要

按后续指令，第6节原第1–3项已处理；第4项暂不处理，第5项只整理下载链接，第6项待真实数据和权重就绪后验证。以下第1–5节保留首次环境维护的历史记录，**当前结果以第7节为准**。补充处理起点commit：`1859ececd837a9ce7919eee020e22e73c40c5288`。

| 本次处理 | 当前结果 |
|---|---|
| qonnx依赖声明、无效`~vitop`残留 | `pip check`通过；保留现有GPU Runtime，残留已备份移出 |
| 第三方弃用警告 | 移除误装的Web Apex；protobuf 5.29.5；严格警告模式下回归通过 |
| 两份old骨干 | 修正语法与历史导入，均可导入；未验证旧算法完整前向 |
| 回归 | CPU 43 passed / 1 skipped；CUDA 1 passed；124个Python文件语法通过 |
| 数据/权重 | 未下载；见[下载清单](04_数据集与权重下载清单.md)，包含MVSEC元数据失效链接说明 |

## 1. 完成结果

初始工作区干净，当前基准commit为 `7ce2fda048a963eba96b246408f216e026296284`（包含上一轮学习文档）。本轮未提交commit、未切换分支。

| 项目 | 实测结果 |
|---|---|
| Python | 3.12.2 |
| PyTorch / torchvision / torchaudio | 2.5.0 / 0.20.0 / 2.5.0，全部保留 |
| Torch CUDA | 12.1，CUDA可用，枚举到8张RTX 4090 |
| NumPy / timm | 1.26.4 / 1.0.15，保留 |
| 新增torch-scatter | 2.1.2+pt25cu121；CPython 3.12、Linux、Torch 2.5 / CUDA 12.1二进制wheel |
| 新增pybind11 | 2.13.6 |
| 已有数据/评估库 | pandas2.2.3、h5py3.12.1、hdf5plugin5.0.0、Pillow10.4.0、OpenCV4.10.0、pycocotools2.0.8；均保留 |
| 已有开发库 | pytest7.4.4，未重新安装 |
| 包版本变化 | 安装前214个有效命名distribution全部保留版本；仅新增上述2个包 |
| GEN1工具 | 已放置官方Prophesee toolbox的src与LICENSE；未改第三方源码 |

查询依据为包元数据、实际import、CUDA小算子结果。未运行真实数据上的HMNet完整前后向、训练、推理或精度评估；没有下载数据集/权重。可以进入数据准备阶段，但不能据此宣称官方算法已跑通。

## 2. 缺失依赖与安装

初始三任务模型导入报 `ModuleNotFoundError: torch_scatter`。先记录全部包版本，再dry-run确认候选，仅安装两个缺失包，使用 `--no-deps` 防止改动共享环境其他依赖。

实际执行的安装命令（根目录，现已完成，无需重复）：

```bash
/opt/miniconda3/envs/pytorch/bin/python -m pip install --no-deps --only-binary=:all: --no-index -f https://data.pyg.org/whl/torch-2.5.0+cu121.html torch-scatter==2.1.2+pt25cu121
/opt/miniconda3/envs/pytorch/bin/python -m pip install --no-deps --only-binary=:all: --index-url https://pypi.org/simple pybind11==2.13.6
```

wheel按照`torch.version.cuda`选择；GPU驱动支持的最高CUDA版本不是PyTorch扩展的选择依据。[PyG官方安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)明确列出Torch2.5对应2.5.0及cu121 wheel组合。

已扩展 [requirements.txt](../requirements.txt) 为当前运行栈清单，加入数据和评估的直接依赖。OpenCV使用4.10系列范围以兼容Conda的`4.10.0`与PyPI带构建后缀的版本；服务器完整版本快照仍保留精确值。

[安装前约束](../requirements/pytorch-baseline-constraints.txt)用于**已存在的这套共享环境**增量维护，包含Conda发行包名/版本，不承诺能单靠pip在另一台机器从零重建。新机器应先明确Python/Torch/CUDA组合，再准备相应环境；不要在base直接安装此清单。

### 官方GEN1工具

源码依赖来自 [Prophesee官方toolbox](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox)，固定commit `c09d34a4fb8dbfd2db7081bf5e26078c2aa94fc7`。本轮下载的是代码工具，不是GEN1数据。

安装位置为`hmnet/utils/psee_toolbox/`，与原HMNet README约定一致，包含`io`、其他上游src文件和LICENSE。该目录沿用原`.gitignore`规则，不纳入Git；[setup_psee_toolbox.py](../scripts/setup_psee_toolbox.py)提供固定版本重建入口，遇到未识别的既有目录会停止，不会覆盖用户文件。已验证主工程需要的`load_td_data`、`filter_boxes`、`reformat_boxes`导入。来源记录见 [psee-toolbox-source.txt](../artifacts/environment/psee-toolbox-source.txt)。

## 3. 兼容性改动

| 原写法/症状 | 当前处理 | 范围和语义 |
|---|---|---|
| `from collections import MutableMapping`在Python3.12导入失败 | `collections.abc.MutableMapping` | `hmnet/utils/common.py`，解决四个Dataset导入链的阻塞 |
| `np.int/np.float`已移除 | 使用对应内置`int/float` | GEN1/DSEC/Eventscape/MVSEC；保持旧别名对应的转换语义，不统一强改成float32 |
| `timm.models.layers`弃用警告 | 改用`timm.layers` | 当前注册骨干的相关import |
| `torch.cuda.amp.autocast/GradScaler`弃用 | `torch.amp.autocast('cuda',...)`、`GradScaler('cuda')` | 三任务train/test、BlockBase和YOLOX匹配；保持AMP开关含义 |
| `SourceFileLoader.load_module()`弃用 | 新增[load_config](../hmnet/utils/config.py)，用spec/module/exec_module | 三任务入口；新命名空间、注册sys.modules支持类pickle，失败时恢复旧模块。参考[Python官方加载示例](https://docs.python.org/3.12/library/importlib.html#importing-a-source-file-directly) |
| 手工TypedStorage共享内存拼批 | Tensor分支委托`torch.utils.data.default_collate` | [collate_keep_dict](../hmnet/dataset/custom_collate_fn.py)保留dict与T×B语义；测试num_workers=0/2 |
| meshgrid未指定indexing | 显式`indexing='ij'` | 保持原坐标顺序，包括事件KV查表网格 |
| `F.upsample` | `F.interpolate` | 分割criterion与PPM，保留mode/align_corners |
| PIL旧枚举位置 | `Image.Resampling.*` | 变换工具中重采样映射 |
| `torch.load`默认值未来变化警告 | 显式`weights_only=False` | 保持Torch2.5原checkpoint读取行为，包括模型/optimizer等；只加载可信的本地训练或官方来源checkpoint |
| Lovasz文件Python3.12警告 | 转义文档字符串中的反斜杠，字符串`is`改`==` | 消除import警告；不修改损失定义 |

本轮没有更换backbone、neck、head、损失定义或训练参数，没有修复上一轮记录的idx_offset、无标签段、跨chunk状态等算法/流程疑点。它们应在真实数据冒烟测试中独立复现和处理。

### MKL/OpenMP多进程问题

初次测试：40通过、1失败、1跳过。失败是`num_workers=2`的DataLoader worker退出，stderr明确为：

```text
MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library
```

使用GNU线程层后，该检查及整套CPU检查通过。新增 [scripts/hmnet-python](../scripts/hmnet-python) 作为项目运行器，在启动解释器**之前**默认设置 `MKL_THREADING_LAYER=GNU`，并补仓库根到`PYTHONPATH`。没有改`.bashrc`、Conda activate脚本、MKL/PyTorch包或其他工程。

运行器保留用户显式设置的线程层；若外部已设为INTEL，本机建议显式覆盖为GNU再运行。仅在测试命令中设OMP/MKL线程数为1，不把训练线程数量永久限制为1。

## 4. 以后如何运行

在工程根目录：

```bash
# 直接使用指定pytorch环境；不要求shell当前激活的是哪个环境
./scripts/hmnet-python -c 'import torch, torch_scatter; print(torch.__version__, torch.version.cuda)'

# 若已有MKL_THREADING_LAYER=INTEL，明确覆盖它
MKL_THREADING_LAYER=GNU ./scripts/hmnet-python -c 'import hmnet.dataset.gen1'

# CPU兼容性检查：不会执行HMNet前向/训练
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./scripts/hmnet-python -B -m pytest tests/test_environment_compatibility.py -q -p no:cacheprovider
```

在任务目录保持原有相对数据/配置路径：

```bash
cd experiments/detection
../../scripts/hmnet-python scripts/train.py --help
# 数据就绪后再按独立smoke配置运行，当前不执行训练命令。
```

如要指定其他已验证解释器，可设置`HMNET_PYTHON=/绝对路径/bin/python`。也可手动`conda activate pytorch`后设置`PYTHONPATH`和`MKL_THREADING_LAYER=GNU`，使用普通python；运行器只是减少遗漏。

CUDA检查需先选空闲GPU，再显式启用：

```bash
# 本轮检查时GPU1空闲；未来需重新核查，不固定占用他人GPU
CUDA_VISIBLE_DEVICES=1 HMNET_TEST_CUDA=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./scripts/hmnet-python -B -m pytest tests/test_environment_compatibility.py -q -p no:cacheprovider -k cuda
```

## 5. 实际验证与证据

测试代码：[test_environment_compatibility.py](../tests/test_environment_compatibility.py)。

| 检查 | 结果 | 证据/范围 |
|---|---|---|
| CPU回归 | 41 passed，1 skipped（CUDA opt-in） | [compatibility-cpu.txt](../artifacts/environment/compatibility-cpu.txt) |
| 配置导入 | 22份全部通过 | 包括检测短序列/TBPTT、分割左右RGB、深度配置；只执行模块定义 |
| 三任务构造 | 检测B3、分割左RGB B3、深度RGB B3成功在CPU构造/初始化 | 检查参数设备/输出通道，未调用HMNet forward |
| DataLoader | 合成Tensor在0/2 worker下拼批一致 | 另外验证变长事件T×B、无标签None、空目标张量保持区别 |
| 配置加载器 | 类pickle、重复加载新命名空间、加载失败回滚通过 | 临时配置文件，不涉及数据 |
| NumPy迁移 | GEN1合成事件时间偏移、极性、xywh→xyxy结果通过 | 直接检查实际_bind函数 |
| checkpoint加载 | 微型Linear层自建可信checkpoint回读匹配 | 验证维护后的加载工具，不代表完整HMNet恢复训练已验证 |
| CUDA扩展 | 1 passed | [compatibility-cuda.txt](../artifacts/environment/compatibility-cuda.txt)：scatter sum/max及梯度、torchvision NMS、FP16 autocast与GradScaler构造；不是性能测试 |
| CLI | 10个入口`--help`返回0 | [cli-checks.txt](../artifacts/environment/cli-checks.txt)，从实际任务工作目录运行；部分脚本help早于模型import，所以不能单靠help判定可训练 |
| Python语法 | 主路径121个`.py`通过内存compile | 排除未注册old骨干和未改第三方toolbox |
| 依赖满足性 | `pip install --dry-run --no-index -r requirements.txt`识别运行依赖均已安装 | 不代表整个共享环境pip check无异常 |
| 版本保护 | 214个既有包无变动 | [package-diff.txt](../artifacts/environment/package-diff.txt) |

安装审计：[before](../artifacts/environment/pytorch-before.txt)、[after](../artifacts/environment/pytorch-after.txt)、[scatter安装报告](../artifacts/environment/scatter-install.txt)、[pybind11安装报告](../artifacts/environment/pybind11-install.txt)。报告使用`.txt`保存结构化JSON，避免仓库原有`*.json`忽略规则隐藏记录。

## 6. 原问题清单的当前状态

1. **已处理**：qonnx使用明确标识的本地GPU依赖修订版`1.0.0+ortgpu`，实际依赖原有`onnxruntime-gpu 1.20.2`；`pip check`返回0。无效`~vitop`目录已备份移出site-packages，正常`nvitop 1.3.2`保留。
2. **已处理原记录中的警告**：移除与NVIDIA Apex同名的Web框架包`apex 0.9.10.dev0`，使timm使用原生PyTorch实现；protobuf升级到5.29.5。HMNet回归及共享依赖检查均在DeprecationWarning/FutureWarning视为错误时通过。未对全环境设置警告屏蔽，也未升级仍留供其他工程使用的Pyramid/WebOb。
3. **已处理所列语法和导入阻塞**：两份`old`文件改用仓库内现有模块，修复`> nn.Module`为`-> nn.Module`，删除未使用的旧工具导入。两份模块已纳入导入回归；没有注册为现行骨干，也未声称历史算法可完整训练。
4. **按指令暂不处理**：TensorRT/ONNX/旧`aot_ts_nvfuser`可选HMNet导出路径。第1项对现有Runtime的合成Add检查仅用于验证qonnx依赖修复，不涉及HMNet导出。
5. **只提供链接，等待用户决定下载**：见[数据集与权重下载清单](04_数据集与权重下载清单.md)。本轮仅读取网页、发布资产清单及HTTP HEAD响应，未下载官方数据、元数据或模型权重。
6. **按指令延后测试**：数据对齐、空监督集合、检测idx_offset、RGB固定通道、跨chunk状态等疑点，待数据与权重准备完毕后，在真实样本完整前后向、短训练、恢复训练及流式推理中复现和判断。本轮不做正式算法改造。

## 7. 第1–3项补充修复记录

### 7.1 共享环境的实际变化

| 包/文件 | 修复前 | 修复后 |
|---|---|---|
| qonnx | 1.0.0，硬编码依赖`onnxruntime`包名 | 1.0.0+ortgpu，依赖`onnxruntime-gpu>=1.16.1` |
| protobuf Python绑定 | 4.25.3，Conda旧C++绑定产生Python3.12弃用警告 | 5.29.5，PyPI二进制wheel，实际后端`upb` |
| apex | 0.9.10.dev0，Pyramid Web工具包 | 卸载；没有安装NVIDIA Apex，timm原生实现可用 |
| 无效nvitop残留 | `~vitop/`、`~vitop-1.3.2.dist-info/` | 已移出site-packages并保存备份 |
| 其他包 | 原有版本 | 均保留，包括Torch2.5、CUDA12.1、NumPy1.26.4、timm1.0.15、onnx1.17.0、onnxruntime-gpu1.20.2、nvitop1.3.2 |

[包差异](../artifacts/environment/followup/package-diff.txt)表明：修改2个包版本、移除1个错误包，未新增其他包；原生`libprotobuf 4.25.3`未更换。GPU Runtime的308个文件逐一校验SHA256，全部未变。

**为什么没有直接安装CPU版onnxruntime：**环境已经有可导入的GPU版。官方要求[同一环境只安装一个Runtime发行包](https://onnxruntime.ai/docs/get-started/with-python.html)，否则两个包会写入同一个Python模块目录。qonnx上游[依赖声明](https://github.com/fastmachinelearning/qonnx/blob/main/setup.cfg)只认CPU发行包名，包名不匹配使pip报告缺依赖。

因此提供[build_qonnx_gpu_wheel.py](../scripts/build_qonnx_gpu_wheel.py)：核验官方qonnx1.0.0 wheel的SHA256，改依赖包名和本地版本标识，重新生成合法wheel及RECORD，再通过pip安装。构建后逐文件验证100个qonnx代码/资源文件未改变。原包来源与哈希见[upstream](../artifacts/environment/followup/qonnx-upstream.txt)，构建见[build](../artifacts/environment/followup/qonnx-build.txt)。未来升级qonnx时需要重新检查上游是否支持GPU依赖，不能直接照搬此补丁到新版本。

protobuf的Python3.12类型弃用问题在[上游issue](https://github.com/protocolbuffers/protobuf/issues/15077)有记录。本机临时目录实测4.25.8仍有同类警告，5.29.5通过严格导入和序列化检查后才安装。使用`--no-deps`，未联动更新Torch、原生libprotobuf或其他依赖。protobuf是共享Python依赖，此次检查覆盖下表所列消费者，其他工程仍应按其自身业务做回归。

### 7.2 安装复现与备份

本机已完成，无需重装。以下命令用于之后在**相同现有环境**重建此次修复，wheel不是数据集/权重：

```bash
mkdir -p /tmp/hmnet-env-followup
./scripts/hmnet-python -m pip download --no-deps --only-binary=:all: --index-url https://pypi.org/simple -d /tmp/hmnet-env-followup qonnx==1.0.0 protobuf==5.29.5
./scripts/hmnet-python scripts/build_qonnx_gpu_wheel.py /tmp/hmnet-env-followup/qonnx-1.0.0-py2.py3-none-any.whl /tmp/hmnet-env-followup
./scripts/hmnet-python -m pip install --no-deps /tmp/hmnet-env-followup/qonnx-1.0.0+ortgpu-py2.py3-none-any.whl /tmp/hmnet-env-followup/protobuf-5.29.5-cp38-abi3-manylinux2014_x86_64.whl
./scripts/hmnet-python -m pip check
```

安装报告：[install](../artifacts/environment/followup/install.txt)、[结构化报告](../artifacts/environment/followup/install-report.txt)。本机恢复材料保存在`.local/environment-backups/20260915-followup/`（Git忽略）：Web Apex原文件归档、原Conda protobuf文件归档、原始/修订qonnx wheel、protobuf新wheel、两份nvitop残留备份。

如确需回退，先卸载修订qonnx/protobuf，再安装备份的原始qonnx wheel，将原protobuf归档按原相对路径恢复到该环境的site-packages；Apex归档也以site-packages为基准保存，包含原console entry point相对路径。回退会重新引入本次已解决的依赖声明/弃用问题。`~vitop`是无效残留，正常使用无需恢复。

原[baseline约束](../requirements/pytorch-baseline-constraints.txt)仅作为首次维护前的历史快照。后续增量维护改用[当前约束](../requirements/pytorch-current-constraints.txt)，**以`-c`使用，不以`-r`安装全部包**。HMNet直接运行依赖仍在`requirements.txt`；qonnx属于共享环境维护，不新增为HMNet必需依赖。

### 7.3 本次验证

测试代码与`artifacts/environment/followup/`审计记录均保留在本机；当前工作区的`.gitignore`忽略`tests/`和`artifacts/`，已保留该设置，因此这些材料不会随普通Git提交自动进入仓库。

| 验证 | 结果与范围 |
|---|---|
| pip check | [退出0，No broken requirements found](../artifacts/environment/followup/pip-check.txt)，无无效distribution警告 |
| CPU严格回归 | [43 passed、1 skipped](../artifacts/environment/followup/compatibility-cpu.txt)；新增两份old导入；包括22份配置、三任务CPU构造和DataLoader两worker |
| CUDA严格回归 | [1 passed](../artifacts/environment/followup/compatibility-cuda.txt)，选当时空闲GPU0；scatter及梯度、NMS、AMP |
| 共享依赖 | [通过](../artifacts/environment/followup/shared-runtime-check.txt)：ORT CPU/QONNX合成Add、protobuf map序列化、TensorBoard/TensorBoardX日志写读、wandb导入 |
| 语法与版本保护 | [124个Python文件compile通过](../artifacts/environment/followup/final-audit.txt)，含old，不含未改第三方toolbox；GPU Runtime 308个文件哈希一致 |
| requirements | [dry-run通过](../artifacts/environment/followup/requirements-check.txt)，使用当前约束、无新增安装需求 |

CPU/CUDA与共享依赖检查均将`DeprecationWarning`和`FutureWarning`作为错误处理，未使用忽略过滤器。可复查：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./scripts/hmnet-python -B -m pytest tests/test_environment_compatibility.py -q -p no:cacheprovider -W error::DeprecationWarning -W error::FutureWarning
```

历史代码本次修复到“语法可编译、模块可导入”。旧文件中另外可见既有运行时缺口，例如`HMBackbone`调用`LatentTrans`时未传入必需的`output_dim`，更新层函数使用未传入的`num_heads`，以及`mvit`分支引用未定义的`MViTStage`。本次未扩展到历史算法恢复；不能用当前主模型测试替代旧模型构造/前向验证。现行22份配置仍选择`HMNet/HMNet1`。
