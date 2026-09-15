# pytorch环境配置与兼容性记录

日期：2026-09-15。环境：`/opt/miniconda3/envs/pytorch`。本轮按用户新指令安装缺失依赖并维护过时API；上一轮仅文档的执行边界不再限制这些已明确授权的环境修改。

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

## 6. 尚未解决或不在本轮范围内

1. 共享环境原本的`pip check`报`qonnx 1.0.0 requires onnxruntime`；pip另报无效distribution `~vitop`。本轮不安装无关ONNX Runtime、不删除其他工程包残留，两个问题仍存在。
2. 测试有第三方包`pkg_resources`/namespace、WebOb/cgi、protobuf的弃用警告，未导致失败；没有为清警告升级共享依赖。
3. `hmnet/models/base/backbone/old/hmnet.py`有原始语法错误（约848行`> nn.Module`），`old/hmnetL1.py`还引用旧`common.utils/torchtools`。它们未被当前builder/22个配置选择，按历史代码保留；不要对整个仓库声称零语法问题。
4. TensorRT/ONNX/旧`aot_ts_nvfuser`等可选导出路径未验证、未补装。当前环境完成的是原生PyTorch三任务主路径的依赖与API维护。
5. 官方数据、元数据、权重仍未下载。下一阶段应先选择数据集与存储目录并准备数据，再做真实样本HMNet完整前后向/短训练/checkpoint恢复/流式推理。
6. 先前文档中的数据对齐、空监督集合、检测idx_offset、RGB固定通道等静态疑点仍保留。环境通过不代表这些问题消失，更不代表算法精度已复现。

本轮修改主要是环境/API兼容层；未实施EfficientViT、Gray融合或S-FIFO正式算法改造。
