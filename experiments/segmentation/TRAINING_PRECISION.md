# 同步双极性输入实验的训练入口

本分支直接来自v1.2_T，保留DVS LiteMLA M=2、FP32状态/归约、其余BF16、TBPTT=1及同步RGB输入。默认配置为本分支的双通道表示。

请使用[POLARITY_INPUTS.md](POLARITY_INPUTS.md)中的缓存、正式训练、恢复、评估和ONNX命令。父版及异步版本的输出目录/检查点不能用于本实验resume；两个点号版本虽同为2通道，也不能互相resume。
