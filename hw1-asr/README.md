# GLM-ASR 学生作业

本作业通过使用 Triton 和 NVIDIA cuTile 实现语音识别模型，帮助您理解 GPU 内核优化。

## 概述

GLM-ASR 是一个将音频转换为文本的语音到文本模型。本次作业 HW1 包含 Triton 和 cuTile 轨道（示例 + 模板），重点在于性能优化。**您只需选择 Triton/cuTile 之一完成即可**，我们推荐 Triton，因为它在许多硬件上具有更好的兼容性。

您将在本次作业中学到：

- **GPU 内核优化** 基础知识
- **编写 Triton 内核** 用于神经网络工作负载
- **编写 NVIDIA cuTile 内核** 作为替代轨道
- **GPU 推理的性能优化** 技术

## 任务

### 需要做什么

打开您所选轨道的模板，并完成以下文件中的 TODO 部分：

**Triton**
- `glm_asr_triton_template/attention.py`
- `glm_asr_triton_template/layers.py`
- `glm_asr_triton_template/rope.py`

**cuTile**
- `glm_asr_cutile_template/attention.py`
- `glm_asr_cutile_template/layers.py`
- `glm_asr_cutile_template/rope.py`

> [!NOTE]
> 您不仅限于填充现有的 TODO 内核。您可以重构和融合内核（例如，在单个 Triton/cuTile 内核中实现当前跨越多个内核的逻辑）。
> 但是，您必须仅使用 Triton/cuTile 实现内核（不要使用预构建的操作符库，如 PyTorch）。

## 快速入门

### Triton 版本快速入门

从**仓库根目录**（`hw1-asr/` 的上一级）：

# 安装：从仓库根目录设置 Triton 环境（`hw1-asr/` 的上一级）
source utils/setup-triton.sh

# 通过运行参考基线验证环境是否正常工作
./benchmark.sh glm_asr_triton_example

# 在模板中填写代码后，运行端到端测试
./benchmark.sh glm_asr_triton_template

### cuTile 版本快速入门

# 安装：从仓库根目录设置 cuTile 环境（`hw1-asr/` 的上一级）
source utils/setup-cutile.sh

# 通过运行参考基线验证环境是否正常工作
./benchmark.sh glm_asr_cutile_example

# 在模板中填写代码后，运行端到端测试
./benchmark.sh glm_asr_cutile_template

## 描述

> **新手？** 阅读 **[详细学生指南 (GUIDE.md)](GUIDE.md)** 获取逐步指导、内核模式和故障排除提示。

### 目录结构

student_version/
├── glm_asr_triton_example/     # 参考：Triton 基线（Torch + Triton）
├── glm_asr_triton_template/    # 您的工作选项 1：在此处完成 TODO（Triton）
├── glm_asr_cutile_example/     # 参考：示例基线（初始 CuPy + cuTile）
├── glm_asr_cutile_template/    # 您的工作选项 2：在此处完成 TODO（cuTile）
├── glm_asr_scratch/            # 参考：PyTorch 基线
├── demo.py                    # Streamlit 交互式演示
├── benchmark.sh               # benchmark_student.py 的 Shell 封装
├── benchmark_student.py       # Python 基准测试脚本
├── benchmark_detailed.sh      # benchmark_detailed.py 的 Shell 封装
├── benchmark_detailed.py      # 详细操作符分析
├── test_audio.wav             # 测试音频文件
└── test_audio.txt             # 预期转录

### 参考实现

| 版本                     | 描述                                                                 |
| ------------------------ | -------------------------------------------------------------------- |
| `glm_asr_scratch`        | PyTorch 参考：明确显示模型结构（仅供理解）                          |
| `glm_asr_triton_example` | Triton 基线：如果您选择了 **Triton** 轨道，请以此为参考              |
| `glm_asr_cutile_example` | cuTile 基线：如果您选择了 **cuTile** 轨道，请以此为参考              |

> [!IMPORTANT]
> 将您的参考与您的轨道匹配：
> - **Triton 轨道** → 研究 `glm_asr_triton_example/` 作为您的基线
> - **cuTile 轨道** → 研究 `glm_asr_cutile_example/` 作为您的基线

### 学生模板

| 版本                      | 描述                    |
| ------------------------- | ----------------------- |
| `glm_asr_triton_template` | Triton 模板（TODO 内核） |
| `glm_asr_cutile_template` | cuTile 模板（TODO 内核） |

> [!IMPORTANT]
> **最低优化要求（选择您的轨道：Triton 或 cuTile）。**  
> 您的提交应包括**至少以下 3 项优化**（我们将在评分/报告审查期间检查这些内容）：
>
> 1. **调整 tile/block 大小**  
>    - 调整关键的平铺超参数（例如，Triton 中的 `BLOCK_M/BLOCK_N/BLOCK_K`、`num_warps`、`num_stages`；cuTile 中对应的 tile 形状/调度参数）。  
>    - 展示您尝试了**至少 2-3 种配置**并选择了最适合您 GPU 的配置。
>
> 2. **内核融合（至少 1 个融合内核）**  
>    - 融合当前分离的两个或多个操作。  
>    - 目标是减少中间读/写和内核启动开销。
>
> 3. **FlashAttention 风格的注意力机制**  
>    - 为自注意力路径实现 **FlashAttention**（或类似 FlashAttention）内核（具有良好内存效率的流式 softmax、分块 QK^T、数值稳定的 softmax，然后乘以 V）。  
>    - 您可以根据需要重构 `attention.py`，但必须保持结果正确性。

### 关键文件说明

- **layers.py**: 基本神经网络层（Linear, LayerNorm, MLP）
- **attention.py**: 自注意力机制
- **rope.py**: 旋转位置嵌入（RoPE）用于位置编码
- **model.py**: 完整模型架构（AudioEncoder, TextDecoder）
- **weight_loader.py**: 加载预训练权重（无需更改）

## 快速开始

在下方选择一个轨道（优先选择 Triton）。

### 环境设置

从仓库根目录，为您的所选轨道执行设置脚本：

# Triton 轨道
source utils/setup-triton.sh
# 可选：演示依赖（如果尚未安装）
# pip install transformers huggingface_hub streamlit soundfile scipy

# cuTile 轨道
source utils/setup-cutile-fix.sh

`setup-cutile-fix.sh` 安装演示使用的常见 ML 工具：
`transformers`、`huggingface_hub`、`streamlit`、`soundfile`、`scipy`。

### Triton 轨道

1. 测试参考实现：

./benchmark.sh glm_asr_triton_example

2. 测试您的实现：

./benchmark.sh glm_asr_triton_template

3. 检查性能：

./benchmark_detailed.sh glm_asr_triton_template

4. 尝试交互式演示：

streamlit run demo.py

### cuTile 轨道

1. 测试参考实现：

./benchmark.sh glm_asr_cutile_example

2. 测试您的实现：

./benchmark.sh glm_asr_cutile_template

3. 检查性能：

./benchmark_detailed.sh glm_asr_cutile_template

4. 尝试交互式演示：

streamlit run demo.py

### 预期输出

Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS

## 基准测试工具

有两种方式运行基准测试：**Shell 脚本**（便捷封装）和 **Python 脚本**（直接执行）。

### Shell 脚本（推荐给初学者）

Shell 脚本提供带有文件夹验证和帮助消息的用户友好封装。

# 显示可用文件夹
./benchmark.sh

# 基本正确性测试（Triton）
./benchmark.sh glm_asr_triton_template

# 基本正确性测试（cuTile）
./benchmark.sh glm_asr_cutile_template

# 测试基线
./benchmark.sh glm_asr_triton_example
./benchmark.sh glm_asr_cutile_example

# 详细性能分析
./benchmark_detailed.sh glm_asr_triton_template
./benchmark_detailed.sh glm_asr_cutile_template

# 详细性能分析（基线）
./benchmark_detailed.sh glm_asr_triton_example
./benchmark_detailed.sh glm_asr_cutile_example

<<<<<<< Updated upstream
# Optional profiling knobs
./benchmark_detailed.sh glm_asr_triton_template --runs 5
./benchmark_detailed.sh glm_asr_triton_template --seq-len 512
./benchmark_detailed.sh glm_asr_triton_template --audio /path/to/test_audio.wav
=======
# 分析特定操作符
./benchmark_detailed.sh --attention-only
./benchmark_detailed.sh --linear-only
>>>>>>> Stashed changes

# 生成 Nsight Systems 性能分析
./benchmark_detailed.sh glm_asr_triton_template --nsys

### Python 脚本（更多控制）

Python 脚本提供更多选项，可直接使用而无需 Shell。

# 带选项的基本基准测试
python benchmark_student.py glm_asr_triton_template
python benchmark_student.py glm_asr_triton_example --warmup 1 --runs 3
python benchmark_student.py glm_asr_cutile_template
python benchmark_student.py glm_asr_cutile_example --warmup 1 --runs 3
# 详细分析
python benchmark_detailed.py glm_asr_triton_template
python benchmark_detailed.py glm_asr_triton_example
python benchmark_detailed.py glm_asr_cutile_template
python benchmark_detailed.py glm_asr_cutile_example

### Streamlit 演示

用于测试转录的交互式 Web UI：

streamlit run demo.py

选择：`Triton 示例（基线）`、`Triton 模板`、`CuTile 示例（基线）`、`CuTile 模板`、`Scratch（PyTorch）`

### 在您的 PC 上查看 Slurm 作业的 WebUI

首先，从 `streamlit run demo.py` 的输出中检查端口。

然后，如果您使用的是 Slurm，请在您的**登录节点/主节点**上运行 `show_tunnel.sh`。该脚本将扫描您正在运行的作业以获取节点名称（第一个正在运行的作业）。

bash show_tunnel.sh <port>

在 `show_tunnel.sh` 的输出中，您将获得在本地 PC 上运行特定命令并打开网站的说明。

## 提示

1. **研究参考实现**：
   - `glm_asr_triton_example/` - Triton 基线，更容易映射到模板
   - `glm_asr_cutile_example/` - 简单基线，更容易理解

2. **增量测试**：实现每一层后，运行基准测试以检查正确性。

3. **使用 CuPy + Triton**（CuTile）/ **使用 Torch + Triton**（Triton）：实现使用 CuPy 进行 CuTile 内核，使用 Torch + Triton 进行 Triton 内核。关键函数：
   - `cp.matmul()` - 矩阵乘法
   - `cp.einsum()` - 爱因斯坦求和
   - `cp.exp()`、`cp.sqrt()` - 元素级操作

4. **检查形状**：调试时打印张量形状：

   print(f"x.shape = {x.shape}")

5. **理解数据流**：

   音频 (wav) → 音频编码器 → 投影器 → 文本解码器 → 文本

## 常见错误

| 错误         | 解决方案                           |
|--------------|------------------------------------|
| 形状不匹配   | 检查输入/输出维度                 |
| NaN 值       | 检查除零情况，使用 epsilon        |
| 空转录       | 验证注意力掩码和位置 ID           |
| 内存不足     | 减少批次大小或序列长度            |

## 参考资料

- [Triton 文档](https://triton-lang.org/)
- [CuPy 文档](https://docs.cupy.dev/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)

## 有问题？

如果您遇到问题：
1. 首先检查示例实现
2. 验证您的张量形状是否符合预期维度
3. 在办公时间提问

祝您好运！
