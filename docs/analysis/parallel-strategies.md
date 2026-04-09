# LLaMA Factory Megatron-Core 并行策略支持分析

## 概述

本文档深入分析 LLaMA Factory 通过 mcore_adapter 支持的 Megatron-Core 并行策略。

**核心问题**: LLaMA Factory 是否支持 TP/PP/SP/CP/ETP/EP/VPP 并行策略？

**结论**: ✅ 支持 6 种主要并行策略（TP/PP/SP/CP/EP/VPP），⚠️ ETP 不作为独立策略存在

---

## 一、支持的并行策略清单

### 1.1 完整支持矩阵

| 并行策略 | 缩写 | 参数名 | 支持状态 | 说明 |
|---------|------|--------|----------|------|
| Tensor Parallelism | TP | `tensor_model_parallel_size` | ✅ 完全支持 | 在张量维度切分权重矩阵 |
| Pipeline Parallelism | PP | `pipeline_model_parallel_size` | ✅ 完全支持 | 按层切分模型 |
| Sequence Parallelism | SP | `sequence_parallel` | ✅ 完全支持 | 与 TP 配合，序列维度并行 |
| Context Parallelism | CP | `context_parallel_size` | ✅ 完全支持 | 在序列长度维度并行 |
| Expert Parallelism | EP | `expert_model_parallel_size` | ✅ 完全支持 | MoE 模型专家并行 |
| Virtual Pipeline Parallelism | VPP | `virtual_pipeline_model_parallel_size` | ✅ 完全支持 | 虚拟流水线，减少 bubble |
| Expert Tensor Parallelism | ETP | ❌ 无独立参数 | ⚠️ 通过 TP+EP 组合 | 非独立策略 |

### 1.2 参数定义源码

所有并行策略参数定义在 `mcore_adapter.training_args.DistributingParallelArguments`:

```python
# src/mcore_adapter/training_args.py:16-43

@dataclass
class DistributingParallelArguments:
    tensor_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Degree of tensor model parallelism."},
    )
    pipeline_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Degree of pipeline model parallelism."},
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={
            "help": "Makes tensor parallelism more memory efficient for LLMs (20B+) "
                   "by parallelizing layer norms and dropout sequentially."
        },
    )
    virtual_pipeline_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Num of virtual pipeline in a pipeline."},
    )
    context_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Degree of context parallelism."},
    )
    expert_model_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Degree of expert model parallelism."},
    )
```

---

## 二、各并行策略详解

### 2.1 TP (Tensor Parallelism) - 张量并行

**参数**: `tensor_model_parallel_size`

**原理**:
- 在张量维度切分单层的权重矩阵
- 例如：将 `[hidden_size, hidden_size]` 的权重切分为 N 份
- 每个设备持有权重的一部分

**配置示例**:
```yaml
tensor_model_parallel_size: 4  # 4 路张量并行
```

**适用场景**:
- 单层模型太大，无法放入单 GPU
- 需要细粒度的并行控制

**显存节省**:
- 单层权重显存: `1 / TP`
- 激活显存: 取决于是否启用 SP

**通信开销**:
- 前向/反向传播中的 All-Reduce 操作
- 通信量: `O(batch_size × seq_len × hidden_size)`

**LLaMAFactory 实际使用**:
- Qwen2-VL: `TP=4, PP=2, SP=true`
- Qwen3-MoE: `TP=1, PP=4, EP=2`

### 2.2 PP (Pipeline Parallelism) - 流水线并行

**参数**: `pipeline_model_parallel_size`

**原理**:
- 按层切分模型到不同设备
- 例如：32 层模型在 4 个设备上，每个设备 8 层
- 使用 micro-batch 减少 pipeline bubble

**配置示例**:
```yaml
pipeline_model_parallel_size: 4  # 4 路流水线并行
gradient_accumulation_steps: 8   # 使用 micro-batch 减少 bubble
```

**流水线调度策略**:
```yaml
# 自定义流水线分层
pipeline_model_parallel_layout: "E,t*3|t*4|t*5,L"
# E: embedding, t: transformer layer, L: loss layer
# | 分隔不同的 pipeline stage
```

**适用场景**:
- 模型层数多，但单层不大
- 多节点训练

**显存节省**:
- 模型权重显存: `1 / PP`
- 激活显存: 仅保留当前 stage 的激活

**通信开销**:
- Pipeline stage 间的 P2P 通信
- 通信量: `O(batch_size × seq_len × hidden_size × num_microbatches)`

**Bubble Time**:
- 理论最小 bubble: `(PP - 1) / (num_microbatches + PP - 1)`
- 增加 gradient_accumulation_steps 可减少 bubble

**LLaMAFactory 实际使用**:
- Qwen3-MoE: `PP=4` (8 GPUs, 每个 stage 2 GPUs)
- Qwen2-VL: `PP=2`

### 2.3 SP (Sequence Parallelism) - 序列并行

**参数**: `sequence_parallel: true/false`

**原理**:
- 必须与 TP 一起使用 (`TP > 1`)
- 在序列维度切分 LayerNorm 和 Dropout 的输入
- 减少激活显存占用

**配置示例**:
```yaml
tensor_model_parallel_size: 4
sequence_parallel: true  # 启用序列并行
```

**前提条件**:
```python
# 仅在 TP > 1 时生效
assert tensor_model_parallel_size > 1, "SP requires TP > 1"
```

**显存节省**:
- LayerNorm/Dropout 激活显存: `1 / TP`
- 总激活显存减少: ~20-30%

**适用场景**:
- 大模型训练 (20B+ 参数)
- 长序列训练 (4k+ tokens)
- 显存紧张的场景

**实现细节**:
```python
# src/mcore_adapter/training_args.py:25-31
sequence_parallel: bool = field(
    default=False,
    metadata={
        "help": "Makes tensor parallelism more memory efficient for LLMs (20B+) "
               "by parallelizing layer norms and dropout sequentially."
    },
)
```

**LLaMAFactory 实际使用**:
- Qwen2-VL: `TP=4, SP=true`
- Qwen3-MoE: `TP=1, SP=false` (TP=1 时 SP 无效)

### 2.4 CP (Context Parallelism) - 上下文并行

**参数**: `context_parallel_size`

**原理**:
- 在序列长度维度进行并行
- 将长序列切分到多个设备
- 主要用于超长上下文训练 (32k+ tokens)

**配置示例**:
```yaml
context_parallel_size: 2  # 2 路上下文并行
cutoff_len: 32768         # 32k 上下文
```

**适用场景**:
- 超长上下文训练 (32k, 128k, 1M tokens)
- Ring Attention 机制
- 单序列无法放入单 GPU 显存

**显存节省**:
- 序列维度激活显存: `1 / CP`

**通信开销**:
- 序列间的 Ring All-Gather
- 通信量: `O(batch_size × (seq_len / CP) × hidden_size)`

**实现位置**:
```python
# src/mcore_adapter/parallel_functions/context_parallel.py
class _ContextParallelGather(torch.autograd.Function):
    def forward(ctx, context_parallel_input, parallel_dim=-1):
        group = mpu.get_context_parallel_group()
        world_size = mpu.get_context_parallel_world_size()
        # All-gather across context parallel group
        ...
```

**与 SP 的区别**:
| 特性 | SP | CP |
|------|----|----|
| 依赖 | 必须配合 TP | 独立使用 |
| 并行维度 | Transformer 内部组件 | 整个序列 |
| 主要优化 | 激活显存 | 长序列支持 |
| 通信模式 | 与 TP 共享 | Ring Attention |

**注意事项**:
```python
# 与某些 MoE dispatcher 不兼容
if variable_seq_lengths and moe_token_dispatcher_type == "allgather":
    raise ValueError("allgather dispatcher不支持可变序列长度")
```

### 2.5 EP (Expert Parallelism) - 专家并行

**参数**: `expert_model_parallel_size`

**原理**:
- 专门用于 MoE (Mixture-of-Experts) 模型
- 将不同的专家分配到不同设备
- 每个设备只持有部分专家的权重

**配置示例**:
```yaml
expert_model_parallel_size: 2  # 2 路专家并行
moe_token_dispatcher_type: alltoall  # Token 路由方式
moe_grouped_gemm: true               # 专家计算优化
```

**MoE 相关配置**:
```yaml
# Token 分发策略
moe_token_dispatcher_type: alltoall  # 或 allgather

# 专家计算优化
moe_grouped_gemm: true               # 组合 GEMM kernel

# 专家容量控制
moe_expert_capacity_factor: 1.25    # 每个专家容量因子
moe_token_drop_policy: probs         # Token 丢弃策略

# 路由器精度
moe_router_dtype: fp32               # 提高路由稳定性
```

**适用场景**:
- MoE 模型 (Qwen3-MoE, Mixtral, DeepSeek-V3)
- 专家数量 > GPU 数量

**显存节省**:
- 专家权重显存: `1 / EP`
- 例如：64 专家模型，EP=8，每个 GPU 只需存储 8 个专家

**通信开销**:
- Token 分发: All-to-All 或 All-Gather
- 通信量: `O(batch_size × seq_len × num_experts_per_token)`

**LLaMAFactory 实际使用**:
```yaml
# Qwen3-MoE 配置
model_name_or_path: Qwen/Qwen3-30B-A3B-Instruct-2507
expert_model_parallel_size: 2        # 64 专家，分配到 2 组
moe_grouped_gemm: true
moe_token_dispatcher_type: alltoall
```

**并行度计算**:
```python
# 总 GPUs = DP × TP × PP × EP
# 例如：8 GPUs, TP=1, PP=4, EP=2
# DP = 8 / (1 × 4 × 2) = 1
```

### 2.6 VPP (Virtual Pipeline Parallelism) - 虚拟流水线并行

**参数**: `virtual_pipeline_model_parallel_size`

**原理**:
- 在每个 pipeline stage 中创建多个虚拟 stage
- 交错执行减少 pipeline bubble
- 提高 GPU 利用率

**配置示例**:
```yaml
pipeline_model_parallel_size: 4
virtual_pipeline_model_parallel_size: 2  # 每个物理 stage 2 个虚拟 stage
# 实际 pipeline stages = 4 × 2 = 8
```

**自动推导**:
```python
# 如果指定了 pipeline_model_parallel_layout
pipeline_model_parallel_layout: "E,t*3|t*3|t*3|t*3,L"  # 4 stages
pipeline_model_parallel_size: 2

# 自动计算 VPP
virtual_pipeline_model_parallel_size = 4 / 2 = 2
```

**调度策略**:
```yaml
overlap_p2p_comm: true  # 重叠通信与计算
# 仅在 VPP > 1 时生效
```

**Bubble Time 优化**:
- 无 VPP: `Bubble = (PP - 1) / num_microbatches`
- 有 VPP: `Bubble = (PP - 1) / (num_microbatches × VPP)`

**示例**:
```
PP=4, VPP=1, microbatches=8
Bubble Time = 3 / 8 = 37.5%

PP=4, VPP=2, microbatches=8
Bubble Time = 3 / (8 × 2) = 18.75%
```

**适用场景**:
- Pipeline 并行度较高 (PP > 2)
- 需要减少 bubble time
- 模型层数足够多

**显存开销**:
- 需要同时持有多个虚拟 stage 的激活
- 显存增加: `~VPP × activation_per_layer`

**与 pipeline_model_parallel_layout 的关系**:
```python
# src/mcore_adapter/training_args.py:230-243
if pipeline_model_parallel_layout and pipeline_model_parallel_size:
    num_stages = PipelineParallelLayerLayout.get_num_stages_from_str(
        pipeline_model_parallel_layout
    )
    virtual_pipeline_model_parallel_size = num_stages // pipeline_model_parallel_size

    if virtual_pipeline_model_parallel_size == 1:
        virtual_pipeline_model_parallel_size = None  # 退化为标准 PP
```

---

## 三、关于 ETP (Expert Tensor Parallelism)

### 3.1 为什么 ETP 不是独立策略？

从 `mcore_adapter` 源码分析：

```python
# src/mcore_adapter/training_args.py
# ❌ 没有 expert_tensor_parallel_size 参数
# ✅ 有 tensor_model_parallel_size
# ✅ 有 expert_model_parallel_size
```

**结论**: ETP 不是独立的并行策略，而是 **TP + EP 的组合使用**。

### 3.2 TP + EP 组合实现 ETP 效果

**配置示例**:
```yaml
# MoE 模型同时使用 TP 和 EP
tensor_model_parallel_size: 2   # TP: 张量并行
expert_model_parallel_size: 4   # EP: 专家并行
```

**并行维度**:
- **TP**: 切分每个专家内部的权重矩阵
- **EP**: 切分不同的专家到不同设备

**组合效果**:
```
假设: 64 专家, 每个专家 14B 参数
TP=2, EP=4
- 每个 GPU 持有: 64 / 4 = 16 个专家
- 每个专家的权重被 TP 切分为 2 份
- 总 GPUs = DP × TP × PP × EP
```

### 3.3 实现细节

**LoRA + EP 支持**:
```python
# src/mcore_adapter/adapters/lora_layer.py:378-397
class LoraRouterParallelLinear(LoraParallelLinear):
    """LoRA layer for TopKRouter"""

    def _create_lora_layers(self, r, lora_bias, **kwargs):
        # 支持路由器的 LoRA，与 EP 兼容
        ...
```

**专家并行通信**:
```python
# 专家权重分布在 EP 维度
# Token 通过 AlltoAll 或 AllGather 路由到对应专家
# 计算完成后 AlltoAll/AllGather 回传
```

### 3.4 为什么通常说 ETP？

在一些文献和讨论中，"ETP" 指的是：
- **Expert-level Tensor Parallelism**: 在专家内部使用张量并行
- 与 **Expert Parallelism** 的区别：
  - EP: 切分专家到不同设备
  - ETP: 切分专家内部的张量（实际就是 TP+EP）

**实际使用建议**:
- 描述时使用: "TP + EP" 更准确
- 避免使用: "ETP" (容易混淆)

---

## 四、并行策略组合模式

### 4.1 并行度计算公式

```python
total_gpus = DP × TP × PP × CP × EP

# DP (Data Parallelism) 是自动推导的
DP = total_gpus / (TP × PP × CP × EP)
```

### 4.2 常见组合模式

#### 模式 1: 纯 DP
```yaml
# 8 GPUs, 纯数据并行
# DP=8, TP=1, PP=1, EP=1, CP=1
```

**适用**: 小模型 (< 7B), 显存充足

#### 模式 2: DP + TP
```yaml
# 8 GPUs
tensor_model_parallel_size: 4
# DP=2, TP=4
```

**适用**: 中等模型 (7B-30B), 单层较大

#### 模式 3: DP + PP
```yaml
# 8 GPUs
pipeline_model_parallel_size: 4
# DP=2, PP=4
```

**适用**: 深度模型 (层数多), 单层不大

#### 模式 4: TP + PP + DP
```yaml
# 8 GPUs
tensor_model_parallel_size: 2
pipeline_model_parallel_size: 2
# DP=2, TP=2, PP=2
```

**适用**: 大模型 (30B-70B)

#### 模式 5: TP + PP + EP (MoE)
```yaml
# 8 GPUs, MoE 模型
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 4
expert_model_parallel_size: 2
# DP=1, TP=1, PP=4, EP=2
```

**适用**: MoE 模型 (Qwen3-MoE, Mixtral, DeepSeek-V3)

**LLaMAFactory 实际配置**:
```yaml
# Qwen3-30B-A3B-Instruct (MoE)
model_name_or_path: Qwen/Qwen3-30B-A3B-Instruct-2507
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 4
expert_model_parallel_size: 2
# 8 GPUs: DP=1, TP=1, PP=4, EP=2
# global_batch_size = (8 // 2 // 4) * 1 * 8 = 8
```

#### 模式 6: TP + PP + SP (视觉模型)
```yaml
# 8 GPUs, 视觉语言模型
tensor_model_parallel_size: 4
sequence_parallel: true
pipeline_model_parallel_size: 2
# DP=1, TP=4, PP=2
```

**适用**: 多模态大模型 (Qwen2-VL, LLaVA)

**LLaMAFactory 实际配置**:
```yaml
# Qwen2-VL-7B-Instruct
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
tensor_model_parallel_size: 4
sequence_parallel: true
pipeline_model_parallel_size: 2
# 8 GPUs: DP=1, TP=4, PP=2
```

#### 模式 7: 超长上下文 (TP + CP)
```yaml
# 8 GPUs, 超长上下文
tensor_model_parallel_size: 2
context_parallel_size: 4
cutoff_len: 131072  # 128k context
# DP=1, TP=2, CP=4
```

**适用**: 超长上下文训练 (32k+)

### 4.3 组合约束条件

**约束 1**: SP 依赖 TP
```python
if sequence_parallel:
    assert tensor_model_parallel_size > 1, "SP requires TP > 1"
```

**约束 2**: VPP 依赖 PP
```python
if virtual_pipeline_model_parallel_size:
    assert pipeline_model_parallel_size > 1, "VPP requires PP > 1"
```

**约束 3**: EP 仅用于 MoE
```python
if expert_model_parallel_size:
    assert model_type in ["mixtral", "qwen3_moe", "deepseek_v3"], \
        "EP only for MoE models"
```

**约束 4**: GPU 总数必须被整除
```python
assert total_gpus % (TP * PP * CP * EP) == 0, \
    "total_gpus must be divisible by TP*PP*CP*EP"
```

**约束 5**: CP 与某些 dispatcher 不兼容
```python
if variable_seq_lengths and moe_token_dispatcher_type == "allgather":
    raise ValueError("allgather dispatcher不支持可变序列长度")
```

---

## 五、LLaMAFactory 中的实际应用

### 5.1 配置示例对比

#### 示例 1: Qwen3-MoE (MoE 模型)
```yaml
# examples/megatron/qwen3_moe_full.yaml
model_name_or_path: Qwen/Qwen3-30B-A3B-Instruct-2507

# 并行配置
tensor_model_parallel_size: 1          # TP=1 (单层不大)
sequence_parallel: false               # SP 需要 TP>1
pipeline_model_parallel_size: 4        # PP=4 (模型深度大)
expert_model_parallel_size: 2          # EP=2 (64 专家分 2 组)

# MoE 优化
moe_grouped_gemm: true                 # 组合 GEMM
moe_token_dispatcher_type: alltoall    # Token 路由

# 重计算
recompute_granularity: full            # 全量重计算

# GPU 配置: 8 * 78GB
# DP = 8 / (1*4*2) = 1
# global_batch_size = 1 * 1 * 8 = 8
```

**并行分析**:
- **PP=4**: 模型切分为 4 个 stage，每个 stage 占 2 GPUs
- **EP=2**: 64 个专家分为 2 组，每组 32 专家
- **DP=1**: 无数据并行（单副本）

#### 示例 2: Qwen2-VL (视觉语言模型)
```yaml
# examples/megatron/qwen2_vl_full.yaml
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct

# 并行配置
tensor_model_parallel_size: 4          # TP=4 (视觉模型大)
sequence_parallel: true                # SP=true (配合 TP)
pipeline_model_parallel_size: 2        # PP=2

# 视觉配置
image_max_pixels: 262144               # 图像分辨率
video_max_pixels: 16384                # 视频分辨率

# GPU 配置: 8 GPUs
# DP = 8 / (4*2) = 1
# global_batch_size = 1 * 1 * 2 = 2
```

**并行分析**:
- **TP=4**: 单层权重切分为 4 份（视觉编码器参数大）
- **SP=true**: LayerNorm/Dropout 在序列维度并行
- **PP=2**: 模型切分为 2 个 stage

**显存优化**:
- TP=4: 权重显存减少 75%
- SP=true: 激活显存额外减少 20-30%
- PP=2: 仅保留当前 stage 的权重

### 5.2 性能优化配置

#### 优化 1: 通信-计算重叠
```yaml
# 分布式优化器
use_distributed_optimizer: true
overlap_grad_reduce: true              # 梯度规约重叠
overlap_param_gather: true             # 参数收集重叠

# Pipeline 通信重叠
overlap_p2p_comm: true                 # P2P 通信重叠 (需要 VPP>1)
```

#### 优化 2: 算子融合
```yaml
bias_activation_fusion: true           # Bias + Activation 融合
apply_rope_fusion: true                # RoPE 算子融合
```

#### 优化 3: 重计算策略
```yaml
recompute_granularity: full            # full 或 selective
recompute_method: uniform              # uniform 或 recompute
recompute_num_layers: 4                # 重计算的层数
```

#### 优化 4: MoE 专属优化
```yaml
moe_grouped_gemm: true                 # 组合 GEMM
moe_token_dispatcher_type: alltoall    # AlltoAll (比 AllGather 快)
moe_router_dtype: fp32                 # 路由器高精度
moe_shared_expert_overlap: false       # 共享专家重叠
```

### 5.3 配置调优建议

#### 场景 1: 7B 模型，8 GPUs
```yaml
# 推荐配置
tensor_model_parallel_size: 2
pipeline_model_parallel_size: 2
sequence_parallel: true
# DP=2, TP=2, PP=2
```

#### 场景 2: 30B MoE 模型，8 GPUs
```yaml
# 推荐配置 (如 Qwen3-MoE)
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 4
expert_model_parallel_size: 2
# DP=1, TP=1, PP=4, EP=2
```

#### 场景 3: 70B 模型，16 GPUs
```yaml
# 推荐配置
tensor_model_parallel_size: 4
pipeline_model_parallel_size: 4
sequence_parallel: true
# DP=1, TP=4, PP=4
```

#### 场景 4: 超长上下文，16 GPUs
```yaml
# 推荐配置
tensor_model_parallel_size: 2
context_parallel_size: 4
pipeline_model_parallel_size: 2
cutoff_len: 131072  # 128k
# DP=1, TP=2, CP=4, PP=2
```

---

## 六、性能与显存分析

### 6.1 显存节省对比

| 配置 | 权重显存 | 激活显存 | 优化器状态 | 总显存 |
|------|---------|---------|-----------|--------|
| Baseline (DP only) | 100% | 100% | 100% | 100% |
| TP=2 | 50% | 100% | 50% | ~67% |
| TP=2, SP=true | 50% | 50% | 50% | ~50% |
| PP=2 | 50% | 50% | 50% | ~50% |
| TP=2, PP=2 | 25% | 50% | 25% | ~33% |
| TP=4, PP=4 | 6.25% | 25% | 6.25% | ~12.5% |

**注**: 使用 Distributed Optimizer 时，优化器状态也按并行度切分

### 6.2 通信开销对比

| 并行策略 | 通信类型 | 通信量 | 频率 |
|---------|---------|--------|------|
| DP | All-Reduce | `O(model_size)` | 每个 step |
| TP | All-Reduce | `O(batch × seq × hidden)` | 每层前向/反向 |
| PP | P2P Send/Recv | `O(batch × seq × hidden)` | Pipeline stages 间 |
| SP | 与 TP 共享 | - | - |
| CP | Ring All-Gather | `O(batch × seq / CP × hidden)` | Attention 计算 |
| EP | All-to-All | `O(batch × seq × num_experts)` | Token 路由 |

### 6.3 吞吐量优化

**提升吞吐的配置**:
```yaml
# 增加 micro-batch 数量
gradient_accumulation_steps: 16       # 减少 pipeline bubble

# VPP 减少 bubble
virtual_pipeline_model_parallel_size: 2

# 通信-计算重叠
overlap_grad_reduce: true
overlap_param_gather: true
overlap_p2p_comm: true

# 算子融合
bias_activation_fusion: true
apply_rope_fusion: true

# MoE 优化
moe_grouped_gemm: true
```

**Bubble Time 优化**:
```
无 VPP:
Bubble = (PP - 1) / num_microbatches

有 VPP:
Bubble = (PP - 1) / (num_microbatches × VPP)

示例: PP=4, microbatches=16
- 无 VPP: 3/16 = 18.75%
- VPP=2: 3/32 = 9.375%
```

---

## 七、故障排查指南

### 7.1 常见错误

#### 错误 1: SP 启用但 TP=1
```
AssertionError: sequence_parallel requires tensor_model_parallel_size > 1
```

**解决**:
```yaml
tensor_model_parallel_size: 2  # 必须 >1
sequence_parallel: true
```

#### 错误 2: GPU 数量不匹配
```
AssertionError: total_gpus must be divisible by TP*PP*CP*EP
```

**解决**:
```yaml
# 8 GPUs, TP=2, PP=3 会报错
# 改为: TP=2, PP=2, EP=2 或其他组合
```

#### 错误 3: VPP 与 PP 不匹配
```
AssertionError: pipeline_model_parallel_layout length must be divisible by PP
```

**解决**:
```yaml
pipeline_model_parallel_layout: "E,t*3|t*4|t*5|t*6,L"  # 4 stages
pipeline_model_parallel_size: 2
# virtual_pipeline_model_parallel_size 自动推导为 2
```

#### 错误 4: MoE dispatcher 与 variable_seq_lengths 冲突
```
ValueError: allgather dispatcher does not support variable sequence length
```

**解决**:
```yaml
moe_token_dispatcher_type: alltoall  # 改用 alltoall
variable_seq_lengths: true
```

### 7.2 性能调优检查清单

- [ ] TP/PP/EP 组合是否合理？
- [ ] 是否启用 SP (当 TP>1 时)？
- [ ] gradient_accumulation_steps 是否足够大？
- [ ] 是否启用 VPP 减少 bubble？
- [ ] 是否启用通信-计算重叠？
- [ ] 是否启用算子融合？
- [ ] MoE 模型是否使用 alltoall dispatcher？
- [ ] 是否启用 Distributed Optimizer？
- [ ] 重计算粒度是否合适？

---

## 八、总结

### 8.1 支持矩阵总览

| 并行策略 | 支持状态 | 参数名 | 适用场景 |
|---------|---------|--------|----------|
| TP | ✅ 完全支持 | `tensor_model_parallel_size` | 单层大，中大型模型 |
| PP | ✅ 完全支持 | `pipeline_model_parallel_size` | 层数多，多节点训练 |
| SP | ✅ 完全支持 | `sequence_parallel` | 配合 TP，大模型长序列 |
| CP | ✅ 完全支持 | `context_parallel_size` | 超长上下文 (32k+) |
| EP | ✅ 完全支持 | `expert_model_parallel_size` | MoE 模型 |
| VPP | ✅ 完全支持 | `virtual_pipeline_model_parallel_size` | 减少 pipeline bubble |
| ETP | ⚠️ 非独立策略 | - | 通过 TP+EP 组合实现 |

### 8.2 关键要点

1. **LLaMAFactory 通过 mcore_adapter 完整支持 6 种主要并行策略**
2. **ETP 不是独立策略**，而是 TP + EP 的组合使用
3. **并行度必须满足**: `total_gpus % (TP × PP × CP × EP) == 0`
4. **SP 依赖 TP**: `sequence_parallel` 仅在 `tensor_model_parallel_size > 1` 时生效
5. **VPP 依赖 PP**: `virtual_pipeline_model_parallel_size` 仅在 `pipeline_model_parallel_size > 1` 时生效
6. **EP 仅用于 MoE**: 普通模型不需要设置 `expert_model_parallel_size`

### 8.3 最佳实践建议

**小模型 (< 7B)**:
- 优先使用 DP
- 如果单卡放不下，使用 TP=2 或 TP=4

**中等模型 (7B-30B)**:
- 使用 TP + PP 组合
- 启用 SP 节省激活显存
- 示例: `TP=2, PP=2, SP=true`

**大模型 (30B-70B)**:
- 使用 TP + PP 组合，提高并行度
- 必须启用 SP
- 考虑使用 VPP 减少 bubble
- 示例: `TP=4, PP=4, SP=true, VPP=2`

**MoE 模型**:
- 使用 PP + EP 组合
- TP 通常设为 1（专家本身不大）
- 启用 MoE 专属优化
- 示例: `TP=1, PP=4, EP=2`

**超长上下文**:
- 使用 TP + CP 组合
- 根据上下文长度选择 CP 大小
- 示例: `TP=2, CP=4` (128k context)

**多模态模型**:
- 使用 TP + PP + SP 组合
- 视觉编码器通常较大，需要较高 TP
- 示例: `TP=4, PP=2, SP=true`

---

## 九、参考资源

### 代码位置

**mcore_adapter 并行参数定义**:
- `src/mcore_adapter/training_args.py:16-243`

**并行函数实现**:
- `src/mcore_adapter/parallel_functions/context_parallel.py`
- `src/mcore_adapter/parallel_functions/vocab_parallel.py`

**LLaMAFactory 配置示例**:
- `examples/megatron/qwen3_moe_full.yaml`
- `examples/megatron/qwen2_vl_full.yaml`

### 相关文档

- [Megatron-LM 官方文档](https://github.com/NVIDIA/Megatron-LM)
- [mcore_adapter README](https://github.com/alibaba/roll/tree/main/mcore_adapter)
- [LLaMAFactory Megatron-Core 集成分析](./megatron-core-integration.md)
