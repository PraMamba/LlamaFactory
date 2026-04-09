# Megatron 集成方案对比分析：mcore_adapter vs Mcore-Bridge

> 作者：LLM Training 框架开发者
> 分析日期：2026-01-02
> 目标：为训练框架开发者提供 Megatron 并行策略集成的选型建议

## 目录

1. [概述](#概述)
2. [核心差异对比](#核心差异对比)
3. [集成方式对比](#集成方式对比)
4. [权重转换机制对比](#权重转换机制对比)
5. [并行策略支持对比](#并行策略支持对比)
6. [LoRA 支持对比](#lora-支持对比)
7. [设计模式与架构对比](#设计模式与架构对比)
8. [扩展性与维护成本对比](#扩展性与维护成本对比)
9. [选型建议](#选型建议)

---

## 概述

### 背景

目前主流的两种为训练框架引入 Megatron 并行策略的方案：

| 方案 | 使用项目 | 核心组件 | 源码路径 |
|------|----------|---------|---------|
| **mcore_adapter** | LLaMA Factory | ROLL Framework 的适配层 | `/home/scbjtfy/ROLL/mcore_adapter` |
| **Mcore-Bridge** | ms-swift | 内置权重转换系统 | `/home/scbjtfy/ms-swift/swift/megatron` |

### 核心问题

两种方案都要解决相同的基础问题：

1. **格式不兼容**: HuggingFace safetensors ↔ Megatron torch_dist checkpoints
2. **权重布局差异**: QKV 分离 vs 融合、MLP 分离 vs 融合、MoE experts 布局
3. **并行策略适配**: TP/PP/SP/CP/EP/VPP 切分与 gather
4. **LoRA 支持**: PEFT 格式 vs Megatron LoRA 格式

但两者的**实现哲学完全不同**。

---

## 核心差异对比

### 本质区别

```
┌────────────────────────────────────────────────────────────────┐
│  mcore_adapter 方案：Wrapper Approach (封装方法)                │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌───────────────┐                   │
│  │ HF Trainer   │────────▶│ mcore_adapter │                   │
│  │              │         │ .McaTrainer   │                   │
│  └──────────────┘         └───────┬───────┘                   │
│                                    │                            │
│                          ┌─────────▼─────────┐                │
│                          │ Megatron-Core     │                │
│                          │ (直接使用原生格式) │                │
│                          └───────────────────┘                │
│                                                                 │
│  特点：                                                         │
│  • 在训练流程层面封装 Megatron-Core                            │
│  • 不做权重转换，直接使用 Megatron 格式                        │
│  • Trainer 层适配器 + 数据层 collator wrapper                  │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  Mcore-Bridge 方案：Bridge Approach (桥接方法)                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌───────────────┐                   │
│  │ HF Format    │◀───────▶│  GPTBridge    │◀────────┐         │
│  │ safetensors  │  双向   │  权重转换系统  │         │         │
│  └──────────────┘  转换   └───────────────┘         │         │
│                                    │                  │         │
│                          ┌─────────▼─────────┐       │         │
│                          │ Megatron Format   │       │         │
│                          │ torch_dist        │       │         │
│                          └───────────────────┘       │         │
│                                                       │         │
│  ┌──────────────┐                          ┌─────────▼───────┐ │
│  │ 用户训练     │─────────────────────────▶│ Megatron 训练   │ │
│  │ (HF 格式输入)│                          │ (Megatron 格式) │ │
│  └──────────────┘                          └─────────┬───────┘ │
│                                                       │         │
│                                             ┌─────────▼───────┐ │
│                                             │ 自动转回 HF 格式│ │
│                                             └─────────────────┘ │
│                                                                 │
│  特点：                                                         │
│  • 在权重层面建立 HF ↔ Megatron 双向桥梁                       │
│  • 运行时自动转换权重格式                                       │
│  • 用户无感知，输入输出都是 HF 格式                            │
└────────────────────────────────────────────────────────────────┘
```

### 关键决策差异

| 维度 | mcore_adapter | Mcore-Bridge |
|------|---------------|--------------|
| **集成层次** | Trainer/训练流程层 | 权重/模型层 |
| **权重格式** | 仅支持 Megatron 格式 | 自动双向转换 HF ↔ Megatron |
| **用户体验** | 需要预先转换权重 | 无需手动转换，透明 |
| **实现复杂度** | 低（依赖 mcore_adapter） | 高（1481 行 bridge 代码） |
| **依赖关系** | 强依赖外部库 mcore_adapter | 自包含，无外部依赖 |
| **维护成本** | 依赖上游维护 | 需自行维护转换逻辑 |

---

## 集成方式对比

### mcore_adapter 方案 (LLaMA Factory)

**核心机制**: 在训练流程层面引入 `McaTrainer`，替代 HuggingFace 的 `Trainer`

#### 集成点 1: 条件路由

```python
# src/llamafactory/train/tuner.py:69-84
if finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
    if not is_mcore_adapter_available():
        raise ImportError("mcore_adapter is not installed...")

    from .mca import run_sft as run_sft_mca  # 导入 MCA 训练流程
    run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
else:
    run_sft(...)  # 原有 HF 训练流程
```

#### 集成点 2: 参数双轨

```python
# src/llamafactory/hparams/parser.py:56-65
if is_env_enabled("USE_MCA"):
    from mcore_adapter import TrainingArguments as McaTrainingArguments
    _TRAIN_MCA_ARGS = [ModelArguments, DataArguments, McaTrainingArguments, ...]
else:
    _TRAIN_ARGS = [ModelArguments, DataArguments, HFTrainingArguments, ...]
```

#### 集成点 3: 数据适配层

```python
# src/llamafactory/train/mca/workflow.py:58-80
def _data_collator_wrapper(data_collator):
    """适配 HF 和 Megatron-Core 的 label shifting 差异"""
    @functools.wraps(data_collator)
    def wrapper(features):
        # Megatron-Core 在模型内部做 shifting，所以需要调整数据
        for feature in features:
            # 去掉最后一个 input token
            feature["input_ids"] = feature["input_ids"][:-1]
            # 去掉第一个 label token
            feature["labels"] = feature["labels"][1:]
        return data_collator(features)
    return wrapper
```

#### 集成点 4: MCA Trainer

```python
# src/llamafactory/train/mca/workflow.py:130-150
from mcore_adapter.trainer import McaTrainer

# 加载模型（直接使用 Megatron 格式）
model = mcore_adapter.models.AutoModel.from_pretrained(
    model_args.model_name_or_path,
    # 不需要转换权重，模型已经是 Megatron 格式
)

# 创建 Trainer
trainer = McaTrainer(
    model=model,
    args=training_args,  # McaTrainingArguments
    data_collator=_data_collator_wrapper(data_collator),
    **trainer_kwargs
)

trainer.train()
```

**优势**:
- ✅ 实现简单：核心代码仅 ~300 行 (workflow.py)
- ✅ 依赖清晰：所有 Megatron 逻辑由 mcore_adapter 处理
- ✅ 最小侵入：新增独立模块，原有代码几乎不变

**劣势**:
- ❌ 强依赖外部库：mcore_adapter 必须安装
- ❌ 需要手动转换权重：训练前必须先转换模型为 Megatron 格式
- ❌ 用户体验断层：需要额外学习 mcore_adapter 的使用

---

### Mcore-Bridge 方案 (ms-swift)

**核心机制**: 在模型加载/保存时自动进行 HF ↔ Megatron 格式转换

#### 集成点 1: 自动格式检测

```python
# swift/megatron/convert.py:34-60
def convert_hf2mcore(args):
    """HF → Megatron 转换全流程"""

    # 1. 准备 HF 模板模型（meta 设备，不分配内存）
    hf_model, tokenizer = prepare_model_template(args.model_dir)

    # 2. 初始化 Megatron 分布式环境
    initialize_megatron(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        # ... 其他并行参数
    )

    # 3. 创建 Megatron 模型（空白，待填充权重）
    mg_models = get_model(model_provider)

    # 4. 创建 Bridge 并加载权重
    bridge = GPTBridge(args, hf_model)
    bridge.load_weights(mg_models, hf_model_dir=args.model_dir)

    # 5. 保存为 Megatron checkpoint（可选）
    if args.save_mcore_ckpt:
        save_checkpoint(mg_models, optimizer=None, opt_param_scheduler=None)

    # 6. 精度验证（可选）
    if args.test_precision:
        test_convert_precision(args, mg_models, hf_model, tokenizer)
```

#### 集成点 2: 权重加载逻辑

```python
# swift/megatron/model/gpt_bridge.py:1396-1403
def load_weights(self, mg_model, hf_model_dir):
    """从 HF 格式加载权重到 Megatron 模型"""

    # 使用 Lazy Loader 延迟加载（节省内存）
    with SafetensorLazyLoader(hf_model_dir) as loader:
        hf_state_dict = loader.get_state_dict()  # 返回 LazyTensor dict

        # 核心转换逻辑
        self._convert([mg_model], hf_state_dict, to_mcore=True)
```

#### 集成点 3: 核心转换引擎

```python
# swift/megatron/model/gpt_bridge.py:486-612
def _convert(self, mg_models, hf_state_dict, hf_prefix, to_mcore):
    """Template Method: 定义转换算法骨架"""

    # 步骤 1: 预处理（embeddings）
    hf_state_dict.update(self._convert_pre_process(...))

    # 步骤 2: 逐层转换
    for layer_idx in range(num_layers):
        # 判断这一层属于哪个 PP rank
        pp_rank = layer_idx // layers_per_pp_rank

        if pp_rank == self.pp_rank:  # 只转换属于当前 rank 的层
            mg_layer = mg_models[0].decoder.layers[layer_idx % layers_per_pp_rank]

            # 转换 Attention
            self._set_attention_state(mg_layer.self_attention, hf_state_dict, ...)

            # 转换 MLP
            self._set_mlp_state(mg_layer.mlp, hf_state_dict, ...)

            # 转换 MoE（如果有）
            if hasattr(mg_layer, 'moe'):
                self._set_moe_state(mg_layer.moe, hf_state_dict, ...)

    # 步骤 3: 后处理（output layer）
    hf_state_dict.update(self._convert_post_process(...))
```

#### 集成点 4: TP 自动切分

```python
# swift/megatron/model/gpt_bridge.py:153-193
def _set_weight(self, mg_weight, hf_weight, mg_key):
    """设置权重 + 自动 TP 切分"""

    # 1. 判断切分维度
    tp_dim = self._get_tp_split_dim(mg_key)
    # 例如: linear_qkv.weight → dim=0 (Column Parallel)
    #      linear_proj.weight → dim=1 (Row Parallel)

    # 2. 执行 TP 切分
    if tp_dim is not None:
        hf_weight = self._split_tp(hf_weight, tp_dim, is_expert=False)
        # 例如: [3072, 4096] → TP=4 → [768, 4096] per rank

    # 3. 复制到 Megatron 参数
    mg_weight.data.copy_(hf_weight)
```

#### 集成点 5: 保存回 HF 格式

```python
# swift/megatron/model/gpt_bridge.py:1421-1475
def save_weights(self, mg_models, output_dir):
    """导出 Megatron 模型为 HF 格式"""

    with StreamingSafetensorSaver(output_dir, max_shard_size='5GB') as saver:
        # 使用 Generator 逐个生成权重（节省内存）
        for key, tensor in self.export_weights(mg_models):
            saver.save_tensor(key, tensor)

        # 自动生成 model.safetensors.index.json
        saver.finalize()
```

**优势**:
- ✅ 用户体验极佳：输入 HF 模型，输出 HF 模型，全程透明
- ✅ 无需手动转换：`--load_safetensors true --save_safetensors true` 即可
- ✅ 自包含实现：不依赖外部库，完全掌控
- ✅ 支持 LoRA：PEFT 格式自动转换

**劣势**:
- ❌ 实现复杂：gpt_bridge.py 1481 行，维护成本高
- ❌ 内存开销：权重转换过程需要额外内存（虽然有懒加载优化）
- ❌ 新模型适配：每个新架构需要写对应的 Bridge 子类

---

## 权重转换机制对比

### 转换时机对比

| 方案 | 转换时机 | 转换工具 | 用户操作 |
|------|----------|---------|---------|
| **mcore_adapter** | 训练前一次性转换 | `mcore_adapter.convert` | 手动运行转换脚本 |
| **Mcore-Bridge** | 训练时自动转换 | GPTBridge.load_weights | 无需手动操作 |

### QKV 融合对比

**HuggingFace 格式**:
```python
{
    'layers.0.self_attn.q_proj.weight': [4096, 4096],  # Q
    'layers.0.self_attn.k_proj.weight': [512, 4096],   # K (GQA)
    'layers.0.self_attn.v_proj.weight': [512, 4096],   # V
}
```

**Megatron 格式**:
```python
{
    'decoder.layers.0.self_attention.linear_qkv.weight': [4608, 4096],
    # 4608 = 4096 (Q) + 512 (K) + 512 (V)
}
```

#### mcore_adapter 处理方式

```python
# mcore_adapter 内部实现（黑盒，用户不可见）
# 推测：直接在初始化时要求权重已经是融合的 QKV 格式
# 用户需要先运行: mcore_adapter convert --input hf_model --output mcore_model
```

**用户侧工作流**:
```bash
# 步骤 1: 下载 HF 模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# 步骤 2: 转换为 Megatron 格式
mcore_adapter convert \
    --input Qwen/Qwen2.5-7B-Instruct \
    --output ./mcore_models/qwen2.5-7b \
    --tensor-parallel-size 4

# 步骤 3: 使用转换后的模型训练
llamafactory-cli train \
    --model_name_or_path ./mcore_models/qwen2.5-7b \
    --tensor_model_parallel_size 4 \
    --use_mca
```

#### Mcore-Bridge 处理方式

```python
# swift/megatron/model/gpt_bridge.py:512-531
def _set_attention_state(self, mg_attn, hf_state_dict, hf_prefix, to_mcore):
    """HF → Megatron Attention 转换"""

    # 1. 加载分离的 Q/K/V（LazyTensor，延迟加载）
    q_weight = hf_state_dict[f'{hf_prefix}q_proj.weight'].load()  # [4096, 4096]
    k_weight = hf_state_dict[f'{hf_prefix}k_proj.weight'].load()  # [512, 4096]
    v_weight = hf_state_dict[f'{hf_prefix}v_proj.weight'].load()  # [512, 4096]

    # 2. 重组为 query groups（支持 GQA/MQA）
    num_query_groups = self.args.num_query_groups  # 8 (GQA)
    q_dim = 4096 // 8 = 512  # per query group
    kv_dim = 512 // 8 = 64   # per query group

    q_weight = q_weight.reshape(num_query_groups, q_dim, hidden_size)  # [8, 512, 4096]
    k_weight = k_weight.reshape(num_query_groups, kv_dim, hidden_size) # [8, 64, 4096]
    v_weight = v_weight.reshape(num_query_groups, kv_dim, hidden_size) # [8, 64, 4096]

    # 3. 拼接为 QKV
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)  # [8, 640, 4096]
    qkv_weight = qkv_weight.reshape(-1, hidden_size)  # [5120, 4096]

    # 4. 设置到 Megatron 参数（自动处理 TP 切分）
    self._set_weight(mg_attn.linear_qkv.weight, qkv_weight, 'linear_qkv.weight')
    # 如果 TP=4: [5120, 4096] → [1280, 4096] per rank
```

**用户侧工作流**:
```bash
# 直接使用 HF 模型，无需转换
megatron sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --tensor_model_parallel_size 4

# 训练完成后，输出仍是 HF 格式 safetensors
```

### MoE Experts 融合对比

**HuggingFace 格式** (分散的 experts):
```python
{
    'layers.0.mlp.experts.0.gate_proj.weight': [2048, 4096],
    'layers.0.mlp.experts.0.up_proj.weight':   [2048, 4096],
    'layers.0.mlp.experts.1.gate_proj.weight': [2048, 4096],
    # ... 重复 60 个 experts
}
```

**Megatron 格式** (聚合为 3D tensor):
```python
{
    'decoder.layers.0.moe.experts.linear_fc1.weight': [60, 2, 2048, 4096],
    # shape = [num_experts, 2 (gate+up), expert_hidden, hidden]
}
```

#### Mcore-Bridge MoE 转换

```python
# swift/megatron/model/gpt_bridge.py:687-752
def _set_moe_state(self, mg_moe, hf_state_dict, hf_prefix, to_mcore):
    """MoE experts 转换"""

    num_experts = self.args.num_experts  # 60

    # 收集所有 experts 的权重
    gate_weights = []
    up_weights = []

    for expert_idx in range(num_experts):
        gate_w = hf_state_dict[f'{hf_prefix}experts.{expert_idx}.gate_proj.weight'].load()
        up_w = hf_state_dict[f'{hf_prefix}experts.{expert_idx}.up_proj.weight'].load()

        gate_weights.append(gate_w)  # [2048, 4096]
        up_weights.append(up_w)      # [2048, 4096]

    # 聚合为 4D tensor
    gate_weights = torch.stack(gate_weights, dim=0)  # [60, 2048, 4096]
    up_weights = torch.stack(up_weights, dim=0)      # [60, 2048, 4096]

    # 融合 gate 和 up
    fc1_weight = torch.stack([gate_weights, up_weights], dim=1)  # [60, 2, 2048, 4096]

    # 设置到 Megatron 参数（自动处理 EP 切分）
    self._set_weight(
        mg_moe.experts.linear_fc1.weight,
        fc1_weight,
        'experts.linear_fc1.weight',
        is_expert=True  # 使用 EP group 进行切分
    )
    # 如果 EP=2: [60, 2, 2048, 4096] → [30, 2, 2048, 4096] per rank
```

---

## 并行策略支持对比

### 支持矩阵

| 并行策略 | mcore_adapter | Mcore-Bridge | 说明 |
|---------|--------------|--------------|------|
| **TP** (Tensor Parallelism) | ✅ | ✅ | 两者都完整支持 |
| **PP** (Pipeline Parallelism) | ✅ | ✅ | 两者都完整支持 |
| **SP** (Sequence Parallelism) | ✅ | ✅ | 依赖 TP，两者都支持 |
| **CP** (Context Parallelism) | ✅ | ✅ | 超长上下文，都支持 |
| **EP** (Expert Parallelism) | ✅ | ✅ | MoE 模型，都支持 |
| **VPP** (Virtual PP) | ✅ | ✅ | 减少气泡，都支持 |
| **ETP** (Expert TP) | ⚠️ | ⚠️ | 两者都是 TP+EP 组合 |

### TP 处理差异

#### mcore_adapter: 黑盒处理

```python
# 用户侧：只需配置参数，内部自动处理
from mcore_adapter.trainer import McaTrainer

trainer = McaTrainer(
    model=model,  # 模型已经是 Megatron 格式，TP 已切分
    args=training_args  # 包含 tensor_model_parallel_size=4
)

# mcore_adapter 内部:
# - 自动初始化 TP group
# - 自动 scatter weights
# - 自动 all_reduce gradients
```

**优点**: 用户无需关心 TP 实现细节
**缺点**: 调试困难，出错时难以定位

#### Mcore-Bridge: 透明化处理

```python
# swift/megatron/model/gpt_bridge.py:100-151
def _get_tp_split_dim(self, mg_key):
    """判断权重应该在哪个维度进行 TP 切分"""

    # ColumnParallelLinear: 输出维度切分（dim=0）
    dim0_keys = {
        'word_embeddings',      # [vocab, hidden] → 每个 rank 负责部分 vocab
        'linear_qkv',           # [3*hidden, hidden] → 每个 rank 负责部分 heads
        'output_layer',         # LM head
    }

    # RowParallelLinear: 输入维度切分（dim=1）
    dim1_keys = {
        'linear_proj',          # Attention output projection
        'linear_fc2',           # MLP down projection
    }

    # 特殊情况：linear_fc1 是 [2, hidden, ffn_hidden]
    if 'linear_fc1' in mg_key:
        return 1  # 第二维切分

    # 解析 key
    key = mg_key.rsplit('.', 2)[-2]

    if key in dim0_keys:
        return 0
    elif key in dim1_keys:
        return 1
    else:
        return None  # 不切分（例如 layer_norm）

def _split_tp(self, hf_weight, tp_dim, is_expert):
    """执行 TP 切分"""
    tp_size = self.etp_size if is_expert else self.tp_size
    tp_rank = self.etp_rank if is_expert else self.tp_rank

    # 按指定维度切分
    chunks = torch.chunk(hf_weight, tp_size, dim=tp_dim)
    return chunks[tp_rank]  # 返回当前 rank 的分片

def _all_gather_tp(self, tensor, tp_dim, is_expert):
    """TP 全收集（保存时）"""
    tp_size = self.etp_size if is_expert else self.tp_size
    tp_group = self.etp_group if is_expert else self.tp_group

    if tp_dim is not None and tp_size > 1:
        if tp_dim == 0:
            # 维度 0 切分：使用 all_gather_into_tensor（更高效）
            output = tensor.new_empty([tensor.shape[0] * tp_size, *tensor.shape[1:]])
            dist.all_gather_into_tensor(output, tensor, group=tp_group)
        else:
            # 维度 1 切分：使用 all_gather + cat
            output = [torch.empty_like(tensor) for _ in range(tp_size)]
            dist.all_gather(output, tensor, group=tp_group)
            output = torch.cat(output, dim=tp_dim)
        return output
    return tensor
```

**优点**:
- 清晰可见 TP 切分逻辑
- 方便调试和定制
- 易于扩展新的并行模式

**缺点**:
- 实现复杂度高
- 需要深入理解 Megatron 并行机制

### EP (Expert Parallelism) 处理

两种方案在 EP 处理上有细微差异：

#### mcore_adapter

```python
# 配置文件
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 4
expert_model_parallel_size: 2  # EP=2，60个experts分配到2个rank

# mcore_adapter 自动处理:
# - Rank 0: experts 0-29
# - Rank 1: experts 30-59
```

#### Mcore-Bridge

```python
# swift/megatron/model/gpt_bridge.py:208-220
def _broadcast_ep_pp(self, tensor, is_expert):
    """EP/PP 广播"""

    if is_expert and self.ep_size > 1:
        # Expert weights 需要在 EP group 内广播
        # 因为每个 EP rank 只拥有部分 experts
        # 保存时需要 gather 所有 experts

        src_rank = get_expert_parallel_src_rank()
        dist.broadcast(tensor, src=src_rank, group=self.ep_group)

    # PP 广播（类似逻辑）
    if self.pp_size > 1:
        src_rank = get_pipeline_model_parallel_src_rank()
        dist.broadcast(tensor, src=src_rank, group=self.pp_group)

    return tensor
```

**关键差异**: Mcore-Bridge 显式处理 EP 的 broadcast/gather，而 mcore_adapter 封装在内部。

---

## LoRA 支持对比

### 支持情况

| 方案 | LoRA 支持 | PEFT 兼容 | 实现方式 |
|------|----------|----------|---------|
| **mcore_adapter** | ❌ 不支持 | N/A | 技术限制（Megatron LoRA 实现复杂） |
| **Mcore-Bridge** | ✅ 完整支持 | ✅ 自动转换 | 1. LoraParallelLinear 封装<br>2. PEFT 格式转换 |

### Mcore-Bridge LoRA 实现

#### 1. Megatron LoRA 架构

```python
# swift/megatron/tuners/lora.py
class LoraParallelLinear(nn.Module):
    """LoRA for ColumnParallelLinear / RowParallelLinear"""

    def __init__(self, base_layer, lora_rank, lora_alpha):
        super().__init__()
        self.base_layer = base_layer

        # LoRA 参数遵循 TP 切分规则
        # 如果 base_layer 是 ColumnParallel (dim=0 split):
        #   lora_A: [hidden, rank] - 不切分
        #   lora_B: [rank, hidden_per_tp] - dim=1 切分

        self.lora_A = nn.Parameter(torch.zeros(in_features, lora_rank))
        self.lora_B = nn.Parameter(torch.zeros(lora_rank, out_features // tp_size))
        self.scaling = lora_alpha / lora_rank

    def forward(self, x):
        # Base forward
        output = self.base_layer(x)

        # LoRA forward: x @ lora_A @ lora_B * scaling
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling

        return output + lora_output
```

#### 2. PEFT 格式转换

**HuggingFace PEFT 格式**:
```python
{
    'model.layers.0.self_attn.q_proj.lora_A.default.weight': [4096, 8],
    'model.layers.0.self_attn.q_proj.lora_B.default.weight': [8, 4096],
    'model.layers.0.self_attn.k_proj.lora_A.default.weight': [512, 8],
    'model.layers.0.self_attn.k_proj.lora_B.default.weight': [8, 512],
    'model.layers.0.self_attn.v_proj.lora_A.default.weight': [512, 8],
    'model.layers.0.self_attn.v_proj.lora_B.default.weight': [8, 512],
}
```

**Megatron 格式** (QKV 融合):
```python
{
    'decoder.layers.0.self_attention.linear_qkv.lora_A.default.weight': [4096, 8],
    'decoder.layers.0.self_attention.linear_qkv.lora_B.default.weight': [8, 5120],
    # 5120 = 4096 (Q) + 512 (K) + 512 (V)
}
```

**转换逻辑**:

```python
# swift/megatron/model/gpt_bridge.py:413-422
def _set_attention_lora(self, mg_attn, hf_state_dict, hf_prefix, adapter_name='default'):
    """PEFT → Megatron LoRA 转换"""

    # 1. 加载 HF LoRA 权重
    q_lora_A = hf_state_dict[f'{hf_prefix}q_proj.lora_A.{adapter_name}.weight'].load()
    k_lora_A = hf_state_dict[f'{hf_prefix}k_proj.lora_A.{adapter_name}.weight'].load()
    v_lora_A = hf_state_dict[f'{hf_prefix}v_proj.lora_A.{adapter_name}.weight'].load()

    # 2. 验证约束：Megatron QKV 融合要求 Q/K/V 共享 lora_A
    assert (q_lora_A == k_lora_A).all() and (k_lora_A == v_lora_A).all(), \
        "Q/K/V must share the same lora_A for QKV fusion"

    # 3. 融合 lora_B
    q_lora_B = hf_state_dict[f'{hf_prefix}q_proj.lora_B.{adapter_name}.weight'].load()
    k_lora_B = hf_state_dict[f'{hf_prefix}k_proj.lora_B.{adapter_name}.weight'].load()
    v_lora_B = hf_state_dict[f'{hf_prefix}v_proj.lora_B.{adapter_name}.weight'].load()

    # 按 query groups 重组
    q_lora_B = q_lora_B.reshape(num_query_groups, -1, lora_rank)  # [8, 512, 8]
    k_lora_B = k_lora_B.reshape(num_query_groups, -1, lora_rank)  # [8, 64, 8]
    v_lora_B = v_lora_B.reshape(num_query_groups, -1, lora_rank)  # [8, 64, 8]

    # 拼接
    qkv_lora_B = torch.cat([q_lora_B, k_lora_B, v_lora_B], dim=1)  # [8, 640, 8]
    qkv_lora_B = qkv_lora_B.reshape(-1, lora_rank)  # [5120, 8]

    # 4. 设置到 Megatron LoRA 参数（自动 TP 切分）
    self._set_weight(
        mg_attn.linear_qkv.lora_A[adapter_name].weight,
        q_lora_A,  # Q/K/V 共享
        f'linear_qkv.lora_A.{adapter_name}.weight'
    )
    self._set_weight(
        mg_attn.linear_qkv.lora_B[adapter_name].weight,
        qkv_lora_B,
        f'linear_qkv.lora_B.{adapter_name}.weight'
    )
```

**关键约束**: 由于 Megatron 融合了 QKV，所以 Q/K/V 的 `lora_A` 必须相同！

#### 3. modules_to_save 处理

PEFT 中，某些模块（如 embeddings, lm_head）可能需要完整训练：

```python
# adapter_config.json
{
    "peft_type": "LORA",
    "r": 8,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "modules_to_save": ["embed_tokens", "lm_head"]  # 这些模块完整训练
}
```

**PEFT 格式**:
```python
{
    'model.embed_tokens.weight': [vocab_size, hidden],  # base 权重
    'model.embed_tokens.modules_to_save.default.weight': [vocab_size, hidden],  # 训练后的权重
}
```

**Mcore-Bridge 转换**:
```python
# swift/megatron/model/gpt_bridge.py:248-259
def _set_module(self, mg_module, hf_state_dict, hf_prefix, to_mcore):
    """处理 modules_to_save"""

    if self._is_peft_format:
        # 检查是否有 modules_to_save 后缀
        new_state_dict = {}
        for k, v in hf_state_dict.items():
            if k.startswith(hf_prefix):
                # 移除 .modules_to_save.{adapter_name} 后缀
                new_k = k.replace(f'.modules_to_save.{self._adapter_name}', '')
                new_state_dict[new_k] = v

        # 加载到 Megatron 模块
        mg_module.load_state_dict(new_state_dict, strict=False)
```

---

## 设计模式与架构对比

### mcore_adapter: Adapter Pattern (适配器模式)

```
┌────────────────────────────────────────────┐
│  Adapter Pattern                            │
├────────────────────────────────────────────┤
│                                             │
│  ┌────────────┐         ┌─────────────┐   │
│  │   Client   │────────▶│  Adapter    │   │
│  │ (LLaMAFact)│         │ (mca.workflow)│   │
│  └────────────┘         └──────┬──────┘   │
│                                 │           │
│                        ┌────────▼────────┐ │
│                        │   Adaptee       │ │
│                        │ (McaTrainer)    │ │
│                        └─────────────────┘ │
└────────────────────────────────────────────┘
```

**核心代码**:

```python
# src/llamafactory/train/mca/workflow.py
def run_sft(model_args, data_args, training_args, finetuning_args, callbacks):
    """Adapter: 适配 LLaMAFactory 接口到 mcore_adapter"""

    # 1. 数据准备（LLaMAFactory 原有逻辑）
    template = get_template_and_fix_tokenizer(...)
    dataset_module = get_dataset(...)

    # 2. Data collator wrapper（适配 label shifting）
    data_collator = get_data_collator(...)
    data_collator = _data_collator_wrapper(data_collator)

    # 3. 模型加载（使用 mcore_adapter API）
    model = mcore_adapter.models.AutoModel.from_pretrained(
        model_args.model_name_or_path,
        # ... Megatron 参数
    )

    # 4. Trainer（使用 mcore_adapter Trainer）
    trainer = mcore_adapter.trainer.McaTrainer(
        model=model,
        args=training_args,  # McaTrainingArguments
        data_collator=data_collator,
        **trainer_kwargs
    )

    # 5. 训练（标准 HF API）
    trainer.train()
```

**优势**:
- 薄适配层，核心逻辑委托给 mcore_adapter
- 接口兼容性高
- 易于理解和维护

**劣势**:
- 强耦合 mcore_adapter
- 受限于 mcore_adapter 的能力边界

---

### Mcore-Bridge: Bridge + Template Method Pattern

#### 1. Bridge Pattern (桥接模式)

```
┌────────────────────────────────────────────────────────┐
│  Bridge Pattern                                         │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐         ┌──────────────────┐        │
│  │ Abstraction  │────────▶│  Implementor     │        │
│  │              │         │                  │        │
│  │ convert_hf2  │         │  GPTBridge       │        │
│  │   mcore      │         │  - load_weights  │        │
│  │ convert_mcore│         │  - save_weights  │        │
│  │   2hf        │         │  - _convert()    │        │
│  └──────────────┘         └────────┬─────────┘        │
│                                    │                   │
│              ┌─────────────────────┼─────────────────┐ │
│              │                     │                 │ │
│     ┌────────▼────────┐  ┌────────▼────────┐  ┌────▼─┴───────┐
│     │  GPTBridge      │  │ Qwen2_5VLBridge │  │ DeepSeekV3   │
│     │  (Base)         │  │ (Multimodal)    │  │  Bridge      │
│     └─────────────────┘  └─────────────────┘  └──────────────┘
└────────────────────────────────────────────────────────┘
```

**实现**:

```python
# swift/megatron/model/gpt_bridge.py
class GPTBridge:
    """Base Bridge: 定义转换接口"""

    def load_weights(self, mg_models, hf_model_dir):
        """抽象接口: 加载权重"""
        pass

    def save_weights(self, mg_models, output_dir):
        """抽象接口: 保存权重"""
        pass

    def _convert(self, mg_models, hf_state_dict, to_mcore):
        """核心转换逻辑（Template Method）"""
        pass

# 子类：特定模型架构
class Qwen2_5VLBridge(MultimodalGPTBridge):
    """Refined Abstraction: 多模态模型"""

    hf_layers_prefix = 'model.language_model.layers'

    def __init__(self, args, hf_model):
        super().__init__(args, hf_model)
        # 定义模块映射
        self.module_mapping = {
            'visual': 'vision_model',
            'aligner': 'vision_projection',
        }

    def _convert_pre_process(self, ...):
        """覆盖预处理：处理 vision encoder"""
        # 调用父类处理 text embeddings
        hf_state_dict = super()._convert_pre_process(...)

        # 额外处理 vision encoder
        for hf_name, mg_name in self.module_mapping.items():
            self._set_module(...)

        return hf_state_dict
```

#### 2. Template Method Pattern (模板方法模式)

```python
# swift/megatron/model/gpt_bridge.py:486-612
class GPTBridge:
    def _convert(self, mg_models, hf_state_dict, hf_prefix, to_mcore):
        """Template Method: 定义转换算法骨架"""

        # 步骤 1: 预处理（Hook，子类可覆盖）
        hf_state_dict.update(self._convert_pre_process(...))

        # 步骤 2: 逐层转换（Hook，子类可覆盖）
        for layer_idx in range(num_layers):
            hf_state_dict.update(self._set_layer_state(...))

        # 步骤 3: 后处理（Hook，子类可覆盖）
        hf_state_dict.update(self._convert_post_process(...))

        return hf_state_dict

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix, to_mcore):
        """Hook method: 子类可以覆盖"""
        # Standard layer: Attention + MLP
        self._set_attention_state(...)
        self._set_mlp_state(...)

        # Optional: MoE
        if hasattr(mg_layer, 'moe'):
            self._set_moe_state(...)

    # Hook methods（钩子方法）
    def _convert_pre_process(self, ...):
        """预处理钩子：embeddings"""
        pass

    def _convert_post_process(self, ...):
        """后处理钩子：output layer"""
        pass

    def _set_attention_state(self, ...):
        """Attention 转换钩子"""
        pass

    def _set_mlp_state(self, ...):
        """MLP 转换钩子"""
        pass

# 子类覆盖特定钩子
class DeepSeekV3Bridge(GPTBridge):
    def _set_attention_state(self, mg_attn, hf_state_dict, hf_prefix, to_mcore):
        """覆盖: 处理 MLA (Multi-Latent Attention)"""

        if self._is_mla_attention(mg_attn):
            # DeepSeek-V3 特殊逻辑
            self._set_attention_state_mla(mg_attn, hf_state_dict, hf_prefix, to_mcore)
        else:
            # 标准 attention
            super()._set_attention_state(mg_attn, hf_state_dict, hf_prefix, to_mcore)
```

**优势**:
- 算法骨架固定，易于扩展新模型
- 代码复用率高
- 遵循 OCP (开闭原则)

**劣势**:
- 继承层次复杂
- 新手学习曲线陡峭

#### 3. Strategy Pattern (策略模式) - 并行策略

```python
class GPTBridge:
    def _get_weight(self, mg_weight, mg_key, is_expert=False):
        """根据 is_expert 选择不同的并行策略"""

        # 策略 1: Dense layer (TP)
        if not is_expert:
            tensor = self._all_gather_tp(tensor, tp_dim, is_expert=False)
            tensor = self._broadcast_ep_pp(tensor, is_expert=False)

        # 策略 2: Expert layer (ETP + EP)
        else:
            tensor = self._all_gather_tp(tensor, tp_dim, is_expert=True)
            tensor = self._broadcast_ep_pp(tensor, is_expert=True)

        return tensor
```

#### 4. Iterator Pattern (迭代器模式) - Generator

```python
def export_weights(self, mg_models):
    """Generator: 按需生成权重，避免 OOM"""

    for key, tensor in self._convert(mg_models, to_mcore=False):
        yield key, tensor  # 逐个返回，而非一次性加载全部

    # 调用者逐个消费
    for key, tensor in bridge.export_weights(mg_models):
        saver.save_tensor(key, tensor)
        del tensor  # 立即释放内存
```

#### 5. Lazy Initialization (延迟初始化)

```python
class LazyTensor:
    """延迟加载 HF 权重"""

    def __init__(self, loader):
        self.loader = loader
        self.tensor = None  # 未初始化

    def load(self):
        if self.tensor is None:  # 首次访问时加载
            self.tensor = self.loader()
        return self.tensor

# 使用
hf_state_dict = {
    'layers.0.q_proj.weight': LazyTensor(loader=lambda: load_from_safetensors('q_proj'))
}

# 只在需要时加载
q_weight = hf_state_dict['layers.0.q_proj.weight'].load()  # 此时才从磁盘读取
```

---

## 扩展性与维护成本对比

### 新增模型架构支持

#### mcore_adapter 方案

**步骤**:
1. 等待 mcore_adapter 上游支持新模型
2. 或者提交 PR 到 mcore_adapter 仓库
3. LLaMA Factory 端无需修改（除非有特殊配置）

**代码量**: ~0 行 (LLaMA Factory 端)

**时间成本**: 取决于 mcore_adapter 维护者

**示例** (假设添加 Llama 4):
```python
# mcore_adapter 上游需要添加:
# - Llama4 model config
# - Llama4 model implementation

# LLaMA Factory 端:
# 无需修改！只需更新 mcore_adapter 版本
```

#### Mcore-Bridge 方案

**步骤**:
1. 创建新的 Bridge 子类
2. 实现特定架构的转换逻辑
3. 注册到 model registry

**代码量**: 100-500 行 (取决于架构复杂度)

**时间成本**: 1-3 天 (熟悉开发者)

**示例** (添加 Llama 4):

```python
# swift/megatron/model/gpt/llama4.py
class Llama4GPTBridge(GPTBridge):
    """Llama 4 架构的 Bridge"""

    hf_layers_prefix = 'model.layers'

    def _set_attention_state(self, mg_attn, hf_state_dict, hf_prefix, to_mcore):
        """Llama 4 特定的 attention 转换"""

        if self._has_grouped_query_attention(mg_attn):
            # GQA 逻辑
            super()._set_attention_state(mg_attn, hf_state_dict, hf_prefix, to_mcore)
        else:
            # 标准 MHA
            ...

    def _set_mlp_state(self, mg_mlp, hf_state_dict, hf_prefix, to_mcore):
        """Llama 4 MLP 转换（假设有新的 GLU 变体）"""

        if to_mcore:
            # HF → Megatron
            gate_weight = hf_state_dict[f'{hf_prefix}gate_proj.weight'].load()
            up_weight = hf_state_dict[f'{hf_prefix}up_proj.weight'].load()

            # Llama 4 特殊融合逻辑
            fc1_weight = self._fuse_llama4_mlp(gate_weight, up_weight)

            self._set_weight(mg_mlp.linear_fc1.weight, fc1_weight, 'linear_fc1.weight')
        else:
            # Megatron → HF
            ...

    def _fuse_llama4_mlp(self, gate_weight, up_weight):
        """Llama 4 专有的 MLP 融合逻辑"""
        # 假设 Llama 4 使用新的激活函数
        return torch.stack([gate_weight, up_weight], dim=0)

# swift/megatron/model/__init__.py
MODEL_BRIDGES = {
    'llama': GPTBridge,
    'llama4': Llama4GPTBridge,  # 注册新 Bridge
    'qwen2': GPTBridge,
    'qwen2_5_vl': Qwen2_5VLBridge,
    # ...
}
```

**优势**:
- ✅ 完全自主控制，不依赖第三方
- ✅ 可以快速响应新模型发布

**劣势**:
- ❌ 每个新模型都需要手动适配
- ❌ 维护成本高

---

### 新增并行策略支持

#### mcore_adapter 方案

**场景**: 假设 Megatron-Core 新增 "Data-Dependent Parallelism" (DDP 2.0)

**步骤**:
1. 等待 mcore_adapter 集成新并行策略
2. LLaMA Factory 端添加配置参数

**代码量**: ~10 行 (添加配置参数)

**示例**:
```python
# LLaMA Factory: examples/megatron/qwen_ddp2.yaml
data_dependent_parallel_size: 2  # 新参数
tensor_model_parallel_size: 4
pipeline_model_parallel_size: 2

# mcore_adapter 内部自动处理新并行策略
```

#### Mcore-Bridge 方案

**步骤**:
1. 实现新的并行组初始化
2. 添加新的 split/gather 逻辑
3. 更新 _get_weight / _set_weight 方法

**代码量**: ~200 行

**示例**:
```python
# swift/megatron/model/gpt_bridge.py
class GPTBridge:
    def __init__(self, args):
        # ... 现有并行组
        self.tp_group = get_tensor_model_parallel_group()
        self.pp_group = get_pipeline_model_parallel_group()
        self.ep_group = get_expert_model_parallel_group()

        # 新增: DDP 2.0 并行组
        self.ddp2_group = get_data_dependent_parallel_group()
        self.ddp2_size = get_data_dependent_parallel_world_size()
        self.ddp2_rank = get_data_dependent_parallel_rank()

    def _get_tp_split_dim(self, mg_key):
        """判断切分维度"""
        # ... 现有逻辑

        # 新增: DDP 2.0 特定 keys
        ddp2_keys = {
            'adaptive_weights',  # 假设 DDP 2.0 引入了动态权重
        }
        if mg_key in ddp2_keys:
            return ('ddp2', 0)  # 新的切分标记

        # ... 原有逻辑

    def _split_ddp2(self, hf_weight, ddp2_dim):
        """DDP 2.0 切分逻辑"""
        chunks = torch.chunk(hf_weight, self.ddp2_size, dim=ddp2_dim)
        return chunks[self.ddp2_rank]

    def _all_gather_ddp2(self, tensor, ddp2_dim):
        """DDP 2.0 全收集"""
        if self.ddp2_size > 1:
            output = [torch.empty_like(tensor) for _ in range(self.ddp2_size)]
            dist.all_gather(output, tensor, group=self.ddp2_group)
            return torch.cat(output, dim=ddp2_dim)
        return tensor

    def _set_weight(self, mg_weight, hf_weight, mg_key):
        """更新: 支持 DDP 2.0"""
        split_info = self._get_tp_split_dim(mg_key)

        if isinstance(split_info, tuple) and split_info[0] == 'ddp2':
            # DDP 2.0 切分
            hf_weight = self._split_ddp2(hf_weight, split_info[1])
        else:
            # 原有 TP 切分
            hf_weight = self._split_tp(hf_weight, split_info)

        mg_weight.data.copy_(hf_weight)
```

**优势**:
- ✅ 可以自定义并行策略实现
- ✅ 不依赖上游更新

**劣势**:
- ❌ 需要深入理解新并行策略的通信模式
- ❌ 实现和测试成本高

---

### Bug 修复与调试

#### mcore_adapter 方案

**场景**: 训练中出现权重加载错误

**调试流程**:
1. 检查 LLaMA Factory 端配置
2. 检查 mcore_adapter 版本
3. 查看 mcore_adapter 源码（如果可访问）
4. 提交 issue 到 mcore_adapter 仓库
5. 等待上游修复

**控制力**: ⭐⭐ (依赖上游)

#### Mcore-Bridge 方案

**调试流程**:
1. 检查配置
2. 直接在 gpt_bridge.py 添加 debug 日志
3. 定位到具体的 _set_weight 或 _get_weight 调用
4. 修改逻辑，立即测试
5. 自行修复

**控制力**: ⭐⭐⭐⭐⭐ (完全自主)

**示例**:

```python
# swift/megatron/model/gpt_bridge.py
def _set_weight(self, mg_weight, hf_weight, mg_key):
    """添加调试信息"""

    # 调试日志
    logger.debug(f"Setting weight: {mg_key}")
    logger.debug(f"  HF shape: {hf_weight.shape}")
    logger.debug(f"  MG shape: {mg_weight.shape}")
    logger.debug(f"  TP split dim: {self._get_tp_split_dim(mg_key)}")

    # ... 原有逻辑

    # 验证
    if mg_weight.shape != expected_shape:
        raise ValueError(f"Shape mismatch: expected {expected_shape}, got {mg_weight.shape}")
```

---

### 维护成本对比表

| 维度 | mcore_adapter | Mcore-Bridge |
|------|--------------|--------------|
| **新模型适配** | ⭐⭐⭐⭐⭐ (依赖上游) | ⭐⭐ (手动实现) |
| **并行策略扩展** | ⭐⭐⭐⭐ (依赖上游) | ⭐⭐ (手动实现) |
| **Bug 修复** | ⭐⭐ (等待上游) | ⭐⭐⭐⭐⭐ (立即修复) |
| **文档维护** | ⭐⭐⭐⭐⭐ (无需文档) | ⭐⭐ (需要详细文档) |
| **代码审查** | ⭐⭐⭐⭐⭐ (仅适配层) | ⭐⭐ (大量转换代码) |
| **长期演进** | ⭐⭐⭐ (上游兼容性风险) | ⭐⭐⭐⭐ (自主演进) |

---

## 选型建议

### 决策树

```
是否需要 LoRA 支持？
├── 是 → Mcore-Bridge（mcore_adapter 不支持 LoRA）
└── 否
    └── 是否愿意依赖外部库？
        ├── 是 → mcore_adapter（更简单）
        └── 否
            └── 团队是否有深厚的 Megatron 经验？
                ├── 是 → Mcore-Bridge（更灵活）
                └── 否 → mcore_adapter（学习成本低）
```

### 场景 1: 小型团队 / 快速原型

**推荐**: **mcore_adapter**

**理由**:
1. ✅ 实现简单，200-300 行代码即可集成
2. ✅ 依赖成熟的外部库，bug 少
3. ✅ 无需深入理解权重转换细节
4. ✅ 维护成本低

**适用条件**:
- 不需要 LoRA 支持
- 可以接受预先转换权重的工作流
- 团队规模 < 10 人
- 项目周期 < 6 个月

**示例项目**: LLaMA Factory

---

### 场景 2: 企业级框架 / 长期项目

**推荐**: **Mcore-Bridge**

**理由**:
1. ✅ 用户体验极佳：输入 HF 模型，输出 HF 模型
2. ✅ 完全自主可控，不依赖外部库
3. ✅ 支持 LoRA，满足微调需求
4. ✅ 可自定义扩展，快速响应新架构

**适用条件**:
- 需要 LoRA 支持
- 愿意投入 1-2 人月开发 Bridge 系统
- 团队有 Megatron 专家（至少 1 人）
- 项目周期 > 1 年，需要长期演进

**示例项目**: ms-swift

---

### 场景 3: 研究项目 / 实验性功能

**推荐**: **mcore_adapter**

**理由**:
1. ✅ 快速验证 idea
2. ✅ 无需投入大量工程时间
3. ✅ 依赖稳定的上游实现

**适用条件**:
- 主要目标是验证算法，而非工程化
- 对 LoRA 需求不强
- 愿意接受 mcore_adapter 的限制

---

### 场景 4: 多模态模型 (VL/VLM)

**推荐**: **Mcore-Bridge**

**理由**:
1. ✅ 灵活处理 vision encoder + language model
2. ✅ 通过 `MultimodalGPTBridge` 支持模块化转换
3. ✅ ms-swift 已有 Qwen2-VL、Qwen2.5-VL 实现可参考

**对比**:
- mcore_adapter 对多模态支持不明确（需查阅文档）

---

### 场景 5: MoE 模型

**推荐**: **两者均可**，但建议 **Mcore-Bridge**

**理由**:
1. ✅ Mcore-Bridge 显式处理 EP 的 gather/broadcast
2. ✅ 更容易调试 experts 权重加载问题
3. ⚠️ mcore_adapter 也支持，但内部细节不透明

**配置示例** (Qwen3-MoE):

```yaml
# mcore_adapter 方案
model_name_or_path: ./mcore_models/qwen3-moe-30b  # 预先转换
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 4
expert_model_parallel_size: 2

# Mcore-Bridge 方案
model: Qwen/Qwen3-30B-A3B-Instruct-2507  # HF 模型直接使用
load_safetensors: true
save_safetensors: true
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 4
expert_model_parallel_size: 2
```

---

### 综合对比表

| 维度 | mcore_adapter | Mcore-Bridge | 优势方 |
|------|--------------|--------------|---------|
| **实现复杂度** | ⭐⭐⭐⭐⭐ | ⭐⭐ | mcore_adapter |
| **用户体验** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Mcore-Bridge |
| **LoRA 支持** | ❌ | ✅ | Mcore-Bridge |
| **多模态支持** | ❓ | ✅ | Mcore-Bridge |
| **调试友好性** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Mcore-Bridge |
| **扩展性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Mcore-Bridge |
| **维护成本** | ⭐⭐⭐⭐⭐ | ⭐⭐ | mcore_adapter |
| **学习成本** | ⭐⭐⭐⭐ | ⭐⭐ | mcore_adapter |
| **自主可控** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Mcore-Bridge |

---

## 最终建议

### 对于新训练框架开发者

**如果你正在从零开始构建训练框架**，我的建议是：

#### 阶段 1: MVP (最小可行产品)

**选择**: **mcore_adapter**

**原因**:
- 快速上线，验证市场需求
- 开发周期 1-2 周
- 代码量 < 500 行

#### 阶段 2: 产品化

**评估**: 是否需要以下任一功能？
- LoRA / PEFT 支持
- 多模态模型
- 自定义并行策略
- 极致用户体验（无需手动转换权重）

**如果是** → 迁移到 **Mcore-Bridge**
**如果否** → 继续使用 **mcore_adapter**

#### 阶段 3: 规模化

**选择**: **Mcore-Bridge**

**原因**:
- 企业客户需要完全自主可控的解决方案
- 需要快速响应客户定制需求
- 长期维护成本更低（不依赖外部库）

---

### 实施路线图 (Mcore-Bridge)

如果你决定实现 Mcore-Bridge，建议按以下步骤进行：

#### Week 1-2: 基础架构

1. 实现 `SafetensorLazyLoader` 和 `StreamingSafetensorSaver`
2. 实现 `GPTBridge` 基类（不包含具体转换逻辑）
3. 实现并行组初始化和 TP split/gather

**代码量**: ~500 行

#### Week 3-4: 标准 Transformer 支持

1. 实现 Llama 架构的 Bridge
   - QKV 融合 / 解融合
   - MLP gate_up 融合 / 解融合
   - LayerNorm / RMSNorm 处理
2. 添加精度验证逻辑

**代码量**: ~400 行

#### Week 5-6: MoE 支持

1. 实现 Mixtral / Qwen3-MoE 架构
   - Experts 聚合 / 分散
   - EP 切分 / gather
   - Router 权重处理

**代码量**: ~300 行

#### Week 7-8: LoRA 支持

1. 实现 `LoraParallelLinear`
2. 实现 PEFT 格式转换
3. 处理 modules_to_save

**代码量**: ~400 行

#### Week 9-10: 多模态支持

1. 实现 `MultimodalGPTBridge`
2. 支持 Qwen2-VL 或 LLaVA

**代码量**: ~300 行

#### Week 11-12: 测试与优化

1. 单元测试
2. 端到端测试
3. 性能优化（内存、速度）

---

### 技术栈对比

| 技术栈 | mcore_adapter | Mcore-Bridge |
|--------|--------------|--------------|
| **核心依赖** | mcore_adapter, Megatron-Core | safetensors, torch.distributed |
| **权重格式** | Megatron torch_dist | HuggingFace safetensors |
| **分布式库** | Megatron-Core (内置) | PyTorch DDP / NCCL |
| **模型加载** | mcore_adapter.models.AutoModel | 自定义 Bridge |
| **Trainer** | mcore_adapter.trainer.McaTrainer | 原生 Megatron Trainer |
| **代码量** | ~300 行 (适配层) | ~1500 行 (Bridge系统) |

---

## 总结

### mcore_adapter 的核心优势

1. **实现简单**: 薄适配层，核心逻辑委托给成熟的外部库
2. **快速上线**: 1-2 周即可完成集成
3. **低维护成本**: 仅需维护适配层，Megatron 逻辑由上游处理

### Mcore-Bridge 的核心优势

1. **用户体验极佳**: 输入 HF 模型，输出 HF 模型，全程透明
2. **功能完整**: 支持 LoRA、多模态、所有并行策略
3. **自主可控**: 完全掌握转换逻辑，可快速定制和修复

### 最终答案

**如果你现在想为训练框架引入 Megatron 并行策略，应该选择哪一种？**

- **短期项目 / 快速验证** → **mcore_adapter**
- **企业级产品 / 长期演进** → **Mcore-Bridge**
- **需要 LoRA 支持** → **Mcore-Bridge**（mcore_adapter 不支持）
- **需要极致用户体验** → **Mcore-Bridge**
- **团队无 Megatron 专家** → **mcore_adapter**（学习成本低）

**我的个人建议**:

对于**生产级训练框架**，我强烈推荐 **Mcore-Bridge** 方案：

1. 虽然初期开发成本高（1-2 人月），但长期收益巨大
2. 用户体验是核心竞争力，自动权重转换是刚需
3. LoRA 是微调的核心功能，不可或缺
4. 自主可控让你能快速响应用户需求和新模型发布

但如果是**研究项目**或**MVP 阶段**，**mcore_adapter** 是更好的选择：

1. 快速验证 idea
2. 低风险低成本
3. 可在未来迁移到 Mcore-Bridge

---

## 附录：参考资源

### LLaMA Factory (mcore_adapter)

- 主仓库: https://github.com/hiyouga/LLaMA-Factory
- PR #9237: https://github.com/hiyouga/LLaMA-Factory/pull/9237
- mcore_adapter: https://github.com/alibaba/roll/tree/main/mcore_adapter
- 分析文档: `/home/scbjtfy/LlamaFactory/docs/analysis/megatron-core-integration.md`

### ms-swift (Mcore-Bridge)

- 主仓库: https://github.com/modelscope/swift
- GPTBridge 源码: `/home/scbjtfy/ms-swift/swift/megatron/model/gpt_bridge.py`
- 分析文档: `/home/scbjtfy/ms-swift/docs/analysis/mcore-bridge-architecture.md`

### Megatron-Core

- 官方仓库: https://github.com/NVIDIA/Megatron-LM
- 文档: https://docs.nvidia.com/megatron-core/

---

**文档版本**: v1.0
**最后更新**: 2026-01-02
