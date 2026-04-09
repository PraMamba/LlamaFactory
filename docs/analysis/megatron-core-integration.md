# LLaMA Factory Megatron-Core Integration Analysis

## 概述

本文档深入分析 LLaMA Factory 如何在不影响原有功能的情况下集成 Megatron-Core 训练后端（通过 mcore_adapter）。

**核心提交**: [#9237](https://github.com/hiyouga/LLaMA-Factory/pull/9237) (commit: `13170577`)
**集成日期**: 2025-10-26
**外部依赖**: [mcore_adapter](https://github.com/alibaba/roll/tree/main/mcore_adapter) - ROLL Framework 的 Megatron-LM 集成层

---

## 一、集成策略：可选性 (Opt-in) 设计模式

### 1.1 核心思想

LLaMA Factory 采用了**可选性**集成策略，通过环境变量 `USE_MCA` 作为功能开关：

```bash
# 不启用 MCA - 使用原有训练流程
llamafactory-cli train config.yaml

# 启用 MCA - 使用 Megatron-Core 训练后端
USE_MCA=1 llamafactory-cli train config.yaml
```

这种设计确保：
- **向后兼容**: 默认情况下，所有原有功能保持不变
- **功能隔离**: MCA 相关代码仅在启用时才会被加载和执行
- **零影响**: 未使用 MCA 的用户不会受到任何性能或功能影响

### 1.2 环境变量检测机制

集成在多个层次进行环境变量检测：

```python
# src/llamafactory/launcher.py:57-58
if is_env_enabled("USE_MCA"):  # force use torchrun
    os.environ["FORCE_TORCHRUN"] = "1"
```

**关键点**:
- 启用 MCA 时自动强制使用 `torchrun` 进行分布式训练
- 这是因为 Megatron-Core 需要分布式环境初始化

---

## 二、架构层次化集成

### 2.1 新增模块: `train/mca/`

创建了独立的 MCA 训练模块，与原有训练流程平行：

```
src/llamafactory/train/
├── mca/                    # 新增：Megatron-Core Adapter 训练模块
│   ├── __init__.py
│   ├── trainer.py          # TODO: 未来可能覆盖原始 trainer
│   └── workflow.py         # 核心：PT/SFT/DPO 训练流程
├── pt.py                   # 原有：预训练
├── sft.py                  # 原有：监督微调
├── dpo.py                  # 原有：DPO 训练
└── tuner.py                # 修改：条件分支路由
```

**设计亮点**:
- 完全独立的模块，不修改原有训练代码
- 通过条件导入 (`if finetuning_args.use_mca`) 动态路由
- 保持与原有训练接口的一致性

### 2.2 条件路由机制

在 `train/tuner.py` 的 `_training_function` 中实现条件分支：

```python
# src/llamafactory/train/tuner.py:69-84
if finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
    if not is_mcore_adapter_available():
        raise ImportError("mcore_adapter is not installed...")

    if finetuning_args.stage == "pt":
        from .mca import run_pt as run_pt_mca
        run_pt_mca(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        from .mca import run_sft as run_sft_mca
        run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "dpo":
        from .mca import run_dpo as run_dpo_mca
        run_dpo_mca(model_args, data_args, training_args, finetuning_args, callbacks)
elif finetuning_args.stage == "pt":
    run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
# ... 原有流程
```

**关键特性**:
- 懒加载 (lazy import): MCA 模块仅在需要时导入
- 运行时依赖检查: 防止未安装 `mcore_adapter` 时出错
- 清晰的条件分支: 不影响原有代码路径

---

## 三、配置系统双轨设计

### 3.1 参数解析分离

创建了独立的 MCA 参数解析流程：

```python
# src/llamafactory/hparams/parser.py:56-65
if is_mcore_adapter_available() and is_env_enabled("USE_MCA"):
    from mcore_adapter import TrainingArguments as McaTrainingArguments
    _TRAIN_MCA_ARGS = [ModelArguments, DataArguments, McaTrainingArguments,
                       FinetuningArguments, GeneratingArguments]
    _TRAIN_MCA_CLS = tuple[ModelArguments, DataArguments, McaTrainingArguments,
                           FinetuningArguments, GeneratingArguments]
else:
    _TRAIN_MCA_ARGS = []
    _TRAIN_MCA_CLS = tuple()
```

**设计要点**:
- 使用 `mcore_adapter.TrainingArguments` 替代 HuggingFace 的 `TrainingArguments`
- 保持其他参数类（ModelArguments, DataArguments 等）不变
- 类型提示支持两种配置模式

### 3.2 参数兼容性处理

```python
# src/llamafactory/hparams/parser.py:219-226
def _configure_mca_training_args(training_args, data_args, finetuning_args) -> None:
    """Patch training args to avoid args checking errors and sync MCA settings."""
    training_args.predict_with_generate = False
    training_args.generation_max_length = data_args.cutoff_len
    training_args.generation_num_beams = 1
    training_args.use_mca = True
    finetuning_args.use_mca = True
```

**兼容性策略**:
- 关闭不兼容的特性（如 `predict_with_generate`）
- 同步必要的参数（如 `cutoff_len` → `generation_max_length`）
- 统一标记 MCA 启用状态

### 3.3 配置文件示例

MCA 训练配置增加了 Megatron-Core 特有的参数：

```yaml
# examples/megatron/qwen3_moe_full.yaml
model_name_or_path: Qwen/Qwen3-30B-A3B-Instruct-2507

# 基础训练参数（与原有相同）
stage: sft
finetuning_type: full  # MCA 目前仅支持 full
dataset: alpaca_en_demo
cutoff_len: 4096

# Megatron-Core 特有的并行和优化参数
tensor_model_parallel_size: 1        # 张量并行度
sequence_parallel: false
pipeline_model_parallel_size: 4      # 流水线并行度
expert_model_parallel_size: 2        # MoE 专家并行度
bias_activation_fusion: true
apply_rope_fusion: true
use_distributed_optimizer: true
overlap_param_gather: true
overlap_grad_reduce: true
moe_grouped_gemm: true
moe_token_dispatcher_type: alltoall
recompute_granularity: full
```

---

## 四、核心实现：MCA Workflow

### 4.1 数据预处理适配

**关键差异**: Megatron-Core 使用不同的 label shifting 策略

```python
# src/llamafactory/train/mca/workflow.py:99-102
# dataset needs +1 then cut back due to MCA shift logic
data_args.cutoff_len += 1
dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
data_args.cutoff_len -= 1
```

**原因分析**:
- Megatron-Core 在模型内部进行 label shifting（将 labels 向左移一位）
- 而 HuggingFace 通常在数据层面处理
- 因此需要多准备一个 token，然后在 collator 中裁剪

### 4.2 Data Collator 包装器

```python
# src/llamafactory/train/mca/workflow.py:58-75
def _data_collator_wrapper(data_collator: Any):
    @functools.wraps(data_collator)
    def wrapper(features: Sequence[dict[str, Any]]):
        labels_key = [k for k in features[0].keys() if k.endswith("labels")]
        input_ids_key = [k for k in features[0].keys() if k.endswith("input_ids")]

        for feature in features:
            if len(labels_key) == 0:  # pt (预训练)
                feature["labels"] = deepcopy(feature["input_ids"])[1:]
            for k in labels_key:
                feature[k] = feature[k][1:]  # 裁剪 labels
            for k in input_ids_key:
                feature[k] = feature[k][:-1]  # 裁剪 input_ids
            for k in ["attention_mask", "position_ids"]:
                if k in feature:
                    feature[k] = feature[k][:-1]

        return data_collator(features)
    return wrapper
```

**包装器职责**:
1. 处理预训练场景（没有 labels 时从 input_ids 生成）
2. 裁剪序列末尾的 token（input_ids 去掉最后一个）
3. 裁剪序列开头的 label（labels 去掉第一个）
4. 同步裁剪 attention_mask 和 position_ids

### 4.3 模型加载和检查

```python
# src/llamafactory/train/mca/workflow.py:78-86
def _check_model_support(model_args: "ModelArguments"):
    from transformers import AutoConfig as HfAutoConfig

    config = HfAutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    if config.model_type not in MCA_SUPPORTED_MODELS:
        raise ValueError(f"Model {config.model_type} is not supported by MCA.")
```

**支持的模型** (来自 `extras/constants.py`):
- `deepseek_v3`
- `llama`, `mistral`, `mixtral`
- `qwen2`, `qwen2_vl`, `qwen2_5_vl`, `qwen3`, `qwen3_vl`, `qwen3_moe`, `qwen3_next`

**模型加载**:
```python
# src/llamafactory/train/mca/workflow.py:105
from mcore_adapter.models import AutoModel
model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
```

**关键区别**:
- 使用 `mcore_adapter.models.AutoModel` 而非 HuggingFace 的 `AutoModel`
- `mcore_adapter` 会将 HuggingFace 模型转换为 Megatron-Core 兼容格式

### 4.4 Trainer 实例化

```python
# src/llamafactory/train/mca/workflow.py:113-120
from mcore_adapter.trainer import McaTrainer

trainer = McaTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
    **dataset_module,
)
```

**McaTrainer 特性**:
- 继承自 HuggingFace Trainer API
- 内部使用 Megatron-Core 的分布式训练逻辑
- 支持 Megatron-Core 的各种并行策略（TP, PP, EP, DP）

### 4.5 多模态支持（Qwen2-VL 系列）

```python
# src/llamafactory/train/mca/workflow.py:165-179
if getattr(model.config, "hf_model_type", None) in ["qwen2_vl", "qwen2_5_vl", "qwen3_vl"]:
    params_to_freeze = []
    if finetuning_args.freeze_vision_tower:
        params_to_freeze.extend(["vision_model.blocks", "vision_model.patch_embed"])

    if finetuning_args.freeze_multi_modal_projector:
        params_to_freeze.extend(["multi_modal_projector"])

    if finetuning_args.freeze_language_model:
        params_to_freeze.extend(["embedding", "decoder", "output_layer"])

    if params_to_freeze:
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in params_to_freeze):
                p.requires_grad_(False)
```

**设计考虑**:
- 支持选择性冻结视觉编码器、多模态投影器、语言模型
- 通过参数前缀匹配进行模块级冻结

### 4.6 DPO 训练支持

```python
# src/llamafactory/train/mca/workflow.py:232-269
from mcore_adapter.trainer import DPOTrainer as McaDPOTrainer
from mcore_adapter.trainer.dpo_config import DPOConfig

if finetuning_args.use_ref_model:
    ref_config = AutoConfig.from_pretrained(model_args.model_name_or_path, training_args)
    ref_model = AutoModel.from_config(ref_config)
    ref_model.load_state_dict(model.state_dict())
else:
    ref_model = None

dpo_config = DPOConfig(
    beta=finetuning_args.pref_beta,
    pref_loss=finetuning_args.pref_loss,
    label_smoothing=finetuning_args.dpo_label_smoothing,
)

trainer = McaDPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_config=dpo_config,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
    **dataset_module,
)
```

**DPO 特性**:
- 可选的参考模型 (`use_ref_model`)
- 支持多种 DPO 损失函数（通过 `pref_loss` 参数）
- 与原有 DPO 配置保持兼容

---

## 五、依赖管理

### 5.1 可选依赖检测

```python
# src/llamafactory/extras/packages.py
def is_mcore_adapter_available():
    try:
        import mcore_adapter
        return True
    except ImportError:
        return False
```

**使用位置**:
- `launcher.py`: 强制启用 torchrun
- `parser.py`: 条件导入 McaTrainingArguments
- `tuner.py`: 运行时依赖检查
- `workflow.py`: 模块导入时检查

### 5.2 安装方式

```bash
# 从 ROLL 项目安装
pip install "git+https://github.com/alibaba/roll.git#subdirectory=mcore_adapter"
```

**依赖链**:
```
LLaMA Factory
    └─> mcore_adapter (可选)
            └─> Megatron-Core
            └─> PyTorch
            └─> Transformers (兼容层)
```

---

## 六、关键设计决策分析

### 6.1 为什么不修改原有训练代码？

**原因**:
1. **风险隔离**: 避免引入 Megatron-Core 相关 bug 影响原有用户
2. **维护性**: 两条代码路径独立演进，互不影响
3. **测试成本**: 不需要为所有模型重新测试 Megatron-Core 兼容性

### 6.2 为什么使用环境变量而非配置参数？

**考虑**:
- 环境变量在进程启动时确定，适合作为系统级开关
- 影响参数解析器的类型选择（TrainingArguments vs McaTrainingArguments）
- 需要在 launcher 层面决定是否强制使用 torchrun

### 6.3 为什么需要 data collator wrapper？

**核心问题**: Megatron-Core 和 HuggingFace 的 label shifting 策略不同

| 特性 | HuggingFace | Megatron-Core |
|------|-------------|---------------|
| Shifting 位置 | Data collator | 模型内部 |
| Input IDs | `[0, 1, 2, 3, 4]` | `[0, 1, 2, 3]` |
| Labels | `[1, 2, 3, 4, -100]` | `[1, 2, 3, 4]` |

**解决方案**: 包装器在 collation 时调整序列长度，适配 Megatron-Core 的期望格式

### 6.4 为什么目前只支持 full-parameter tuning？

**技术限制**:
- LoRA/QLoRA 在 Megatron-Core 中需要特殊处理（张量并行、流水线并行）
- `mcore_adapter` 目前的实现重点在全参数训练
- 未来可能通过扩展 `mcore_adapter` 支持 LoRA

---

## 七、模型转换和导出

### 7.1 HuggingFace → Megatron 转换

`mcore_adapter` 能够直接加载 HuggingFace 模型：

```python
# 内部自动转换
model = AutoModel.from_pretrained("Qwen/Qwen3-30B", training_args)
```

**转换过程** (在 `mcore_adapter` 内部):
1. 加载 HuggingFace checkpoint
2. 重新映射权重名称（`query_key_value` vs. `q_proj`, `k_proj`, `v_proj`）
3. 应用张量并行切分
4. 应用流水线并行切分

### 7.2 Megatron → HuggingFace 转换

使用提供的转换脚本：

```bash
# scripts/megatron_merge.py
python scripts/megatron_merge.py \
    --checkpoint_path path/to/megatron/checkpoint \
    --output_path path/to/hf/model
```

**转换步骤**:
1. 收集分布式 checkpoint 分片
2. 合并张量并行/流水线并行的权重
3. 重新映射为 HuggingFace 格式
4. 保存为标准 HuggingFace checkpoint

---

## 八、性能优化特性

### 8.1 Megatron-Core 特有的优化

MCA 训练配置中可用的优化选项：

```yaml
# 融合操作
bias_activation_fusion: true        # 融合 bias 和 activation
apply_rope_fusion: true              # 融合 RoPE 操作

# 分布式优化器
use_distributed_optimizer: true      # 跨设备分片优化器状态
overlap_param_gather: true           # 重叠参数收集
overlap_grad_reduce: true            # 重叠梯度规约

# MoE 优化
moe_grouped_gemm: true               # MoE 分组 GEMM
moe_token_dispatcher_type: alltoall  # Token 分发策略

# 重计算
recompute_granularity: full          # 激活重计算粒度
```

### 8.2 并行策略

支持的并行维度：

1. **Tensor Parallelism (TP)**: `tensor_model_parallel_size`
   - 切分单层的权重矩阵

2. **Pipeline Parallelism (PP)**: `pipeline_model_parallel_size`
   - 按层切分模型

3. **Expert Parallelism (EP)**: `expert_model_parallel_size`
   - MoE 模型的专家并行

4. **Data Parallelism (DP)**: 自动推导
   - `total_gpus / (TP * PP * EP)`

**并行度计算示例**:
```yaml
# 8 GPUs, TP=1, PP=4, EP=2
# DP = 8 / (1 * 4 * 2) = 1
# Global batch size = per_device_batch * gradient_accumulation * DP
#                   = 1 * 8 * 1 = 8
```

---

## 九、集成的优势与局限

### 9.1 优势

1. **大规模训练能力**
   - 支持 3D 并行（TP + PP + DP）
   - MoE 专家并行
   - 分布式优化器减少显存占用

2. **性能优化**
   - 融合操作减少 kernel launch
   - 重叠通信和计算
   - 高效的 MoE token 分发

3. **向后兼容**
   - 原有功能完全保留
   - 可以在同一仓库中使用两种训练后端

4. **生态兼容**
   - 复用 LLaMA Factory 的数据处理
   - 复用 LLaMA Factory 的模型支持

### 9.2 局限

1. **仅支持全参数微调**
   - 暂不支持 LoRA/QLoRA
   - 显存需求较高

2. **模型支持有限**
   - 仅支持 11 种模型架构
   - 需要 `mcore_adapter` 显式支持

3. **学习曲线**
   - 需要理解 Megatron-Core 的并行策略
   - 配置复杂度增加

4. **依赖外部项目**
   - 依赖 `mcore_adapter` 的维护
   - 版本兼容性需要关注

---

## 十、总结

LLaMA Factory 的 Megatron-Core 集成展示了**可选性架构**的最佳实践：

### 核心设计原则

1. **最小侵入性**: 新增独立模块，不修改原有代码
2. **条件激活**: 通过环境变量控制功能启用
3. **接口一致性**: 保持与原有训练流程相同的 API
4. **渐进式集成**: 先支持核心功能（PT/SFT/DPO），后续可扩展

### 技术亮点

- **双轨配置系统**: 根据 `USE_MCA` 选择不同的参数解析器
- **数据适配层**: 通过 collator wrapper 桥接两种 label shifting 策略
- **懒加载机制**: 仅在需要时导入 MCA 相关依赖
- **模型支持检查**: 运行时验证模型兼容性

### 对 LLM 框架开发者的启示

1. **功能开关设计**: 使用环境变量或配置文件控制大型特性
2. **模块化架构**: 新功能作为独立模块，而非修改现有代码
3. **适配器模式**: 使用 wrapper/adapter 桥接不同的 API 差异
4. **渐进式支持**: 先支持核心场景，逐步扩展到边缘情况

这种集成方式值得其他 LLM 训练框架借鉴，尤其是在需要支持多种训练后端时。
