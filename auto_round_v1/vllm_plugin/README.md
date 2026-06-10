# vLLM SpinQuant MXFP4 Plugin — 设计文档

## 概述

本插件为 vLLM 提供 SpinQuant/QuaRot 在线旋转 + MXFP4 量化推理支持，作为 auto-round 的
out-of-tree 扩展实现。用户无需修改 vLLM 源码，仅需 `import auto_round.vllm_plugin` 即可
注册 `spinquant_mxfp4` 量化方法。

### 性能对比

| 推理方式 | 速度 (tok/s) | 备注 |
|---------|-------------|------|
| HuggingFace eager (Python butterfly) | 3.7 | 原始方式，Hadamard 占 94% 时间 |
| HuggingFace + Triton patch | 4.1 | 仅加速 dequant+GEMM |
| **vLLM plugin** | **236.4** | torch.compile + CUDA graph + cuBLAS |

---

## 文件结构

```
auto_round/vllm_plugin/
├── __init__.py                  # 包入口，import 即注册 + 应用权重加载补丁
├── spinquant_mxfp4.py           # 核心实现：Config + LinearMethod + Custom Op
├── _weight_loading_patch.py     # 权重加载 monkey-patch (处理 spinquant_R2_head)
├── register.py                  # entry_point 注册函数（pyproject.toml 用）
├── test_plugin.py               # 单元测试
└── test_e2e.py                  # 端到端真实模型测试
```

---

## 使用方法

### 方式 1: 显式 import（推荐）

```python
import auto_round.vllm_plugin  # 注册插件

from vllm import LLM, SamplingParams

llm = LLM(
    model="path/to/spinquant_mxfp4_model",
    quantization="spinquant_mxfp4",
    gpu_memory_utilization=0.8,
)

outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=64))
```

### 方式 2: entry_point 自动加载

在 `pyproject.toml` 中添加：

```toml
[project.entry-points."vllm.general_plugins"]
spinquant_mxfp4 = "auto_round.vllm_plugin.register:register"
```

然后 vLLM 启动时会自动加载插件，无需手动 import。

### 方式 3: 环境变量控制

```bash
VLLM_PLUGINS=spinquant_mxfp4 python -m vllm.entrypoints.openai.api_server \
    --model path/to/model --quantization spinquant_mxfp4
```

### 自动检测（无需手动指定 quantization）

如果模型的 `config.json` 中包含：
- `quant_method: "auto-round"`
- `data_type: "mx_fp"` 或 `"mxfp4"` 或任何包含 "mxfp" 的值
- `spinquant_config.online_r1_rotation: true`

插件会通过 `override_quantization_method` 自动接管，无需 `quantization="spinquant_mxfp4"`：

```python
import auto_round.vllm_plugin
from vllm import LLM
# quantization 参数可省略，会自动检测
llm = LLM(model="path/to/model")
```

---

## 完整推理调用链

### Phase 1: 插件注册

```
import auto_round.vllm_plugin
  │
  ├─ from .spinquant_mxfp4 import SpinQuantMXFP4Config
  │    └─ @register_quantization_config("spinquant_mxfp4") 装饰器执行
  │         └─ QUANTIZATION_METHODS.append("spinquant_mxfp4")
  │         └─ _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["spinquant_mxfp4"] = SpinQuantMXFP4Config
  │    └─ direct_register_custom_op("spinquant_mxfp4_linear", ...)
  │         └─ torch.library 注册自定义算子 torch.ops.auto_round.spinquant_mxfp4_linear
  │         └─ 注册 fake_impl（用于 torch.compile 符号追踪）
  │
  └─ from ._weight_loading_patch import apply_weight_loading_patch
       └─ apply_weight_loading_patch()
            └─ monkey-patch AutoWeightsLoader.__init__
                 添加 "spinquant_R" 到 ignore_unexpected_prefixes
                 (使 spinquant_R2_head 等 top-level key 不报错)
```

### Phase 2: 模型配置解析与量化方法检测

```
LLM(model=path, quantization="spinquant_mxfp4")
  │
  └─ vllm/config/model.py:
       读取 config.json → quantization_config:
       {
         "quant_method": "auto-round",
         "data_type": "mx_fp",
         "group_size": 32,
         "spinquant_config": {
           "r1": true, "r2": true,
           "online_r1_rotation": true,
           "random_r1": false,
           "hidden_size": 1024, "head_dim": 128
         }
       }
       │
       └─ 遍历所有已注册的量化方法，调用 override_quantization_method:
            SpinQuantMXFP4Config.override_quantization_method(quant_cfg, user_quant)
              ├─ 检查 user_quant == "spinquant_mxfp4" → 直接返回
              └─ 或者自动检测:
                   sq = quant_cfg["spinquant_config"]
                   sq["online_r1_rotation"] == True ✓
                   "mx_fp" contains "mx_fp" ✓
                   → return "spinquant_mxfp4"
       │
       └─ get_quantization_config("spinquant_mxfp4") → SpinQuantMXFP4Config
       └─ SpinQuantMXFP4Config.from_config(quantization_config)
            └─ 解析 spinquant_config 字段
            └─ 创建: SpinQuantMXFP4Config(bits=4, group_size=32,
                       online_r1=True, r1_type="hadamard",
                       hidden_size=1024, head_dim=128)
```

### Phase 3: 模型层构建 (create_weights)

```
对模型中每个 LinearBase 层:
  例如 model.layers.0.self_attn.qkv_proj (QKVParallelLinear)

  config.get_quant_method(layer, prefix)
    └─ isinstance(layer, LinearBase) → True
    └─ 返回 SpinQuantMXFP4LinearMethod(config)

  method.create_weights(layer, input_size=1024, output_sizes=[512,128,128], ...)
    │
    ├─ layer.weight_packed = Parameter([768, 512], dtype=uint8)
    │     attrs: {input_dim=1, output_dim=0, packed_dim=1, pack_factor=2}
    │     → 告知 vLLM 如何对 TP 分片和解包
    │
    ├─ layer.weight_scale = Parameter([768, 32], dtype=uint8)
    │     attrs: {input_dim=1, output_dim=0}
    │
    ├─ layer.spinquant_r1_type = Parameter(scalar, dtype=int32)
    │     attrs: {ignore_warning=True}
    │     → ignore_warning 使 QKVParallelLinear.weight_loader 不报 scalar 警告
    │
    └─ layer.spinquant_r1_size = Parameter(scalar, dtype=int32)
          attrs: {ignore_warning=True}
```

### Phase 4: 权重加载 (load_weights)

这是最复杂的部分，涉及 vLLM 的 AutoWeightsLoader 和模型特定的 stacked_params_mapping。

```
DefaultModelLoader.load_weights(model)
  │
  └─ Qwen3ForCausalLM.load_weights(weights_iterator)
       │
       └─ AutoWeightsLoader(self, skip_prefixes=["lm_head."])
            │  ← 我们的 patch 生效:
            │     ignore_unexpected_prefixes += ["spinquant_R"]
            │
            └─ .load_weights(weights)
                 │
                 ├─ _groupby_prefix(weights): 按第一个 "." 分组
                 │
                 ├─ prefix="model" → child_modules["model"] = Qwen3Model
                 │    └─ Qwen3Model 有 load_weights() 方法
                 │         └─ 调用 Qwen2Model.load_weights(child_weights)
                 │              (下面详述)
                 │
                 └─ prefix="spinquant_R2_head" → 不是 child module/param
                      └─ _can_ignore_unexpected("spinquant_R2_head") → True
                           (因为 startswith "spinquant_R")
                      └─ 安全跳过，不报错 ✓

Qwen2Model.load_weights(weights) 中处理 per-layer 权重:
  │
  │  stacked_params_mapping = [
  │    ("qkv_proj", "q_proj", "q"),   # q_proj.* → qkv_proj.* shard_id="q"
  │    ("qkv_proj", "k_proj", "k"),   # k_proj.* → qkv_proj.* shard_id="k"
  │    ("qkv_proj", "v_proj", "v"),   # v_proj.* → qkv_proj.* shard_id="v"
  │    ("gate_up_proj", "gate_proj", 0),  # gate_proj.* → gate_up_proj.* shard_id=0
  │    ("gate_up_proj", "up_proj", 1),    # up_proj.* → gate_up_proj.* shard_id=1
  │  ]
  │
  ├─ 输入: "layers.0.self_attn.q_proj.spinquant_r1_type" (scalar, int32, 值=0)
  │    匹配 stacked: "q_proj" → "qkv_proj", shard_id="q"
  │    重写名称: "layers.0.self_attn.qkv_proj.spinquant_r1_type"
  │    在 params_dict 中查找 → 找到! (create_weights 注册的)
  │    调用: qkv_proj.weight_loader(spinquant_r1_type_param, tensor, shard_id="q")
  │      └─ QKVParallelLinear.weight_loader:
  │           output_dim = None (scalar 没有 output_dim)
  │           ignore_warning = True → 不报警告
  │           param_data.copy_(loaded_weight) → 简单复制 ✓
  │
  ├─ 输入: "layers.0.self_attn.q_proj.weight_packed" ([1024, 512], uint8)
  │    匹配 stacked: "q_proj" → "qkv_proj", shard_id="q"
  │    重写名称: "layers.0.self_attn.qkv_proj.weight_packed"
  │    调用: qkv_proj.weight_loader(weight_packed_param, tensor, shard_id="q")
  │      └─ output_dim=0, packed_dim=1
  │           shard_offset = 0 (q 在第一位)
  │           shard_size = num_heads * head_dim / tp_size
  │           param_data.narrow(0, offset, size).copy_(loaded_weight_shard) ✓
  │
  ├─ 输入: "layers.0.self_attn.o_proj.weight_packed" ([1024, 512], uint8)
  │    不匹配 stacked_params (o_proj 不在列表中)
  │    → else 分支: params_dict["layers.0.self_attn.o_proj.weight_packed"]
  │    调用: RowParallelLinear.weight_loader(param, tensor) (无 shard_id) ✓
  │
  └─ 输入: "layers.0.mlp.gate_proj.spinquant_r1_size" (scalar, int32, 值=1024)
       匹配 stacked: "gate_proj" → "gate_up_proj", shard_id=0
       重写名称: "layers.0.mlp.gate_up_proj.spinquant_r1_size"
       调用: gate_up_proj.weight_loader(param, tensor, shard_id=0)
         └─ MergedColumnParallelLinear.weight_loader:
              output_dim = None → ignore_warning = True → copy ✓

注意: o_proj 和 down_proj 的 spinquant_r1_type/size 在 safetensors 中不存在
      → create_weights 创建的参数保持默认值 0
      → process_weights_after_loading 检测到 rot_size=0 → _rotation_matrix=None
      → apply() 不执行旋转 (正确行为: 这些层不需要 R1 旋转)
```

### Phase 5: 权重后处理（关键步骤）

```
method.process_weights_after_loading(layer)
  └─ rot_type = layer.spinquant_r1_type.item() = 0 (HADAMARD)
  └─ rot_size = layer.spinquant_r1_size.item() = 1024

  └─ _build_full_hadamard(n=1024, device=cuda)
       │
       │  ┌─────────────────────────────────────────────────┐
       │  │ 在 load time 构建完整 [1024, 1024] Hadamard 矩阵  │
       │  │                                                  │
       │  │ 1. get_hadamard_K(1024) → (H_K=[1×1], K=1)     │
       │  │    (1024 = 2^10, 纯 power-of-2)                 │
       │  │                                                  │
       │  │ 2. Butterfly 展开:                               │
       │  │    H = I(1024)                                   │
       │  │    for level in [512, 256, 128, ..., 1]:         │
       │  │      B = butterfly_matrix(n=1024, size=level)    │
       │  │      H = B @ H                                   │
       │  │                                                  │
       │  │ 3. 归一化: H = H / sqrt(1024)                    │
       │  │                                                  │
       │  │ 结果: 正交矩阵, H @ H^T = I                      │
       │  └─────────────────────────────────────────────────┘
       │
  └─ layer._rotation_matrix = H.to(float16)   # [1024, 1024] 缓存
  └─ layer._rot_size = 1024

  注意: 这一步只在模型加载时执行一次。
  将 O(n log n) 的 butterfly 算法"展开"为 [n, n] 矩阵，
  后续推理只需 x @ R 一次 matmul，可被 torch.compile 编译和 CUDA graph 捕获。
```

### Phase 6: torch.compile 编译

```
vLLM v1 默认使用 fullgraph=True 编译:

  torch.compile(model.forward)
    └─ Dynamo bytecode transform (追踪 Python → FX Graph)
    │
    │  对于 qkv_proj.forward → quant_method.apply(layer, x, bias):
    │    - x @ R.to(dtype=x.dtype)
    │        → 编译为 aten.mm (cuBLAS 高效 matmul)
    │    - torch.ops.auto_round.spinquant_mxfp4_linear(x, w_packed, w_scale, 32)
    │        → 编译器看到已注册的 custom op，用 fake_impl 推导输出 shape/dtype
    │        → 不展开内部实现，作为不透明节点插入计算图
    │    - output + bias
    │        → aten.add
    │
    └─ AOT Autograd → Inductor → 生成优化的 CUDA kernel code
    └─ 缓存编译结果到 ~/.cache/vllm/torch_compile_cache/

  编译耗时 ~10s（首次），后续从缓存加载。
```

### Phase 7: CUDA Graph 捕获

```
vLLM 捕获 CUDA graphs 加速推理:

  Piecewise graphs (prefill+decode 混合): 51 个
    - 捕获不同 batch size 下的计算图
    - 每个 graph 包含: rotation matmul + custom op + attention + MLP

  Decode-only graphs (FULL): 35 个
    - batch=1,2,4,...,256 的纯 decode 计算图
    - 每次 decode 只需 graph.replay()，无 kernel launch overhead

  捕获耗时 ~4s，之后推理极快。
```

### Phase 8: 推理执行（每个 token）

```
╔═══════════════════════════════════════════════════════════════╗
║  vLLM Scheduler 调度一个 batch 的请求                          ║
║    → 选择 CUDA graph (按 batch_size 匹配)                     ║
║    → graph.replay()                                          ║
╚═══════════════════════════════════════════════════════════════╝
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Qwen3ForCausalLM.forward(input_ids, positions, kv_cache)       │
│                                                                  │
│  embed = embed_tokens(input_ids)  → [B, 1, 1024]               │
│                                                                  │
│  for layer_idx in range(28):  # Qwen3-0.6B 有 28 层             │
│    ┌────────────────────────────────────────────────────────┐   │
│    │ DecoderLayer.forward(hidden_states, residual)           │   │
│    │                                                         │   │
│    │  # Self-Attention                                       │   │
│    │  norm = input_layernorm(hidden_states)                   │   │
│    │                                                         │   │
│    │  qkv = qkv_proj(norm)  ← SpinQuantMXFP4LinearMethod    │   │
│    │    ┌──────────────────────────────────────────────┐     │   │
│    │    │ apply(layer, x=[B,1,1024], bias=None)        │     │   │
│    │    │                                              │     │   │
│    │    │ Step 1: R1 Rotation                          │     │   │
│    │    │   R = layer._rotation_matrix  [1024,1024]    │     │   │
│    │    │   x_rot = x @ R.to(bfloat16)                 │     │   │
│    │    │   → cuBLAS gemm (在 CUDA graph 中重放)       │     │   │
│    │    │                                              │     │   │
│    │    │ Step 2: MXFP4 Dequant + GEMM                │     │   │
│    │    │   torch.ops.auto_round.spinquant_mxfp4_linear│     │   │
│    │    │   (x_rot, weight_packed, weight_scale, 32)   │     │   │
│    │    │   → Triton kernel 或 Python fallback:        │     │   │
│    │    │     - 解包 uint8 → 2个 E2M1 FP4 值           │     │   │
│    │    │     - 查表还原浮点值                           │     │   │
│    │    │     - 应用 e8m0 shared exponent 缩放          │     │   │
│    │    │     - 融合 GEMM: output = x_rot @ W_dequant^T│     │   │
│    │    │                                              │     │   │
│    │    │ → output: [B, 1, 768] (Q+K+V 合并)          │     │   │
│    │    └──────────────────────────────────────────────┘     │   │
│    │                                                         │   │
│    │  q, k, v = split(qkv)                                   │   │
│    │  attn_out = flash_attention(q, k, v, kv_cache)          │   │
│    │                                                         │   │
│    │  out = o_proj(attn_out)  ← apply() (无旋转, rot_size=0) │   │
│    │  hidden = residual + out                                 │   │
│    │                                                         │   │
│    │  # MLP                                                  │   │
│    │  norm2 = post_attention_layernorm(hidden)                │   │
│    │  gate_up = gate_up_proj(norm2)  ← apply() (有旋转)      │   │
│    │  gate, up = split(gate_up)                               │   │
│    │  mlp_out = down_proj(silu(gate) * up)  ← apply() (无旋转)│   │
│    │  hidden = hidden + mlp_out                               │   │
│    └────────────────────────────────────────────────────────┘   │
│                                                                  │
│  logits = lm_head(final_norm(hidden))  → [B, 1, vocab_size]    │
│  next_token = sample(logits)                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心设计决策

### 1. 预计算完整旋转矩阵（而非运行时 butterfly）

**问题**: Python butterfly Hadamard 有 `while` 循环和动态 shape，无法被 `torch.compile(fullgraph=True)` 编译。

**解决方案**: 在 `process_weights_after_loading()` 中一次性构建完整 `[n, n]` 矩阵，推理时只做 `x @ R`。

```python
# 加载时 (一次性):
R = _build_full_hadamard(1024, device)  # [1024, 1024]
layer._rotation_matrix = R

# 推理时 (每个 token):
x_rotated = x @ R  # 简单 matmul, torch.compile 友好
```

**代价**: 额外 `1024*1024*2 = 2MB` 内存/层（共 ~280MB for 140 layers）。对 vLLM 来说微不足道。

**收益**: cuBLAS matmul 在 L20 GPU 上 < 0.01ms，远快于 Python butterfly 的 0.74ms。

### 2. Custom Op 注册（解决 fullgraph 编译）

**问题**: vLLM v1 使用 `fullgraph=True`，不允许 graph break。`torch.compiler.disable` 会导致 graph break。Triton kernel 和 Python dequant 都无法直接被 Dynamo 追踪。

**解决方案**: 使用 `direct_register_custom_op` 注册为 PyTorch 自定义算子：

```python
_auto_round_lib = torch.library.Library("auto_round", "DEF")

direct_register_custom_op(
    op_name="spinquant_mxfp4_linear",
    op_func=_spinquant_mxfp4_linear_impl,   # 真实实现 (Triton 或 Python fallback)
    fake_impl=_spinquant_mxfp4_linear_fake,  # shape 推导 (compile tracing 用)
    target_lib=_auto_round_lib,
    dispatch_key="CUDA",
)
```

编译器将 custom op 视为**不透明节点**：
- 追踪时调用 `fake_impl` 推导输出 shape/dtype（不执行真实计算）
- 运行时调用 `op_func` 执行真实计算
- 不破坏计算图完整性，可被 CUDA graph 捕获

### 3. 权重加载补丁 (Weight Loading Patch)

**问题**: auto-round 导出模型时，将 `spinquant_R2_head` 作为 top-level key 保存在 safetensors 中。
vLLM 的 `AutoWeightsLoader` 遍历所有 key 并尝试映射到模型参数，找不到 `spinquant_R2_head`
对应的模块就抛出 `ValueError`。

**解决方案**: monkey-patch `AutoWeightsLoader.__init__` 添加 ignore 规则：

```python
# _weight_loading_patch.py
_orig_init = AutoWeightsLoader.__init__

def _patched_init(self, module, *, ignore_unexpected_prefixes=None, **kwargs):
    if ignore_unexpected_prefixes is None:
        ignore_unexpected_prefixes = []
    else:
        ignore_unexpected_prefixes = list(ignore_unexpected_prefixes)
    # spinquant_R2_head, spinquant_R3_matrix 等 top-level key 会被安全忽略
    ignore_unexpected_prefixes.append("spinquant_R")
    _orig_init(self, module, ignore_unexpected_prefixes=ignore_unexpected_prefixes, **kwargs)

AutoWeightsLoader.__init__ = _patched_init
```

**为什么安全**:
- 只影响以 "spinquant_R" 开头的 top-level key
- 普通模型没有这类 key → 补丁无实际效果
- per-layer 的 `spinquant_r1_type/size` 不受影响（它们在 `model.layers.X.Y.Z.` 前缀下）

### 4. 与 vLLM stacked_params_mapping 的协作

**问题**: vLLM 将 `q_proj`, `k_proj`, `v_proj` 融合为 `qkv_proj` (QKVParallelLinear)，
`gate_proj`, `up_proj` 融合为 `gate_up_proj` (MergedColumnParallelLinear)。
但 safetensors 中保存的是未融合的名称。

**vLLM 的解决方案** (已内置于 Qwen2Model.load_weights):

```python
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]
```

**我们的 spinquant 参数如何通过此映射**:

1. 文件中: `model.layers.0.self_attn.q_proj.spinquant_r1_type`
2. 匹配 stacked: `"q_proj"` → `"qkv_proj"`, shard_id = `"q"`
3. 映射后: `model.layers.0.self_attn.qkv_proj.spinquant_r1_type`
4. 在 params_dict 中查找 → 找到 (create_weights 注册的)
5. 调用 weight_loader(param, tensor, shard_id="q"):
   - `output_dim = None` (scalar 无维度)
   - `ignore_warning = True` (我们在 create_weights 设置的)
   - 直接 `param_data.copy_(loaded_weight)` ✓

**为什么 scalar 参数可以被多次 load (q/k/v 各一次)**:
- 同一层的 q_proj/k_proj/v_proj 都有相同的 spinquant_r1_type/size 值
- 每次 load 只是覆盖相同的值，结果正确

### 5. 哪些层有旋转，哪些没有

| 层类型 | safetensors 中是否有 spinquant buffers | 是否执行在线旋转 |
|--------|---------------------------------------|-----------------|
| q_proj | ✓ spinquant_r1_type=0, r1_size=1024 | ✓ R1 旋转 |
| k_proj | ✓ spinquant_r1_type=0, r1_size=1024 | ✓ R1 旋转 |
| v_proj | ✓ spinquant_r1_type=0, r1_size=1024 | ✓ R1 旋转 |
| gate_proj | ✓ spinquant_r1_type=0, r1_size=1024 | ✓ R1 旋转 |
| up_proj | ✓ spinquant_r1_type=0, r1_size=1024 | ✓ R1 旋转 |
| o_proj | ✗ 无 spinquant buffers | ✗ 不旋转 (rot_size=0) |
| down_proj | ✗ 无 spinquant buffers | ✗ 不旋转 (rot_size=0) |

**原理**: R1 旋转应用于 hidden states → linear projection 的路径。
- `q/k/v/gate/up` proj 的输入是 hidden_size 维的 hidden states → 需要 R1
- `o_proj` 的输入是 head_dim × num_heads → 属于 R2 变换后的空间，不需要 R1
- `down_proj` 的输入是 intermediate_size → 属于 R3 变换后的空间，不需要 R1

### 6. dtype 兼容性

vLLM 可能以 float16 或 bfloat16 运行模型。旋转矩阵存储为 float16（最小内存），
在 `apply()` 中动态 cast：

```python
R = R.to(dtype=x.dtype)  # 匹配输入 dtype
x = x @ R
```

---

## auto-round 模型格式详解

### config.json 中的 quantization_config

```json
{
  "quant_method": "auto-round",
  "data_type": "mx_fp",
  "bits": 4,
  "group_size": 32,
  "packing_format": "auto_round:llm_compressor",
  "spinquant_config": {
    "r1": true,
    "r2": true,
    "r3": false,
    "r4": false,
    "online_r1_rotation": true,
    "rotation_size": null,
    "random_r1": false,
    "random_r2": false,
    "trainable_rotation": false,
    "head_dim": 128,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "algorithm": "spinquant"
  }
}
```

### model.safetensors 结构

```
Top-level keys (特殊):
  spinquant_R2_head          [128, 128] float32   # R2 头部旋转矩阵 (已被离线应用)

Per-layer keys (需要在线旋转的层):
  model.layers.{i}.self_attn.{q,k,v}_proj.weight_packed    [N, K//2] uint8
  model.layers.{i}.self_attn.{q,k,v}_proj.weight_scale     [N, K//32] uint8
  model.layers.{i}.self_attn.{q,k,v}_proj.spinquant_r1_type scalar int32
  model.layers.{i}.self_attn.{q,k,v}_proj.spinquant_r1_size scalar int32
  model.layers.{i}.mlp.{gate,up}_proj.weight_packed         [N, K//2] uint8
  model.layers.{i}.mlp.{gate,up}_proj.weight_scale          [N, K//32] uint8
  model.layers.{i}.mlp.{gate,up}_proj.spinquant_r1_type     scalar int32
  model.layers.{i}.mlp.{gate,up}_proj.spinquant_r1_size     scalar int32

Per-layer keys (不需要在线旋转的层):
  model.layers.{i}.self_attn.o_proj.weight_packed    [N, K//2] uint8
  model.layers.{i}.self_attn.o_proj.weight_scale     [N, K//32] uint8
  model.layers.{i}.mlp.down_proj.weight_packed       [N, K//2] uint8
  model.layers.{i}.mlp.down_proj.weight_scale        [N, K//32] uint8

Non-quantized layers:
  model.layers.{i}.input_layernorm.weight            [1024] bfloat16
  model.layers.{i}.post_attention_layernorm.weight   [1024] bfloat16
  model.layers.{i}.self_attn.{q,k}_norm.weight       [128] bfloat16
  model.embed_tokens.weight                          [V, 1024] bfloat16
```

### MXFP4 编码格式

```
weight_packed: 每个 uint8 字节存储 2 个 E2M1 FP4 值
  low nibble (bits 0-3):  偶数列的 FP4 值
  high nibble (bits 4-7): 奇数列的 FP4 值

  E2M1 编码表:
    0b0000 → +0.0    0b1000 → -0.0
    0b0001 → +0.5    0b1001 → -0.5
    0b0010 → +1.0    0b1010 → -1.0
    0b0011 → +1.5    0b1011 → -1.5
    0b0100 → +2.0    0b1100 → -2.0
    0b0101 → +3.0    0b1101 → -3.0
    0b0110 → +4.0    0b1110 → -4.0
    0b0111 → +6.0    0b1111 → -6.0

weight_scale: e8m0 shared exponent (每 32 个元素共享一个)
  实际缩放: scale = 2^(uint8_value - 127)
  
  反量化: dequant_value = FP4_decode(nibble) * 2^(scale_byte - 127)
```

---

## 与现有 auto_round_extension/vllm_ext 的关系

```
auto_round_extension/vllm_ext/     ← 现有扩展 (VLLM_ENABLE_AR_EXT=1)
  ├── auto_round_ext.py            # patch AutoRoundConfig → AutoRoundExtensionConfig
  ├── linear_impl_mxfp4.py         # MXFP4 线性层实现 (无旋转)
  └── vllm_oot_patches.py          # patch maybe_remap_kv_scale_name

auto_round/vllm_plugin/           ← 本插件 (spinquant + MXFP4)
  ├── spinquant_mxfp4.py           # 注册独立量化方法 "spinquant_mxfp4"
  └── _weight_loading_patch.py     # patch AutoWeightsLoader

区别:
  - vllm_ext: 扩展现有 "auto-round" 量化方法，通过 monkey-patch AutoRoundConfig
  - vllm_plugin: 注册全新 "spinquant_mxfp4" 量化方法，不干扰 "auto-round" 的正常流程
  - vllm_ext: 仅支持 MXFP4 qdq (无旋转)
  - vllm_plugin: 支持 online R1 rotation + MXFP4
  - 两者可以共存: 无 spinquant_config 的模型走 vllm_ext，有的走 vllm_plugin
```

---

## 与 vLLM 内置 fp_quant 的对比

| 特性 | fp_quant (内置) | spinquant_mxfp4 (本插件) |
|------|----------------|--------------------------|
| 旋转类型 | group-wise Hadamard (32×32) | full-dim R1 (1024×1024) |
| 旋转实现 | CUTLASS `fusedQuantizeMx` (SM89+) | cuBLAS matmul (SM70+) |
| GEMM | CUTLASS `matmul_mxf4_bf16_tn` | Triton fused kernel / PyTorch fallback |
| 量化时机 | 在线激活量化 | 离线权重量化 + 在线旋转 |
| 最低 GPU | SM89 (Ada Lovelace) | SM70 (Volta) |
| 安装要求 | 编译 vLLM _C 模块 | pip install auto-round |
| 模型来源 | 任何 BF16 模型 (在线量化) | auto-round 导出的 MXFP4 模型 |

---

## Tensor Parallelism (TP) 支持

### 使用方式

```python
import auto_round.vllm_plugin
from vllm import LLM

llm = LLM(
    model="path/to/large_model",
    quantization="spinquant_mxfp4",
    tensor_parallel_size=4,  # 直接指定 TP 数量
)
```

### 正确性分析

**核心数学：**
```
原始: y = x @ W^T
旋转后: y = (x @ R) @ W_rot^T，其中 W_rot = W @ R（离线已吸收旋转到权重中）
```

**ColumnParallel (q/k/v/gate/up) — R1 旋转层：✓ 完全支持**

```
分片方式: 沿 output_dim (行) 切分
  W_rot: [N, K] → 每个 rank: [N/tp, K]

关键数学性质 — 行分片与旋转可交换:
  (W @ R)[row_shard, :] = W[row_shard, :] @ R  ✓

推理流程 (每个 rank):
  1. x 是完整 hidden_states (所有 rank 相同，ColumnParallel 输入不分片)
  2. x_rot = x @ R  ← 每个 rank 独立计算，结果相同
  3. output_shard = custom_op(x_rot, weight_packed_shard)  ← 各自的权重分片
  4. all-gather output shards → 完整输出
```

**RowParallel (o_proj/down_proj) — 当前无旋转：✓ 安全**

```
分片方式: 沿 input_dim (列) 切分
  W: [N, K] → 每个 rank: [N, K/tp]

当前: rot_size = 0 → 不执行旋转 → TP 正常工作

假设未来要加旋转 (如 R3 for down_proj):
  问题: 列分片与全维度旋转 不可交换!
    (W @ R)[:, col_shard] ≠ W[:, col_shard] @ R_sub
  因为旋转矩阵 R 混合了所有输入列
```

### 旋转矩阵的 TP 行为

| 组件 | TP 行为 | 说明 |
|------|---------|------|
| `spinquant_r1_type` | 所有 rank 相同值 | scalar, weight_loader 直接 copy |
| `spinquant_r1_size` | 所有 rank 相同值 | scalar, weight_loader 直接 copy |
| `_rotation_matrix` | 每个 rank 独立构建相同矩阵 | 确定性 Hadamard 算法 → 结果一致 |
| `weight_packed` | 每个 rank 不同 shard | 沿 output_dim 自动切分 |
| `weight_scale` | 每个 rank 不同 shard | 沿 output_dim 自动切分 |

### rotation_size 与分片的约束

`apply()` 中的 block-wise 旋转逻辑:
```python
if rot_size == in_features:
    x = x @ R                          # 全维度旋转
else:
    x = x.reshape(..., -1, rot_size)    # 分块旋转
    x = (x @ R).reshape(shape)          # 要求: in_features % rot_size == 0
```

**约束规则:**

| 层类型 | input per rank | rotation 约束 | 当前是否满足 |
|--------|---------------|---------------|-------------|
| ColumnParallel (q/k/v/gate/up) | K (完整, 不分片) | rot_size ≤ K 且 K % rot_size == 0 | ✓ rot_size=1024=K |
| RowParallel (o_proj/down_proj) | K/tp (被分片) | rot_size ≤ K/tp 且 (K/tp) % rot_size == 0 | ✓ 当前 rot_size=0 |

**大模型场景示例 (Qwen3-32B, hidden_size=5120, tp=4):**

```
ColumnParallel:
  input_per_rank = 5120 (不分片)
  rot_size = 5120
  5120 % 5120 == 0 ✓

RowParallel (假设未来加 R3, intermediate_size=27648):
  input_per_rank = 27648 / 4 = 6912
  若 rot_size = 27648 → 6912 < 27648 ✗ 无法工作!
  若 rot_size = 128 (per-head) → 6912 % 128 == 0 ✓ 可以工作
```

### 未来 RowParallel 旋转的解决方案

如果需要在 RowParallel 层（o_proj, down_proj）上做全维度旋转:

1. **per-head/per-block 旋转** — 使用 rot_size = head_dim (128)，每个分片内独立旋转
2. **旋转前 all-gather** — 收集完整 input → 旋转 → 再分片 (增加通信开销)
3. **导出时预分片** — 按 TP 拓扑导出已正确分片的旋转权重
4. **结构化旋转矩阵** — 使用 block-diagonal R，使得列分片与旋转可交换

### 内存开销 (TP 场景)

每个 rank 存储完整旋转矩阵 (冗余但必要):

| 模型 | hidden_size | rot_matrix 大小/层 | 层数 | 每 rank 总开销 |
|------|-------------|-------------------|------|---------------|
| Qwen3-0.6B | 1024 | 2 MB | 140* | ~280 MB |
| Qwen3-8B | 4096 | 32 MB | 180* | ~5.6 GB |
| Qwen3-32B | 5120 | 50 MB | 320* | ~16 GB |
| Llama-70B | 8192 | 128 MB | 400* | ~50 GB |

*注: 层数 = (num_layers × 每层需旋转的 linear 数), 实际约 5 个/decoder layer

**对于大模型，预计算矩阵的内存开销可能成为瓶颈。** 解决方案:
- 使用 `fast_hadamard_transform` CUDA kernel (O(n log n) 计算, O(1) 额外内存)
- 但需要解决 torch.compile fullgraph 兼容性

---

## 局限性与未来工作

### 当前局限

1. **仅支持 R1 online rotation** — R2 已保存在模型中(离线应用), R3/R4 需修改 attention/MLP 层
2. **旋转矩阵内存** — 每层存储 [n, n] float16 矩阵，对大 hidden_size 模型(如 8192)可能占 ~50GB
3. **Triton kernel 精度** — 使用 TF32 accumulation，不如 IEEE FP32 精确
4. **TP 下旋转矩阵冗余** — 每个 rank 存完整矩阵（因为 ColumnParallel input 不分片）

### 未来优化方向

1. **`fast_hadamard_transform` CUDA kernel** — 替代预计算矩阵，降为 O(n log n) 且省内存
2. **R3/R4 online support** — 需要 hook attention output 和 MLP intermediate
3. **CUTLASS MXFP4 GEMM** — SM89+ 上使用硬件加速的 `matmul_mxf4_bf16_tn`
4. **与 vllm_ext 合并** — 统一为一个扩展，自动根据模型配置选择 rotation vs no-rotation 路径

---

## 调试与问题排查

### 常见错误

**1. `ValueError: There is no module or parameter named 'spinquant_R2_head'`**
- 原因: 未导入 `auto_round.vllm_plugin`，权重加载补丁未生效
- 解决: 确保在 `from vllm import LLM` 之前执行 `import auto_round.vllm_plugin`

**2. `torch._dynamo.exc.Unsupported: Import failure`**
- 原因: forward 路径中有动态 import
- 本插件已修复: 所有 import 在模块顶部完成

**3. `torch._dynamo.exc.Unsupported: Skip calling torch.compiler.disable()d function`**
- 原因: vLLM v1 使用 fullgraph=True，不允许 @torch.compiler.disable
- 本插件已修复: 使用 custom op 代替 compiler.disable

**4. `c10::Half != c10::BFloat16`**
- 原因: 旋转矩阵 dtype 与输入不匹配
- 本插件已修复: `R = R.to(dtype=x.dtype)` 动态 cast

### 验证方法

```python
# 快速验证插件加载
import auto_round.vllm_plugin
from auto_round.vllm_plugin.spinquant_mxfp4 import SpinQuantMXFP4Config
print(f"Registered: {SpinQuantMXFP4Config.get_name()}")

# 验证 override 检测
import json
with open("model_path/config.json") as f:
    qc = json.load(f)["quantization_config"]
result = SpinQuantMXFP4Config.override_quantization_method(qc, None)
print(f"Detection: {result}")  # should be "spinquant_mxfp4"
```

---

## 测试

```bash
# 单元测试 (无需大 GPU 内存)
python auto_round/vllm_plugin/test_plugin.py

# 端到端测试 (需要已保存的 MXFP4 模型)
python auto_round/vllm_plugin/test_e2e.py
```

### 验证输出示例

```
[PASS] Plugin registration
[PASS] Config parsing
[PASS] Weight creation
[PASS] Forward pass (device=cuda)
[PASS] Override detection
✅ All vLLM plugin tests passed!

Loading weights took 0.18 seconds
Model loading took 0.64 GiB memory and 12.8 seconds
torch.compile took 14.85 s in total
Graph capturing finished in 4 secs
Generated: "The capital of France is Paris..."
Speed: 236.4 tok/s
✅ vLLM inference with SpinQuant MXFP4 plugin successful!
```
