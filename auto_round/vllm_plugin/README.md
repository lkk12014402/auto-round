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

## 推理性能调优

### 推理流水线架构

```
input activation x
    │
    ├─ Step 1: R1/R4 online rotation (Hadamard butterfly / matrix matmul)
    │
    ├─ Step 2: Activation QDQ  ← 由 AUTO_ROUND_MXFP4_QDQ_BACKEND 控制
    │     选择: local_ext | quark | vllm_ext | fallback
    │
    └─ Step 3: GEMM           ← 由 AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND 控制
          选择: quark_like_dense | preunpack_fp8 | packed_fused
          │
          output
```

**两个环境变量独立控制不同阶段，互不冲突。**

### 环境变量一览

| 环境变量 | 作用 | 可选值 | 默认 |
|---------|------|--------|------|
| `AUTO_ROUND_MXFP4_QDQ_BACKEND` | Activation QDQ 实现选择 | `local_ext`, `quark`, `vllm_ext`, `fallback` | 自动（local_ext > quark > vllm_ext > fallback） |
| `AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND` | GEMM 计算后端 | `quark_like_dense`, `preunpack_fp8`, `packed_fused` | `packed_fused` |
| `AUTO_ROUND_DISABLE_LOCAL_MXFP4_QDQ` | 禁用本地 CUDA kernel 编译 | `1` | 未设置 |

### Activation QDQ 后端详情

| 后端 | 实现 | 说明 |
|------|------|------|
| `local_ext` | 本地 vendored CUDA kernel (`csrc/mxfp4_qdq/`) | 从 Quark MIT 源码迁移，JIT 编译，**最快** |
| `quark` | `quark.torch.kernel.mx.qdq_mxfp4` | Quark 原生 C++ extension，需安装 Quark |
| `vllm_ext` | `auto_round_extension.vllm_ext.mxfp4_qdq_utils` | PyTorch 实现，无 CUDA 依赖 |
| `fallback` | 纯 PyTorch（plugin 内置） | 任何环境可用，最慢 |

**性能对比** (NVIDIA L20, bf16, 256×4096):

| 后端 | 延迟 | 相对 |
|------|------|------|
| local_ext | 0.013ms | 1.00x |
| quark_cuda | 0.015ms | 1.12x |
| vllm_ext | 0.179ms | 13.2x |
| fallback | 0.236ms | 17.5x |

> `local_ext` 和 `quark_cuda` 是同一份 CUDA kernel 源码，local_ext 略快因为 dispatch 路径更短（直接 pybind11 vs torch.library 包装）。

### GEMM Runtime 后端详情

| 后端 | 实现 | 说明 |
|------|------|------|
| `quark_like_dense` | 模型加载时将 MXFP4 packed weight 解量化为 dense bf16，推理时走 `torch.nn.functional.linear` → **cuBLAS GEMM** | 小 batch 最快，内存占用较高 |
| `preunpack_fp8` | 模型加载时将 packed weight 解量化为 FP8 (E4M3)，推理时走 `vllm._custom_ops.cutlass_scaled_mm` → **CUTLASS FP8 scaled GEMM** | 大 batch 最快，需 FP8 硬件支持 (sm89+) |
| `packed_fused` | 保持 packed uint8 weight，推理时走 Triton fused kernel（在 GEMM 内逐 tile 解量化） | 内存最省，但 Triton kernel 未充分优化 |

**显存占用对比** (以 Qwen3-32B 为例, 约 32B 参数):

| 后端 | 权重存储格式 | 单卡显存 (TP=1) | TP=4 每卡 | 说明 |
|------|-------------|----------------|-----------|------|
| `packed_fused` | uint8 packed (4bit 有效) + uint8 scale | ~16 GB | ~4 GB | 保持原始压缩格式 |
| `preunpack_fp8` | FP8 E4M3 (8bit) + bf16 scale | ~32 GB | ~8 GB | 加载时解包为 FP8，删除 packed 原始存储 |
| `quark_like_dense` | bf16 dense (16bit) | ~64 GB | ~16 GB | 加载时完全 dequant 为 bf16，删除 packed 原始存储 |

> **注意**：`quark_like_dense` 和 `preunpack_fp8` 在模型加载后会 **释放** packed weight 和 scale（`del layer.weight_packed`），
> 最终显存中只保留 dequant 后的 dense/FP8 weight。因此实际占用是 dequant 后的大小，不是叠加。

**性能对比** (NVIDIA L20, bf16, N_out=4096):

| 后端 | 1×4096 | 16×4096 | 256×4096 | 1024×4096 |
|------|--------|---------|----------|-----------|
| quark_like_dense | 0.054ms | **0.030ms** | 0.093ms | 0.324ms |
| preunpack_fp8 | **0.035ms** | 0.035ms | **0.089ms** | **0.214ms** |
| packed_fused | 0.226ms | 0.223ms | 0.560ms | 1.578ms |

> cuBLAS GEMM 由 PyTorch 的 `torch.nn.functional.linear` 调用，底层走 NVIDIA cuBLAS 库。
> CUTLASS FP8 GEMM 由 vLLM 封装的 `cutlass_scaled_mm` 调用，底层走 NVIDIA CUTLASS 库。

### 推荐配置

```bash
# 最优性能（显存充足，如 TP=4 每卡 ≥16GB 空闲）
export AUTO_ROUND_MXFP4_QDQ_BACKEND=local_ext
export AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND=quark_like_dense

# 性能/显存均衡（sm89+ GPU，如 L4/L40/H100）
export AUTO_ROUND_MXFP4_QDQ_BACKEND=local_ext
export AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND=preunpack_fp8

# 极限省显存（性能最差，仅在显存极度受限时使用）
export AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND=packed_fused
```

**选择建议：**
- 显存充足 → `quark_like_dense`（GEMM 最快，代价是 4x 权重显存）
- 显存有限但有 FP8 硬件 → `preunpack_fp8`（GEMM 接近最快，显存仅 2x）
- 极限场景 → `packed_fused`（保持 4bit 压缩，但 Triton GEMM 慢 6-8x）

### 性能 benchmark 工具

```bash
# 完整 benchmark（QDQ + runtime）
python examples/benchmark_qdq_kernels.py

# 只测 activation QDQ
python examples/benchmark_qdq_kernels.py --mode qdq

# 只测 GEMM runtime
python examples/benchmark_qdq_kernels.py --mode runtime

# 自定义 shape
python examples/benchmark_qdq_kernels.py --shapes 256x4096,1024x25600 --iters 200
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
      → process_weights_after_loading 检测到 rot_size=0
      → _r1_rotation_runtime = NONE, _r1_rotation_matrix = None
      → apply() 不执行旋转 (正确行为: 这些层不需要 R1 旋转)
```

### Phase 5: 加载后旋转状态准备（关键步骤）

```
method.process_weights_after_loading(layer)
  └─ rot_type = layer.spinquant_r1_type.item() = 0 (HADAMARD)
  └─ rot_size = layer.spinquant_r1_size.item() = 1024

  └─ get_hadamard_K(1024) → (hadamard_K=[1×1], K=1)
  └─ 因为 rot_size == in_features:
       layer._r1_rotation_runtime = HADAMARD
       layer._r1_hadamard_K = hadamard_K
       layer._r1_hadamard_factor = K
       layer._r1_rotation_matrix = None
       layer._r1_rot_size = 1024

  注意:
  1. 这一步只在模型加载时执行一次
  2. 这里只准备 online rotation 的 runtime state，不会修改 layer.weight_packed
  3. full-size Hadamard 不再展开成 [n, n] 显式矩阵，而是 forward 时走 butterfly
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
    │    - torch.ops.auto_round.spinquant_mxfp4_act_qdq(x, 32)
    │    - torch.ops.auto_round.spinquant_mxfp4_linear(x_qdq, w_packed, w_scale, 32)
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
│    │    │   full-size Hadamard:                        │     │   │
│    │    │     x_rot = matmul_hadU(x, hadamard_K, K)    │     │   │
│    │    │   random/trained/block:                      │     │   │
│    │    │     x_rot = x @ R                            │     │   │
│    │    │                                              │     │   │
│    │    │ Step 2: MXFP4 Activation QDQ                │     │   │
│    │    │   x_qdq = qdq_mxfp4(x_rot)                   │     │   │
│    │    │                                              │     │   │
│    │    │ Step 3: MXFP4 Dequant + GEMM                │     │   │
│    │    │   torch.ops.auto_round.spinquant_mxfp4_linear│     │   │
│    │    │   (x_qdq, weight_packed, weight_scale, 32)   │     │   │
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

### Activation QDQ 实现来源

当前 plugin 对 `torch.ops.auto_round.spinquant_mxfp4_act_qdq` 的自动选择优先级是：

1. **本地 vendored CUDA kernel**（从 Quark 的 MIT-licensed MXFP4 QDQ kernel 迁入 `auto_round/vllm_plugin/csrc/mxfp4_qdq/`）
2. **Quark kernel**（懒加载：仅在 local_ext 不可用时才 import，避免触发 Quark 的 import-time 编译）
3. **`auto_round_extension.vllm_ext.qdq_mxfp4`**
4. **纯 PyTorch fallback**

可通过 `AUTO_ROUND_MXFP4_QDQ_BACKEND` 环境变量强制指定后端（详见"推理性能调优"章节）。

首次选择后端时，plugin 会输出一条日志：
```
INFO: MXFP4 activation QDQ backend: local_ext
```

这意味着现在 vLLM plugin **不再需要依赖 Quark 才能拿到 Quark 同语义的 fast path**；只要本地 extension 编译成功，就会优先走 auto-round 自己 vendored 的 CUDA kernel。

---

## 核心设计决策

### 1. 旋转只准备 runtime state，不在 plugin 中重写 weight

这次实现里，必须先区分两件事：

1. **weight side 的旋转/量化处理**
   - 发生在导出侧
   - 导出的就是已经处理好的 `weight_packed/weight_scale`
2. **activation side 的 online rotation**
   - 由 vLLM plugin 在推理时执行
   - plugin load 只负责准备它需要的 runtime state

```python
def process_weights_after_loading(layer):
    # 只准备旋转状态，不改 packed weight
    self._process_rotation(layer, prefix="r1")
    self._process_rotation(layer, prefix="r4")
```

换句话说：

- **改前后都不是 plugin 在“提前处理 weight”**
- **改前后都是导出侧先把 weight 处理好**
- 这次改的是：**plugin 如何准备和执行 activation-side rotation**

### 2. 当前实现是 hybrid rotation runtime

`process_weights_after_loading()` 现在按 rotation 类型准备不同的 runtime path：

| 类型 | load 时准备什么 | forward 时怎么做 |
|------|------------------|------------------|
| full-size Hadamard | `hadamard_K + K + runtime=HADAMARD` | `matmul_hadU(x, hadamard_K, K)` |
| block Hadamard | 显式 block 矩阵 `R` | `x @ R` |
| random / trained | 从 checkpoint 读取显式矩阵 `R` | `x @ R` |
| 无旋转 | `runtime=NONE` | 跳过 |

代码上对应的是：

```python
if rot_type == ROTATION_TYPE_HADAMARD and rot_size == in_features:
    layer._r1_rotation_runtime = ROTATION_RUNTIME_HADAMARD
    layer._r1_hadamard_K = hadamard_K
    layer._r1_hadamard_factor = K
elif rot_type == ROTATION_TYPE_HADAMARD:
    layer._r1_rotation_runtime = ROTATION_RUNTIME_MATRIX
    layer._r1_rotation_matrix = _build_block_hadamard(...)
elif rot_type in (ROTATION_TYPE_RANDOM, ROTATION_TYPE_TRAINED):
    layer._r1_rotation_runtime = ROTATION_RUNTIME_MATRIX
    layer._r1_rotation_matrix = loaded_matrix
```

这和旧实现最大的差别是：

- **旧实现**：Hadamard / random / trained 基本都尽量变成显式矩阵 `_rotation_matrix`
- **新实现**：只有 block Hadamard、random、trained 继续走显式矩阵；full-size Hadamard 改成结构化 butterfly runtime

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
    "r4": true,
    "online_r1_rotation": true,
    "rotation_size": null,
    "random_r1": false,
    "random_r2": false,
    "random_r4": false,
    "trainable_rotation": false,
    "head_dim": 128,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "r1_rotation_size": 1024,
    "r4_rotation_size": 3072,
    "r4_hadamard_K": 12,
    "algorithm": "spinquant"
  }
}
```

**字段说明：**
- `rotation_size`: 用户配置值（null 表示使用默认）
- `r1_rotation_size`: 实际使用的 R1 旋转维度 = `rotation_size or hidden_size`
- `r4_rotation_size`: 实际使用的 R4 旋转维度 = `rotation_size or intermediate_size`
- `r4_hadamard_K`: 仅当 R4 旋转维度不是 power-of-2 时出现，记录 Hadamard 分解的 K 值
  - 例如 3072 = 12 × 256，K=12 表示使用 [12×12] 基础 Hadamard + butterfly(256)
- `r1_rotation_size` / `r4_rotation_size` 为 null 表示对应旋转未启用

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

## 为什么 Quark 的 online R1 推理看起来更快

### 先说结论

Quark 示例里的 **online R1** 看起来很快，主要不是因为它把 R1 本身做得“更高级”，而是因为它走的是一条
**更轻的推理路径**：

1. **评测直接使用内存中的 PyTorch 模型对象**，不是重新加载导出的 packed 低比特模型。
2. **线性层最终仍走 `F.linear` / cuBLAS dense GEMM**，不是每层都执行 packed weight dequant + custom low-bit GEMM。
3. **激活 MXFP4 有专门的 dynamic fast path**：`mx.qdq_mxfp4(...)`。
4. 示例配置里 **R1 只做 block rotation (`rotation_size=128`)**，不是 full hidden-size R1。

因此，Quark 的“快”主要来自：

- **更便宜的 activation QDQ**
- **标准 dense linear**
- **较小的 R1 block size**

而不是说明 “在线 R1 本身几乎没有成本”。

### Quark 的 dynamic MXFP4 fast code path 是什么

Quark 在动态 MXFP4 激活量化时，有一条专门的快路径：

```text
QuantLinear.forward_with_weight(...)
  -> QuantMixin.get_quant_input(x)
     -> input_quantizer(x)
        -> DynamicScaledFakeQuantize.forward(x)
           -> self.fake_quantize_func(x, ...)
              -> mxfp4_dynamic_fake_quantize(x, fake_quantizer, scale_calculation_mode)
                 -> mx.qdq_mxfp4(x, scale_calculation_mode="even")
                    -> qdq_mxfp4_hip(...)   或   qdq_mxfp4_triton(...)
```

关键代码位置：

- `quark/torch/quantization/nn/modules/mixin.py`
  - `get_quant_input()` 会调用 input quantizer
- `quark/torch/quantization/tensor_quantize.py`
  - `DynamicScaledFakeQuantize.forward()`
  - 对 OCP MXFP4 + `scale_calculation_mode="even"`，直接走 `mxfp4_dynamic_fake_quantize`
- `quark/torch/kernel/__init__.py`
  - `mxfp4_dynamic_fake_quantize()` 直接调用 `mx.qdq_mxfp4(...)`
- `quark/torch/kernel/mx/__init__.py`
  - 根据 `QUARK_MXFP4_IMPL` 选择 HIP 或 Triton 实现
- `quark/torch/kernel/mx/hip.py`
  - `qdq_mxfp4_hip(x, ...) -> kernel_ext.qdq_mxfp4(x, 32)`
- `quark/torch/kernel/mx/triton.py`
  - `qdq_mxfp4_triton(...)` 走 Triton 的 `downcast_to_mxfp` + `upcast_from_mxfp`

这条 fast path 的本质是：

> **对激活直接做“设备端 fused qdq”**，避免 Python 侧 observer + pack/unpack + 普通张量运算组合。

### Quark 为什么没有像我们当前 vLLM plugin 那样贵

Quark 示例的训练/评测脚本是：

- `examples/torch/language_modeling/rotation/train_rotation.py`
- `quark/contrib/llm_eval/evaluation.py`

这里的 `eval_model(...)` 直接把当前内存中的 `model` 对象传给 `lm_eval`，而不是重新从导出的
real-quantized safetensors 模型里恢复一个 packed low-bit inference 模型。

此外，Quark 在 `ModelQuantizer.freeze(model)` 之后：

1. **非 dynamic 的 quantizer 会被冻结**
2. 如果 `quantize=True`，权重会先被 fake-quant 一次，写回高精度 weight tensor
3. 之后 `QuantLinear.forward_with_weight()` 里的 `get_quant_weight()` 看到 `frozen_params=True`，
   就不会每次 forward 再做权重量化
4. forward 最终仍然是：

```python
output = F.linear(quant_input, quant_weight, bias=quant_bias)
```

这意味着 Quark 的评测路径本质更接近：

> **activation 走 dynamic MXFP4 qdq fast path + weight 用已量化好的高精度 tensor + cuBLAS dense linear**

而不是：

> **packed uint8 weight + shared exponent + 每层在线 dequant + 自定义 low-bit GEMM**

### Quark R1 路径与我们当前 vLLM plugin 路径的逐层调用对比

下表以 **R1 online + MXFP4** 为例，对比两条路径的关键步骤：

| 阶段 | Quark example / `llm_eval` | 当前 vLLM plugin |
|------|-----------------------------|------------------|
| 模型来源 | 直接评测内存中的 PyTorch 模型对象 | 从导出的 auto-round 模型目录加载 |
| R1 配置 | 示例常用 `rotation_size=128` block R1 | 常见为 full-size R1 或 block R1 |
| R1 预处理 | 先对目标 linear weight 做一次 R1 融合，再包 `InputRotationWrapperHadamard` | 导出侧已处理好 weight；plugin load 只准备 R1 runtime state，不改 packed weight |
| 激活 R1 执行 | `InputRotationWrapperHadamard.forward()` -> `HadamardTransform.forward()` | `_process_rotation()` 准备 Hadamard runtime 或显式矩阵，`apply()` 里先做 rotation，再做 activation qdq |
| 激活量化 | `DynamicScaledFakeQuantize.forward()` -> `mx.qdq_mxfp4(...)` fast path | `spinquant_mxfp4_act_qdq`，优先对齐 `auto_round_extension/vllm_ext` / Quark 的 `qdq_mxfp4(even)` 语义 |
| 权重形式 | freeze 后通常是“已 fake-quant 的高精度 weight” | `weight_packed:uint8` + `weight_scale:uint8` |
| GEMM | `F.linear(...)`（dense GEMM, cuBLAS） | `triton_mxfp4_gemm(...)` 或 PyTorch fallback |
| 每层是否反量化权重 | 通常**不需要**每层重建完整权重 | 需要在 GEMM 中按 tile dequant packed weight |
| 主要瓶颈 | R1 transform + activation qdq | R1 transform + packed MXFP4 GEMM 临时 buffer / dequant |
| 典型感受 | “online R1 也很快” | “online R1 + MXFP4 插件开销明显” |

### 逐层解释：Quark 为什么更像“dense model + fast qdq”

#### 1. R1 不是完全“裸在线”

Quark 在 `RotationProcessor.apply_online_r1()` 中，并不是只在前向时对输入做旋转。
它还会先对目标 layer 的权重做一次 R1 融合：

```python
layer.weight.data = matmul_hadU(layer.weight.data, hadamard_K=hadamard_K, K=K)
layer_with_input_rotation = InputRotationWrapperHadamard(...)
```

也就是说它仍然用了：

- **weight 预融合**
- **input wrapper 在线补偿**

这和我们当前的思路在数学上是一致的。

#### 2. 它的线性层仍是普通 dense linear

`QuantLinear.forward_with_weight()` 最终还是：

```python
quant_input = self.get_quant_input(args[0])
quant_weight = self.get_quant_weight(weight)
output = F.linear(quant_input, quant_weight, bias=quant_bias)
```

这意味着核心 GEMM 仍然交给成熟的 dense GEMM backend（通常是 cuBLAS），而不是走 packed low-bit kernel。

#### 3. dynamic MXFP4 的快路径只处理 activation qdq

Quark 的 `mx.qdq_mxfp4(...)` fast path 是：

- 观测 + 计算 scale
- 做 MXFP4 风格 qdq
- 返回 **原 dtype 的激活 tensor**

它解决的是“**激活量化快**”的问题，不是“**packed low-bit weight GEMM 快**”的问题。

#### 4. block R1 比 full-size R1 便宜得多

Quark 示例配置：

```json
"online_r1_rotation": true,
"rotation_size": 128
```

这意味着它做的是 block-wise R1，不是 full hidden-size R1。
对于 hidden_size 很大的模型，`rotation_size=128` 的代价远低于 full-size 旋转。

### 为什么这不能直接推出“我们的实现有问题”

Quark 快，并不自动说明我们当前 vLLM plugin 的 R1 实现有 correctness 或 engineering 问题。
两边跑的不是同一类路径：

- **Quark example**
  - 更接近“高精度 weight + dynamic activation qdq + dense GEMM”
- **当前 vLLM plugin**
  - 更接近“packed real-quantized MXFP4 weight + online rotation + low-bit custom GEMM”

两者比较时，最容易忽略的差异是：

1. **权重是否还是高精度 tensor**
2. **GEMM 是 dense 还是 low-bit packed**
3. **R1 是 full-size 还是 block-size**
4. **激活量化是否有专门 fused qdq fast path**

### 对当前工程判断的启发

这次对比给出的结论是：

1. **Quark 的快，不代表 online R1 没成本**
   - 只是它把大头开销放在了更成熟、更便宜的 dense 路径上

2. **我们当前最贵的不是“R1 这个数学变换本身”，而是它和 packed MXFP4 inference path 叠加后的整体开销**

3. **如果未来要缩小和 Quark 的差距，最有效的方向不是单独优化 Python hook，而是让 R1 更贴近 GEMM 执行路径**
   - 例如 block-R1 融入 GEMM tile
   - 或者引入更低峰值、更高效的 MXFP4 execution path

4. **在没有 fused R1+GEMM 之前，Quark 这种“dynamic qdq + dense GEMM”路径在体感上很可能仍然更轻**

### 一句话总结

Quark 的 online R1 推理之所以“看起来很快”，本质上是因为它评测时仍然更接近：

> **dense linear + activation dynamic MXFP4 fast qdq**

而不是我们当前这条：

> **packed real-quantized MXFP4 weight + online rotation + activation qdq + custom low-bit GEMM**

所以它快的关键，不是 “R1 online 零成本”，而是 **它避开了我们这条部署路径中最贵的那部分**。

---

## Quark-like / preunpack 运行时后端

### 目标

如果当前目标不是只看默认的 packed 部署路径，而是同时比较：

- 更接近 **Quark eval 语义**
- 更接近 **auto_round_extension/vllm_ext 预解包语义**
- 当前 **packed fused 部署语义**

那么现在插件已经支持多运行时后端：

| runtime backend | 目标 | 当前状态 |
|------|------|------|
| `packed_fused` | 当前默认部署路径：packed weight + activation qdq + fused dequant+GEMM | 已实现，默认 |
| `quark_like_dense` | 更接近 Quark：rotate(x) + activation qdq + frozen dense weight + `F.linear` | 已实现 |
| `preunpack_fp8` | 更接近 `auto_round_extension/vllm_ext`：load 时预解包到 FP8+scale | 已实现 |

### 选择方式

不需要改导出格式，可以通过环境变量或 `config.json` 中的可选字段切换：

```bash
export AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND=quark_like_dense
# 或
export AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND=preunpack_fp8
```

也可以在模型 `config.json` / `quantization_config` 中显式加入：

```json
"runtime_backend": "quark_like_dense"
```

支持的值：

- `packed_fused`
- `quark_like_dense`
- `preunpack_fp8`

### 为什么 `quark_like_dense` 最接近 Quark

Quark 示例评测时，本质更像：

```text
rotate(x)
-> qdq_mxfp4(x)
-> F.linear(x_qdq, frozen_fake_quant_weight)
```

也就是说：

1. **activation 走 dynamic MXFP4 qdq**
2. **weight 在 eval 时已经是“冻结后的量化值”**
3. **forward 最终仍然是 dense GEMM**

`quark_like_dense` 现在的实现是：

```text
load time:
  packed weight -> 一次性 dequant 成 bf16/fp16 frozen weight
  保留 R1/R4 runtime state

forward:
  x -> rotate(x) -> qdq_mxfp4(x) -> F.linear(x_qdq, weight_dense_qdq)
```

这和当前 `packed_fused` 的关键区别是：

| 项目 | 当前 `packed_fused` | 提议 `quark_like_dense` |
|------|---------------------|--------------------------|
| weight 存储 | `weight_packed + weight_scale` | load 后转成 `weight_dense_qdq` |
| weight dequant | 每次 forward 融合在 GEMM 里 | **load 时一次性完成** |
| activation | `rotate -> qdq` | `rotate -> qdq` |
| GEMM | low-bit custom GEMM | dense `F.linear` / cuBLAS |
| 目标 | 部署/性能路径 | Quark 语义对齐路径 |

### 代码层面需要怎么改

建议只在 plugin 中加一个 runtime backend 选择，不改导出格式。

#### 1. `SpinQuantMXFP4Config` 增加 backend 选项

可以从 `quantization_config` 或环境变量读取，例如：

```python
SpinQuantMXFP4Config(
    ...,
    runtime_backend="packed_fused",  # default
)
```

建议支持：

- `packed_fused`
- `quark_like_dense`
- 预留 `preunpack_fp8`

#### 2. 在 `process_weights_after_loading()` 中按 backend 分支

当前：

```python
process_weights_after_loading():
    _process_rotation(layer, "r1")
    _process_rotation(layer, "r4")
```

提议改成：

```python
process_weights_after_loading():
    _process_rotation(layer, "r1")
    _process_rotation(layer, "r4")

    if runtime_backend == "quark_like_dense":
        layer.weight_dense_qdq = dequant_mxfp4_weight_once(
            layer.weight_packed, layer.weight_scale
        )
```

这里的 `dequant_mxfp4_weight_once(...)` 建议直接复用/参考已有实现：

- 当前 plugin 中的 `_mxfp4_dequant_linear_fallback` 里的 weight unpack 逻辑
- `auto_round_extension/vllm_ext/mxfp4_qdq_utils.py`
  - `dequant_mxfp4_to_fp8(...)`
  - `mxfp4_fp8_weight_to_bf16(...)`

如果目标是**最接近 Quark 语义**，当前 backend 会直接得到：

- `layer.weight_dense_qdq : bf16/fp16`

而不是只得到：

- `weight_unpacked_fp8 + scale_bf16`

因为后者仍然更像 `vllm_ext`，不是 Quark 的 frozen dense eval 形式。

#### 3. `apply()` 按 backend 分支

当前：

```python
x = rotate(x)
x = spinquant_mxfp4_act_qdq(x, 32)
out = spinquant_mxfp4_linear(x, weight_packed, weight_scale, 32)
```

提议：

```python
x = rotate(x)
x = spinquant_mxfp4_act_qdq(x, 32)

if runtime_backend == "quark_like_dense":
    out = F.linear(x, layer.weight_dense_qdq, bias)
else:
    out = spinquant_mxfp4_linear(x, weight_packed, weight_scale, 32)
```

这样就得到三条清晰的 A/B 路径：

- **同一个导出模型**
- **同一套 rotation runtime**
- **同一套 activation qdq**
- 只对比：
  - dense frozen weight
  - pre-unpacked FP8 weight
  - packed fused low-bit GEMM

这对分析 accuracy / perplexity / lm_eval 差异会非常干净。

### `preunpack_fp8` 是什么

`auto_round_extension/vllm_ext` 的思路是：

```text
load:
  packed FP4 -> unpacked FP8 + bf16 scale

forward:
  qdq(x)
  -> 再把 weight 乘 scale 恢复到 bf16
  -> matmul
```

它更像是一个**工程折中态**：

- 比完全 packed path 更容易调试
- 比 dense frozen weight 更省一点 load-time 转换复杂度
- 但 **forward 里仍然有 weight 恢复成本**

所以如果目标是“更像 Quark”，它不是第一选择；但如果目标是看：

> **每次 forward 的 packed unpack / dequant 开销到底占多少**

那它是一个很合适的中间对照组。

### 现在可以直接怎么比较

建议直接做三组 benchmark：

1. `packed_fused`
2. `preunpack_fp8`
3. `quark_like_dense`

在相同模型、相同 batch、相同 `rotation_size` 下比较：

- tokens/s
- TTFT
- decode latency
- 显存占用

### 这个方案最大的价值

它能把当前问题拆开：

| 问题 | `quark_like_dense` 能否帮助回答 |
|------|-------------------------------|
| rotation + qdq 语义是否对 | **能** |
| packed weight / Triton GEMM 是否引入额外误差 | **能** |
| 与 Quark lm_eval 结果为什么不同 | **能** |
| 最终部署性能是否最好 | **不能，`packed_fused` 才是那条路** |

### 一句话建议

现在这三个 backend 都已经有了：

- **如果看最终部署**：重点看 `packed_fused`
- **如果看 unpack/dequant 代价**：重点看 `preunpack_fp8`
- **如果看最像 Quark 的体感路径**：重点看 `quark_like_dense`

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
  2. x_rot = rotate(x)
       - full-size Hadamard: matmul_hadU(x, hadamard_K, K)
       - 其它路径: x @ R
     每个 rank 独立计算，结果相同
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
| rotation runtime state | 每个 rank 独立准备相同状态 | full-size Hadamard 用 `hadamard_K+K`，其余路径用显式矩阵 |
| `weight_packed` | 每个 rank 不同 shard | 沿 output_dim 自动切分 |
| `weight_scale` | 每个 rank 不同 shard | 沿 output_dim 自动切分 |

### rotation_size 与分片的约束

`apply()` 中的 block-wise 旋转逻辑:
```python
if runtime == HADAMARD and rot_size == in_features:
    x = matmul_hadU(x, hadamard_K, K)   # 全维度 Hadamard
elif rot_size == in_features:
    x = x @ R                           # 全维度显式矩阵
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

## R4 (down_proj) 在线旋转支持

R4 对 MLP `down_proj` 层的输入做在线旋转，原理与 R1 完全一致：

```
原始:   y = down_proj(activation)
R4 旋转: y = down_proj(activation @ R4)

其中 R4 矩阵大小 = intermediate_size (Qwen3-0.6B 为 3072)
```

### R4 checkpoint 格式

auto-round 导出时，在 `down_proj` 层上添加额外 buffers:

```
model.layers.{i}.mlp.down_proj.spinquant_r4_type   scalar int32  (0=hadamard, 1=random, 2=trained)
model.layers.{i}.mlp.down_proj.spinquant_r4_size   scalar int32  (3072)
model.layers.{i}.mlp.down_proj.spinquant_r4_matrix [3072, 3072] float32  (仅 random/trained 时)
```

### R4 插件实现细节

1. **Config 扩展** — `SpinQuantMXFP4Config` 新增 `online_r4`, `r4_type`, `intermediate_size`
2. **create_weights** — 为 `down_proj` 层注册 `spinquant_r4_type/size/matrix` 参数
3. **process_weights_after_loading** — 调用通用 `_process_rotation(layer, "r4")` 准备 R4 runtime state
4. **apply** — 调用通用 `_apply_rotation(x, layer, "r4")` 执行 `matmul_hadU(x)` 或 `x @ R4`

### TP 安全性 (R4)

`down_proj` 是 RowParallel 层，输入按列分片:
- `intermediate_size / tp` 是每个 rank 的 input features
- 使用 block-wise 旋转: `rot_size=128`，每个 rank 独立旋转各自的 blocks
- 约束: `(intermediate_size / tp) % rot_size == 0`
- Qwen3-0.6B: 3072/2=1536, 1536%128=0 ✓

### 支持的旋转组合

| 配置 | R1 | R2 | R3 | R4 | vLLM 支持 |
|------|----|----|----|----|-----------|
| R1 only | ✓ online | — | — | — | ✅ |
| R1+R2 | ✓ online | ✓ offline | — | — | ✅ |
| R1+R4 | ✓ online | — | — | ✓ online | ✅ |
| R1+R2+R4 | ✓ online | ✓ offline | — | ✓ online | ✅ |
| R1+R2+R3 | ✓ online | ✓ offline | ✓ online | — | ❌ (R3 未实现) |
| R1+R2+R3+R4 | ✓ online | ✓ offline | ✓ online | ✓ online | ❌ (R3 未实现) |

---

## 旋转矩阵加载与自定义 rotation_size

### 加载保存的旋转矩阵

当模型使用 `random` 或 `trained` 旋转时，checkpoint 中包含 `spinquant_r{1,4}_matrix` 张量:

```
process_weights_after_loading():
  1. 读取 rot_type: 0=hadamard, 1=random, 2=trained
  2. 若 rot_type != 0: 加载 layer.spinquant_r{1,4}_matrix → 缓存为 runtime 显式矩阵
  3. 若 rot_type == 0:
       - full-size Hadamard: 缓存 hadamard_K + K
       - block Hadamard: 构建显式 block 矩阵
  4. 删除原始 checkpoint 参数，只保留 runtime state
```

这里的“显式矩阵路径”指的是：

> 旋转矩阵 `R` 本身被直接保存/加载，并在 forward 中直接执行 `x @ R`。

它适用于：

- `random`
- `trained`
- block-wise Hadamard

而 full-size Hadamard 当前不再保存/缓存完整 `[n, n]` 的显式矩阵，而是走 `matmul_hadU`。

### 2026-05-20 代码修改前后对照

这次最容易误解的一点是：**改动不在 weight side，而在 plugin 的 rotation runtime 准备方式。**

| 问题 | 修改前 | 修改后 |
|------|--------|--------|
| weight 是谁处理的 | 导出侧先处理，再保存成 `weight_packed/weight_scale` | **不变** |
| plugin load 会不会再改 weight | 不会 | **不会** |
| full-size Hadamard | load 时展开整块 `_rotation_matrix` | load 时只缓存 `hadamard_K + K` |
| block Hadamard | 显式矩阵 | 显式矩阵 |
| random / trained | 加载显式矩阵 | 加载显式矩阵 |
| activation qdq | 无 | `rotate(x)` 后增加 MXFP4 qdq |
| forward 的 R1/R4 | 统一 `x @ R` | Hadamard(full) 走 `matmul_hadU`，其余仍 `x @ R`，随后统一做 `qdq_mxfp4` |

一句话总结：

> **weight side 没变；变的是 plugin 不再把 full-size Hadamard 强行展开成完整矩阵，而是改成加载时准备 butterfly runtime state。**

### 自定义 rotation_size (block-wise rotation)

当 `rotation_size < hidden_size` 时:

```python
# apply() 中的分块旋转
x_shape = x.shape                          # [B, seq, 1024]
x = x.reshape(*x_shape[:-1], -1, rot_size)  # [B, seq, 8, 128]
x = x @ R                                   # R: [128, 128]
x = x.reshape(x_shape)                      # [B, seq, 1024]
```

优势:
- **内存**: rot_size=128 → 128×128×2 = 32KB/层（vs 全维度 2MB/层）
- **TP 友好**: 只要 `input_per_rank % rot_size == 0`
- **精度折衷**: block-wise 旋转数学上弱于全维度旋转

### TP 下旋转矩阵一致性

所有 rank 构建相同的旋转矩阵，通过确定性算法保证:
- Hadamard: butterfly 算法是纯数学的，无随机性
- Random: 使用固定 seed (42) + `torch.linalg.qr` → 所有 rank 结果相同

```python
def _generate_random_orthogonal(n, device):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(42)  # 固定 seed → TP 所有 rank 一致
    A = torch.randn(n, n, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q.to(device)
```

---

## Entry Point 自动注册机制

### 工作原理

vLLM 的 out-of-tree plugin 机制基于 Python `entry_points` (PEP 621):

```
┌─────────────────────────────────────────────────────────────────────┐
│  pip install auto-round  (editable: pip install -e .)               │
│    │                                                                 │
│    └─ 注册 entry point: vllm.general_plugins → spinquant_mxfp4     │
│                                                                      │
│  启动 vLLM (任何方式: LLM(), lm_eval --model vllm, api_server)     │
│    │                                                                 │
│    └─ vllm 扫描所有 installed packages 的 "vllm.general_plugins" 组 │
│         └─ 发现 "spinquant_mxfp4" → import auto_round.vllm_plugin   │
│              └─ 执行 register() 函数                                 │
│                   ├─ @register_quantization_config 注册量化方法       │
│                   ├─ direct_register_custom_op 注册自定义算子         │
│                   └─ apply_weight_loading_patch() 打 monkey-patch    │
└─────────────────────────────────────────────────────────────────────┘
```

### pyproject.toml 配置

```toml
[project.entry-points."vllm.general_plugins"]
spinquant_mxfp4 = "auto_round.vllm_plugin.register:register"
```

### 验证 entry point 是否生效

```bash
# 方法 1: 直接检查
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm.general_plugins')
print([e.name for e in eps])
# 应包含: ['spinquant_mxfp4']
"

# 方法 2: 验证插件加载
python -c "
import auto_round.vllm_plugin
from auto_round.vllm_plugin.spinquant_mxfp4 import SpinQuantMXFP4Config
print(f'Plugin loaded: {SpinQuantMXFP4Config.get_name()}')
"
```

### lm_eval 中如何使用

由于 entry point 自动注册，使用 `lm_eval --model vllm` 时**无需任何额外参数**:

```bash
# vLLM 启动 → 自动加载插件 → 读到模型 spinquant_config → 激活 online rotation
lm_eval \
    --model vllm \
    --model_args "pretrained=./rotated_model,dtype=bfloat16,add_bos_token=True" \
    --tasks piqa,hellaswag \
    --limit 100
```

完整推理链路:
1. lm_eval 创建 vLLM engine
2. vLLM 初始化时扫描 entry_points → 发现并加载我们的 plugin
3. 读取模型 config.json → 检测到 `spinquant_config.online_r1_rotation=true`
4. `override_quantization_method()` 返回 `"spinquant_mxfp4"`
5. 使用 `SpinQuantMXFP4LinearMethod` 构建模型（注册旋转参数 + 量化权重）
6. 加载 checkpoint → 构建旋转矩阵 → 推理时自动执行 online rotation

---

## 端到端评估流程

### 评估脚本

提供一套完整的评估工具，验证各种旋转配置下 HF 和 vLLM 推理精度一致:

```
new_commit/
├── save_rotated_models.py   # 导出各配置的 rotated + quantized 模型
├── eval_hf.sh               # lm_eval HF backend 评估
├── eval_vllm.sh             # lm_eval vLLM backend 评估
├── compare_results.py       # 对比 HF vs vLLM 精度
└── run.sh                   # 一键运行全部流程
```

### run.sh — 一键端到端流程

对每种旋转配置，依次执行: 保存模型 → vLLM 评估 → HF 评估:

```bash
# 运行全部 9 种配置 (R1, R1+R2, R1+R4, R1+R2+R4, R1_size128, R1_random, R1+R4_size128, R1+R2+R3, R1+R2+R3+R4)
bash run.sh

# 只运行指定配置
bash run.sh R1 R1+R4

# 模型已保存,跳过 save 步骤
SKIP_SAVE=1 bash run.sh

# 只跑 HF 评估 (不跑 vLLM)
SKIP_VLLM=1 bash run.sh

# 换 GPU 和 limit
CUDA_VISIBLE_DEVICES=6 LIMIT=50 bash run.sh

# 多 GPU (tensor parallel)
CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 bash run.sh
```

每种配置的流程:
```
╔══════════════════════════════════════════════════╗
║  [1/9] Config: R1+R4                            ║
╚══════════════════════════════════════════════════╝
  Step 1/3: python save_rotated_models.py --configs R1+R4
    → ./rotated_models_Qwen3-0.6B/R1+R4/Qwen3-0.6B-mxfp-w4g32/
  Step 2/3: lm_eval --model vllm ...
    → ./lm_eval_results_vllm/R1+R4/results.json
  Step 3/3: lm_eval --model hf ...
    → ./lm_eval_results_hf/R1+R4/results.json
  ⏱️  R1+R4 completed in 185s
```

### save_rotated_models.py — 模型导出

```bash
# 导出所有配置 (9种)
python save_rotated_models.py --device cuda:7

# 导出指定配置
python save_rotated_models.py --device cuda:7 --configs R1 R1+R4 R1+R2+R3

# 自定义输出路径
python save_rotated_models.py --device cuda:7 --output-base ./my_models
```

导出目录结构:
```
rotated_models_Qwen3-0.6B/
├── R1/Qwen3-0.6B-mxfp-w4g32/
│   ├── config.json              # 含 quantization_config + spinquant_config
│   ├── model.safetensors        # 量化权重 + spinquant_r1_type/size 参数
│   └── tokenizer files...
├── R1+R4/Qwen3-0.6B-mxfp-w4g32/
│   ├── config.json              # spinquant_config.r4=true
│   ├── model.safetensors        # 额外含 spinquant_r4_type/size on down_proj
│   └── ...
├── R1_random/Qwen3-0.6B-mxfp-w4g32/
│   ├── model.safetensors        # 额外含 spinquant_r1_matrix [1024,1024]
│   └── ...
└── R1+R2+R3/...                 # R3 模型 (仅 HF eval 支持)
```

### eval_hf.sh — HuggingFace 推理评估

```bash
# 评估所有已保存的模型 + FP16 baseline
bash eval_hf.sh

# 评估单个模型
bash eval_hf.sh ./rotated_models_Qwen3-0.6B/R1+R4

# 自定义 tasks 和 limit
TASKS="piqa,hellaswag" LIMIT=200 bash eval_hf.sh
```

### eval_vllm.sh — vLLM 推理评估

```bash
# 评估所有模型 (自动跳过 R3 配置)
bash eval_vllm.sh

# 多 GPU
CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 bash eval_vllm.sh
```

含 R3 的配置会自动检测并跳过:
```
  ⏭️  Skipping R1+R2+R3 (R3 not supported in vLLM plugin)
```

### compare_results.py — 精度对比

```bash
python compare_results.py --hf-dir ./lm_eval_results_hf --vllm-dir ./lm_eval_results_vllm
```

输出示例:
```
╔══════════════════════════════════════════════════════════════╗
║              Accuracy Comparison: HF vs vLLM                ║
╠══════════════════════════════════════════════════════════════╣
║  Config      │ Task       │ HF acc │ vLLM acc │ Δ          ║
║─────────────┼────────────┼────────┼──────────┼────────────║
║  FP16       │ piqa       │ 0.742  │ —        │ —          ║
║  R1         │ piqa       │ 0.738  │ 0.738    │ 0.000 ✓   ║
║  R1+R4      │ piqa       │ 0.735  │ 0.735    │ 0.000 ✓   ║
║  R1+R2+R3   │ piqa       │ 0.737  │ —        │ (HF only) ║
╚══════════════════════════════════════════════════════════════╝
```

### 单元测试

```bash
# 插件基础测试 (无需 GPU)
python auto_round/vllm_plugin/test_plugin.py

# R4 导出+加载端到端测试
python test_r4_export_and_plugin.py

# 数学正确性验证 (旋转矩阵正交性等)
python test_rotation_scheme_matrix_v2.py
```

---

## 局限性与未来工作

### 当前局限

1. **R3 不支持** — R3 需要在 attention 层内（RoPE 后、attention 计算前）插入旋转，需修改 attention 模块
2. **旋转矩阵内存** — 每层存储 [n, n] float16 矩阵，对大 hidden_size 模型(如 8192)可能占 ~50GB
3. **Triton kernel 精度** — 使用 TF32 accumulation，不如 IEEE FP32 精确
4. **TP 下旋转矩阵冗余** — 每个 rank 存完整矩阵（因为 ColumnParallel input 不分片）
5. **R4 TP 限制** — full-size R4 旋转（rotation_size=intermediate_size）不支持 TP > 1，详见下方

### R4 与 Tensor Parallelism (TP) 的兼容性

**核心问题：** `down_proj` 是 RowParallelLinear，TP > 1 时输入被分片：

```
TP=1: input shape = [batch, seq, intermediate_size]         → full rotation OK
TP=2: input shape = [batch, seq, intermediate_size / 2]     → full rotation FAILS
TP=4: input shape = [batch, seq, intermediate_size / 4]     → full rotation FAILS
```

**原因：** 保存时 `spinquant_r4_size = intermediate_size`（如 18432），但 TP>1 推理时
每个 rank 只看到 `intermediate_size/tp` 个元素。代码尝试用 `[18432, 18432]` 矩阵旋转
长度为 `9216` 的向量 → reshape 失败。

**R1 不受影响：** R1 作用在 ColumnParallelLinear（q/k/v/gate/up）的输入上，TP 下
ColumnParallel 的输入是完整的 hidden_size（不分片），因此任意 TP 都安全。

**解决方案：对 R4 使用 block rotation（rotation_size ≤ intermediate_size/tp）：**

| rotation_size | TP=1 | TP=2 | TP=4 | TP=8 |
|---------------|------|------|------|------|
| 32            | ✅   | ✅   | ✅   | ✅   |
| 128           | ✅   | ✅   | ✅   | ✅   |
| 1024          | ✅   | ✅   | ✅   | 需验证整除性 |
| None (full)   | ✅   | ❌   | ❌   | ❌   |

约束公式：`(intermediate_size / tp) % rotation_size == 0`

示例（Qwen3-32B, intermediate_size=18432）：
- TP=4: 每 rank 有 4608 元素, 4608 % 128 = 0 ✓, 4608 % 32 = 0 ✓

### 未来优化方向

1. **R3 实现** — 在 attention 模块注册 pre-hook (RoPE 后 matmul head_dim rotation)，backend-agnostic
2. **`fast_hadamard_transform` CUDA kernel** — 替代预计算矩阵，降为 O(n log n) 且省内存
3. **CUTLASS MXFP4 GEMM** — SM89+ 上使用硬件加速的 `matmul_mxf4_bf16_tn`
4. **与 vllm_ext 合并** — 统一为一个扩展，自动根据模型配置选择 rotation vs no-rotation 路径
5. **TP-aware full-size R4** — 通过 all-gather 支持 full intermediate_size 旋转（通信开销大，需评估）

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

**5. `AssertionError` in RowParallelLinear.weight_loader (R4 scalar shape)**
- 原因: `spinquant_r4_type/size` 注册为 shape `()` 的 scalar，但 RowParallelLinear
  的 weight_loader 会将 shape `()` reshape 为 `(1,)` 再做 assert
- 修复: 注册 R4 scalar 为 shape `(1,)` 而非 `()`

**6. R4 精度差 (non-power-of-2 维度如 3072)**
- 原因: Hadamard butterfly 矩阵构建方式不一致（见下方详细说明）
- 修复: `_build_full_hadamard()` 改用 `matmul_hadU(I)` 构建

### Hadamard Butterfly 结构不一致问题（已修复）

这是一个关键的数学正确性 bug，影响所有 **non-power-of-2 维度** 的 Hadamard 旋转。

**背景：** 对 n=3072 (K=12, n/K=256)，Hadamard 矩阵由两部分组成：
1. Butterfly 阶段：递归将维度从 n 降至 K
2. K-block 阶段：应用 [K×K] Hadamard 基础矩阵

**Bug：** `_build_full_hadamard`（旧版）和 `matmul_hadU`（save/hook 使用）的 butterfly
操作使用了不同的索引排列方式：

```python
# matmul_hadU: interleaved even/odd (view as [n/2, 2])
inp = inp.view(batch, n//2, 2, last_dim)
output[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]  # even indices
output[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]  # odd indices

# 旧 _build_full_hadamard: first-half / second-half split
B[start+i, start+i] = 1; B[start+i, start+half+i] = 1       # first half
B[start+half+i, start+i] = 1; B[start+half+i, start+half+i] = -1  # second half
```

**影响：**
- n 是 power-of-2（如 1024, K=1）：两种方式得到相同结果（误差 < 1e-6）
- n 非 power-of-2（如 3072, K=12）：完全不同的矩阵（误差 ~ 5.0）

**修复：** `_build_full_hadamard` 改为通过 `matmul_hadU(torch.eye(n))` 来构建显式矩阵，
确保与 save 时 offline fuse 和 online hook 使用完全相同的 Hadamard 变换。

```python
def _build_full_hadamard(n: int, device: torch.device) -> torch.Tensor:
    from auto_round.algorithms.transforms.spinquant.rotation_utils import matmul_hadU
    I = torch.eye(n, device=device, dtype=torch.float32)
    H = matmul_hadU(I)  # 等价于 I @ H = H
    return H
```

**验证：** 修复后所有维度一致性误差 < 1e-5 (float32 精度限制)。

### HF 推理 bfloat16 精度问题（已修复）

**问题：** HF 推理端使用 butterfly 算法在 bfloat16 精度下计算 Hadamard，
每次加减操作引入 ~0.4% 相对误差，经过 10 层 butterfly（n=1024=2^10）后
累积 ~2% 相对误差，28 层 × 5 个 linear = 140 次旋转 → 精度严重恶化。

**修复：** 在 `serialize.py` 的 `_apply_rotation_from_buffer()` 中，将输入上转为
float32 执行 butterfly 计算，结果再转回 bfloat16：

```python
def _apply_rotation_from_buffer(module, x, prefix):
    # Upcast for precision
    x_f32 = x.float()
    result = matmul_hadU(x_f32, ...)
    return result.to(x.dtype)
```

### 插件日志输出

vLLM 启动时，插件会输出详细的旋转配置信息：

```
INFO [register.py] [AutoRound] SpinQuant MXFP4 vLLM plugin loaded.
  Capabilities: R1/R4 online rotation + MXFP4 dequant (R2 offline in weights)

INFO [spinquant_mxfp4.py] [AutoRound] Model quantization: MXFP4 (group_size=32) |
  Online rotations: [R1(online, hadamard, size=1024, power-of-2),
                     R4(online, hadamard, size=3072, K=12 (non-power-of-2))] |
  hidden_size=1024, intermediate_size=3072

INFO [spinquant_mxfp4.py] [AutoRound] R4 rotation: size=3072
  (non-power-of-2, K=12, using matmul_hadU butterfly)
```

日志说明：
- **power-of-2**: 纯 Walsh-Hadamard butterfly（如 1024=2^10）
- **non-power-of-2, K=N**: 使用 K×K Hadamard 基矩阵 + butterfly 组合（如 3072=12×256）
- **R2(offline, fused into weights)**: R2 已经在保存时融入权重，推理时无需额外操作

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

### 单元测试

```bash
# 插件基础测试 (注册, config 解析, weight 创建, forward, override 检测)
python auto_round/vllm_plugin/test_plugin.py

# R4 导出→加载端到端测试 (导出 R1+R4 模型, 验证 checkpoint, 验证 plugin 加载)
python test_r4_export_and_plugin.py
```

### 数学正确性验证

```bash
# 旋转矩阵正交性, 不同 rotation_size, random vs hadamard 对比
python test_rotation_scheme_matrix_v2.py
```

### 端到端精度评估

```bash
# ═══ Qwen3-0.6B (快速验证，单卡) ═══

# 一键运行全流程: save → vLLM eval → HF eval (per config)
bash run.sh

# 指定配置
bash run.sh R1 R1+R4

# 单独步骤
python save_rotated_models.py --device cuda:7 --configs R1
bash eval_vllm.sh ./rotated_models_Qwen3-0.6B/R1
bash eval_hf.sh ./rotated_models_Qwen3-0.6B/R1

# 对比 HF vs vLLM 精度
python compare_results.py --hf-dir ./lm_eval_results_hf --vllm-dir ./lm_eval_results_vllm


# ═══ Qwen3-32B (完整评估，多卡) ═══

# 全部 10 种配置 (4 卡推理)
bash run_qwen3_32b.sh

# 指定配置
bash run_qwen3_32b.sh R1 R1+R4_size128

# 自定义 GPU 配置
SAVE_DEVICE=cuda:0 EVAL_GPUS=4,5,6,7 TP=4 bash run_qwen3_32b.sh

# 用 2 卡
EVAL_GPUS=0,1 TP=2 bash run_qwen3_32b.sh

# 模型已保存，只跑评估
SKIP_SAVE=1 bash run_qwen3_32b.sh
```

**Qwen3-32B 测试矩阵：**

| Config | R1 | R2 | R3 | R4 | rotation_size | TP-safe |
|--------|----|----|----|----|---------------|---------|
| R1 | ✓ | | | | 5120 (hidden) | ✅ |
| R1+R2 | ✓ | ✓ | | | 5120 | ✅ |
| R1+R2+R3 | ✓ | ✓ | ✓ | | 5120 | ⏭ skip vLLM |
| R1_random | ✓ | | | | 5120 | ✅ |
| R1_size32 | ✓ | | | | 32 | ✅ |
| R1_size128 | ✓ | | | | 128 | ✅ |
| R1+R4_size32 | ✓ | | | ✓ | 32 | ✅ |
| R1+R4_size128 | ✓ | | | ✓ | 128 | ✅ |
| R1+R2+R4_size32 | ✓ | ✓ | | ✓ | 32 | ✅ |
| R1+R2+R4_size128 | ✓ | ✓ | | ✓ | 128 | ✅ |

评测任务：`piqa, hellaswag, mmlu, gsm8k` 全集（无 limit）。

**注意：** R4 使用 `rotation_size=32` 或 `128`（block rotation）以兼容 TP > 1。
Full-size R4（rotation_size=intermediate_size）只能 TP=1 使用。

### Qwen3-32B 多卡 OOM 报错分析与解决方案

在 4 卡 TP=4 的 Qwen3-32B 评测中，`run_qwen3_32b.sh` 可能在第一个配置
（例如 `R1_size32`）的 vLLM 阶段直接失败。典型报错如下：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.56 GiB.
...
File ".../spinquant_mxfp4.py", line 554, in _spinquant_mxfp4_linear_impl
    output = _triton_mxfp4_gemm(...)
File ".../mxfp4_gemm.py", line 247, in triton_mxfp4_gemm
    output = torch.empty((M, N), dtype=torch.float32, device=input.device)
```

#### 结论

这类失败的主因是**显存不足**，不是 R1/R4 旋转逻辑错误，也不是 TP/NCCL 通信错误。
根因是：**vLLM 已经把大部分显存分配给权重、KV cache 和 CUDA graph，随后
SpinQuant MXFP4 Triton GEMM 还需要申请一个额外的 fp32 临时输出 buffer，最终触发 OOM。**

#### 触发链路

从日志可以看到，engine 初始化阶段显存已经非常紧张：

```text
Available KV cache memory: 33.83 GiB
GPU KV cache size: 553,984 tokens
CUDA graph pool memory: 1.67 GiB (actual)
```

而运行参数又比较激进：

```text
tensor_parallel_size=4
max_model_len=4096
max_num_batched_tokens=32768
max_num_seqs=128
gpu_memory_utilization=0.95
dtype=bfloat16
```

此时再进入插件的 MXFP4 线性层：

```python
# auto_round/triton_kernels/mxfp4_gemm.py
output = torch.empty((M, N), dtype=torch.float32, device=input.device)
```

这里会为 GEMM 分配一个 **`(M, N)` 的 float32 临时输出张量**。

#### 为什么恰好是 1.56 GiB

对 Qwen3-32B：

- `hidden_size = 5120`
- `intermediate_size = 25600`
- `TP = 4`

在 MLP 的 merged `gate_up_proj` 上，每个 rank 的输出维度为：

```text
N = 2 * intermediate_size / TP
  = 2 * 25600 / 4
  = 12800
```

日志里使用了：

```text
max_num_batched_tokens = 32768
```

在 chunked prefill / compile 路径下，`M` 可以达到这个 token budget，于是临时张量大小为：

```text
shape = (M, N) = (32768, 12800)
dtype = float32
bytes = 32768 * 12800 * 4
      = 1,677,721,600
      = 1.5625 GiB
```

这与日志中的：

```text
Tried to allocate 1.56 GiB
```

完全一致，因此可以确认是 **MXFP4 GEMM 临时 buffer** 导致的 OOM。

#### 为什么 TP=4 也会爆

虽然 TP=4 把权重分到了 4 张卡上，但每张卡仍然要同时容纳：

1. 本 rank 的量化权重
2. vLLM 预留的 KV cache
3. CUDA graph capture / compile 的额外显存
4. 当前 step 的 activation / temporary tensors
5. SpinQuant MXFP4 GEMM 的 fp32 输出 buffer

在日志里，OOM 前每卡只剩大约 **1.09~1.12 GiB** 可用显存，而插件一次就要申请
**1.56 GiB**，因此四个 rank 都会几乎同时失败。

#### 日志里哪些不是根因

以下 warning 容易让人误判，但它们不是这次 crash 的根因：

- `SymmMemCommunicator: Device capability 8.9 not supported`
  - 只影响某些通信优化路径
- `Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs`
  - 只影响性能，不会直接导致 OOM
- tokenizer regex warning
  - 可能影响评测精度，不会导致显存申请失败

#### 参数与显存的直接关系

对当前插件实现，`max_num_batched_tokens` 直接决定 MXFP4 GEMM 临时输出的大小。

以 Qwen3-32B、TP=4、merged `gate_up_proj` 为例：

| `max_num_batched_tokens` | 临时 buffer 大小 |
|---:|---:|
| 32768 | 1.5625 GiB |
| 24576 | 1.1719 GiB |
| 16384 | 0.7812 GiB |
| 12288 | 0.5859 GiB |
| 8192  | 0.3906 GiB |

这也是为什么**先降 token budget**往往最有效。

#### 推荐解决方案

建议按下面顺序收紧参数：

1. **先降 `max_num_batched_tokens`**
   - 从 `32768` 降到 `16384`
   - 如果还不稳，先用 `8192`

2. **降低 `gpu_memory_utilization`**
   - 从 `0.95` 降到 `0.85`
   - 更保守可用 `0.80`
   - 这样 vLLM 不会把 KV cache 撑得过满

3. **降低并发**
   - `batch_size: 64 -> 32` 或 `16`
   - `max_num_seqs: 128 -> 64`

4. **必要时降低 graph / compile 压力**
   - 关闭部分 graph capture 或 compile 路径可以回收一部分显存
   - 代价是吞吐下降

#### 推荐的保守起步参数

对 Qwen3-32B + TP=4 + SpinQuant MXFP4，建议先从更保守的参数起步：

```text
gpu_memory_utilization = 0.80 ~ 0.85
max_num_batched_tokens = 8192 ~ 16384
max_num_seqs = 64
batch_size = 16 ~ 32
```

如果这些参数稳定，再逐步放大吞吐。

#### 一个实用判断准则

如果日志里已经出现：

- KV cache 很大（例如 30+ GiB）
- CUDA graph pool 还有 1~2+ GiB
- OOM 前 free memory 只剩 1 GiB 左右

那么只要你的 MXFP4 GEMM 临时输出再需要 **1 GiB 以上**，就很容易在第一个大 prefill
batch 上直接爆掉。

#### 后续优化方向

当前 Triton kernel 的实现是：

```python
output = torch.empty((M, N), dtype=torch.float32, device=input.device)
```

这对精度是安全的，但峰值显存较高。后续若需要进一步优化 32B/70B 场景，可以考虑：

- 降低 `M`（调度层面）
- 避免一次性 materialize 整个大输出
- 研究更低峰值的分块输出 / accumulate 策略

不过在现阶段，**先收紧 vLLM 的并发与 token budget** 是最直接、最稳定的解决办法。

### 验证输出示例

```
[PASS] Plugin registration
[PASS] Config parsing (R1 + R4)
[PASS] Weight creation (spinquant_r1_type/size + spinquant_r4_type/size)
[PASS] Forward pass (R1 rotation + MXFP4 GEMM)
[PASS] R4 rotation (down_proj input)
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
