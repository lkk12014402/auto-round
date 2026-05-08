# Rotation 模型保存与加载方案（完整版）

## 1. 问题描述

Auto-round 使用 `forward_pre_hook` 实现在线旋转（online R1、R4），与量化管线完美兼容。
但 **hook 不是模型状态的一部分**——`model.state_dict()` 不包含 hook，`model.save_pretrained()` 不会保存 hook 逻辑。

量化后保存模型时，旋转信息丢失，推理时无法重建旋转。

---

## 2. 三种旋转矩阵类型及其持久化需求

| 类型 | 初始化方式 | 能否从参数重建 | 必须保存完整矩阵 |
|------|-----------|--------------|----------------|
| **Hadamard (固定)** | `scipy.linalg.hadamard(N)` 确定性构造 | ✅ 只需 `rotation_size` | ❌ 不用 |
| **QuaRot (随机)** | `random_±1_diagonal × Hadamard` | ❌ 随机种子不可复现 | ✅ 必须 |
| **SpinQuant (训练)** | Cayley 优化 on Stiefel 流形 | ❌ 训练结果不可复现 | ✅ 必须 |

### 2.1 Hadamard（确定性）

```python
# Quark: hadamard.py
hadamard_K = scipy.linalg.hadamard(N)  # 固定的 ±1 矩阵
# 归一化: H / sqrt(N)
```

- 给定 `rotation_size`，任何时候都能重建完全相同的矩阵
- Quark 用 `int8` dtype 存储（±1 值，N×N）
- Auto-round 只需存 `rotation_size` 整数即可

### 2.2 QuaRot（随机正交）

```python
# Quark: hadamard.py — random_hadamard_matrix()
Q = torch.randint(0, 2, (size,)).float() * 2 - 1  # 随机 ±1 向量
Q = torch.diag(Q)                                   # 随机 ±1 对角矩阵
rotation = _matmul_hadU(Q, hadamard_K, K)            # Q @ H → 随机正交矩阵
```

- **随机 ±1 对角矩阵 × Hadamard** → 结构化随机正交矩阵
- 每次生成不同（除非固定种子），**必须保存完整矩阵**
- Quark 用 `float64` dtype 存储（N×N 完整正交矩阵）

### 2.3 SpinQuant（训练优化）

```python
# Quark: rotation.py + cayley.py
# 初始化：
rotation_matrix = get_rotation_matrix(size, random=True)  # 随机或 Hadamard 初始化
rotation_param = nn.Parameter(rotation_matrix)             # 变为可训练参数

# 训练：
optimizer = AdamAndSGDG(...)  # SGDG 在 Stiefel 流形上优化，保持正交性
for step in range(num_steps):
    loss = kl_divergence(rotated_model(x), teacher(x))
    loss.backward()
    optimizer.step()  # Cayley retraction → 保持正交约束

# 训练后：
trained_rotation = rotation_param.data  # float64, N×N
```

- 初始化可以是随机或 Hadamard，但训练后矩阵不再是任何确定性函数
- **必须保存完整矩阵**（float64, N×N）
- 每个 R1/R2/R4 可以有独立的训练矩阵

---

## 3. Quark 如何解决持久化

Quark 使用 **三重持久化机制**：

### 3.1 `input_rotation` Buffer 存入 state_dict

```python
# rotation_utils.py — InputRotationWrapper.__init__()
if isinstance(self, InputRotationWrapperHadamard):
    self.register_buffer("input_rotation", rotation_matrix.to(torch.int8))   # ±1 值
elif isinstance(self, InputRotationWrapperOrthogonal):
    self.register_buffer("input_rotation", rotation_matrix.to(torch.float64))  # 完整矩阵
```

State dict 结构（经 `state_dict()` 展平后）：
```
model.layers.0.self_attn.q_proj.weight           # 来自 original_module
model.layers.0.self_attn.q_proj.bias             # 来自 original_module
model.layers.0.self_attn.q_proj.input_rotation   # int8 (Hadamard) 或 float64 (随机/训练)
```

### 3.2 通过 dtype 区分矩阵类型

```python
# 加载时：
if submodule.input_rotation.dtype == torch.int8:
    # Hadamard → 用 rotation_size 重建快速算法
    wrapper = InputRotationWrapperHadamard(submodule, rotation_size=size)
elif submodule.input_rotation.dtype == torch.float64:
    # 随机/训练 → 直接用保存的矩阵做矩阵乘法
    wrapper = InputRotationWrapperOrthogonal(submodule, rotation_matrix=matrix)
```

### 3.3 RotationConfig 序列化到 quantization_config

```json
{
  "algo_config": [{
    "name": "rotation",
    "r1": true, "r2": true, "r3": false, "r4": false,
    "trainable": true,
    "online_r1_rotation": true,
    "rotation_size": 128
  }]
}
```

---

## 4. Auto-round 的完整方案（支持三种矩阵类型）

### 4.1 方案概述

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 保存时                                                                   │
│  1. Hadamard: 只在 JSON 中记录 rotation_size（确定性重建）                 │
│  2. Random/Trained: 在模块上注册 buffer 保存完整矩阵 (float64)             │
│  3. rotation_config 写入 quantization_config.json（含矩阵类型标识）        │
│  4. model.save_pretrained() → buffer 随 state_dict 自动保存               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 加载时                                                                   │
│  1. model = AutoModel.from_pretrained(path) → 自动加载 buffer            │
│  2. 读取 quantization_config.json 中的 rotation_config                   │
│  3. 根据 rotation_type 决定重建策略：                                     │
│     - hadamard: 从 rotation_size 重建 → matmul_hadU hook                │
│     - random/trained: 从 buffer 读取矩阵 → matmul hook                  │
│  4. 注册 forward_pre_hook                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 具体实现

#### Step 1: 注册旋转 Buffer（按矩阵类型区分）

```python
def _apply_online_r1(self, model, config):
    rotation_size = config.rotation_size
    rotation_type = config.rotation_type  # "hadamard" | "random" | "trained"
    
    if rotation_type == "hadamard":
        hadamard_K = get_hadamard_K(rotation_size)
        rotation_matrix = None  # 不需要保存，可重建
    elif rotation_type == "random":
        rotation_matrix = random_hadamard_matrix(rotation_size)  # float64, N×N
        hadamard_K = None  # 不用快速算法，用矩阵乘
    elif rotation_type == "trained":
        rotation_matrix = config.trained_rotation_matrix  # 训练结果, float64
        hadamard_K = None
    
    for name, module in model.named_modules():
        if self._should_rotate_r1(name):
            # 1. 注册 hook
            if rotation_type == "hadamard":
                hook = self._make_hadamard_hook(hadamard_K, rotation_size)
            else:
                hook = self._make_matmul_hook(rotation_matrix)
            module.register_forward_pre_hook(hook)
            
            # 2. 注册持久化 buffer
            if rotation_type == "hadamard":
                # 只存一个标量标记 rotation_size（或不存，JSON 里有）
                module.register_buffer(
                    "_rotation_info",
                    torch.tensor([rotation_size], dtype=torch.int32)
                )
            else:
                # 必须保存完整矩阵
                module.register_buffer(
                    "_rotation_matrix",
                    rotation_matrix.clone().to(torch.float64)
                )
```

#### Step 2: rotation_config 写入 quantization_config.json

```json
{
  "bits": 4,
  "data_type": "mxfp4",
  "rotation_config": {
    "rotation_type": "hadamard",
    "r1": true,
    "r2": true,
    "r3": false,
    "r4": true,
    "online_r1_rotation": true,
    "rotation_size": 128,
    "r1_modules": ["model.layers.*.self_attn.q_proj", "..."],
    "r4_modules": ["model.layers.*.mlp.down_proj"],
    "has_rotation_buffers": false
  }
}
```

对于 random/trained 类型：
```json
{
  "rotation_config": {
    "rotation_type": "trained",
    "r1": true,
    "r2": true,
    "rotation_size": 128,
    "online_r1_rotation": true,
    "has_rotation_buffers": true,
    "buffer_name": "_rotation_matrix"
  }
}
```

#### Step 3: 加载时重建 Hook

```python
# auto_round/inference/rotation_loader.py

def restore_rotation_hooks(model, rotation_config: dict) -> None:
    """从 rotation_config + buffers 重建 forward_pre_hook。"""
    rotation_type = rotation_config.get("rotation_type", "hadamard")
    rotation_size = rotation_config["rotation_size"]
    
    if rotation_type == "hadamard":
        # 确定性重建，不需要 buffer
        hadamard_K = get_hadamard_K(rotation_size)
        make_hook = lambda mod: _make_hadamard_hook(hadamard_K, rotation_size)
    else:
        # 从 buffer 读取矩阵
        make_hook = lambda mod: _make_matmul_hook(mod._rotation_matrix)
    
    # R1 hooks
    if rotation_config.get("online_r1_rotation") and rotation_config.get("r1"):
        for name, module in model.named_modules():
            if _matches_r1_pattern(name, rotation_config):
                hook = make_hook(module)
                module.register_forward_pre_hook(hook)
    
    # R4 hooks
    if rotation_config.get("r4"):
        for name, module in model.named_modules():
            if _matches_r4_pattern(name, rotation_config):
                if rotation_type == "hadamard":
                    r4_size = module.in_features
                    h4 = get_hadamard_K(r4_size)
                    hook = _make_hadamard_hook(h4, r4_size)
                else:
                    hook = _make_matmul_hook(module._rotation_matrix)
                module.register_forward_pre_hook(hook)
    
    # R3: monkeypatch (独立于 buffer)
    if rotation_config.get("r3"):
        from auto_round.algorithms.transforms.spinquant.monkeypatch import inject_r3
        inject_r3(model)


def _make_hadamard_hook(hadamard_K, rotation_size):
    """Hadamard 快速算法 hook (O(N log N))。"""
    def hook(module, args):
        x = args[0]
        return (matmul_hadU(x, hadamard_K, rotation_size),) + args[1:]
    return hook


def _make_matmul_hook(rotation_matrix):
    """通用矩阵乘法 hook (O(N²))。"""
    R = rotation_matrix.float()  # 转为 float32 用于推理
    def hook(module, args):
        x = args[0]
        return (x @ R.to(x.device),) + args[1:]
    return hook
```

---

## 5. 对 Auto-round 代码的侵入性分析

### 5.1 需要修改的文件

| 文件 | 修改内容 | 侵入程度 |
|------|---------|---------|
| `export/utils.py` (line 374) | 从 `clean_list` 中移除 `"rotation_configs"`，改为序列化 | **1 行** |
| `compressors_new/base.py` (line 1162-1170) | 在 `save_quantized()` 中序列化 rotation_config | **~10 行** |
| `preprocessor.py` — `_apply_online_r1()` | 注册 buffer（已有 hook 注册逻辑旁） | **~5 行** |
| `inplace/apply.py` — R4 hook | 注册 buffer | **~3 行** |
| `inference/convert_model.py` (line 819-835) | 扩展已有的 rotation_config 处理逻辑 | **~15 行**（已有框架） |

### 5.2 需要新增的文件

| 文件 | 内容 | 大小 |
|------|------|------|
| `inference/rotation_loader.py` | `restore_rotation_hooks()` + `load_rotated_model()` | ~100 行 |

### 5.3 侵入性总结

```
┌──────────────────────────────────────────────────────┐
│ 修改 auto-round 核心代码:                              │
│   export/utils.py:         1 行（移除 clean_list 项）  │
│   compressors_new/base.py: ~10 行（序列化 config）     │
│   inference/convert_model.py: ~15 行（已有位置扩展）   │
│                                                      │
│ 修改 spinquant 自己的代码:                             │
│   preprocessor.py:  ~5 行（注册 buffer）              │
│   inplace/apply.py: ~3 行（注册 buffer）              │
│                                                      │
│ 新增文件:                                             │
│   inference/rotation_loader.py: ~100 行               │
│                                                      │
│ 总计: 修改 auto-round 核心 ≈ 26 行                    │
│       新增/修改 spinquant 模块 ≈ 108 行               │
└──────────────────────────────────────────────────────┘
```

**核心侵入点只有 1 个关键修改：** `export/utils.py` line 374 把 `rotation_configs` 从 clean_list 中移除。其余都是在已有的扩展点上工作。

---

## 6. 三种矩阵类型的存储开销

以 Qwen3-0.6B (hidden_size=1024, rotation_size=128) 为例：

| 类型 | 每个模块的 Buffer 大小 | 所有 R1 模块总计 (24层×5模块) | 备注 |
|------|----------------------|------------------------------|------|
| **Hadamard** | 4 bytes (int32 标记) | 480 bytes | 可忽略 |
| **Random** | 128×128×8 = 128 KB (float64) | 15.36 MB | 可共享（所有模块用同一矩阵） |
| **Trained** | 128×128×8 = 128 KB (float64) | 15.36 MB（独立）或 128 KB（共享） | 取决于是否 per-layer 训练 |

**优化：** 如果所有模块共享同一旋转矩阵（Quark 默认行为：`shared_parallel=True`），只需在一个地方保存矩阵，其余模块引用同一 buffer。

### 6.1 共享矩阵的持久化策略

```python
# 方案 A: 只在第一个模块存 buffer，JSON 中标记共享
"rotation_config": {
    "rotation_type": "random",
    "shared": true,
    "reference_module": "model.layers.0.self_attn.q_proj",
    ...
}

# 方案 B: 存为独立文件
# 保存: rotation_matrices.pt (包含 R1_matrix, R2_matrix, R4_matrix)
# 加载: torch.load("rotation_matrices.pt") → 注册 hook
```

**推荐方案 B**：独立文件更清晰，不侵入 state_dict。

```python
# 保存
torch.save({
    "r1_matrix": r1_rotation_matrix,   # [rotation_size, rotation_size], float64
    "r2_matrix": r2_rotation_matrix,   # [head_dim, head_dim], float64 (可选)
    "r4_matrix": r4_rotation_matrix,   # [intermediate_size, intermediate_size] (可选)
}, f"{output_dir}/rotation_matrices.pt")

# 加载
matrices = torch.load(f"{model_path}/rotation_matrices.pt")
```

---

## 7. 最终方案对比

| 维度 | Hadamard | Random (QuaRot) | Trained (SpinQuant) |
|------|----------|-----------------|---------------------|
| **JSON 中存储** | rotation_size | rotation_size + 引用 | rotation_size + 引用 |
| **矩阵保存位置** | 不需要保存 | `rotation_matrices.pt` 或 buffer | `rotation_matrices.pt` 或 buffer |
| **加载重建** | `get_hadamard_K(size)` | `torch.load(...)` | `torch.load(...)` |
| **Hook 类型** | `matmul_hadU` (O(N log N)) | `x @ R` (O(N²)) | `x @ R` (O(N²)) |
| **推理性能** | 快（蝴蝶算法） | 慢（需要实际矩阵乘法） | 慢（需要实际矩阵乘法） |
| **State dict 影响** | 无（或 4 bytes 标记） | 可选: buffer 在 state_dict 中 | 可选: buffer 在 state_dict 中 |

---

## 8. Quark 的完整参考（三种类型如何区分）

```python
# Quark 用 buffer dtype 区分：
# int8 → Hadamard (确定性，用快速算法)
# float64 → 随机或训练（通用矩阵乘法）

# 加载时：
if input_rotation.dtype == torch.int8:
    rotation_size = input_rotation.shape[0]
    # 重建快速 Hadamard → InputRotationWrapperHadamard
elif input_rotation.dtype == torch.float64:
    rotation_matrix = input_rotation
    # 直接矩阵乘 → InputRotationWrapperOrthogonal
```

---

## 9. 实现路线图

### Phase 1: Hadamard Only（最小可行）

适用于 QuaRot 的默认非训练模式。

1. `export/utils.py`: 从 clean_list 移除 `rotation_configs` → **1 行**
2. `base.py save_quantized()`: 序列化 `rotation_config` dict → **~10 行**
3. `preprocessor.py`: 注册 `_rotation_info` int32 buffer → **~5 行**
4. `inference/convert_model.py` 或新文件: 检测 config → 重建 hook → **~50 行**

**侵入 auto-round 核心：~11 行**

### Phase 2: + Random/Trained 支持

增加完整矩阵的保存/加载。

5. `preprocessor.py`: 增加 `_rotation_matrix` float64 buffer 注册 → **~10 行**
6. 保存路径增加 `rotation_matrices.pt` 写入 → **~15 行**
7. 加载路径增加矩阵读取 + `_make_matmul_hook` → **~30 行**

**额外侵入 auto-round 核心：~5 行**（保存时多写一个文件）

### Phase 3: + R3 Monkeypatch 持久化

8. R3 只需要在 JSON 中标记 `"r3": true`，加载时 `inject_r3(model)` → **~5 行**

---

## 10. 与 Quark 的等价性验证

```
Quark (Hadamard):
  x → InputRotationWrapperHadamard.forward()
    → x_rot = matmul_hadU(x, H, K)     # 快速 Hadamard
    → output = QuantLinear(x_rot)

Quark (Random/Trained):
  x → InputRotationWrapperOrthogonal.forward()
    → x_rot = x @ rotation_matrix      # 矩阵乘法
    → output = QuantLinear(x_rot)
    
Auto-round (Hadamard):
  x → forward_pre_hook(module, (x,))
    → x_rot = matmul_hadU(x, H, K)     # 同样的快速 Hadamard
    → return (x_rot,)
  → QuantizedLinear.forward(x_rot)

Auto-round (Random/Trained):
  x → forward_pre_hook(module, (x,))
    → x_rot = x @ R.float()            # 同样的矩阵乘法
    → return (x_rot,)
  → QuantizedLinear.forward(x_rot)
```

数学上完全等价，区别仅在于旋转代码的"容器"（wrapper vs hook）。

---

## 11. 关于"为什么不直接保存 wrapper"的解释

即使在保存时（量化已完成）将 hook 转换回 wrapper，也存在问题：

1. **transformers 不认识自定义类**：`from_pretrained()` 通过 `config.json` 的 `architectures` 字段决定模型类，不认识 `InputRotationWrapperHadamard`
2. **State dict key 不匹配**：如果保存时模块路径是 `q_proj.original_module.weight`，但加载时模型结构没有 wrapper，key 对不上
3. **推理生态不兼容**：vLLM、TGI、llama.cpp 等不认识自定义 wrapper 模块
4. **Quark 能用是因为它自己的加载流程会重建 wrapper**：先建模型结构 → 注册空 buffer → load_state_dict → 检测 buffer 重建 wrapper。这是 Quark 的闭环，外部框架做不到。

**Hook + Config 方案的优势：**
- 模型文件格式完全标准（`model.safetensors` 中 key 正常）
- 任何推理框架都能加载权重（只是没有旋转，精度会差）
- 旋转 hook 的注册是可选的增强步骤
- 对于 Hadamard，甚至不需要额外的 buffer（零存储开销）
