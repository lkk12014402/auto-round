# SpinQuant/QuaRot Code Quality Fixes — Changelog

> 本文档记录了对 `/data/lkk/quarot/latest/auto-round` 中 SpinQuant/QuaRot 实现的所有代码修复，
> 按严重级别排列。每个修复包含：问题描述、影响分析、修复前后代码对比。

---

## Critical（安全/正确性）

### C1: `torch.load()` 缺少 `weights_only=True` — 任意代码执行风险

**文件:** `spinquant/trainer.py` 第 392 行

**问题:** `torch.load` 底层使用 Python pickle 反序列化。pickle 可以在加载时执行任意 Python 代码。
如果用户加载了恶意构造的 checkpoint 文件，攻击者可以执行任意命令（删除文件、植入后门等）。
PyTorch 从 1.13 开始对此发出 deprecation warning，未来版本将默认禁止。

**影响:** 安全漏洞。仅在调用 `RotationTrainer.load_checkpoint()` 加载外部 checkpoint 时触发。

**修复前:**
```python
ckpt = torch.load(path, map_location="cpu")
```

**修复后:**
```python
ckpt = torch.load(path, map_location="cpu", weights_only=True)
```

加 `weights_only=True` 后，只允许反序列化张量、基础类型和 dict/list，拒绝任何可执行 Python 对象。

---

### C2: `remove_spinquant_hooks_from_model()` 误删所有 hook

**文件:** `spinquant/preprocessor.py` 第 1095–1110 行，`spinquant/inplace/apply.py` 第 163 行

**问题:** 原函数遍历模型所有 module，**无差别删除全部** `_forward_hooks` 和 `_forward_pre_hooks`。
如果其他框架（auto-round 量化层、DeepSpeed、FSDP、自定义 profiler 等）也注册了 hook，会被一起
误删，导致行为异常且无任何报错——这是一个静默 bug。

**影响:** 在训练模式下调用 `clone_model_for_reference()` 时触发（会调用此函数清理克隆模型的 hook）。
当前仅影响训练流程，但如果在推理流程中误调用，会破坏量化 hook。

**修复（两步）:**

**第一步 — 注册 hook 时打标记:**

`preprocessor.py`（R1 online hook）:
```python
# 修复前
handle = module.register_forward_pre_hook(hook)
self._r1_hook_handles.append(handle)

# 修复后
hook._spinquant_hook = True  # 在 hook 函数上打标记
handle = module.register_forward_pre_hook(hook)
self._r1_hook_handles.append(handle)
```

`inplace/apply.py`（R4 hook）:
```python
# 修复前
handle = module.register_forward_pre_hook(hook)

# 修复后
hook._spinquant_hook = True
handle = module.register_forward_pre_hook(hook)
```

> **关键细节:** 标记必须打在 **hook 函数对象** 上，不是 `register_forward_pre_hook` 返回的
> handle 上。因为 PyTorch 的 `module._forward_pre_hooks` 字典存储的是函数对象本身，不是 handle。
> 第一次修复时标记打在了 handle 上，UT 验证失败后才发现这个细节。

**第二步 — 删除时只删有标记的:**

```python
# 修复前
def remove_spinquant_hooks_from_model(model):
    for module in model.modules():
        for hook_id in list(module._forward_pre_hooks.keys()):
            try:
                del module._forward_pre_hooks[hook_id]  # 删除所有！
            except Exception:
                pass

# 修复后
def remove_spinquant_hooks_from_model(model):
    for module in model.modules():
        for hook_id in list(module._forward_pre_hooks.keys()):
            hook = module._forward_pre_hooks[hook_id]
            if getattr(hook, "_spinquant_hook", False):  # 只删自己的
                del module._forward_pre_hooks[hook_id]
```

---

### C3: `_fuse_rmsnorm_with_layer_paths()` 静默空操作

**文件:** `spinquant/rotation_utils.py` 第 553–559 行

**问题:** 这是一个为非 Llama 架构预留的 RMSNorm 融合函数，但函数体只有 `pass`——什么都不做，
静默返回。当 `get_scaling_layers()` 为某个架构返回了 layer paths 时，RMSNorm 融合会被跳过，
但上层代码不会报错，用户以为 R1 rotation 正常工作了。

**影响:** RMSNorm 没有被吸收进 linear 层，导致数学等价性被破坏。R1 offline rotation 需要
`W' = R @ W`，同时需要把 RMSNorm 的 `γ` 吸收进权重，否则 `γ * (x @ R)` ≠ `(γ * x) @ R`。
这会导致精度静默退化，且很难排查（因为没有报错信息）。

**修复前:**
```python
def _fuse_rmsnorm_with_layer_paths(model, layer_paths):
    for path in layer_paths:
        parts = path.split(".")
        pass  # Placeholder for full integration.
```

**修复后:**
```python
def _fuse_rmsnorm_with_layer_paths(model, layer_paths):
    raise NotImplementedError(
        "Model-config-driven RMSNorm fusion is not yet implemented. "
        "Use fuse_rmsnorm_into_linear() for Llama-family models, or set "
        "r1=False to skip R1 rotation for unsupported architectures."
    )
```

用户会立刻知道此路径不支持，可以选择：使用已实现的 Llama 路径，或设置 `r1=False` 跳过 R1。

---

## High（功能性 bug）

### H1: `AdamAndSGDG.step()` 对 closure 调用两次

**文件:** `spinquant/cayley_optimizer.py` 第 236–239 行

**问题:** `step(closure)` 中，Adam 和 SGDG 各自调用一次 closure。closure 包含前向传播和 loss
计算——被执行两次意味着：(1) 性能浪费 (2) 第二次调用可能看到不同的 dropout/random 状态，
导致两个优化器用不同的梯度更新，训练结果不可复现。

**影响:** 仅在 SpinQuant 训练模式（`trainable_rotation=True`）时触发。QuaRot 不受影响。

**修复前:**
```python
def step(self, closure=None):
    if self.adam_optimizer is not None:
        self.adam_optimizer.step(closure)   # closure 被调用第 1 次
    if self.sgdg_optimizer is not None:
        self.sgdg_optimizer.step(closure)   # closure 被调用第 2 次
```

**修复后:**
```python
def step(self, closure=None):
    # 只调用一次 closure，两个优化器共享同一次前向传播的梯度
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    if self.adam_optimizer is not None:
        self.adam_optimizer.step()       # 不传 closure
    if self.sgdg_optimizer is not None:
        self.sgdg_optimizer.step()       # 不传 closure
```

---

### H2: `torch.cuda.empty_cache()` 未检查 CUDA 可用性

**文件:** `spinquant/trainer.py` 第 485 行，`spinquant/preprocessor.py` 第 573 行

**问题:** 在 CPU-only 环境中，`torch.cuda.empty_cache()` 会抛出 RuntimeError。
虽然当前代码预期在 GPU 上运行，但作为库代码应该对运行环境做防御性检查。

**修复前:**
```python
torch.cuda.empty_cache()
```

**修复后:**
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

### H3: `_ensure_registry_populated()` 导入路径错误

**文件:** `transforms/base.py` 第 161 行

**问题:** 尝试导入 `auto_round.algorithms.transforms.hadamard`，但实际子包名是 `rotation`。
导入失败被 `except ImportError: pass` 静默吞掉，导致 HadamardRotation 不会通过这条路径
被注册到 registry。它能工作纯粹是因为 `__init__.py` 中有显式导入作为 fallback。

**影响:** 如果未来有人移除 `__init__.py` 中的显式导入，依赖 lazy registry 发现，
HadamardRotation 会找不到。

**修复前:**
```python
for sub in ("hadamard", "spinquant"):
```

**修复后:**
```python
for sub in ("rotation", "spinquant"):
```

---

## Medium（代码质量/维护性）

### M1: 删除死代码函数（cayley_optimizer.py）

**文件:** `spinquant/cayley_optimizer.py` 第 19–68 行

**删除了 5 个从未被调用的函数:**
- `unit()` — 向量归一化
- `_norm()` — L2 norm 封装
- `_matrix_norm_one()` — 矩阵 1-范数
- `Cayley_loop()` — Cayley 变换 retraction
- `qr_retraction()` — QR retraction

这些是从 Quark 移植时带过来的遗留代码。`SGDG.step()` 内联实现了自己的 QR retraction，
不依赖这些函数。同时删除了 `import copy` 和 `import random` 两个未使用的导入。

---

### M2: 删除未使用导入

**`training.py`:** 删除 `SGDG`（只用了 `AdamAndSGDG`）
```python
# 修复前
from ...cayley_optimizer import AdamAndSGDG, SGDG

# 修复后
from ...cayley_optimizer import AdamAndSGDG
```

**`preprocessor.py`:** 删除 `field`（只用了 `dataclass`）
```python
# 修复前
from dataclasses import dataclass, field

# 修复后
from dataclasses import dataclass
```

---

### M3: `normalize_rotation_config` dict 参数无 key 过滤

**文件:** `transforms/__init__.py` 第 104 行

**问题:** 当用户传入 dict 配置时（如 `{"algorithm": "spinquant", "r1": True, "typo_field": 42}`），
未过滤无效 key，直接 `SpinQuantConfig(**filtered)` 会抛出 `TypeError: unexpected keyword argument`。
相比之下，Hadamard 分支用的是 Pydantic 的 `model_validate` 会自动忽略未知字段。

**修复前:**
```python
filtered = {k: v for k, v in config.items() if k != "algorithm"}
return SpinQuantConfig(**filtered)
```

**修复后:**
```python
import dataclasses
valid_fields = {f.name for f in dataclasses.fields(SpinQuantConfig)}
filtered = {k: v for k, v in config.items() if k != "algorithm" and k in valid_fields}
return SpinQuantConfig(**filtered)
```

---

### M4: `_print_transformation_summary` 冗余条件

**文件:** `spinquant/preprocessor.py` 第 998 行和第 1050 行

**问题:** 两处代码 `"-" if self.config.r1 else "-"`，无论条件真假都返回 `"-"`。

**修复前:**
```python
r1_embed = "-" if self.config.r1 else "-"  # 两个分支结果相同
r1_lm = "-" if self.config.r1 else "-"
```

**修复后:**
```python
r1_embed = "-"  # online R1 doesn't touch embed
r1_lm = "-"     # online R1 doesn't touch lm_head
```

---

### M5: `__init__.py` 模块文档遗漏 spinquant

**文件:** `transforms/__init__.py` 第 14–23 行

**问题:** 模块 docstring 只提到了 `hadamard` 算法，没有提到 `spinquant`。

**修复:** 补充了 spinquant 的描述：
```python
# Current algorithms
# * hadamard — Simple Hadamard rotations.
# * spinquant — SpinQuant/QuaRot multi-level rotation (R1–R4) with optional
#   online hooks, trainable rotations, and known Hadamard matrices for non-pow2.
```

---

### M6: `AutoRound._resolve_config` 缺少 spinquant/quarot 支持

**文件:** `compressors_new/entry.py` 第 154–168 行

**问题:** `AutoRound` 新架构入口的 `_CONFIG_ALIASES` 只有 `"hadamard"`，不识别
`"quarot"` 和 `"spinquant"` 字符串。用户通过 `AutoRound(alg_configs=["quarot"])` 会报错。

**修复:** 在 `_resolve_config` 中加入对 `"quarot"`/`"spinquant"` 的分发，复用
`normalize_rotation_config()` 统一处理。

---

## 修复文件汇总

| 文件 | 修复项 |
|------|--------|
| `spinquant/trainer.py` | C1 (torch.load), H2 (cuda guard) |
| `spinquant/preprocessor.py` | C2 (hook 标记+选择性删除), H2, M2, M4 |
| `spinquant/inplace/apply.py` | C2 (R4 hook 标记) |
| `spinquant/rotation_utils.py` | C3 (NotImplementedError) |
| `spinquant/cayley_optimizer.py` | H1 (double closure), M1 (死代码) |
| `spinquant/training.py` | M2 (未使用导入) |
| `transforms/__init__.py` | M3 (key 过滤), M5 (docstring) |
| `transforms/base.py` | H3 (导入路径) |
| `compressors_new/entry.py` | M6 (resolve_config) |
