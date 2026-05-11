# SpinQuant / QuaRot 代码审查修复报告

本文档详细记录了代码审查中发现的 7 个问题，包括问题分析、修复方案和验证结果。

---

## 目录

1. [单元测试默认值断言不匹配](#1-单元测试默认值断言不匹配)
2. [scipy 依赖移除](#2-scipy-依赖移除)
3. [R2 融合忽略已存储的旋转矩阵](#3-r2-融合忽略已存储的旋转矩阵)
4. [QKRotationWrapper 未使用缓存的 Hadamard 矩阵](#4-qkrotationwrapper-未使用缓存的-hadamard-矩阵)
5. [R3 Monkeypatch 清理失败（isinstance 类型错误）](#5-r3-monkeypatch-清理失败isinstance-类型错误)
6. [训练时参考模型克隆时机](#6-训练时参考模型克隆时机)
7. [Online R1 Hook 不可序列化](#7-online-r1-hook-不可序列化)

---

## 1. 单元测试默认值断言不匹配

**严重级别:** 🔴 高  
**类型:** 测试 Bug  
**文件:** `test/test_cuda/transform/test_spinquant.py`

### 问题描述

`SpinQuantConfig` 的默认值已从 `r3=True, r4=True` 改为 `r3=False, r4=False`（默认只开启 R1+R2），但单元测试中的断言仍然检查旧的默认值：

```python
# 修复前 ❌
def test_default_config(self):
    cfg = SpinQuantConfig()
    assert cfg.r3 is True    # 旧默认值
    assert cfg.r4 is True    # 旧默认值
```

这会导致测试失败，因为实际默认值已变更。

### 修复方案

更新断言以匹配新的默认值：

```python
# 修复后 ✅
def test_default_config(self):
    cfg = SpinQuantConfig()
    assert cfg.r3 is False   # 新默认值
    assert cfg.r4 is False   # 新默认值
```

### 背景

改变默认值的原因：`"quarot"` 和 `"spinquant"` 字符串快捷方式默认只开启 R1+R2，因为 R3/R4 是在线 hook，会带来推理开销。用户需要显式配置 `SpinQuantConfig(r3=True, r4=True)` 才能开启全部旋转层级。

---

## 2. scipy 依赖移除

**严重级别:** 🟡 中  
**类型:** 依赖管理  
**文件:** `auto_round/algorithms/transforms/spinquant/rotation_utils.py`

### 问题描述

`get_hadamard_K()` 函数在处理 power-of-2 维度时，使用了 `scipy.linalg.hadamard` 来生成 Hadamard 矩阵：

```python
# 修复前 ❌
from scipy.linalg import hadamard as scipy_hadamard

if is_pow2(n):
    had = torch.Tensor(scipy_hadamard(n))
    return had, 1
```

问题：
- `scipy` 不在 auto-round 的 `requirements.txt` 中，属于隐性依赖
- 如果用户环境没有安装 scipy，在运行时会抛出 `ImportError`
- 代码中已有纯 PyTorch 的 `deterministic_hadamard_matrix()` 函数（Sylvester 构造法），完全可以替代

### 修复方案

用 PyTorch 的 Sylvester 构造法替代 scipy：

```python
# 修复后 ✅
if is_pow2(n):
    # Sylvester construction (unnormalized) — replaces scipy.linalg.hadamard
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H, 1
```

### 技术细节

**为什么不直接用 `deterministic_hadamard_matrix()`？**

因为 `get_hadamard_K()` 返回的是**未归一化**的 Hadamard 矩阵（H），而 `deterministic_hadamard_matrix()` 返回的是**归一化**的（H/√N）。归一化操作由下游的 `matmul_hadU()` 在最后统一执行（`result / math.sqrt(n)`），所以 `get_hadamard_K()` 必须返回未归一化版本。

**Sylvester 构造法原理：**
```
H_1 = [1]
H_2 = [H_1  H_1] = [1  1]
      [H_1 -H_1]   [1 -1]
H_4 = [H_2  H_2] = [1  1  1  1]
      [H_2 -H_2]   [1 -1  1 -1]
                    [1  1 -1 -1]
                    [1 -1 -1  1]
```
这与 `scipy.linalg.hadamard` 产生完全相同的矩阵，已通过正交性验证：`H @ H^T = n * I`。

---

## 3. R2 融合忽略已存储的旋转矩阵

**严重级别:** 🔴 高  
**类型:** 功能 Bug  
**文件:** `auto_round/algorithms/transforms/spinquant/preprocessor.py`

### 问题描述

`_fuse_r2_rotation()` 方法负责将 R2 旋转离线融合到 `v_proj` 和 `o_proj` 的权重中。代码先获取了存储的 R2 矩阵 `spinquant_R2_head`，但随后**完全忽略了它**，转而使用 `apply_hadamard_to_linear()` 生成一个全新的确定性 Hadamard 矩阵：

```python
# 修复前 ❌
def _fuse_r2_rotation(self):
    R2_head = self._get_rotation_tensor("spinquant_R2_head")  # 获取了 ← 但没使用！
    if R2_head is None:
        return
    for layer in self._get_layers():
        attn = layer.self_attn
        # 下面这两行总是生成新的确定性 Hadamard，忽略了 R2_head
        apply_hadamard_to_linear(attn.v_proj, had_dim=self.head_dim, output=True)
        apply_hadamard_to_linear(attn.o_proj, had_dim=self.head_dim, output=False)
```

**影响场景：**

| R2 类型 | 旧代码行为 | 是否正确 |
|---------|-----------|---------|
| 确定性 Hadamard（QuaRot 默认） | 生成新 Hadamard ≡ R2_head | ✅ 碰巧正确（因为对称 Hadamard H=H^T） |
| 随机 Hadamard（`random_r2=True`） | 生成确定性 Hadamard ≠ R2_head | ❌ 错误 |
| 可训练矩阵（SpinQuant，训练后） | 生成确定性 Hadamard ≠ R2_head | ❌ 错误 |

对于 QuaRot 默认配置（确定性 Hadamard），旧代码**碰巧正确**，因为 Hadamard 矩阵是对称的（H = H^T）。但对于随机或可训练的 R2，旧代码完全错误。

### 修复方案

直接使用存储的 R2_head 矩阵进行权重融合：

```python
# 修复后 ✅
def _fuse_r2_rotation(self):
    R2_head = self._get_rotation_tensor("spinquant_R2_head")
    if R2_head is None:
        return
    R2 = R2_head.data.to(torch.float64)
    R2_T = R2.t()

    for layer in self._get_layers():
        attn = layer.self_attn
        # v_proj: W_new = R2^T @ W per head (output dimension)
        W = attn.v_proj.weight.data.to(torch.float64)
        n_heads = W.shape[0] // self.head_dim
        W_reshaped = W.reshape(n_heads, self.head_dim, W.shape[1])
        W_reshaped = torch.einsum('ij,kjl->kil', R2_T, W_reshaped)
        attn.v_proj.weight.data = W_reshaped.reshape(W.shape).to(dtype)

        # o_proj: W_new = W @ R2^T per head (input dimension)
        W = attn.o_proj.weight.data.to(torch.float64)
        n_heads = W.shape[1] // self.head_dim
        W_reshaped = W.reshape(W.shape[0], n_heads, self.head_dim)
        W_reshaped = torch.einsum('ijk,lk->ijl', W_reshaped, R2)
        attn.o_proj.weight.data = W_reshaped.reshape(W.shape).to(dtype)
```

### 数学推导

R2 旋转的在线行为：对每个 attention head，value 向量乘以 R2：`v_rotated = v @ R2`

**融合到 v_proj（output 侧）：**
```
v = x @ W_v^T          (nn.Linear 的计算方式)
v_rotated = v @ R2      (R2 在线旋转)
         = x @ W_v^T @ R2
```
要融合：令 `W_v_new^T = W_v^T @ R2`，则 `W_v_new = R2^T @ W_v`（对每个 head 分块操作）

**融合到 o_proj（input 侧）：**
```
o_proj 的输入已经被 R2 旋转过 → 需要在 o_proj 权重中补偿
W_o_new = W_o @ R2^T（对每个 head 分块操作）
```

**einsum 解释：**
- `einsum('ij,kjl->kil', R2_T, W)`: 对每个 head k，计算 `R2^T @ W[k,:,:]`，即 R2^T 乘以 v_proj 权重
- `einsum('ijk,lk->ijl', W, R2)`: 对每个位置 (i,j)，计算 `sum_k W[i,j,k] * R2[l,k]`，等价于 `W @ R2^T`

**验证：** 当 R2 是对称 Hadamard（H = H^T）时，`R2^T @ W = H^T @ W = H @ W`，与旧代码的 `apply_hadamard_to_linear` 行为完全一致。✅

---

## 4. QKRotationWrapper 未使用缓存的 Hadamard 矩阵

**严重级别:** 🔴 高  
**类型:** 性能 Bug + 功能问题  
**文件:** `auto_round/algorithms/transforms/spinquant/monkeypatch.py`, `auto_round/algorithms/transforms/spinquant/inplace/apply.py`

### 问题描述

`QKRotationWrapper` 是 R3 在线旋转的核心组件，它在 RoPE 之后对 Q 和 K 应用 Hadamard 变换。问题有两个方面：

**问题 A：存储的矩阵从未使用**

```python
# 修复前 ❌
class QKRotationWrapper(nn.Module):
    def __init__(self, original_func):
        self._had_matrix = None    # 存储矩阵
        self._head_dim = 0

    def set_hadamard(self, had_matrix, head_dim):
        self._had_matrix = had_matrix   # 存储了 ←
        self._head_dim = head_dim

    def forward(self, *args, **kwargs):
        q, k = self.original_func(*args, **kwargs)
        if self._had_matrix is not None:   # 用作布尔标志
            q = matmul_hadU(q.float())     # ← 没有传入 _had_matrix！
            k = matmul_hadU(k.float())     # ← 每次重新计算 Hadamard
        return q, k
```

调用方传入的是一个空 tensor：
```python
# inplace/apply.py ❌
wrapper.set_hadamard(torch.empty(0), head_dim)  # 占位符，从未使用
```

**问题 B：每次 forward 都重新计算**

`matmul_hadU()` 在没有传入 `hadamard_K` 参数时，会在内部调用 `get_hadamard_K(n)` 重新计算 Hadamard 分解。对于一个有 28 层的模型，每个 forward pass 会调用 28×2 = 56 次 `get_hadamard_K()`，每次都生成相同的矩阵。

### 修复方案

**monkeypatch.py — 预计算并缓存 Hadamard 分解：**

```python
# 修复后 ✅
class QKRotationWrapper(nn.Module):
    def __init__(self, original_func):
        self._had_K = None    # 缓存 Hadamard K×K 矩阵
        self._K = 1           # Hadamard 分解的 K 值
        self._head_dim = 0

    def set_hadamard(self, had_matrix, head_dim):
        from .rotation_utils import get_hadamard_K
        self._head_dim = head_dim
        had_K, K = get_hadamard_K(head_dim)  # 一次性计算
        self._had_K = had_K
        self._K = K

    def forward(self, *args, **kwargs):
        q, k = self.original_func(*args, **kwargs)
        if self._had_K is not None:
            had_K = self._had_K.to(device=q.device, dtype=torch.float32)
            q = matmul_hadU(q.float(), hadamard_K=had_K, K=self._K)  # ✅ 传入缓存
            k = matmul_hadU(k.float(), hadamard_K=had_K, K=self._K)
        return q, k
```

**inplace/apply.py — 不再传入占位符：**

```python
# 修复后 ✅
wrapper.set_hadamard(None, head_dim)        # had_matrix 参数被忽略
module._spinquant_r3_patched = True         # 新增：标记模块以便后续清理
```

### 效果

- **消除每次 forward 的重复计算：** Hadamard 分解只在 `set_hadamard()` 时计算一次
- **消除 scipy 运行时依赖：** 不再在推理热路径中触发 `get_hadamard_K()` → scipy 导入
- **正确传递预计算的矩阵：** `matmul_hadU` 直接使用缓存的 `hadamard_K` 和 `K`

---

## 5. R3 Monkeypatch 清理失败（isinstance 类型错误）

**严重级别:** 🟡 中  
**类型:** Bug  
**文件:** `auto_round/algorithms/transforms/spinquant/inplace/apply.py`, `auto_round/algorithms/transforms/spinquant/preprocessor.py`

### 问题描述

有两个相关问题：

**问题 A：`remove_spinquant_hooks()` 中的 isinstance 类型错误**

```python
# 修复前 ❌ (inplace/apply.py)
if isinstance(h, tuple) and h[0] == "r3_monkeypatch":
    _, name, module, wrapper = h
    # isinstance 的第二个参数必须是类型，但 type(module).forward 是一个函数
    if hasattr(module, "forward") and not isinstance(module.forward, type(module).forward):
        #                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                                            这是一个函数对象，不是一个类型！
        #                                            → 抛出 TypeError
        delattr(module, "forward")
```

`isinstance()` 的第二个参数必须是一个类型（class），但 `type(module).forward` 是一个函数对象。这会抛出 `TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union`。

虽然外层有 `except Exception: pass` 捕获，但后果是 **R3 monkeypatch 永远不会被清理**。

**问题 B：`remove_spinquant_hooks_from_model()` 不处理 R3**

```python
# 修复前 ❌ (preprocessor.py)
def remove_spinquant_hooks_from_model(model):
    """Remove only SpinQuant-tagged forward hooks / pre-hooks from a model."""
    for module in model.modules():
        # 只检查 _forward_hooks 和 _forward_pre_hooks（R1/R4 的 hook）
        # 完全忽略了 R3 的 monkeypatch！
```

R3 不是通过 PyTorch hook 机制实现的，而是通过替换 attention 模块的 `forward` 方法（monkeypatch）。这个函数只处理 hook，不处理 monkeypatch。

### 修复方案

**A. 修复 handle-based 清理 (inplace/apply.py)：**

```python
# 修复后 ✅
if isinstance(h, tuple) and h[0] == "r3_monkeypatch":
    _, name, module, wrapper = h
    # 检查 module.__dict__ 中是否有实例级别的 forward 覆盖
    if "forward" in module.__dict__:
        del module.__dict__["forward"]     # 删除实例覆盖，回退到类方法
    if hasattr(module, "_spinquant_r3_patched"):
        delattr(module, "_spinquant_r3_patched")
```

关键改进：
- 用 `"forward" in module.__dict__` 替代有 bug 的 `isinstance` 检查
- `module.__dict__` 只包含实例级别的属性，不包含从类继承的方法
- `del module.__dict__["forward"]` 删除 monkeypatch 后，Python 会自动回退到类级别的原始 `forward`

**B. 添加 R3 标记和全局清理 (inplace/apply.py + preprocessor.py)：**

注册 R3 时标记模块：
```python
# inplace/apply.py ✅
wrapper = add_qk_rotation_after_rope(module, ...)
module._spinquant_r3_patched = True    # 新增标记
```

全局清理函数现在也处理 R3：
```python
# preprocessor.py ✅
def remove_spinquant_hooks_from_model(model):
    for module in model.modules():
        # ... 原有的 hook 清理逻辑 ...

        # 新增：R3 monkeypatch 清理
        if getattr(module, "_spinquant_r3_patched", False):
            if "forward" in module.__dict__:
                del module.__dict__["forward"]
            delattr(module, "_spinquant_r3_patched")
```

### R3 Monkeypatch 工作原理

理解这个修复需要了解 R3 的注入机制：

```
1. 原始: module.forward 是从类继承的方法
   → module.__dict__ 中没有 "forward"
   → Python 查找 type(module).forward 并绑定

2. Monkeypatch: setattr(module, "forward", new_method.__get__(module))
   → module.__dict__["forward"] = <bound patched method>
   → Python 优先使用实例级别的 forward

3. 清理: del module.__dict__["forward"]
   → 实例级别的 forward 被删除
   → Python 回退到 type(module).forward（原始方法）
```

---

## 6. 训练时参考模型克隆时机

**严重级别:** 🟢 无  
**类型:** 非 Bug（设计符合预期）

### 问题描述

Review 指出：在 `_setup_training()` 中，参考模型是在 R3/R4 hook 注册之后克隆的，导致克隆的模型也带有 R3/R4 的 monkeypatch，KL loss 可能会塌陷。

### 分析结论：非 Bug

实际的执行顺序（在 `preprocess()` 方法中）是：

```
Step 1: 验证维度
Step 2: 融合 RMSNorm gamma
Step 3: 替换 TrainableRMSNorm
Step 4: 初始化旋转矩阵
Step 5: 训练 (_train_rotations)    ← 克隆发生在这里
Step 6: 应用旋转（在线/离线）
Step 7: 注册在线 hook（R3/R4）    ← hook 在这里注册
Step 8: 清理
```

克隆发生在 Step 5，而 R3/R4 hook 在 Step 7 才注册。所以参考模型克隆时，**模型还没有 R3/R4 hook**，不存在 KL loss 塌陷的问题。

### 不需要修改

---

## 7. Online R1 Hook 不可序列化

**严重级别:** 🟡 中  
**类型:** 设计局限性  
**文件:** `auto_round/algorithms/transforms/spinquant/preprocessor.py`

### 问题描述

Online R1 使用 PyTorch 的 `forward_pre_hook` 机制在推理时对激活值应用 Hadamard 旋转。但 PyTorch 的 hook **不会被 `save_pretrained()` / `state_dict()` 保存**。

这意味着：
1. 用户用 online R1 处理模型
2. 保存模型 (`save_pretrained()`)
3. 重新加载模型 (`from_pretrained()`)
4. **激活值旋转 hook 丢失**，但权重已经被旋转过 → 模型输出错误

### 修复方案

添加运行时警告，让用户在使用 online R1 时知道这个限制：

```python
# preprocessor.py ✅
def _apply_online_r1(self):
    """..."""
    logger.warning(
        "[SpinQuant] Online R1 uses forward_pre_hooks which are NOT saved by "
        "save_pretrained(). The saved model will lose activation rotation hooks. "
        "Use SpinQuantConfig(online_r1_rotation=False) for offline R1 if you "
        "need to save/reload the model."
    )
```

同时更新了 docstring 中的 warning：

```python
.. warning::
    Hook-based online R1 is **not serializable** — ``save_pretrained()``
    will NOT save the activation hooks.  If you need to save and reload
    the rotated model, use offline R1 instead
    (``SpinQuantConfig(online_r1_rotation=False)``).
```

### 备注

代码中已有 `InputRotationWrapperHadamard` 类（nn.Module 包装器），它是可序列化的替代方案。它将原始的 `nn.Linear` 替换为一个包含 Hadamard 旋转的 wrapper module，由于是 `nn.Module`，可以通过 `save_pretrained()` / `state_dict()` 正常保存和恢复。未来可以考虑将 online R1 的实现从 hook 迁移到 `InputRotationWrapperHadamard`。

---

## 修改文件汇总

| 文件 | 修改内容 |
|------|---------|
| `test/test_cuda/transform/test_spinquant.py` | Fix 1: 更新 r3/r4 默认值断言 |
| `auto_round/.../spinquant/rotation_utils.py` | Fix 2: 用 torch Sylvester 构造替代 scipy |
| `auto_round/.../spinquant/preprocessor.py` | Fix 3: R2 融合使用存储的矩阵<br>Fix 5b: `remove_spinquant_hooks_from_model` 支持 R3 清理<br>Fix 7: 添加 online R1 序列化警告 |
| `auto_round/.../spinquant/monkeypatch.py` | Fix 4: `QKRotationWrapper` 预计算并缓存 Hadamard |
| `auto_round/.../spinquant/inplace/apply.py` | Fix 4b: 调用方不再传入占位符 `torch.empty(0)`<br>Fix 5a: 修复 isinstance bug，添加 R3 模块标记<br>Fix 5a: R3 清理使用 `module.__dict__` 检查 |

## 验证结果

所有 25 个单元测试通过：

```
test_spinquant.py::TestSpinQuantConfig          5/5   PASSED
test_spinquant.py::TestNormalizeRotationConfig   9/9   PASSED
test_spinquant.py::TestBaseRotationRegistry      3/3   PASSED
test_spinquant.py::TestHookLifecycle             2/2   PASSED
test_spinquant.py::TestRotationCorrectness       3/3   PASSED
test_spinquant.py::TestPipelineIntegration       3/3   PASSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
25 passed in 492.76s
```

额外验证：
- ✅ `get_hadamard_K`: 所有 power-of-2 尺寸 (2~128) 通过正交性验证 `H @ H^T = n * I`
- ✅ R2 融合：identity R2 不改变权重，symmetric Hadamard R2 结果与旧代码完全一致
- ✅ `QKRotationWrapper`: `set_hadamard()` 正确缓存，`forward()` 使用缓存矩阵
- ✅ R3 清理：handle-based 和 model-scan 两种方式都能正确清理 monkeypatch
