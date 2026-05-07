# Auto-Round SpinQuant 训练兼容性改进方案

## 现状分析

### Quark 的架构

Quark 用 `RotationLinear` 包裹原始 `nn.Linear`，在 forward 中做**在线旋转**：

```
RotationLinear.forward(x):
    weight = self.linear.weight                     # 原始权重，训练中不直接修改
    x, weight = apply_in_transform(x, weight)       # R_in 旋转 x 和 weight
    weight, bias = apply_out_transform(weight, bias) # R_out 旋转 weight
    return F.linear(x, weight, bias)                 # 用旋转后的 weight 做 matmul
```

核心设计：
- **旋转矩阵是 `nn.Parameter`**，可以反向传播
- **训练时**：每次 forward 在线计算 `W @ R`，梯度流过 R
- **训练后**：`post_process_trained_rotation()` 将学到的 R 融合进权重，并替换为 `InputRotationWrapperHadamard`（推理用的轻量 wrapper）

### Quark 各旋转的 online/offline 情况

| 旋转 | 训练模式 | 非训练模式 | 有 online 版本？ |
|------|----------|-----------|-----------------|
| R1 | `RotationLinear(rotation_in=R1)` 包裹每个 linear | online_r1=True: `InputRotationWrapperHadamard`; online_r1=False: 直接融合权重 | ✓ |
| R2 | `RotationLinear(rotation_out=R2)` 在 v_proj，`rotation_in=R2` 在 o_proj | 直接 `rotate_out_channels_` / `rotate_in_channels_` 融合 | ✗（训练用 RotationLinear，推理融合） |
| R3 | monkeypatch `apply_rotary_pos_emb`（不可训练） | 同左 | ✓（始终在线） |
| R4 | `RotationLinear(rotation_in=R4)` 在 down_proj | `InputRotationWrapperHadamard` 包裹 down_proj | ✓ |

**结论：Quark 没有 online R2**。R2 在训练时用 `RotationLinear` 做可微旋转，训练后融合进权重。

---

## 改进方案

### Phase 1: 支持 `rotation_size` (block Hadamard R1)

**目标：** R1 旋转支持 `rotation_size < hidden_size`

改动点：
1. `SpinQuantConfig` 增加 `rotation_size: int | None = None`
2. `_init_rotation_matrices()` 中 R1 矩阵大小从 `hidden_size` 改为 `rotation_size or hidden_size`
3. `_fuse_offline_rotations()` 中 R1 融合逻辑适配 block 旋转
4. 当 `rotation_size < hidden_size` 时，R1 变为 block-diagonal 旋转

估计改动量：~50 行

### Phase 2: `RotationLinear` 训练兼容 wrapper

**目标：** 支持 SpinQuant 单独训练（KL loss 优化旋转矩阵）

需要实现 Quark 的 `RotationLinear` 等价物：

```python
class RotationLinear(nn.Module):
    """训练时的可微旋转 wrapper"""
    def __init__(self, linear, rotation_in=None, rotation_out=None, 
                 smooth_values_in=None, rotate_activation=False):
        self.linear = linear
        self.rotation_in = rotation_in      # nn.Parameter，可训练
        self.rotation_out = rotation_out     # nn.Parameter，可训练
        self.smooth_values_in = smooth_values_in  # SmoothQuant
        self.rotate_activation = rotate_activation

    def forward(self, x):
        weight = self.linear.weight
        if self.rotation_in is not None:
            weight = weight @ self.rotation_in       # 旋转权重输入通道
            if self.rotate_activation:
                x = x @ self.rotation_in             # 旋转激活
        if self.rotation_out is not None:
            weight = self.rotation_out.T @ weight    # 旋转权重输出通道
        return F.linear(x, weight, self.linear.bias)
```

改动点：
1. 新增 `rotation_linear.py`（~150 行）
2. `preprocessor.py` 训练分支：用 `RotationLinear` 包裹 linear layers
3. `preprocessor.py` 训练后：`_post_process_trained_rotation()` 融合 + 替换

流程：
```
preprocess(trainable=True):
  1. fuse_rmsnorm
  2. 包裹 linear → RotationLinear (R_in/R_out 作为 nn.Parameter)
  3. train(dataloader, loss=KL(rotated_model, original_model))
  4. post_process: RotationLinear → 融合权重 + InputRotationWrapper(推理用)
  5. register R3/R4 hooks
```

估计改动量：~300 行

### Phase 3: 与 Auto-Round tuning 联合训练

**目标：** rotation 训练和 auto-round 量化优化同时进行

这里有两种策略：

#### 策略 A：串行（简单，推荐先做）

```
Step 1: SpinQuant 训练旋转矩阵 (KL loss, ~200 iters)
Step 2: 融合旋转到权重
Step 3: Auto-Round tuning 优化量化 (iters>0, round loss)
```

改动：
- `AutoRound` 初始化时接受 `SpinQuantConfig`，在 quantize() 之前调用 preprocess()
- 或者用户手动先 preprocess 再传给 AutoRound（当前已支持）

#### 策略 B：联合训练（复杂）

```
单一训练循环:
  for batch in dataloader:
      # Forward: x → RotationLinear(含 QDQ) → loss
      loss = KL(quantized_rotated_model, original_model) + round_loss
      loss.backward()
      # 更新旋转矩阵（Cayley SGD）+ 量化参数（Adam）
```

这需要：
- `RotationLinear` 的 forward 中插入 QDQ（量化-反量化）
- 两组 optimizer：SGDG for 旋转矩阵，Adam for 量化 scale/round
- Loss 函数融合

这本质上是 Quark 的 `train_rotation.py` + AutoRound 的 tuning loop 合并。
复杂度高，建议在 Phase 2 稳定后再考虑。

---

## 建议实施顺序

1. **Phase 1** (rotation_size) — 最小改动，立即可做
2. **Phase 2** (RotationLinear + 训练) — 核心功能，需要仔细设计
3. **Phase 3A** (串行：先 rotation 训练，再 auto-round tuning) — Phase 2 完成后自然支持
4. **Phase 3B** (联合训练) — 长期目标

---

## 关键设计决策

| 决策 | Quark 的选择 | 建议 auto-round 的选择 |
|------|------------|----------------------|
| R1 online vs offline | 支持两种 (`online_r1_rotation` flag) | 先做 offline，后续加 online |
| 训练 wrapper | `RotationLinear` 包裹 `nn.Linear` | 同样思路，实现 `RotationLinear` |
| 训练后处理 | `post_process_trained_rotation()` 逐层融合 | 同样思路 |
| 旋转优化器 | SGDG (Cayley manifold SGD) | 已有 `cayley_optimizer.py`，可复用 |
| R3 训练 | 不可训练（Hadamard 固定） | 同 |
| R4 训练 | `shared_parallel` 选项（层间共享 R4） | 可选，先不共享 |
