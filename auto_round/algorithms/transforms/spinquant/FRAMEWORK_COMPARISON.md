# QuaRot / SpinQuant 框架对比：auto-round vs Quark vs llm-compressor

> 三个框架的 R1–R4 rotation 实现方式、online/offline 策略、rotation 类型、
> block-diagonal 支持、模型序列化及量化集成的全面对比。

---

## 1. 总览对比表

| 特性 | **auto-round** | **Quark** (AMD) | **llm-compressor** (Neural Magic) |
|------|---------------|-----------------|-----------------------------------|
| **基础框架** | Intel auto-round | AMD Quark | Neural Magic vLLM 生态 |
| **Rotation 论文** | QuaRot + SpinQuant | QuaRot + SpinQuant | QuaRot + SpinQuant |
| **R1 模式** | online (默认) / offline | online / offline (默认) | offline only |
| **R2 模式** | offline (fuse) | offline (fuse) | offline (fuse) |
| **R3 模式** | online (monkeypatch) | online (monkeypatch) | online (attention hook) |
| **R4 模式** | online (hook) + offline fuse | online (hook) | online (hook) + offline fuse |
| **Hadamard** | ✅ butterfly O(n log n) | ✅ butterfly O(n log n) | ✅ Sylvester construction |
| **Random rotation** | ✅ random Hadamard (±1) | ✅ random Hadamard | ✅ random-hadamard + random-matrix |
| **Trained rotation** | ⚠️ 框架已有，未完全启用 | ✅ Cayley SGD 优化 | ❌ 不支持 |
| **R1 custom size** | ✅ rotation_size (power-of-2, 如 128) | ✅ rotation_size (power-of-2) | ✅ transform_block_size |
| **R2 custom size** | ❌ 仅 head_dim | ❌ 仅 head_dim | ❌ 仅 head_dim |
| **R3 custom size** | ❌ 仅 head_dim | ❌ 不支持 | ✅ transform_block_size |
| **R4 custom size** | ✅ rotation_size (power-of-2, 如 128) | ✅ rotation_size (power-of-2) | ✅ transform_block_size |
| **非 power-of-2** | ✅ 复合分解 (K × 2^m) | ✅ Kronecker 乘积 | ❌ 必须 power-of-2 |
| | | | |
| **R1 保存 (det)** | 仅 type+size (2个int32=8字节), 运行时 butterfly 重建 | 不存 (已 fuse 或 RotationLinear 重建) | 不需要 (已 fuse 到权重) |
| **R1 保存 (random)** | type+size+int8 矩阵 (±1), per-module buffer | RotationLinear 参数 (float) | N/A (不支持 online R1) |
| **R2 保存** | 不需要 (已 fuse 到 v_proj + o_proj 权重) | 不需要 (已 fuse) 或训练参数导出 | 不需要 (已 fuse) |
| **R3 保存** | 不需要 (Q@R·(K@R)^T 数学抵消) | 不需要 (运行时重建) | 子模块参数存储 |
| **R4 保存 (det)** | 仅 type+size (2个int32=8字节), 运行时 butterfly 重建 | 不存 (RotationLinear 重建) | 子模块参数存储 |
| **R4 保存 (random)** | type+size+int8 矩阵 (±1), down_proj buffer | RotationLinear 参数 (float) | 子模块参数存储 |
| **模型加载** | preregister 空 buffer → HF 填充 → rebuild hooks | RotationLinear 包装 | from_pretrained 自动恢复 |
| **量化方案** | W4A16, MXFP4, NVFP4, FP8 | W4A16, MXFP4, FP8 | W4A16, W8A8 |
| **支持的模型** | Llama/Qwen/Mistral/Phi/Gemma | Llama/Qwen3-MoE/GPT-OSS | Llama (显式) + 通用 regex |
| | | | |
| **推理额外延迟 (R1 online)** | Hadamard: O(n log n) butterfly, 低延迟; Random: O(n²) matmul | Hadamard: O(n log n); Random: O(n²) | N/A (仅 offline, 零开销) |
| **推理额外延迟 (R3)** | O(d log d) butterfly per head, 每 token 每层 | O(d log d) butterfly per head | O(d²) dense matmul per head |
| **推理额外延迟 (R4)** | O(m log m) butterfly (det) 或 O(m²) matmul (rand) | O(m²) RotationLinear matmul | O(m²) dense matmul |
| **推理总开销估计** | 确定性: <2% latency; 随机: 5-15% latency | 确定性: <2%; 随机/训练: 5-15% | offline 零开销; online R3/R4: 5-10% |
| | | | |
| **Rotation 组合灵活性** | ✅ 任意组合: R1/R2/R3/R4 独立开关 | ✅ 任意组合: r1/r2/r3/r4 config flags | ⚠️ 预定义 scheme (R1+R2, R1+R2+R4 等) |
| **Per-rotation 类型混合** | ✅ random_r1/r2/r3/r4 独立控制 | ⚠️ random_r1/r2 仅部分 | ❌ 全局统一 factory 类型 |
| **支持的组合示例** | none/R1/R1+R2/R1+R2+R3/R1+R2+R3+R4/R2 only/... | R1/R1+R2/R1+R2+R3+R4 | R1+R2/R1+R2+R4 |
| | | | |
| **KV-cache 量化配合** | ⚠️ R3 已实现, 但 KV-cache quant 未集成 | ✅ R3 + KV-cache INT4/FP8 量化 | ✅ R3 + QuantizedKVCache |
| **R3 的设计目的** | 为 KV-cache 量化预留 (当前改善权重量化) | 显式为 KV-cache 量化服务 | 显式为 KV-cache 量化服务 |
| | | | |
| **GQA 适配** | ✅ 自动检测 num_kv_heads, R2 按 head 独立旋转 | ✅ 配置指定 num_kv_heads | ✅ 正则匹配 kv_proj |
| **MQA 适配** | ✅ num_kv_heads=1 时正常工作 | ✅ | ⚠️ 未显式测试 |
| **GQA 时 R2 处理** | R2 分别作用于 V 的每个 kv_head (head_dim×head_dim) | R2 per kv_head 独立 | R2 per kv_head 独立 |
| **GQA 时 R3 处理** | 同一 R3 作用于所有 Q heads 和 KV heads | 同一 Hadamard 作用于所有 heads | 独立 hook per Q/K |

---

## 2. R1–R4 实现方式详细对比

### 2.1 R1: Embedding → Attention/MLP 输入旋转

R1 是最基础的旋转，将隐藏状态从标准基旋转到 Hadamard 基，使权重分布更均匀。

| 细节 | auto-round | Quark | llm-compressor |
|------|-----------|-------|----------------|
| **默认模式** | Online (hook) | Offline (fuse) | Offline (fuse) |
| **可切换** | ✅ `online_r1_rotation` | ✅ `online_r1_rotation` | ❌ 仅 offline |
| **Online 实现** | forward_pre_hook on q/k/v/gate/up_proj | RotationLinear 包装类 | N/A |
| **Offline fuse 目标** | embed/lm_head + q/k/v/gate/up/o/down_proj | embed→QKV + lm_head | embed/attn_o/mlp_out (output), q/k/v/mlp_in/lm_head (input inverse) |
| **RMSNorm fuse** | ✅ offline 时 fuse 到 linear | ✅ offline 时 fuse | ✅ fuse 到 linear |
| **Block-diagonal** | ✅ rotation_size 参数 | ✅ rotation_size 参数 | ✅ transform_block_size |
| **R1 矩阵存储** | online + random/trained 时存 buffer | online 时存 RotationLinear 参数 | 不需要（已 fuse） |

**数学**：
```
Online:  x' = x @ R1    (hook 在每层输入处应用)
Offline: W' = R1^T @ W  (预融合到权重中，运行时零开销)
```

**三框架差异解读**：
- auto-round 默认 online 是因为 online 对 ShardWriter 序列化更友好（每层独立保存）
- Quark 默认 offline 是因为推理性能更优（无 hook 开销）
- llm-compressor 只支持 offline，设计哲学是"所有能 fuse 的都 fuse"

---

### 2.2 R2: Attention Value → Output 旋转

R2 在 attention 的 value 路径上应用，改善 per-head 量化的精度。

| 细节 | auto-round | Quark | llm-compressor |
|------|-----------|-------|----------------|
| **模式** | 始终 offline | 始终 offline | 始终 offline |
| **Fuse 方式** | einsum per-head | rotate_out/in_channels | weight fuse per-head |
| **Fuse 目标** | v_proj (output) + o_proj (input) | v_proj (output) + o_proj (input) | attn_v (output) + attn_o (input inverse) |
| **矩阵维度** | head_dim × head_dim | head_dim × head_dim | head_dim × head_dim |
| **每层独立** | ✅ | ✅ (可共享) | ✅ |
| **存储** | 不需要（已 fuse） | 训练时存参数 | 不需要（已 fuse） |

**数学**：
```
V' = R2^T @ V_per_head     (v_proj output channels 旋转)
O' = O_per_head @ R2       (o_proj input channels 旋转)
推理时: softmax(QK^T)V' → result → O' → 完全等价于原始计算
```

**关键一致性**：三个框架的 R2 实现本质相同——都是 offline fuse 到 v_proj + o_proj，因为 R2 可以完全被权重吸收，无需运行时开销。

---

### 2.3 R3: RoPE 后 Q/K 旋转

R3 在 Rotary Position Embedding 之后对 Q 和 K 同时旋转，用于改善 KV-cache 量化精度。

| 细节 | auto-round | Quark | llm-compressor |
|------|-----------|-------|----------------|
| **模式** | 始终 online | 始终 online | 始终 online |
| **实现方式** | monkeypatch `apply_rotary_pos_emb` | monkeypatch `apply_rotary_pos_emb` | attention forward hook + KVCache hook |
| **Hook 类型** | 替换函数引用 (QKRotationWrapper) | 替换函数 (QKRotation wrapper) | register_query_hook + register_key_hook |
| **支持 random** | ✅ (matrix mode) | ❌ (仅 Hadamard) | ✅ (random-hadamard, random-matrix) |
| **Custom size** | ❌ 仅 head_dim | ❌ 不支持 | ✅ transform_block_size |
| **矩阵存储** | 不需要 (R 互相抵消) | 不需要 | 在线子模块存储 |

**数学**（R3 无损性证明）：
```
attention_score = softmax((Q @ R3) @ (K @ R3)^T / √d)
               = softmax(Q @ R3 @ R3^T @ K^T / √d)    // R3 正交: R3 @ R3^T = I
               = softmax(Q @ K^T / √d)                  // 与原始完全相同
```

**为什么 R3 不需要存储矩阵**：因为相同的 R3 同时作用于 Q 和 K，在 QK^T 点积中互相抵消，所以数学上完全等价。任何正交矩阵都可以用，结果相同。

**三框架差异**：
- auto-round 和 Quark 都用 monkeypatch 替换 `apply_rotary_pos_emb`，实现最直接
- llm-compressor 用独立的 attention/KVCache hook 系统，解耦性更好但更复杂

---

### 2.4 R4: MLP Down Projection 旋转

R4 在 MLP 的 gate/up → down 路径上旋转，改善 MLP 中间激活的量化精度。

| 细节 | auto-round | Quark | llm-compressor |
|------|-----------|-------|----------------|
| **模式** | Online hook + offline fuse | Online hook only | Online hook + offline fuse |
| **Online 实现** | forward_pre_hook on down_proj | RotationLinear 包装 | forward pre-hook |
| **Offline fuse** | ✅ `_fuse_r4_rotation()` 到 down_proj | ❌ | ✅ WEIGHT_INPUT fuse |
| **Block-diagonal** | ✅ rotation_size 参数 | ✅ rotation_size 参数 | ✅ transform_block_size |
| **矩阵维度** | intermediate_size × intermediate_size | intermediate_size × intermediate_size | intermediate_size |
| **矩阵存储** | random/trained 时存 buffer | 训练时存参数 | 在线子模块存储 |

**数学**：
```
Online:  x_down = (activation @ R4) @ W_down^T
Offline: W_down' = W_down @ R4,  然后 x_down = activation @ W_down'^T
         (等价: activation @ R4 @ W_down^T = activation @ (W_down @ R4)^T)

实际实现: 同时做 offline fuse + online hook (互相抵消保持数学一致)
```

---

## 3. Rotation 类型对比

### 3.1 确定性 vs 随机旋转

| 属性 | **确定性 Hadamard** | **随机 Hadamard** | **随机正交矩阵** | **训练正交矩阵** |
|------|-------------------|------------------|-----------------|-----------------|
| **数学形式** | H = H_K ⊗ H_{2^m} | D · H (D=diag(±1)) | QR 分解的 Q | Cayley(A) 优化 |
| **正交性** | ✅ 严格正交 | ✅ 严格正交 | ✅ 严格正交 | ✅ 严格正交 |
| **无损性** | ✅ 无量化时完全等价 | ✅ 无量化时完全等价 | ✅ 无量化时完全等价 | ✅ 无量化时完全等价 |
| **可复现** | ✅ 完全确定性 | ⚠️ 需固定 seed | ⚠️ 需固定 seed | ✅ 训练后固定 |
| **计算复杂度** | O(n log n) butterfly | O(n²) dense matmul | O(n²) dense matmul | O(n²) dense matmul |
| **存储需求** | 0 (从 size 重建) | int8 矩阵 (±1) | float32 矩阵 | float32 矩阵 |
| **存储开销/层** | ~8 bytes (type+size) | ~n² bytes (int8) | ~4n² bytes (float32) | ~4n² bytes (float32) |
| **量化后精度** | 好 | 理论略优* | 理论略优* | 最优 (数据驱动) |
| **推理开销** | 最低 (butterfly) | 中等 (matmul) | 中等 (matmul) | 中等 (matmul) |
| **适用场景** | 默认首选 | 资源充足时探索 | 非 power-of-2 维度 | 追求极致精度 |

> \* 随机旋转理论上可以更均匀地分散权重异常值，但实际增益通常很小。

### 3.2 三框架对 rotation 类型的支持

| 类型 | auto-round | Quark | llm-compressor |
|------|-----------|-------|----------------|
| **Hadamard (确定性)** | ✅ R1/R2/R3/R4 | ✅ R1/R2/R3/R4 | ✅ `hadamard` factory |
| **Random Hadamard** | ✅ per-rotation 控制 (random_r1/r2/r3/r4) | ✅ random_r1/r2 标志 | ✅ `random-hadamard` factory |
| **Random 正交矩阵** | ❌ | ❌ | ✅ `random-matrix` factory (任意大小) |
| **Trained (Cayley SGD)** | ⚠️ 代码框架存在但未完全启用 | ✅ 完整实现 | ❌ |

### 3.3 确定性 Hadamard 实现对比

| 实现细节 | auto-round | Quark | llm-compressor |
|---------|-----------|-------|----------------|
| **构造方法** | 预计算 H_K + butterfly 分解 | 预计算 H_K + butterfly 分解 | Sylvester 递归构造 |
| **butterfly 加速** | ✅ `matmul_hadU()` O(n log n) | ✅ `matmul_hadU()` O(n log n) | ❌ 密集矩阵乘法 |
| **非 power-of-2** | ✅ K ∈ {12,20,28,...} × 2^m | ✅ K ∈ {12,20,28,...} × 2^m | ❌ 要求 power-of-2 |
| **Kronecker 分解** | ✅ H = H_K ⊗ H_{2^m} | ✅ H = H_K ⊗ H_{2^m} | ❌ |

### 3.4 随机 Hadamard 实现对比

| 实现细节 | auto-round | Quark | llm-compressor |
|---------|-----------|-------|----------------|
| **生成方式** | D · H, D=random ±1 对角矩阵 | numpy random Hadamard | seeded random permutation |
| **存储格式** | int8 (±1 值) ≈ n² bytes | float 参数 | 子模块参数 |
| **加载还原** | `R.float() / √n` 归一化 | 直接加载 | 直接加载 |
| **seed 控制** | ✅ `--seed` 参数 | ✅ numpy seed | ✅ seed 参数 |
| **Per-rotation 控制** | ✅ random_r1/r2/r3/r4 独立 | ⚠️ random_r1/r2 仅部分 | ❌ 全局 factory 类型 |

---

## 4. Custom Rotation Size (Block-Diagonal) 对比

### 4.1 支持矩阵

| Rotation | auto-round | Quark | llm-compressor |
|----------|-----------|-------|----------------|
| **R1** | ✅ rotation_size | ✅ rotation_size | ✅ transform_block_size |
| **R2** | ❌ 固定 head_dim | ❌ 固定 head_dim | ❌ 固定 head_dim |
| **R3** | ❌ 固定 head_dim | ❌ 不支持 | ✅ transform_block_size |
| **R4** | ✅ rotation_size | ✅ rotation_size | ✅ transform_block_size |

### 4.2 Block-Diagonal 原理

当 `rotation_size < full_dimension` 时，旋转矩阵变为分块对角矩阵：

```
Full rotation (rotation_size = hidden_size = 1024):
  R = [1024 × 1024 正交矩阵]

Block-diagonal (rotation_size = 128, hidden_size = 1024):
  R = diag(R_128, R_128, R_128, R_128, R_128, R_128, R_128, R_128)
      ↑ 8 个独立的 128×128 块
```

**实现方式** (三框架类似)：
```python
# 输入: x shape = (batch, seq, hidden_size=1024)
x = x.reshape(*shape[:-1], -1, rotation_size)   # → (batch, seq, 8, 128)
x = x @ R                                        # R: (128, 128) 广播到每个块
x = x.reshape(*shape)                             # → (batch, seq, 1024)
```

**权衡**：
- 更小的 rotation_size → 更快的 butterfly/matmul（存储更少）
- 更大的 rotation_size → 更好的异常值分散效果
- 通常 128 是一个好的平衡点（= head_dim）

---

## 5. 模型保存与加载对比

### 5.1 保存策略

| 特性 | auto-round | Quark | llm-compressor |
|------|-----------|-------|----------------|
| **保存格式** | safetensors (buffer 注入) | safetensors (参数导出) | save_pretrained (HF 标准) |
| **Offline rotation** | 已 fuse 到权重，无需额外存储 | 已 fuse 到权重，无需额外存储 | 已 fuse 到权重，无需额外存储 |
| **Online R1 (det)** | 存 type + size (8 bytes) | 不存（运行时重建） | N/A (不支持 online R1) |
| **Online R1 (random)** | 存 type + size + int8 matrix | 存 RotationLinear 参数 | N/A |
| **R3** | 不存储（数学抵消） | 不存储 | 存为子模块参数 |
| **R4 (det)** | 存 type + size (8 bytes) | 不存（RotationLinear 重建） | 存为子模块参数 |
| **R4 (random)** | 存 type + size + int8 matrix | 存 RotationLinear 参数 | 存为子模块参数 |
| **配置信息** | config.json → spinquant_config | 单独 rotation config | config.json → transform_config |

### 5.2 auto-round Buffer 命名规则

```
每个需要 online rotation 的 module 上注册：
  {module}.spinquant_{R}_type     # int32: 0=HADAMARD, 1=RANDOM, 2=TRAINED
  {module}.spinquant_{R}_size     # int32: rotation 矩阵维度
  {module}.spinquant_{R}_matrix   # tensor: 仅 RANDOM/TRAINED 时存在

示例 (Qwen3-0.6B, R1 online + R4, 第 0 层):
  model.layers.0.self_attn.q_proj.spinquant_R1_type   = 0 (HADAMARD)
  model.layers.0.self_attn.q_proj.spinquant_R1_size   = 1024
  model.layers.0.mlp.down_proj.spinquant_R4_type      = 1 (RANDOM)
  model.layers.0.mlp.down_proj.spinquant_R4_size      = 2048
  model.layers.0.mlp.down_proj.spinquant_R4_matrix    = tensor(2048, 2048, dtype=int8)
```

### 5.3 加载流程对比

| 步骤 | auto-round | Quark | llm-compressor |
|------|-----------|-------|----------------|
| **1. 读取配置** | config.json → spinquant_config | rotation config dict | config.json → transform_config |
| **2. 创建占位** | preregister_buffers (空 buffer) | N/A (RotationLinear 类) | N/A (子模块自动创建) |
| **3. 加载权重** | HF load_state_dict 填充 buffer | HF load_state_dict | HF from_pretrained |
| **4. 重建 hooks** | rebuild_spinquant_online() | post_process_trained_rotation() | apply_transform_config() |
| **5. 推理** | QuantLinear.forward 自动应用 R1/R4 | RotationLinear.forward 应用 | hook 自动应用 |

---

## 6. 量化集成对比

| 特性 | auto-round | Quark | llm-compressor |
|------|-----------|-------|----------------|
| **pipeline 顺序** | rotation → calibration → quantize → save | rotation → train → quantize → save | SpinQuantModifier → QuantizationModifier |
| **量化时机** | rotation 后再量化 | rotation 后再量化 | rotation 后再量化 |
| **支持的量化** | W4A16 (INT4), MXFP4, NVFP4, FP8 | W4A16, MXFP4, FP8 | W4A16, W8A8 |
| **RTN (无调优)** | ✅ iters=0 | ✅ | ✅ |
| **Tuning** | ✅ AutoRound (iters=200+) | ✅ SGDG rotation training | ✅ GPTQ/SmoothQuant |
| **训练时量化** | weight + activation QDQ | activation only QDQ | weight + activation QDQ |
| **ShardWriter 集成** | ✅ per-layer buffer 注入 | ❌ (整体导出) | ❌ (整体 save_pretrained) |

---

## 7. Online vs Offline 决策树

```
                    ┌─────────────────────────────────────┐
                    │  Rotation 可以完全 fuse 到权重吗？     │
                    └──────────────┬──────────────────────┘
                         ┌────────┴────────┐
                         │                 │
                        YES               NO
                         │                 │
                    ┌────┴────┐      ┌────┴────┐
                    │ OFFLINE │      │ ONLINE  │
                    └─────────┘      └─────────┘
                         │                 │
              ┌──────────┤          ┌──────┤
              │          │          │      │
           R1 fuse    R2 fuse   R3 hook  R4 hook
        (embed→all)  (v→o per  (QK^T    (down_proj
         weights)    head)     cancels)  activation)

  R1 特殊: 可选 online/offline
  - Offline: 性能最优，但需要 fuse RMSNorm
  - Online: 序列化更简单，支持 per-layer save
```

**为什么 R2 只能 offline**：R2 作用于 v_proj 输出和 o_proj 输入，可以完全吸收到这两层权重中。

**为什么 R3 只能 online**：R3 必须在 RoPE 之后应用（RoPE 依赖 position，无法预计算），且应用于激活而非权重。

**为什么 R4 需要 online**：R4 作用于 MLP 中间激活（gate × up 的结果），这个值是运行时动态计算的，无法预 fuse。但 R4 可以同时 fuse 到 down_proj 权重以抵消一半计算。

---

## 8. 支持的模型架构对比

| 架构 | auto-round | Quark | llm-compressor |
|------|-----------|-------|----------------|
| **Llama / Llama2 / Llama3** | ✅ | ✅ | ✅ (显式注册) |
| **Qwen2 / Qwen3** | ✅ | ✅ (含 MoE) | ⚠️ (通用 regex 兜底) |
| **Mistral** | ✅ | ❌ 未列出 | ⚠️ (通用 regex 兜底) |
| **Phi / Phi-2 / Phi-3** | ✅ | ❌ 未列出 | ❌ |
| **Gemma** | ✅ | ❌ 未列出 | ❌ |
| **GPT-OSS** | ❌ | ✅ | ❌ |
| **GQA 支持** | ✅ | ✅ | ✅ |
| **MQA 支持** | ✅ | ✅ | ⚠️ |
| **MoE 支持** | ❌ | ✅ (Qwen3-MoE) | ❌ |
| **架构检测** | HF config + 结构探测 fallback | 显式配置 JSON | 显式注册 + regex fallback |

---

## 9. 特色与差异总结

### 9.1 auto-round 独有特性

| 特性 | 说明 |
|------|------|
| **ShardWriter per-layer 注入** | 量化时逐层保存，支持超大模型低内存导出 |
| **3 种 buffer 类型** | HADAMARD (0 存储) / RANDOM (int8) / TRAINED (float32)，按需最小化存储 |
| **Per-rotation random 控制** | random_r1/r2/r3/r4 独立控制每个旋转的类型 |
| **R4 同时 fuse + hook** | 预 fuse 到 down_proj 权重 + hook 抵消，减少运行时计算 |
| **非 power-of-2 Hadamard** | 支持 12,20,28... 等复合维度的模型 |
| **Benchmark 框架** | 12-part 全矩阵测试 + 多 GPU 调度 |

### 9.2 Quark 独有特性

| 特性 | 说明 |
|------|------|
| **完整 SpinQuant 训练** | Cayley SGD 优化器训练正交旋转矩阵 |
| **RotationLinear 包装类** | 统一的 online rotation 抽象，包含量化集成 |
| **共享旋转矩阵** | 多层可共享同一旋转矩阵 (`shared_parallel`) |
| **MoE 支持** | 支持 Qwen3 Mixture-of-Experts |
| **Rotation 导出** | 训练好的旋转矩阵可导出为 `rotations.safetensors` |

### 9.3 llm-compressor 独有特性

| 特性 | 说明 |
|------|------|
| **TransformFactory 注册机制** | 统一的 transform 工厂，hadamard/random-hadamard/random-matrix 可插拔 |
| **random-matrix (非 Hadamard)** | 支持完全随机正交矩阵（QR 分解），适用于任意维度 |
| **R3 custom block size** | R3 也支持 block-diagonal，其他框架不支持 |
| **HF 原生 save/load** | 完全兼容 `save_pretrained` / `from_pretrained` 流程 |
| **compressed_tensors 生态** | 与 vLLM 推理引擎深度集成 |

---

## 10. 选型建议

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| **Intel 平台部署** | auto-round | 原生集成，MXFP4/NVFP4 支持 |
| **AMD 平台部署** | Quark | AMD 官方，MoE 支持 |
| **vLLM 部署** | llm-compressor | compressed_tensors 原生支持 |
| **追求最高精度** | Quark | SpinQuant 训练（Cayley SGD）|
| **超大模型 (70B+)** | auto-round | ShardWriter per-layer 低内存 |
| **快速实验** | llm-compressor | HF 原生 save/load 最简单 |
| **非 power-of-2 模型** | auto-round / Quark | 复合 Hadamard 支持 |
| **Random rotation 实验** | auto-round | per-rotation 独立控制 |

---

## 附录 A: R1–R4 位置示意图

```
┌─────────────────────────────────────────────────────────────┐
│  Transformer Layer                                          │
│                                                             │
│  Input Hidden State                                         │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                                │
│  │ RMSNorm │                                                │
│  └────┬────┘                                                │
│       │                                                     │
│       ├──── [R1 online] ──→ x' = x @ R1                    │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                   │
│  │ Q_proj  │   │ K_proj  │   │ V_proj  │                    │
│  └────┬────┘   └────┬────┘   └────┬────┘                    │
│       │             │             │                          │
│       ▼             ▼             ├──── [R2 offline fused]   │
│  ┌─────────┐   ┌─────────┐       ▼                          │
│  │  RoPE   │   │  RoPE   │   V_rotated                      │
│  └────┬────┘   └────┬────┘       │                          │
│       │             │             │                          │
│       ├──── [R3 online] ──→ Q'=Q@R3, K'=K@R3               │
│       │             │             │                          │
│       ▼             ▼             ▼                          │
│  ┌──────────────────────────────────┐                        │
│  │      Attention: Q'K'^T V'       │                        │
│  └─────────────┬───────────────────┘                        │
│                │                                             │
│                ▼                                             │
│  ┌─────────────────┐                                        │
│  │ O_proj          │──── [R2 offline fused]                  │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼ (+residual)                                      │
│  ┌─────────┐                                                │
│  │ RMSNorm │                                                │
│  └────┬────┘                                                │
│       │                                                     │
│       ├──── [R1 online] ──→ x' = x @ R1                    │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐   ┌─────────┐                                  │
│  │gate_proj│   │ up_proj │                                  │
│  └────┬────┘   └────┬────┘                                  │
│       │             │                                        │
│       ▼             ▼                                        │
│  ┌─────────────────────┐                                    │
│  │  SiLU(gate) × up    │                                    │
│  └──────────┬──────────┘                                    │
│             │                                                │
│             ├──── [R4 online] ──→ act' = act @ R4           │
│             ▼                                                │
│  ┌─────────────────┐                                        │
│  │ down_proj       │──── [R4 offline fused into weight]     │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼ (+residual)                                      │
│       Output Hidden State                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 附录 B: 存储开销估算

以 Llama-3.1-8B 为例 (hidden=4096, heads=32, head_dim=128, intermediate=14336, 32 layers):

| Rotation | 类型 | 每层存储 | 32 层总计 | 占模型比例 |
|----------|------|---------|----------|-----------|
| R1 online (Hadamard) | type+size | ~40 bytes (5 modules × 8B) | ~1.3 KB | ≈ 0% |
| R1 online (random) | type+size+matrix | ~80 MB (5 × 4096²) | ~2.5 GB | ~16% |
| R1 online (random, size=128) | type+size+matrix | ~80 KB (5 × 128²) | ~2.5 MB | ≈ 0% |
| R2 | 已 fuse | 0 | 0 | 0% |
| R3 | 不存储 | 0 | 0 | 0% |
| R4 (Hadamard) | type+size | ~8 bytes | ~256 B | ≈ 0% |
| R4 (random) | type+size+matrix | ~196 MB (14336²) | ~6.3 GB | ~40% |
| R4 (random, size=128) | type+size+matrix | ~16 KB (128²) | ~512 KB | ≈ 0% |

**结论**：random rotation 全维度存储开销巨大，实际应用中应使用 block-diagonal (rotation_size=128) 将开销降至可忽略。

---

## Appendix C: English Overview Comparison Table

| Feature | **auto-round** (Intel) | **Quark** (AMD) | **llm-compressor** (Neural Magic) |
|---------|----------------------|-----------------|-----------------------------------|
| **R1 mode** | online (default) / offline | online / offline (default) | offline only |
| **R2 mode** | offline (fused) | offline (fused) | offline (fused) |
| **R3 mode** | online (monkeypatch) | online (monkeypatch) | online (attention hook) |
| **R4 mode** | online (hook) + offline fuse | online (hook) | online (hook) + offline fuse |
| | | | |
| **Hadamard (deterministic)** | ✅ butterfly O(n log n) | ✅ butterfly O(n log n) | ✅ Sylvester construction |
| **Random Hadamard** | ✅ per-rotation control (random_r1/r2/r3/r4) | ✅ random_r1/r2 flags | ✅ random-hadamard factory |
| **Random orthogonal (non-Hadamard)** | ❌ | ❌ | ✅ random-matrix (QR decomposition) |
| **Trained rotation (Cayley SGD)** | ⚠️ framework exists, not fully enabled | ✅ full implementation | ❌ |
| | | | |
| **R1 custom block size** | ✅ rotation_size (power-of-2) | ✅ rotation_size (power-of-2) | ✅ transform_block_size |
| **R2 custom block size** | ❌ head_dim only | ❌ head_dim only | ❌ head_dim only |
| **R3 custom block size** | ❌ head_dim only | ❌ not supported | ✅ transform_block_size |
| **R4 custom block size** | ✅ rotation_size (power-of-2) | ✅ rotation_size (power-of-2) | ✅ transform_block_size |
| **Non-power-of-2 support** | ✅ composite decomposition (K × 2^m) | ✅ Kronecker product | ❌ must be power-of-2 |
| | | | |
| **R1 save (deterministic)** | type+size only (8 bytes), butterfly rebuild at runtime | not stored (fused or RotationLinear rebuild) | not needed (fused into weights) |
| **R1 save (random)** | type+size+int8 matrix (±1), per-module buffer | RotationLinear param (float) | N/A (no online R1) |
| **R2 save** | not needed (fused into v_proj + o_proj) | not needed (fused) or trained param export | not needed (fused) |
| **R3 save** | not needed (Q@R·(K@R)^T cancels mathematically) | not needed (runtime rebuild) | submodule parameter |
| **R4 save (deterministic)** | type+size only (8 bytes), butterfly rebuild at runtime | not stored (RotationLinear rebuild) | submodule parameter |
| **R4 save (random)** | type+size+int8 matrix (±1), down_proj buffer | RotationLinear param (float) | submodule parameter |
| **Model loading** | preregister empty buffers → HF fill → rebuild hooks | RotationLinear wrappers | from_pretrained auto-restore |
| | | | |
| **Inference latency (R1 online)** | Hadamard: O(n log n) butterfly, low; Random: O(n²) matmul | Hadamard: O(n log n); Random: O(n²) | N/A (offline only, zero overhead) |
| **Inference latency (R3)** | O(d log d) butterfly per head per token per layer | O(d log d) butterfly per head | O(d²) dense matmul per head |
| **Inference latency (R4)** | O(m log m) butterfly (det) or O(m²) matmul (rand) | O(m²) RotationLinear matmul | O(m²) dense matmul |
| **Total overhead estimate** | Deterministic: <2%; Random: 5-15% | Deterministic: <2%; Random/Trained: 5-15% | Offline: 0%; Online R3/R4: 5-10% |
| | | | |
| **Rotation combo flexibility** | ✅ arbitrary: R1/R2/R3/R4 independent on/off | ✅ arbitrary: r1/r2/r3/r4 config flags | ⚠️ predefined schemes (R1+R2, R1+R2+R4) |
| **Per-rotation type mixing** | ✅ random_r1/r2/r3/r4 independently | ⚠️ random_r1/r2 only partial | ❌ global factory type |
| **Supported combos** | none/R1/R1+R2/R1+R2+R3/R1+R2+R3+R4/R2-only/... | R1/R1+R2/R1+R2+R3+R4 | R1+R2/R1+R2+R4 |
| | | | |
| **KV-cache quantization** | ⚠️ R3 implemented, but KV-cache quant not integrated | ✅ R3 + KV-cache INT4/FP8 | ✅ R3 + QuantizedKVCache |
| **R3 design purpose** | Reserved for KV-cache quant (currently improves weight quant) | Explicitly for KV-cache quantization | Explicitly for KV-cache quantization |
| | | | |
| **GQA adaptation** | ✅ auto-detect num_kv_heads, R2 per kv-head | ✅ config-specified num_kv_heads | ✅ regex match kv_proj |
| **MQA adaptation** | ✅ num_kv_heads=1 works correctly | ✅ | ⚠️ not explicitly tested |
| **R2 with GQA** | R2 applied per kv_head independently (head_dim × head_dim) | R2 per kv_head independent | R2 per kv_head independent |
| **R3 with GQA** | Same R3 applied to all Q-heads and KV-heads | Same Hadamard for all heads | Independent hook per Q/K |
| | | | |
| **Quantization schemes** | W4A16, MXFP4, NVFP4, FP8 | W4A16, MXFP4, FP8 | W4A16, W8A8 |
| **Supported architectures** | Llama/Qwen/Mistral/Phi/Gemma | Llama/Qwen3-MoE/GPT-OSS | Llama (explicit) + generic regex |
| **MoE support** | ❌ | ✅ (Qwen3-MoE) | ❌ |
