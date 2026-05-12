# SpinQuant/QuaRot 时序图

本文档以 Mermaid 时序图描述 rotation + quantization 的完整流程，分为三大阶段：

1. **预处理阶段**（Rotation 应用 + 量化前准备）
2. **保存阶段**（量化后模型序列化）
3. **加载阶段**（反序列化 + 在线旋转重建）

---

## 图 1: 总览 — 三阶段全流程

```mermaid
sequenceDiagram
    participant User as 用户脚本
    participant AR as AutoRound<br/>(autoround.py)
    participant TF as transforms/__init__.py<br/>(dispatch)
    participant PP as SpinQuantPreprocessor
    participant Model as HF Model
    participant Export as export_to_*
    participant Disk as Safetensors<br/>+ config.json
    participant Load as convert_model.py<br/>(加载侧)

    Note over User,Load: ═══════ 阶段 1: 预处理 (Rotation) ═══════

    User->>AR: AutoRound(model, rotation_config="spinquant")
    AR->>TF: apply_rotation(model, config)
    TF->>PP: SpinQuantPreprocessor.preprocess(model)
    PP->>Model: 初始化 R1/R2/R3/R4 矩阵
    PP->>Model: 融合离线旋转 (R2, R4 offline fuse)
    PP->>Model: 注册在线 hooks (R1/R3/R4)
    PP-->>AR: 返回已旋转模型 (带 hooks)

    Note over User,Load: ═══════ 阶段 2: 量化 + 保存 ═══════

    AR->>AR: quantize() — RTN / iters=200 签名量化
    AR->>Export: save_quantized(model, save_dir)
    Export->>TF: inject_rotation_buffers (per-layer 或 bulk)
    TF->>Model: 注入 spinquant_r1/r4 buffers 到 QuantLinear
    Export->>Disk: model.save_pretrained() → safetensors
    Export->>TF: save_rotation_config(model, save_dir)
    TF->>Disk: 写 spinquant_config 到 config.json

    Note over User,Load: ═══════ 阶段 3: 加载 + 重建 ═══════

    User->>Load: AutoModelForCausalLM.from_pretrained(save_dir)
    Load->>TF: preregister_rotation_buffers(model, quant_config)
    TF->>Model: 预注册空 buffers (type/size/matrix)
    Load->>Load: HF state_dict loader → 填充 buffer 数据
    Load->>TF: rebuild_rotation_if_needed(model)
    TF->>Model: patch QuantLinear.forward (R1/R4 buffer 旋转)
    TF->>Model: 重建 R3 monkeypatch hook
    Load-->>User: 返回可推理模型
```

---

## 图 2: 预处理阶段详细流程

展示 `SpinQuantPreprocessor.preprocess()` 的 8 个步骤：

```mermaid
sequenceDiagram
    participant Caller as apply_rotation()
    participant PP as SpinQuantPreprocessor
    participant Model as HF Model
    participant RotUtils as rotation_utils.py
    participant Apply as inplace/apply.py

    Caller->>PP: preprocess(model, dataloader)

    Note over PP: Step 1: 模型 dtype 转换
    PP->>Model: model.to(config.dtype)

    Note over PP: Step 2: RMSNorm 融合 (offline R1 only)
    alt offline R1
        PP->>Model: fuse_rmsnorm_in_model(model)
    else online R1
        PP->>PP: skip (不需要融合)
    end

    Note over PP: Step 3: 可训练 SmoothQuant (可选)
    opt trainable_smooth=True
        PP->>Model: replace RMSNorm → TrainableRMSNorm
    end

    Note over PP: Step 4: 初始化旋转矩阵
    PP->>RotUtils: random_hadamard_matrix() / deterministic_hadamard_matrix()
    RotUtils-->>PP: R1, R2, R3, R4 矩阵

    alt random_r1
        PP->>Model: register_parameter("spinquant_R1", Random H×D)
    else deterministic online
        PP->>PP: 不注册 (butterfly on-the-fly)
    end
    PP->>Model: register_parameter("spinquant_R2_head", R2)
    PP->>Model: register_buffer("spinquant_R3_head", R3)
    opt random_r4
        PP->>Model: register_buffer("spinquant_R4_matrix", R4)
    end

    Note over PP: Step 5: 训练 (可选)
    opt trainable_rotation / trainable_smooth
        PP->>PP: _train_rotations(dataloader) — Adam 优化 R 和 smooth
    end

    Note over PP: Step 6: 应用旋转到权重
    alt online R1
        PP->>Model: W = W @ R1^T (q/k/v/gate/up_proj)
        PP->>Model: 注册 R1 forward_pre_hook: x → x @ R1
        PP->>Model: R2 fuse: W_qkv 按 head 旋转
        PP->>Model: R4 fuse: W_gate/up @ R4^T, W_down = R4 @ W_down
    else offline R1
        PP->>Model: 全模型权重矩阵融合 R1/R2/R4
    end

    Note over PP: Step 7: 注册在线 hooks (R3 / R4)
    PP->>Apply: register_spinquant_hooks(model, config)
    Apply->>Model: R3: monkeypatch attention (Q@R3, K@R3)
    Apply->>Model: R4: forward_pre_hook on down_proj (x → x@R4)
    Apply-->>PP: hook handles

    Note over PP: Step 8: 清理
    PP->>Model: freeze parameters
    PP->>PP: clear tracking lists
    PP->>Model: model._rotation_config = config
    PP-->>Caller: return model (旋转完成, hooks 就绪)
```

---

## 图 3: 保存阶段详细流程

展示量化模型如何序列化 rotation buffers：

```mermaid
sequenceDiagram
    participant Export as export_to_autoround.py
    participant Dispatch as transforms/__init__.py
    participant Algo as algorithm.py<br/>(RotationSerializer)
    participant Ser as serialize.py
    participant QL as QuantLinear 模块
    participant Disk as safetensors<br/>+ config.json

    Note over Export,Disk: ═══ 路径 A: ShardWriter (per-layer) ═══

    loop 每个量化层 (pack_layer)
        Export->>Dispatch: inject_rotation_buffers_on_layer(name, qlayer, model)
        Dispatch->>Algo: serializer.inject_buffers_on_layer(name, qlayer, model)

        alt name ∈ r1_targets (q/k/v/gate/up_proj)
            Algo->>Ser: _inject_rotation_buffers(qlayer, "spinquant_r1", size, ...)

            alt random_r1
                Algo->>Algo: matrix = model.spinquant_R1 (nn.Parameter)
                Ser->>QL: register_buffer("spinquant_r1_type", RANDOM=1)
                Ser->>QL: register_buffer("spinquant_r1_size", 1024)
                Ser->>QL: register_buffer("spinquant_r1_matrix", int8 ±1)
            else deterministic
                Ser->>QL: register_buffer("spinquant_r1_type", HADAMARD=0)
                Ser->>QL: register_buffer("spinquant_r1_size", 1024)
                Note over QL: 无 _matrix (运行时重建)
            end
        end

        alt name ∈ r4_targets (down_proj)
            Algo->>Ser: _inject_rotation_buffers(qlayer, "spinquant_r4", size, ...)
            Note over Ser,QL: 同 R1 逻辑 (HADAMARD/RANDOM/TRAINED)
        end
    end

    Note over Export,Disk: ═══ 路径 B: Bulk 注入 (非 ShardWriter) ═══

    Export->>Dispatch: inject_rotation_buffers_bulk(model, quant_config)
    Dispatch->>Algo: serializer.inject_buffers_bulk(model, quant_config)
    Algo->>Ser: inject_spinquant_buffers(model, config)
    Ser->>Ser: 遍历所有 QuantLinear, 批量注入 buffers

    Note over Export,Disk: ═══ 保存到磁盘 ═══

    Export->>Disk: model.save_pretrained(save_dir)
    Note over Disk: safetensors 包含:<br/>• 量化权重 (qweight, scales, zeros)<br/>• spinquant_r1_type, _size, [_matrix]<br/>• spinquant_r4_type, _size, [_matrix]

    Export->>Dispatch: save_rotation_config(model, save_dir)
    Dispatch->>Algo: serializer.save_config(model, save_dir)
    Algo->>Ser: save_spinquant_config(model, save_dir, config)
    Ser->>Disk: 读取 config.json, 写入 spinquant_config 字段
    Note over Disk: config.json 新增:<br/>"spinquant_config": {<br/>  "algorithm": "spinquant",<br/>  "r1": true, "r2": true,<br/>  "r3": true, "r4": true,<br/>  "online_r1_rotation": true,<br/>  "random_r1": false, ...<br/>}
```

---

## 图 4: 加载阶段详细流程

展示从磁盘恢复完整推理能力：

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant HF as HuggingFace<br/>from_pretrained
    participant Conv as convert_model.py
    participant Dispatch as transforms/__init__.py
    participant Ser as serialize.py
    participant QL as QuantLinear
    participant Model as 推理模型

    User->>HF: AutoModelForCausalLM.from_pretrained(save_dir)
    HF->>Conv: convert_hf_model(model, quant_config)

    Note over Conv: Step 1: 替换 Linear → QuantLinear
    Conv->>Model: 按 backend 替换所有 Linear 层

    Note over Conv: Step 2: 预注册空 buffers
    Conv->>Dispatch: preregister_rotation_buffers(model, quant_config)
    Dispatch->>Ser: preregister_spinquant_buffers(model, spinquant_config)

    loop 每个 QuantLinear
        alt name ∈ r1_targets
            Ser->>QL: register_buffer("spinquant_r1_type", empty int32)
            Ser->>QL: register_buffer("spinquant_r1_size", empty int32)
            opt needs_matrix (random/trained)
                Ser->>QL: register_buffer("spinquant_r1_matrix", empty tensor)
            end
        end
        alt name ∈ r4_targets
            Ser->>QL: register_buffer("spinquant_r4_type/size/[matrix]", empty)
        end
    end

    Note over Conv: Step 3: HF state_dict loader 填充数据
    HF->>QL: load_state_dict(safetensors) → 填充所有 buffer 值
    Note over QL: spinquant_r1_type=0(HAD)/1(RND)<br/>spinquant_r1_size=1024<br/>spinquant_r1_matrix=int8 (if random)

    Note over Conv: Step 4: 后处理 (post_init)
    Conv->>Conv: backend post_init (GPTQ pack, etc.)
    Conv->>Conv: model dtype conversion

    Note over Conv: Step 5: 重建在线旋转
    Conv->>Dispatch: rebuild_rotation_if_needed(model)
    Dispatch->>Ser: rebuild_spinquant_online(model)

    Note over Ser: 5a: Patch QuantLinear.forward
    Ser->>Ser: _patch_quantlinear_forward_spinquant(model)
    Ser->>QL: cls.forward = forward_with_spinquant
    Note over QL: 新 forward:<br/>x = apply_r1_from_buffer(x)<br/>x = apply_r4_from_buffer(x)<br/>return original_forward(x)

    Note over Ser: 5b: 重建 R3 Hook (如启用)
    opt config.r3 = True
        Ser->>Model: register_spinquant_hooks(model, r3_config)
        Note over Model: attention forward 被 monkeypatch:<br/>Q = Q @ R3, K = K @ R3
    end

    Ser-->>Conv: 重建完成
    Conv-->>HF: 返回模型
    HF-->>User: 可推理模型
```

---

## 图 5: 推理时前向传播 — 旋转作用路径

展示单个 Transformer Layer 中，旋转在前向传播中的作用位置：

```mermaid
sequenceDiagram
    participant X as 输入 x
    participant RMS as RMSNorm
    participant QKV as q/k/v_proj<br/>(QuantLinear)
    participant RoPE as RoPE
    participant Attn as Attention<br/>(QK^T / softmax / V)
    participant O as o_proj
    participant RMS2 as RMSNorm2
    participant GU as gate/up_proj<br/>(QuantLinear)
    participant Act as SiLU × gate
    participant Down as down_proj<br/>(QuantLinear)
    participant Out as 输出

    Note over X,Out: ═══ Self-Attention Block ═══

    X->>RMS: LayerNorm(x)
    RMS->>QKV: 进入 q/k/v_proj

    Note over QKV: R1 online (buffer-based):<br/>x' = x @ R1 (from buffer)<br/>然后正常 W^T @ x'<br/>(W 已融合 R1^T)
    QKV->>QKV: forward_with_spinquant:<br/>x = apply_r1_from_buffer(x)<br/>return orig_forward(x)

    QKV->>RoPE: Q, K, V

    Note over RoPE: R2 已离线融合进 W_qkv<br/>(per-head rotation, 不需要 hook)

    RoPE->>Attn: Q', K' (after RoPE)

    Note over Attn: R3 online (monkeypatch):<br/>Q'' = Q' @ R3<br/>K'' = K' @ R3<br/>score = Q''K''^T = Q'K'^T<br/>(R3 抵消, 数学等价)
    Attn->>Attn: Q@R3, K@R3, then QK^T/√d, softmax, @V

    Attn->>O: attention output
    O->>RMS2: residual + norm

    Note over RMS2,Out: ═══ MLP Block ═══

    RMS2->>GU: 进入 gate/up_proj

    Note over GU: R1 online (同上):<br/>x' = x @ R1
    GU->>GU: forward_with_spinquant (R1 buffer)

    GU->>Act: gate_proj(x) * SiLU(up_proj(x))

    Note over Act,Down: R4 已离线融合进 gate/up 权重<br/>(W_gate = W_gate @ R4^T)

    Act->>Down: 进入 down_proj

    Note over Down: R4 online (buffer-based):<br/>x' = x @ R4 (from buffer)<br/>然后正常 W_down^T @ x'<br/>(W_down 已融合 R4)
    Down->>Down: forward_with_spinquant:<br/>x = apply_r4_from_buffer(x)<br/>return orig_forward(x)

    Down->>Out: MLP output (+ residual)
```

---

## 图 6: Buffer 存储类型决策树

```mermaid
flowchart TD
    A[_inject_rotation_buffers<br/>prefix, size, random, is_trained, rotation_matrix] --> B{is_trained?}
    B -->|Yes| C[rot_type = TRAINED<br/>存 float32 完整矩阵]
    B -->|No| D{random?}
    D -->|Yes| E[rot_type = RANDOM<br/>存 int8 ±1 矩阵]
    D -->|No| F[rot_type = HADAMARD<br/>仅存 type + size<br/>运行时用 butterfly 重建]

    C --> G[buffers 注册到 QuantLinear]
    E --> G
    F --> G

    G --> H{加载时: rot_type?}
    H -->|HADAMARD=0| I[get_hadamard_K(size)<br/>matmul_hadU butterfly<br/>O(n log n)]
    H -->|RANDOM=1| J[matrix.float() / √n<br/>dense matmul x @ R<br/>O(n²)]
    H -->|TRAINED=2| K[matrix (float32)<br/>dense matmul x @ R<br/>O(n²)]
```

---

## 图 7: R1/R2/R3/R4 各自的作用方式总结

| Rotation | 应用方式 | 何时执行 | 是否需要存矩阵 |
|----------|---------|---------|---------------|
| **R1 offline** | 权重融合 `W = R1 @ W @ R1^T` | 预处理时 | ❌ 不需要 |
| **R1 online** | 权重半融合 `W = W @ R1^T` + hook `x' = x @ R1` | 预处理 + 推理时 | ✅ random/trained 需要 |
| **R2** | 权重融合 per-head `W_qkv[:, h] = W @ R2` | 预处理时 | ❌ 已融合 |
| **R3** | Attention monkeypatch `Q@R3, K@R3` | 推理时 | ❌ 任何正交 R 都等价 |
| **R4 offline** | 权重融合 `W_gate/up @ R4^T`, `R4 @ W_down` | 预处理时 | — |
| **R4 online** | hook `x' = x @ R4` on down_proj | 推理时 | ✅ random/trained 需要 |

**关键公式**：
- R1: `x @ R1 @ (W @ R1^T)^T = x @ R1 @ R1 @ W^T = x @ W^T` ✓ 数学等价
- R3: `(Q@R3)(K@R3)^T = Q@R3@R3^T@K^T = Q@K^T` ✓ 正交矩阵抵消
- R4: `(activation @ R4) @ (R4^T @ W_down^T) = activation @ W_down^T` ✓ 等价

---

## 附录: 文件职责映射

```
auto_round/
├── autoround.py                          # 入口: AutoRound 主类
├── algorithms/transforms/
│   ├── __init__.py                       # 调度层: apply_rotation, inject, save, preregister, rebuild
│   └── spinquant/
│       ├── preprocessor.py               # 预处理: 矩阵创建、权重旋转、hook 注册
│       ├── serialize.py                  # 序列化: buffer 注入、config 保存、加载重建
│       ├── algorithm.py                  # RotationSerializer mixin 实现
│       ├── inplace/
│       │   ├── apply.py                  # register_spinquant_hooks (R3/R4 hook 注册)
│       │   ├── r3_monkeypatch.py         # R3 attention monkeypatch 实现
│       │   └── rotation_utils.py         # matmul_hadU, Hadamard 分解
│       └── SERIALIZATION_ARCHITECTURE.md # 架构文档
└── inference/
    └── convert_model.py                  # 加载侧: preregister + rebuild 调用点
```
