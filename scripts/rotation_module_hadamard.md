# auto-round `algorithms/transforms/rotation` 模块详解

> 对应代码：`auto_round/algorithms/transforms/rotation/`
> 与之并列的是 `auto_round/algorithms/transforms/spinquant/`（SpinQuant/QuaRot 的
> R1–R4 多级旋转，我们前面调试的就是它）。
>
> 这两个子包共享同一个统一入口 `apply_rotation(model, config, data_type=...)`，
> 由 config 的 `algorithm` 字段决定走哪条实现：
> - `algorithm="hadamard"` → **本文档的 `rotation/` 模块**
> - `algorithm="spinquant"`（或字符串 `"quarot"`/`"spinquant"`）→ `spinquant/` 模块

---

## 1. 两个旋转系统的定位

| | `rotation/`（Hadamard） | `spinquant/`（R1–R4） |
|---|---|---|
| 配置类 | `RotationConfig`（pydantic） | `SpinQuantConfig`（dataclass） |
| 算法注册名 | `"hadamard"` | `"spinquant"` |
| 旋转粒度 | **每个 Linear 独立**做块对角 Hadamard（或整条残差流） | 残差流级别的 R1 + 注意力头 R2/R3 + MLP R4 |
| 主要目标 | MXFP4 / NVFP4（activation 量化抑制 outlier） | 通用 INT/FP 全方案，结构化 spinquant 论文复现 |
| 是否需要训练 | 否（RTN/AutoRound 调优即可） | 可选（trainable_rotation/Cayley） |
| 字符串快捷方式 | `"default"`, `"hadamard"`, `"random_hadamard"` | `"quarot"`, `"spinquant"` |

`rotation/` 模块本身又有 **两个后端（backend）**，下面分别讲。

---

## 2. `rotation/` 模块结构

```
rotation/
├── config.py        # RotationConfig：唯一的 schema + normalize_rotation_config
├── dispatcher.py    # resolve_hadamard_backend：auto/inplace/transform 路由
├── apply.py         # HadamardRotation（transform 后端）+ apply_rotation_transform
├── transforms.py    # HadamardTransform / RandomHadamardTransform（nn.Module）
├── patch.py         # 把 transform 注入 WrapperLinear/WrapperWALayer/QuantLinear
├── utils/
│   ├── math.py      # deterministic_hadamard_matrix / random_hadamard_matrix
│   ├── matrix.py    # apply_transform_weight / multihead_matmul（块对角乘法）
│   ├── hadamards.safetensors  # 非 2 的幂尺寸的预计算 Hadamard 矩阵
│   └── triton/mxfp4.py        # MXFP4 融合 forward kernel
└── inplace/
    ├── apply.py     # QuaRot 残差流旋转（fuse layernorm、rotate weights、online hook）
    ├── hooks.py     # online Hadamard 的 forward hook
    └── model_config.py  # RotationMapping（不同模型结构的映射）
```

### 2.1 `RotationConfig` 关键字段（config.py）

| 字段 | 默认 | 说明 |
|---|---|---|
| `algorithm` | `"hadamard"` | 注册名（frozen） |
| `backend` | `"auto"` | `auto` / `inplace` / `transform` |
| `block_size` | `None` | 块对角 Hadamard 的块大小；mx_fp 默认 32，nv_fp 默认 16 |
| `hadamard_type` | `"hadamard"` | `hadamard`（确定性 Sylvester）/ `random_hadamard`（带符号随机） |
| `fuse_online_to_weight` | `None` | 仅 inplace：是否把 online Hadamard 融进权重 |
| `allow_online_rotation` | `True` | 仅 inplace：是否允许需要补偿 hook 的旋转 |

`normalize_rotation_config(config, data_type)` 会：
- 字符串/dict/RotationConfig → 统一成校验过的 dict；
- 根据 `data_type` 自动补 `block_size`（mx_fp=32，nv_fp=16），不匹配时 warning。

### 2.2 后端路由（dispatcher.py）

`resolve_hadamard_backend(config, data_type)`：

```
backend="inplace"   → 永远 inplace
backend="transform" → 校验：必须 MX/NV-FP，且不能 fuse_online_to_weight
backend="auto"      → 若要求 fuse_online_to_weight → inplace
                      elif data_type 是 mx_fp/nv_fp  → transform
                      else                            → inplace
```

> 实务结论：用字符串 `"default"`/`"hadamard"`/`"random_hadamard"` + `scheme="MXFP4"/"NVFP4"`，
> 走的就是 **transform 后端**（每 Linear 块对角 Hadamard + triton 融合 kernel）。

---

## 3. transform 后端：每个 Linear 的块对角 Hadamard

入口：`apply.py::HadamardRotation.apply_to_model` → `_apply_to_module`。
对每个 `nn.Linear`（跳过 `lm_head`）做两件事：

### 3.1 权重侧（`_apply_weight_transform`）
- 构造 `w_transform = HadamardTransform(block_size, location="weight")`，
  权重矩阵 `H = deterministic_hadamard_matrix(block_size) / sqrt(block_size)`（正交归一）。
- 数学（`utils/matrix.py::apply_transform_weight`，对 `nn.Linear`）：
  - location="weight" → `value @ H.T`，即把权重旋转成 `W' = W @ H.T`（块对角，按 `block_size` 切块）。
  - `random_hadamard` 时还会 `patch_quantlinear(w_transform)`，把矩阵作为 buffer 存进 QuantLinear，便于推理/序列化。

### 3.2 激活侧（`_apply_input_transform`，online hook）
- 构造 `inp_transform = HadamardTransform(..., location="input", inverse=True)`，作为 `W` 的逆旋转。
- 注册 `forward_pre_hook`：对输入 `x` 做 `x @ H`（或随机时 `x @ H.T`）。
- 若有 triton 且是 MXFP4：hook 直接调用 `mxfp4_forward_kernel_wrapper(x_flat, w)`，
  **在一个 kernel 内完成「Hadamard 旋转 + MXFP4 量化/反量化」**，并设 `module.pre_dequantized_input=True`。
- 等价性：`(x @ H) @ (W @ H.T).T = x @ H @ H @ W.T = x @ W.T`（H 对称正交，`H@H=I`）。
  > 注意：这里用确定性 Hadamard（对称）所以成立；random 用 `inverse=True` 存 `H.T`，
  > 也保证 weight 侧与 hook 侧互逆。这与我们在 spinquant 里修过的「分块路径 R vs R.T」是同一类约束。

### 3.3 校准（AutoRound 调优循环里）—— patch.py
AutoRound 的 RTN/迭代调优会用 `WrapperLinear`/`WrapperWALayer` 包住 Linear，
在每次 forward 里重量化。`patch.py` 把 Hadamard 注入这些 wrapper（**类级 monkeypatch，幂等**）：
- `patch_wrapperlinear_to_apply_transform`：
  - `_qdq_weight`：第一次调用时把 `W` 原地替换成 `w_transform(W)`（只对 <16bit）。
  - `_qdq_act`：量化前先 `x = inp_transform(x)`。
- `patch_wrapperwalayer_forward_to_apply_transform`：W+A 层 forward 前对激活做旋转再做 act 量化。

### 3.4 序列化 / 推理 —— patch.py::patch_quantlinear
- 打包成 `QuantLinear`（auto_round fp4 格式）时，`pack` 被 patch，
  额外 `register_buffer("hadamard_matrix", w_transform.weight)`。
- 推理：导出的 QuantLinear 自带 `hadamard_matrix`，forward 时对输入做 Hadamard（online），
  对随机 Hadamard 因为矩阵已持久化，所以**能保存并重新加载后正确推理**。

> ⚠️ inplace 后端会打印 warning：`this backend does not support real exporting, please export
> the model to fake format` —— 即 inplace（QuaRot 残差流）目前不支持真实打包导出，只能 fake 量化导出；
> 真正能"旋转 + 量化 + 保存 + 加载推理"闭环的是 **transform 后端**。

---

## 4. inplace 后端：QuaRot 残差流旋转（R1–R4 等价）

入口：`inplace/apply.py::apply_rotation_transform`。本质是 QuaRot/SpinQuant 的"整条残差流旋转"：
1. `_fuse_layer_norms` / `_replace_layernorms_with_rmsnorm`：把 LayerNorm 的 scale 融进相邻 Linear。
2. `_untie_word_embeddings`：必要时解绑 embedding / lm_head。
3. `_rotate_weights`：
   - **fuse 模式**（`fuse_online_to_weight=True`）：QuaRot 风格——
     q/k/v、gate/up 输入侧乘残差旋转 Q；o_proj/down_proj 输出侧乘 Q；
     down_proj 输入侧额外叠加 online Hadamard（+ 对应 hook）；attention 头维度做 R2/R3 类旋转。
   - **unfused 模式**：每个 Linear 自包含（输入侧 Hadamard + online hook 互相抵消），不动 embedding/lm_head。
4. online hook（`inplace/hooks.py`）：对需要补偿的算子（down_proj 输入、Q/K 头维度等）在 forward 时做在线 Hadamard。

特点：
- 对**任意 dtype**有效（不限 MX/NV-FP）；
- 与 spinquant 模块在概念上重叠（都是残差流 R1 + 头维 R2/R3 + MLP R4），
  但 spinquant 模块功能更全（可训练旋转、完整 save/load、R 级别开关）。

---

## 5. 端到端：rotation + quantization 在 AutoRound 里的流程

```python
from auto_round import AutoRound

ar = AutoRound(
    model_name_or_path,
    scheme="MXFP4",                 # 或 "NVFP4"
    rotation_config="default",      # = hadamard, block_size=32, transform 后端
    # 也可：rotation_config="random_hadamard"
    #      rotation_config={"algorithm":"hadamard","backend":"inplace","fuse_online_to_weight":True}
)
ar.quantize_and_save(output_dir, format="auto_round")
```

内部时序（`compressors/base.py`）：
1. `rotation_config` 经 `entry.py` 归一化后塞进 `self.rotation_configs`。
2. **Phase 3** `_patch_model`：先定型模型拓扑（MoE 合并等）。
3. **Phase 4** `_build_layer_config`：构建每层量化配置。
4. **Phase 4.5** `_apply_rotations` → 对每个 rotation_config 调
   `apply_rotation(model, cfg, data_type=quantize_config.data_type)`：
   - hadamard/transform：给每个 Linear 装 weight 旋转 + activation hook + patch wrapper。
   - hadamard/inplace 或 spinquant：做残差流旋转。
5. **Phase 5** 正式量化（RTN 或迭代调优）；wrapper 在调优循环里透明地应用 Hadamard。
6. 导出：transform 后端把 `hadamard_matrix` 写进 QuantLinear buffer。

推理：`AutoModelForCausalLM.from_pretrained(quantized_dir)` 加载后，
QuantLinear 的 online hook（带持久化的 `hadamard_matrix`）在 forward 时对激活做旋转，
对 MXFP4 走 triton 融合 kernel。

---

## 6. 关键代码索引

| 功能 | 位置 |
|---|---|
| 统一入口 `apply_rotation` | `algorithms/transforms/__init__.py` |
| 后端路由 | `rotation/dispatcher.py::resolve_hadamard_backend` |
| transform 后端主体 | `rotation/apply.py::HadamardRotation.apply_to_model` |
| 权重/激活旋转数学 | `rotation/utils/matrix.py::apply_transform_weight` |
| Hadamard 矩阵构造 | `rotation/utils/math.py` |
| 调优 wrapper 注入 | `rotation/patch.py` |
| MXFP4 融合 kernel | `rotation/utils/triton/mxfp4.py` |
| QuaRot 残差流旋转 | `rotation/inplace/apply.py` |
| Phase 4.5 接线 | `compressors/base.py::_apply_rotations` |
| 文档/示例 | `docs/step_by_step.md`（"Hadamard Transform" 一节） |

---

## 7. 与 spinquant 模块的对照小结

- **想做结构化、可训练、可完整 save/load、按 R1/R2/R3/R4 精细开关** → 用 `spinquant/`（`rotation_config="quarot"` 或 `SpinQuantConfig(...)`）。
- **想要 MXFP4/NVFP4 上简单、有融合 kernel、能真实导出推理的"每 Linear Hadamard"** → 用 `rotation/` transform 后端（`rotation_config="default"`/`"random_hadamard"`）。
- 二者都通过 `apply_rotation` 进入，可在同一 `AutoRound(...)` 调用里通过 `rotation_config` 选择。

对比脚本见同目录 `test_rotation_module_compare.py`（与 `test_rotation_scheme_matrix_v2.py` 的
spinquant 路径并列，可一起跑 lm_eval 看真实数据集精度）。它在**同一模型 + 同一量化 scheme**
（默认 MXFP4，`iters=0` RTN、不训练 rotation）下，把两套系统的多个变体并排评测，输出一张表：

- `none`：不做 rotation 的基线
- `spinquant_r1` / `spinquant_r1234` / `spinquant_r1_rand`：`spinquant/` 系统（`SpinQuantConfig`）
- `had_default` / `had_random`：`rotation/` transform 后端（`rotation_config="default"`/`"random_hadamard"`）
- `had_inplace`：`rotation/` inplace 后端（QuaRot 残差流，`fuse_online_to_weight=True`）

用法示例：

```bash
# 默认 Qwen3-0.6B + MXFP4 + piqa，跑全部变体
python test_rotation_module_compare.py --device cuda:4

# 选变体 + 多任务 + limit 快速冒烟
python test_rotation_module_compare.py --device cuda:4 \
    --variants none,spinquant_r1,had_default,had_random \
    --tasks piqa,hellaswag --limit 200

# 换 NVFP4 scheme
python test_rotation_module_compare.py --device cuda:4 --scheme NVFP4
```
