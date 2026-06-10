# Triton MXFP4 Kernel 社区调研报告

> 调研日期: 2026-05-18  
> 目的: 调研社区中 MXFP4/FP4 Triton kernel 的实现，为 auto-round 的 Triton 加速方案提供参考

---

## 1. 调研总结

搜索了 11 个主流仓库，**核心发现**：

- **没有任何仓库实现了纯 Triton `@triton.jit` 的 fused FP4 dequant + GEMM kernel**
- FP4 GEMM 的生产方案全部依赖：CUTLASS SM100 硬件 Tensor Core 或 OpenAI `triton-kernels` 库
- Triton 手写 fused dequant+GEMM 仅存在于 **INT4 (GPTQ/AWQ)** 场景
- E2M1 解码主流方案：硬件 PTX `cvt.rn.satfinite.e2m1x2`（SM89+）> 软件 LUT > 算术计算

---

## 2. 各仓库详细分析

### 2.1 vllm-project/vllm ⭐⭐⭐

**License:** Apache 2.0 ✅  
**FP4 覆盖度:** 最全面 — 11 个 CUDA/CUTLASS kernel + W4A4 linear + 14 种 MoE 后端

#### CUDA/CUTLASS Kernels (`csrc/libtorch_stable/quantization/fp4/`)

| 文件 | 用途 | 架构要求 |
|------|------|----------|
| `nvfp4_quant_kernels.cu` | BF16/FP16 → FP4 量化 (PTX `cvt_warp_fp16_to_fp4`) | SM100+ |
| `nvfp4_scaled_mm_kernels.cu` | CUTLASS FP4×FP4 scaled matmul | SM100 |
| `nvfp4_scaled_mm_sm120_kernels.cu` | 同上，桌面 Blackwell | SM120 |
| `mxfp4_blockwise_moe_kernel.cu` | MXFP4 MoE grouped GEMM | SM100 |
| `mxfp4_experts_quant.cu` | Expert 激活在线 MXFP4 量化 | SM100 |
| `nvfp4_experts_quant.cu` | Expert 激活在线 NVFP4 量化 | SM100 |
| `activation_nvfp4_quant_fusion_kernels.cu` | Fused activation + FP4 quant | SM100 |

#### Triton Kernels

| 文件 | 用途 | 备注 |
|------|------|------|
| `qutlass_utils.py` | E8M0 scale swizzle (128×4 block → TMA 格式) | `@triton.jit`，仅做 scale 重排 |
| `triton_w4a16.py` | **INT4 W4A16 fused GEMM** | `@triton.jit`，ROCm MI300 专用 |
| `awq_triton.py` | AWQ INT4 dequant + fused GEMM | `@triton.jit` |

#### W4A4 FP4 Linear Layer (`fp_quant.py`)

```python
# 核心推理流程（FPQuantLinearMethod）:
def quantized_forward(x, qweight, weight_scales, ...):
    # 1. Hadamard 旋转激活
    # 2. CUDA kernel: fused_quantize_mx (激活 → E2M1 + E8M0 scale)
    x_q, x_scales = torch.ops.vllm.fused_quantize_mx(x, hadamard_matrix, method)
    # 3. CUTLASS/triton-kernels: FP4×FP4 → BF16 GEMM
    y = torch.ops.vllm.matmul_mxf4_bf16(x_q, qweight, x_scales, weight_scales, alpha)
```

- Hadamard 旋转 + 在线激活量化 + 硬件 FP4 GEMM
- E2M1 解码由 Tensor Core 隐式完成，不需要软件 LUT
- Scale 通过 Triton swizzle kernel 预处理为 TMA 格式

#### MoE 后端系统（14 种）

```python
# mxfp4.py 中的 Mxfp4MoeBackend enum:
DEEPGEMM_MXFP4          # DeepGEMM FP8×FP4, SM100+
FLASHINFER_TRTLLM_*     # TRT-LLM, SM100
FLASHINFER_CUTLASS_*    # CUTLASS, SM90+
TRITON / TRITON_UNFUSED # OpenAI triton-kernels (matmul_ogs)
BATCHED_MARLIN / MARLIN # Marlin kernel
AITER_*                 # ROCm CK
```

优先级: FLASHINFER_TRTLLM > AITER > **TRITON** > FLASHINFER_CUTLASS > MARLIN

**关键洞察**: vllm 的 `TRITON` 后端调用的是 OpenAI 的 `triton-kernels` 外部库的 `matmul_ogs`，不是手写 `@triton.jit`。

---

### 2.2 sgl-project/sglang ⭐⭐

**License:** Apache 2.0 ✅  
**FP4 覆盖度:** MoE + KV-cache，无 dense linear FP4

#### 核心文件

| 文件 | 用途 |
|------|------|
| `mxfp4.py` (54KB) | MXFP4 MoE 配置、weight swizzle、多后端调度 |
| `mxfp4_tensor.py` | E2M1 参考 LUT 量化/反量化（offline） |
| `kvfp4_tensor.py` | KV-cache FP4: BlockFP4 + NVFP4 |
| `fp4_kv_cache_quant_method.py` | KV-cache 量化抽象层 |
| `quark_w4a4_mxfp4_moe.py` | ROCm AITER MXFP4 MoE |
| `triton_kernels_moe.py` | OpenAI `triton_kernels.matmul_ogs` 接口 |

#### Weight Swizzle（用于 triton-kernels 库）

```python
def _swizzle_mxfp4(quant_tensor, scale, num_warps):
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    # SM120: StridedLayout (non-persistent)
    # SM100/SM90: make_default_matmul_mxfp4_w_layout (TMA persistent)
    quant_tensor = convert_layout(wrap_torch_tensor(quant_tensor.T, dtype=FP4), value_layout)
    scale = convert_layout(wrap_torch_tensor(scale.T), scale_layout)
```

#### E2M1 Software LUT（参考实现）

```python
class MXFP4QuantizeUtil:
    E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
    
    @classmethod
    def quantize(cls, input, block_size=32):
        # Scale: ceil(log2(amax / 6.0)) + 127 → uint8 e8m0
        # Quantize: threshold comparison → nibble index
        ord_ = sum((abs(x) > E2M1_bounds))
        fp4_val = sign_bit * 8 + ord_
    
    @classmethod
    def dequantize(cls, quantized_data, dtype, scale, block_sizes):
        # LUT: values[magnitude_idx]
        # Scale: exp2(e8m0 - 127)
```

#### KV-Cache FP4

- BlockFP4 (block_size=16): `torch.compile` 量化，LUT 反量化
- NVFP4 (two-level scale): SM100 用 FlashInfer native PTX，SM90 用 PyTorch fallback

**关键洞察**: SGLang 的 dense linear 不支持 FP4 — `get_quant_method(LinearBase)` 直接返回 `UnquantizedLinearMethod()`。

---

### 2.3 pytorch/FBGEMM ⭐⭐

**License:** BSD-3-Clause ✅

**文件:** `fbgemm_gpu/experimental/gemm/triton_gemm/fp4_quantize.py` (215KB)

生产级 MX4 Triton 量化 kernel，使用 **PTX inline assembly**:

```python
@triton.jit
def _kernel_quantize_mx4_unpack(...):
    # PTX: 一次打包 8 个 FP4 值到 1 个 int32
    packed_result = tl.inline_asm_elementwise(
        asm="""
        {
            cvt.rn.satfinite.e2m1x2.f32  byte0, $2, $1;
            cvt.rn.satfinite.e2m1x2.f32  byte1, $4, $3;
            cvt.rn.satfinite.e2m1x2.f32  byte2, $6, $5;
            cvt.rn.satfinite.e2m1x2.f32  byte3, $8, $7;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        dtype=tl.int32, is_pure=True, pack=1,
    )
```

- 支持 group_size 16/32，rounding mode: ceil/floor/stochastic
- Tensor Core scale swizzling (tcgen05 `mma-scale-factor-b-layout-4x`)
- **仅量化 kernel，无 GEMM**

---

### 2.4 pytorch/ao (torchao) ⭐

**License:** BSD-3-Clause ✅

**文件:** `torchao/prototype/mx_formats/kernels.py` (48KB)

- **MXFP8** Triton kernels（production quality，heavily autotuned）
- **FP4 E2M1**: 仅 Python 参考实现，非 Triton JIT
- `_e8m0_to_fp32` helper: `tl.exp2(scale_e8m0 - 127)` with NaN guard
- Scale 支持 `floor` 和 `rceil` rounding modes
- SM100 上有 PTX `cvt.rp.satfinite.ue8m0x2.f32` 条件路径

**关键洞察**: torchao 的 FP4 是未来计划项，当前仅 MXFP8 有 Triton kernel。

---

### 2.5 IST-DASLab/FP-Quant ⭐

**License:** ⚠️ 无 LICENSE 文件，默认保留所有权利

**文件:** `inference_lib/src/fp_quant/module/triton/mxfp4.py`

训练/伪量化路径的 Triton kernel:

```python
@triton.jit
def mxfp4_forward_kernel(x_ptr, hadamard_matrix_ptr, output_ptr, ...):
    # 1. Hadamard transform (rotation)
    x_had = tl.dot(x, hadamard_matrix)
    # 2. Per-group scale
    scales = tl.max(tl.abs(x_grouped), axis=-1, keep_dims=True)
    shared_exps = tl.exp2(tl.floor(tl.log2(scales)) - 2) / (3/4)
    # 3. FP4 E2M1 量化 (7-level nested tl.where)
    x_fp4 = tl.where(abs > 5, 6, tl.where(abs > 3.5, 4, ...))
    # 4. Dequant
    x_dequantized = x_fp4 * shared_exps
```

- 软件 E2M1 lookup（7 层嵌套 `tl.where`）
- 包含 Hadamard rotation fusion
- 仅伪量化路径（quant → dequant，不产生打包格式）
- 实际 GEMM 由 QuTLASS (CUTLASS wrapper) 执行

---

### 2.6 OpenAI triton-kernels（外部库）⭐⭐

vllm 和 sglang 的 `TRITON` 后端都使用 `triton_kernels.matmul_ogs`:

```python
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FusedActivation
# 接收 MXFP4 packed weight + E8M0 scale via PrecisionConfig
# 内部处理 E2M1 decode + GEMM
```

这是 OpenAI 维护的闭源/半闭源 Triton kernel 库，不在 GitHub 公开仓库中。

---

### 2.7 无相关实现的仓库

| 仓库 | 状态 |
|------|------|
| openai/triton | 无 FP4 examples/tutorials |
| flashinfer-ai/flashinfer | 仅 FP8，无 FP4 Triton |
| Dao-AILab/flash-attention | 无 FP4/量化 linear |
| neuralmagic/llm-compressor | 无 FP4 Triton kernel |
| NVIDIA/Megatron-LM | 仅 MXFP8 Triton |
| NVIDIA/TensorRT-Model-Optimizer | 无公开 Triton FP4 kernel |

---

## 3. E2M1 解码方式对比

| 方式 | 性能 | 精度 | GPU 要求 | 用于 |
|------|------|------|----------|------|
| **PTX `cvt.rn.satfinite.e2m1x2`** | 最快 | 硬件精确 | SM89+(Ada) | vllm, FBGEMM 量化 |
| **CUTLASS native `float_e2m1_t`** | 最快 | 硬件精确 | SM100(Blackwell) | vllm GEMM |
| **Software 7-level `tl.where`** | 中等 | 精确 | 任意 CUDA | IST-DASLab 训练 |
| **Software LUT indexing** | 中等 | 精确 | 任意 CUDA | sglang 参考 |
| **Arithmetic (exp/mantissa)** | 较慢 | 精确 | 任意 CUDA | 我们的实现 |

---

## 4. Fused Dequant + GEMM 模式总结

| 模式 | 实现 | FP4 适用 | 备注 |
|------|------|----------|------|
| **Triton tl.interleave INT4** | vllm `triton_w4a16.py` | ❌ 仅 INT4 | ROCm MI300 |
| **Triton nested tl.where FP4** | IST-DASLab | ⚠️ 伪量化 | 训练用，无打包 |
| **CUTLASS FP4 Tensor Core** | vllm `nvfp4_scaled_mm` | ✅ 生产级 | 需 SM100+ |
| **OpenAI triton-kernels** | vllm/sglang MoE | ✅ 生产级 | 闭源库 |
| **DeepGEMM** | vllm MoE | ✅ FP8×FP4 | SM100+ |
| **我们的 Triton 实现** | auto-round | ✅ 任意 CUDA | 性能待优化 |

---

## 5. 对 auto-round 的建议

### 5.1 可直接参考/复用的代码

| 来源 | 内容 | License |
|------|------|---------|
| vllm `triton_w4a16.py` | INT4 fused GEMM 的 tile-loop 结构 | Apache 2.0 ✅ |
| vllm `awq_triton.py` | AWQ fused GEMM + split-K | Apache 2.0 ✅ |
| vllm `qutlass_utils.py` | E8M0 scale swizzle Triton kernel | Apache 2.0 ✅ |
| FBGEMM `fp4_quantize.py` | PTX E2M1 encode（如果目标 SM89+） | BSD-3 ✅ |
| torchao `kernels.py` | MXFP8 dequant kernel 结构（可改 FP4） | BSD-3 ✅ |
| sglang `mxfp4_tensor.py` | E2M1 LUT 参考实现 | Apache 2.0 ✅ |

### 5.2 推荐的实现策略

#### 短期目标（L20/A100 兼容，无 SM100 要求）

1. **Standalone Triton dequant kernel**（已实现，验证通过 ✅）
   - 参考：torchao MXFP8 dequant 结构
   - 我们的 E2M1 arithmetic 解码在 L20 上已正确工作

2. **Fused dequant + GEMM**（需修复精度问题）
   - 参考：vllm `triton_w4a16.py` 的 tile-loop 结构
   - 关键差异：INT4 用 `tl.interleave` 解包，FP4 需 nibble + LUT
   - 建议改用 7-level `tl.where`（IST-DASLab 方案）替代我们的 arithmetic 解码

3. **RotatedMXFP4Linear 模块**
   - 参考：vllm `FPQuantLinearMethod` 的 Hadamard + quant + GEMM 流程
   - 差异：vllm 用 CUDA kernel 做 Hadamard，我们可以用 Triton `tl.dot`

#### 中期目标（SM89+ Ada Lovelace）

4. **PTX `cvt.rn.satfinite.e2m1x2` 加速**
   - 参考：FBGEMM `fp4_quantize.py` 和 vllm `fused_indexer_q.py`
   - 用 `tl.inline_asm_elementwise` 在 Triton 中调用 PTX
   - 可以加速量化（activation quant）和反量化（weight dequant）

#### 长期目标（SM100+ Blackwell）

5. **CUTLASS FP4 Tensor Core 路径**
   - 直接复用 vllm 的 `nvfp4_scaled_mm` 或集成 DeepGEMM
   - FP4×FP4 → BF16 在 Tensor Core 上原生支持
   - 需要 E8M0 scale 预处理为 TMA swizzle 格式

### 5.3 我们实现的独特价值

社区中没有的：
- **纯 Triton fused rotation + FP4 dequant + GEMM**（社区全部分离处理）
- **兼容 pre-SM100 GPU 的 FP4 GEMM**（社区生产级都要求 SM100+）
- **与 auto-round 量化框架集成的端到端方案**

这意味着我们的实现填补了一个真实的空白：在 L20/A100/A6000 等 non-Blackwell GPU 上做 MXFP4 推理加速。

---

## 6. 关键代码片段参考

### 6.1 vllm INT4 Fused GEMM（最接近我们的 FP4 需求）

```python
# vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py
@triton.jit
def triton_w4a16_gemm_kernel(a_ptr, b_ptr, scales_ptr, zeros_ptr, c_ptr, ...):
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        # Load packed INT4 weights
        b_packed = tl.load(b_ptrs, ...)
        # Unpack: 3x interleave to extract nibbles
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xF  # extract 4-bit
        # Dequantize: subtract zero, multiply scale
        b_fp = (b - z).to(a.dtype) * scales
        # Accumulate
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)
```

### 6.2 IST-DASLab E2M1 Software Quantize（7-level tl.where）

```python
# IST-DASLab/FP-Quant triton/mxfp4.py
x_fp4 = (
    tl.where(x_abs > 5, 6,
    tl.where(x_abs > 3.5, 4,
    tl.where(x_abs > 2.5, 3,
    tl.where(x_abs > 1.75, 2,
    tl.where(x_abs > 1.25, 1.5,
    tl.where(x_abs > 0.75, 1,
    tl.where(x_abs > 0.25, 0.5, 0)))))))
) * x_sign
```

### 6.3 vllm E8M0 Scale Swizzle

```python
# vllm/model_executor/layers/quantization/qutlass_utils.py
@triton.jit
def triton_scale_swizzle(scale_ptr, scale_rows, scale_cols, output_ptr, ...):
    # 128×4 block swizzle for NVIDIA TMA format
    # dest_offset = (r % 32) * 16 + (r // 32) * 4 + col
    row_in_block = row_idx % 32
    row_group = row_idx // 32
    dest = row_in_block * 16 + row_group * 4 + col
    tl.store(output_ptr + block_offset + dest, val)
```

### 6.4 FBGEMM PTX E2M1 Encode

```python
# pytorch/FBGEMM fp4_quantize.py
packed_result = tl.inline_asm_elementwise(
    asm="""
    {
        cvt.rn.satfinite.e2m1x2.f32  byte0, $2, $1;
        cvt.rn.satfinite.e2m1x2.f32  byte1, $4, $3;
        cvt.rn.satfinite.e2m1x2.f32  byte2, $6, $5;
        cvt.rn.satfinite.e2m1x2.f32  byte3, $8, $7;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
    }
    """,
    constraints="=r,f,f,f,f,f,f,f,f",
    args=[f_one, f_five, f_three, f_seven, f_two, f_six, f_four, f_eight],
    dtype=tl.int32, is_pure=True, pack=1,
)
```

### 6.5 vllm MXFP4 Scale 计算（DeepSeek-V4 风格）

```python
# vllm DeepSeek-V4 ops
@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi):
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 6.0 * (2**-126))  # 最小 clamp
    log2_ratio = tl.math.ceil(tl.math.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    scale = tl.math.exp2(log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)
    inv_scale = 1.0 / scale
    packed = _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale)
    return packed, ue8m0
```

---

## 7. 附录：License 兼容性

| 仓库 | License | 可用于 Apache 2.0 项目 |
|------|---------|------------------------|
| vllm-project/vllm | Apache 2.0 | ✅ 直接兼容 |
| sgl-project/sglang | Apache 2.0 | ✅ 直接兼容 |
| pytorch/FBGEMM | BSD-3-Clause | ✅ 兼容 |
| pytorch/ao | BSD-3-Clause | ✅ 兼容 |
| NVIDIA/Megatron-LM | Apache 2.0 | ✅ 直接兼容 |
| IST-DASLab/FP-Quant | ⚠️ 无 License | ❌ 不可直接使用 |
| OpenAI triton-kernels | 未公开 | ❓ 不确定 |
