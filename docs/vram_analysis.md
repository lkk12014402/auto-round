# Auto-Round 显存（VRAM）与内存管理机制分析

## 1. 概述

本文档记录了对 auto-round 框架中显存（GPU VRAM）和内存（CPU RAM）管理机制的完整分析，
包括 block-wise quantization 的内存节省原理、blockwise rotation 的实际 VRAM 影响、
PyTorch/safetensors 的 mmap 机制，以及 auto-round 各层内存管理策略。

---

## 2. CPU Device vs Meta Device vs torch.empty(0)

### 2.1 三者对比

| 方式 | 内存占用 | 能计算吗 | 能恢复吗 | auto-round 用在哪里 |
|---|---|---|---|---|
| **CPU device** | 占 CPU RAM（真实 tensor） | 能 | 直接 `.to("cuda")` | 默认模式暂存 |
| **Meta device** | **零**（只有 shape/dtype 元信息） | 不能 | 从磁盘重新加载 | 写盘后永久释放 |
| **`torch.empty(0)` on CPU** | **几乎零**（size=0 的空 tensor） | 不能 | 从磁盘临时文件恢复 | OffloadManager 临时卸载 |

### 2.2 Meta device 在 auto-round 中的使用场景

**场景 1：即时写盘（ShardWriter / Immediate Saving）** — 最核心的用途

```python
# shard_writer.py: 权重写入 safetensors 后，立刻释放内存
module.to("meta")  # 已写盘，不再需要内存
```

当 `low_cpu_mem_usage=True + is_immediate_saving=True` 时，每个 block 量化完 → 打包 →
写入 safetensors 文件 → `to("meta")`。已保存的 block 不占任何内存。

**场景 2：零样本量化（RTN / zero_shot.py）**

```python
# zero_shot.py: 量化完一个 block 后
block.to("meta")  # 释放内存，已经写盘了
```

RTN（iters=0）不需要 calibration 数据，直接逐 block 量化 + 写盘 + `to("meta")`。

**场景 3：MoE 模型的 Expert 替换（fused_moe/）**

```python
# 用 meta device 创建新模块结构（不分配内存）
with torch.device("meta"):
    fused_expert = FusedMoEExpert(...)  # 只有 shape，不分配

# 从原始 expert 拷贝权重后释放原始的
original.to_empty(device="meta")  # 释放原始 expert 内存
```

MoE 模型有大量 expert，用 meta 创建替换模块的"骨架"，避免同时持有新旧两份内存。

**场景 4：wrapper 量化优化后（wrapper.py）**

```python
# 量化调优完成后，如果 layer 支持 update()（如 FP8）
self.orig_layer.update()
self.orig_layer.to("meta")  # 更新完释放
```

---

## 3. Auto-Round Block-wise Quantization 的内存管理

### 3.1 核心流程

auto-round 的量化是 **per-block** 的：每次只把一个 decoder block（如 `model.layers.0`）
搬到 GPU，量化完搬回 CPU，再处理下一个 block。

```
quantize() 核心流程:
1. post_init()          → 模型在 CPU
2. calibration          → 模型搬到 GPU 做 forward pass（或 CPU，取决于 low_gpu_mem_usage）
3. mv_module_from_gpu() → 模型搬回 CPU
4. block 循环:
   for block in all_blocks:
     block.to("cuda")   → 1个 block 上 GPU
     quantize_block()   → 量化
     mv_module_from_gpu(block)  → 搬回 CPU
```

### 3.2 三种内存管理模式

auto-round 根据配置，提供三级递进的内存节省：

**模式 A：默认模式（`low_cpu_mem_usage=False`）**

```
calibration 后:  model.to("cpu")  ← 全部在 CPU RAM
block 循环:
  block_i.to("cuda")   → 量化 → block_i.to("cpu")

CPU RAM:  [block0][block1]...[blockN]  ← 全部权重常驻
GPU VRAM: [block_i]                    ← 一次只一个 block
```

- GPU 峰值：calibration 阶段（整个模型上 GPU）或单个 block
- CPU RAM：始终持有完整模型

**模式 B：`low_cpu_mem_usage=True`**

```
calibration 后:  offloader(model, all_blocks)  ← 存磁盘 + torch.empty(0) 清权重
block 循环:
  offloader.reload(block_i)      ← 从磁盘恢复到 CPU
  block_i.to("cuda")             → 量化
  offloader(block_i, overwrite)  ← 存回磁盘 + 清权重

磁盘:    block0.tmp, block1.tmp, ...
CPU RAM:  [block_i]  ← 只有当前处理的
GPU VRAM: [block_i]
```

- GPU 峰值：单个 block
- CPU RAM：只有单个 block（其余存磁盘）
- 额外磁盘：临时文件存量化后的权重

**模式 C：`low_cpu_mem_usage=True + immediate_saving=True`**

```
block 循环:
  offloader.reload(block_i)    ← 从磁盘恢复
  block_i.to("cuda")           → 量化
  shard_writer.write(block_i)  ← 写入最终 safetensors 文件
  block_i.to("meta")           ← 彻底释放

磁盘:    model-00001.safetensors (最终输出文件)
CPU RAM:  []     ← 空
GPU VRAM: [block_i]
```

- GPU 峰值：单个 block
- CPU RAM：最小（block 写入最终文件后 `to("meta")` 释放）
- 无临时文件（直接写最终输出）

### 3.3 auto-round 其它 block 的设备状态

| 模式 | 其它 block 的状态 | 内存占用 |
|---|---|---|
| 默认 | **CPU** device（真实 tensor） | 占 CPU RAM |
| `low_cpu_mem_usage` | `torch.empty(0)` on CPU（权重存磁盘临时文件）| 几乎零 CPU RAM |
| `low_cpu_mem_usage + immediate_saving` | **meta** device（权重已写入最终 safetensors） | 零 |

---

## 4. Calibration 阶段的 VRAM

calibration 是量化前的数据采集步骤：用校准数据做 forward pass，缓存每个 block 的输入。

### 4.1 默认 calibration（GPU）

```python
# calib.py try_cache_inter_data_gpucpu()
# 默认路径：整个模型搬到 GPU
self.model_context.model = self.model_context.model.to(self.compress_context.device)
all_inputs = self.cache_inter_data(...)
```

这是 **VRAM 的最大瓶颈**：需要整个模型在 GPU 上做 forward pass。
对于 70B 模型（~140GB FP16），这一步就可能 OOM。

### 4.2 `low_gpu_mem_usage=True` 的 calibration（CPU）

```python
# calib.py 第136行
if self.compress_context.low_gpu_mem_usage:
    calibrate_on_cpu = True  # 在 CPU 上做 forward，不搬模型到 GPU
    all_inputs = self.cache_inter_data(...)  # model 留在 CPU
```

当 `low_gpu_mem_usage=True` 时，calibration 在 **CPU** 上做。
速度更慢，但 GPU VRAM 峰值只有后续 block 循环的单个 block。

### 4.3 `device_map="auto"` 的 calibration（多卡分布）

```python
# 使用 accelerate 的 dispatch_model 跨多卡
device_map = infer_auto_device_map(model, max_memory=...)
model = dispatch_model(model, device_map=device_map)
```

模型按层分布到多张 GPU 上，每张卡只持有部分层。

---

## 5. Blockwise Rotation 的 VRAM 分析

### 5.1 Full-model Rotation 路径

```
阶段                           CPU RAM              GPU VRAM
───────────────────────────────────────────────────────────────
1. 加载模型                     全部权重              0
2. apply_to_model() (rotation)  全部权重(原地修改)    0  ← rotation 在 CPU 做
3. calibration                  全部权重              全部模型 ← VRAM 峰值
4. mv_module_from_gpu           全部权重              0
5. block 循环 (量化)            全部权重              1个 block
```

关键：**rotation 在 CPU 上执行**，不消耗 GPU VRAM。
VRAM 峰值在 calibration 阶段（整个模型上 GPU）。

### 5.2 Layerwise (Blockwise) Rotation 路径

```
阶段                           CPU RAM              GPU VRAM
───────────────────────────────────────────────────────────────
1. 加载模型                     全部权重              0
2. prepare_layerwise()          全部权重 + R矩阵      0  ← 只准备矩阵
3. calibration                  全部权重              全部模型 ← 同样的峰值
4. mv_module_from_gpu           全部权重              0
5. block 循环 (rotation+量化)   全部权重              1个 block
```

### 5.3 对比结论

**两者 GPU VRAM 峰值完全相同** — 都在 calibration 阶段（步骤3）。

blockwise rotation **本身不节省 GPU VRAM**，因为：
1. rotation 的数学运算本来就在 CPU 上执行（full-model 也是）
2. calibration 阶段是 VRAM 峰值，两条路径行为一致
3. block 循环阶段两者都是 1 个 block 在 GPU

### 5.4 实测数据验证（Qwen3-0.6B）

从 `layerwise-compare` 测试日志中提取的 VRAM 数据：

| 路径 | GPU VRAM 峰值 | GPU VRAM 平均 |
|---|---|---|
| Full-model | 1.67 GB | 1.58 GB |
| Layerwise  | 1.67 GB | 1.53 GB |

**差异不显著**，符合分析预期。Qwen3-0.6B 只有 ~1.2GB（FP16），本身就能完整放入 GPU。

### 5.5 Blockwise Rotation 的真正价值

虽然不直接节省 GPU VRAM，blockwise rotation 在以下场景有价值：

**1. 配合 `low_cpu_mem_usage=True` 节省 CPU RAM**

```
Full-model + low_cpu_mem:
  rotation 在 offload 之前执行 → 所有层权重必须同时在 CPU RAM

Layerwise + low_cpu_mem:
  rotation 推迟到 block 循环 → 每次只有1个 block 在内存中做 rotation
```

对于 700B+ 模型（FP16 ~1.4TB），连 CPU RAM 都放不下时，
layerwise rotation 可以配合 offload 实现逐 block 加载→rotation→量化→写盘。

**2. 架构优雅性**

rotation 作为 block lifecycle hook 的一环，与量化 pipeline 自然融合：
```python
with self._block_lifecycle(block, name, idx):
    # _on_block_ready() → rotation (自动)
    reference_output = ...
    quantize_block(...)
    # _on_block_quantized() → 后处理 (自动)
```

**3. 可扩展性**

未来任何 per-block transform（如 SmoothQuant、channel pruning）都可以通过
同样的 lifecycle hook 接入，无需修改量化主循环。

---

## 6. ShardWriter 实现原理

ShardWriter 实现**流式分片写盘**——量化一个 block，立刻写入磁盘，释放内存。

### 6.1 核心流程

```
Block 0 量化完
  ↓
save_module(block_0)
  → _add_tensor("model.layers.0.q_proj.weight", tensor)
  → _add_tensor("model.layers.0.k_proj.weight", tensor)
  → ...
  → 累计大小 > max_shard_size?
     YES → _flush_shard()
            → 写 model-shard-00001.safetensors
            → _offload_to_meta()  → block.to("meta") 释放内存
            → 清空 current_shard_tensors

Block 1 量化完 → 同上
...
Block N 量化完
  ↓
finalize()
  → 收集剩余未保存参数 (embed_tokens, lm_head 等)
  → flush 最后一个 shard
  → 临时文件重命名: shard-00001 → model-00001-of-00005
  → 写 model.safetensors.index.json
```

### 6.2 关键设计

**单例模式**：全局只有一个 ShardWriter，跨 block 循环累积 tensor。

**分片大小控制**：根据模型大小自动计算 `max_shard_size`，
单个 shard 通常 1-5 GB，避免单文件过大。

**写完即释放**：`_offload_to_meta()` 检查某个 module 的所有参数是否都已写盘，
如果是则 `module.to("meta")` 释放。

**tied weights 处理**：通过 storage pointer 去重，
检测 `embed_tokens` 和 `lm_head` 共享权重只写一次。

**HuggingFace 标准格式输出**：最终文件命名和 index.json 完全兼容
`from_pretrained()` 加载。

### 6.3 它解决的核心问题

70B 模型量化后仍然 ~35GB（INT4）。如果等全部量化完再 `save_pretrained()`，
需要在内存中同时持有完整模型。ShardWriter 让内存峰值降到 ≈ 1 个 block 的大小。

---

## 7. safetensors mmap 机制

### 7.1 模型加载阶段

auto-round 通过 `AutoModelForCausalLM.from_pretrained(torch_dtype="auto")` 加载模型，
内部使用 `safetensors.safe_open()`，它**始终使用 mmap**：

```python
# safetensors 内部
# safe_open: "Opens a safetensors lazily and returns tensors as asked"
with safe_open(file, framework="pt") as f:
    tensor = f.get_tensor("model.layers.0.q_proj.weight")
    # tensor 是 mmap 支撑的，访问时 OS 按需从磁盘加载页面
```

对于 PyTorch 的 `.bin` 文件，transformers 也使用 `torch.load(file, mmap=True)`。

### 7.2 mmap 的工作原理

```
虚拟地址空间:  [page0][page1][page2]...[pageN]  ← 映射到文件
物理内存:      [    ]  ← 初始为空

访问 page2 → page fault → OS 从磁盘加载 page2 到物理内存
访问 page5 → page fault → OS 加载 page5
```

- **优点**：按需加载，初始不占物理内存
- **限制**：一旦 `.to("cuda")` 或做计算，tensor 被拷贝成真实内存，脱离 mmap

### 7.3 mmap 在 auto-round 中的角色

mmap **只影响初始加载阶段**。进入量化流程后，auto-round 用自己的内存管理：

```
mmap (safetensors)        → OS 级，按需加载，只管初始加载
per-block .to("cuda")     → 框架级，手动控制 GPU VRAM
OffloadManager            → 框架级，手动控制 CPU RAM（存磁盘临时文件）
ShardWriter + to("meta")  → 框架级，彻底释放（写最终文件）
low_gpu_mem_usage         → 框架级，calibration 走 CPU
```

auto-round 的核心节省显存策略是 **per-block 手动搬运**，不依赖 mmap 物化机制。

### 7.4 auto-round 中与 mmap 相关的代码

`base.py` 第658-661行的注释提到了 mmap：

```python
# This also detaches any parameter tensors that are still backed by
# safetensors' mmap, preventing per-block RSS growth (~14 MB/block)
# when .to(device) page-faults the underlying file pages into physical memory.
```

这说明 auto-round 意识到 mmap 的副作用：当 block `.to("cuda")` 时会触发 page fault，
把文件页面调入物理内存（RSS 增长）。通过 `.to(bfloat16)` dtype 转换来 detach mmap 支撑，
避免这种增长。

---

## 8. 完整 VRAM 时间线对比

### 8.1 默认模式（无 rotation）

```
阶段                        GPU VRAM           CPU RAM
────────────────────────────────────────────────────────
加载模型                     0                  全部 (mmap)
calibration                  全部模型 ← 峰值    全部
mv_module_from_gpu           0                  全部
block_0.to(cuda)             block_0            全部
  quantize_block()           block_0            全部
  mv_from_gpu(block_0)       0                  全部
block_1.to(cuda)             block_1            全部
  ...
完成                         0                  全部(已量化)
save_pretrained              0                  全部
```

GPU VRAM 峰值 = **整个模型**（calibration 阶段）

### 8.2 `low_cpu_mem_usage=True`

```
阶段                        GPU VRAM           CPU RAM
────────────────────────────────────────────────────────
加载模型                     0                  全部 (mmap)
calibration (CPU)            0 ← 不上 GPU      全部
offload(all_blocks)          0                  ≈ 0 (存磁盘)
reload(block_0) + to(cuda)   block_0            block_0
  quantize_block()           block_0            block_0
  offload(block_0)           0                  ≈ 0
reload(block_1) + to(cuda)   block_1            block_1
  ...
reload_all                   0                  全部(已量化)
save_pretrained              0                  全部
```

GPU VRAM 峰值 = **单个 block**

### 8.3 `low_cpu_mem_usage=True + immediate_saving=True`

```
阶段                        GPU VRAM           CPU RAM
────────────────────────────────────────────────────────
加载模型                     0                  全部 (mmap)
calibration (CPU)            0                  全部
offload(all_blocks)          0                  ≈ 0
reload(block_0) + to(cuda)   block_0            block_0
  quantize_block()           block_0            block_0
  shard_writer.write()       block_0            block_0
  block_0.to("meta")         0                  0 ← 彻底释放
reload(block_1) + to(cuda)   block_1            block_1
  ...
shard_writer.finalize()      0                  剩余(embed/lm_head)
```

GPU VRAM 峰值 = **单个 block**
CPU RAM 峰值 = **单个 block**（其余已写盘释放）

---

## 9. 代码位置索引

| 功能 | 文件 | 关键行 |
|---|---|---|
| 模型加载 (mmap) | `auto_round/utils/model.py` | `llm_load_model()` L289, `from_pretrained` L353 |
| calibration GPU/CPU 选择 | `auto_round/compressors_new/calib.py` | `try_cache_inter_data_gpucpu()` L136 |
| 模型搬回 CPU | `auto_round/compressors_new/calib.py` | `mv_module_from_gpu()` L1133 |
| block 搬到 GPU | `auto_round/compressors_new/calib.py` | `m.to(device)` L966 |
| block lifecycle hooks | `auto_round/compressors_new/base.py` | `_block_lifecycle()` L972 |
| rotation 初始化 | `auto_round/compressors_new/base.py` | `_init_rotation()` L878 |
| OffloadManager | `auto_round/utils/offload.py` | `OffloadManager` L231 |
| _clear_module_weights | `auto_round/utils/offload.py` | `torch.empty(0)` L126 |
| ShardWriter | `auto_round/compressors_new/shard_writer.py` | `ShardWriter` L33 |
| ShardWriter 写盘释放 | `auto_round/compressors_new/shard_writer.py` | `_offload_to_meta()` L246 |
| mmap 副作用注释 | `auto_round/compressors_new/base.py` | L658-661 |
| Meta device 检查 | `auto_round/compressors_new/calib.py` | L158 |

---

## 10. 总结

1. **auto-round 节省 GPU VRAM 的核心机制**是 per-block 手动搬运（`.to("cuda")` / `.to("cpu")`），
   不依赖 PyTorch mmap 物化机制。

2. **VRAM 峰值瓶颈**在 calibration 阶段（整个模型上 GPU 做 forward pass），
   `low_gpu_mem_usage=True` 通过在 CPU 做 calibration 来消除这个瓶颈。

3. **blockwise rotation 不直接节省 GPU VRAM**，因为 rotation 本来就在 CPU 上执行。
   它的价值在于配合 `low_cpu_mem_usage` 节省 CPU RAM，以及架构上的优雅性和可扩展性。

4. **三级内存管理**递进：默认（全部在 CPU）→ OffloadManager（存磁盘临时文件）→
   ShardWriter（写最终文件 + `to("meta")` 彻底释放）。

5. **safetensors mmap** 只影响模型初始加载，进入量化流程后由 auto-round 自己管理内存。
