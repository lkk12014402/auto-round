# 校准数据 concat / EOS 处理讨论记录

> 记录本次排查与讨论,供后续修复参考。涉及文件:
> - `auto-round/auto_round/calib_dataset.py`
> - `calib_experiments/scripts/run_quantize.py`
> - `calib_experiments/configs/experiments.py`

---

## 0. 背景

在做"校准数据集对比研究"(`calib_experiments`),对 Qwen3-0.6B 做 MXFP4 量化,
比较不同校准数据(pile-10k / ultrachat / IF / math / opencode / swe / 各种 mix)
对量化精度的影响。排查日志时发现并讨论了以下几个问题。

---

## 1. 已修复:concat 路径 `attention_mask` 维度 bug

**现象**:带 `concat=true` 的数据集(ultrachat 等)在 `dataset.cast(...)` 报错:
```
TypeError: Couldn't cast array of type list<item: int8> to int8
```
`pile-10k` 不报错(未走 concat 路径)。

**根因**:`calib_dataset.py` 的通用拼接函数 `concat_dataset_element`(约 1173 行)里
```python
attention_mask = torch.ones([1, seqlen]).to(torch.int64)   # 二维,错误
```
导致每条 `attention_mask` 是二维 `list<list<int8>>`,而下游 cast 目标是一维
`Sequence(Value("int8"))`,类型不匹配。

**修复**(已应用):改成一维,与 input_ids 维度一致、也与另一处正确实现
(OpenCodeInstruct 内置 concat,约 428 行的 `torch.ones([seqlen])`)保持一致:
```python
attention_mask = torch.ones([seqlen]).to(torch.int64)
```

---

## 2. 已修复(规避):torch.compile 触发的循环导入 ImportError

**现象**:修复 #1 后,ultrachat 数据 cast 通过、`caching done`,但量化第 0 层崩溃:
```
ImportError: cannot import name '_normalize_function_or_error'
             from 'torch.fx.operator_schemas'
```
`pile-10k` 全程正常。

**根因(非本项目数据 bug)**:
- 符号 `_normalize_function_or_error` **确实存在**(定义在 `torch/fx/operator_schemas.py`
  第 344 行,位置靠后),单独导入正常。
- 这是 **torch 内部的懒加载循环导入(half-initialized circular import)**:
  `torch/_subclasses/schema_check_mode.py` 顶层 `from torch.fx.operator_schemas import
  _normalize_function_or_error`,而该模块又被 `_python_dispatch._disable_current_modes()`
  懒加载。当 dynamo 首次编译 block 触发时,若 `operator_schemas` 恰好还没执行到第 344 行
  就被重入,符号尚未定义 → ImportError。
- auto-round 在 **Linux + CUDA 下默认开启 `torch.compile`**
  (`compressors/base.py:356` → `utils/device_manager.py:897`
  `default_enable_torch_compile`:非 xpu、非 win32 即 True),所以走到 dynamo 才踩坑。
- ultrachat vs pile 的差异只是**模块导入时序**不同(ultrachat 额外拉起 HF datasets 流式
  加载 + chat template),恰好让循环导入在 dynamo 编译时首次发生。**与 concat/数据内容无关**。

**责任归属**:根因是 **PyTorch 的脆弱循环导入**(torch 2.11.0);auto-round 只是默认开
compile 的"触发者",不算功能性 bug;本项目脚本/数据无错。

**修复(方案 A 预导入,已应用)**:在 `scripts/run_quantize.py` 顶部、`import auto_round`
之前,提前完整加载这两个模块,打破循环导入 race:
```python
try:
    import torch.fx.operator_schemas            # noqa: F401
    import torch._subclasses.schema_check_mode  # noqa: F401
except Exception:
    pass
```
备选方案 B(未采用):`AutoRound(..., enable_torch_compile=False)` / `--disable_torch_compile`,
彻底关掉 compile,更稳但量化更慢。

---

## 3. 待修复(重点):concat 时的 EOS / 文档边界处理

### 3.1 concat 是怎么做的

通用函数 `concat_dataset_element`(`calib_dataset.py` ~1173):贪心缓冲区打包。
逐条把样本塞进 `buffer_input_id`,填满 `seqlen`(2048)就切出一条,产出的每条长度恰好
= seqlen,`attention_mask` 全 1;放不下的样本从切点截断,余下部分续接到下一条。
末尾未攒满 seqlen 的 buffer 被丢弃。

EOS/BOS 处理:每条原始样本自带的 bos/eos **先被剥掉**,打包后只在整条 2048 序列的
**最前补 1 个 bos、最后补 1 个 eos**,且**条件性**(`have_bos/have_eos` 为真才补,
即原始样本本身带 bos/eos 才补)。**文档之间不插任何 eos/分隔符**。

> 注意:多数据集"混合"是**两层不同含义**的:
> - `concat=true` = 每个数据集**内部** token 级打包成 2048;
> - mixture(逗号分隔 + `num=`) = 各数据集**已打包好的整条样本按配额行级拼接再 shuffle**
>   (`_get_dataset_impl` 里 `concatenate_datasets` + `shuffle`),**跨数据集 token 不会
>   混进同一条序列**。

### 3.2 EOS 是否影响量化结果 —— 严谨结论(纠正过度断言)

先前有一句"现在的 MXFP4 量化 EOS 处理对结果无影响",**这是过度断言,已纠正**。

- **说对的部分**:auto-round **默认 loss 是逐块激活重构 MSE**
  (`sign_round/quantizer.py:412,680` `mse_loss(pred_output, ref_output)`),**不是**
  next-token 交叉熵。所以**不存在"EOS 作为监督标签缺失/少了梯度信号"的问题**——从
  "LLM 建模 next token"角度担心的那类影响,在默认路径下不成立(无 label、无 shift、
  EOS 不作 target)。
  - 唯一走 next-token CE 的是 `lfq_loss`(`quantizer.py:253`),但 `enable_lfq`
    **默认 False**、标注 experimental、且只作用于最后一个 block;当前实验未启用。

- **说错/需纠正的部分**:"对结果无影响"不成立。校准是**数据驱动**的,MSE 的目标
  `ref_output` 完全由输入 token 流决定。EOS/拼接方式会通过以下机制**实实在在改变结果**:
  1. **改变输入 token**:EOS 位置喂的是 EOS embedding 而非普通 token → 激活不同。
  2. **改变 attention 分布**:文档间**不插 EOS** 时,同一条 2048 内 A/B 文档 token
     相互 attend(串味),边界处激活被污染;插 EOS 可起软分隔。
  3. **改变被拟合位置集合**:我们建的 `attention_mask` 全为 1,**未 mask 任何位置**,
     每个位置(含边界、含 EOS)都计入 MSE,边界污染会进 loss → 影响选出的 round/scale。

- **正确表述**:EOS/拼接**会**影响量化权重(经由"校准激活分布"这一层,而非"loss 标签"
  这一层),但**量级通常很小**(2048 里只有个位数 EOS/边界位置,激活重构对少量扰动鲁棒)。
  **量级小 ≠ 无影响**;从产品严谨角度不应当成"零"。

### 3.3 `concat_dataset_element` 现存的严谨性问题(待修)

1. **文档间无 EOS 分隔**:同一条 2048 内多个短文档首尾直连,无 boundary 标记。
   - 对 MSE 影响小;
   - 一旦启用 `enable_lfq`(CE),会出现"用 A 文档末 token 预测 B 文档首 token"的
     错误跨文档监督,污染更明显。规范做法:文档间显式插 1 个 EOS 作分隔。

2. **`os_cnt / have_bos / have_eos` 在循环里从不重置**(初始化后再未清零):
   - 一旦某条样本带 bos/eos,标志**永久 True**、`os_cnt` 单调累加,影响后续
     `idx_keep` 长度计算与 `== seqlen` 判定 → 可能**打包长度偏移 / 边界 token 错放**。
   - 对 Qwen(纯文本 tokenize 通常不加 bos/eos,全程 False、os_cnt=0)无影响;
     但换 Llama 等**自动加 BOS/EOS** 的 tokenizer 时是**潜在 bug**。

3. **EOS 补加逻辑对 tokenizer 行为不稳健**:当前对 Qwen 实际结果是"整条 2048 无任何
   EOS 的纯 token 流"。是否/如何补 EOS 应对"tokenizer 是否加特殊符"稳健、可控。

### 3.4 后续修复建议(优先级)

- **当前实验(MXFP4 + 默认 MSE)**:可继续跑,EOS 影响小、不改变对比研究的主要结论。
- **产品级修复(后续做)**:重写 `concat_dataset_element`,做到
  1. 文档之间显式插入 EOS 作分隔(或提供开关);
  2. 修复 `os_cnt / have_bos / have_eos` 每条重置的问题;
  3. EOS/BOS 补加逻辑对不同 tokenizer 行为稳健、可配置;
  4. 保持 `attention_mask` 一维(已修),并考虑是否需要对边界/padding 位置做 loss mask。
- 同步检查 OpenCodeInstruct 内置 concat(~428 行)是否有相同的计数不重置 / 边界问题。

---

## 4. 单跑命令备忘

只跑单个 cell(过滤器 `--models/--schemes/--recipes`):
```bash
cd /home/hshen/lkk/calib_feat/calib_experiments
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_experiments.py \
  --schemes MXFP4 --recipes ultrachat --skip-eval
```
最小复现(直接调原子量化脚本):
```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_quantize.py \
  --model Qwen/Qwen3-0.6B --scheme MXFP4 \
  --dataset "ultrachat_200k:concat=true" \
  --output-dir /tmp/test_ultrachat \
  --nsamples 128 --seqlen 2048 --iters 200 --seed 42 --device-map 0
```

---

## 5. 状态小结

| 项 | 状态 |
|---|---|
| #1 attention_mask 二维 → 一维 | ✅ 已修 |
| #2 torch 循环导入(预导入规避) | ✅ 已修(规避) |
| #3 concat EOS / 边界 / 计数重置 | ⬜ 待修(本文档重点) |
