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

## 3. concat 时的 EOS / 文档边界处理(问题分析,修复见 §3.5)

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

## 3.5 ✅ 已修复:concat 打包重写(EOS 分隔 + BOS 可配 + 计数 bug)

> 状态:**已实现并跨 6 个模型系列验证通过**。修改文件:`auto-round/auto_round/calib_dataset.py`。

### 3.5.1 实现

抽出模块级共享函数,原来 **两处重复** 的内联打包逻辑
(`concat_dataset_element` 通用路径 + `get_opencode_instruct_dataset` 内置 concat)
统一调用它,彻底消除重复与 `os_cnt/have_bos/have_eos` 不重置的 bug:

```python
def tokenizer_adds_bos(tokenizer): ...      # 探测 tokenizer 是否原生在首位加 BOS
def pack_documents(input_ids_iterable, seqlen, tokenizer,
                   add_eos_separator=True, add_bos="auto"): ...
def concat_dataset_element(dataset, seqlen, tokenizer,
                           add_eos_separator=True, add_bos="auto"): ...
```

**贪心打包算法**:逐文档独立处理(不再有跨样本残留状态)——
1. 剥掉该文档自带的前导 BOS / 末尾 EOS(归一化);
2. 追加到 buffer,并在文档之间插入 1 个 EOS 作分隔(`add_eos_separator`);
3. buffer 满 `content_len` 就切出一条,若需要在首位补 1 个 BOS(`add_bos`);
   `content_len = seqlen - (1 if 补BOS else 0)`,保证每条恰好 `seqlen`;
4. 末尾未攒满的残余丢弃(与旧行为一致)。

每条输出:`input_ids` / `attention_mask` 均为一维 `int64`,`attention_mask` 全 1。

### 3.5.2 默认值(设计取舍)

指导原则:**校准数据应尽量贴近推理时模型真实看到的输入分布**
(PTQ 是数据驱动的,激活分布决定选出的 round/scale)。

| 参数 | 默认 | 理由 |
|---|---|---|
| `add_eos_separator`(文档间 EOS 分隔) | **True** | 避免同一条 2048 内 A/B 文档跨界"串味"(注意力污染);EOS-then-text 模型训练时(多轮/packed 预训练)见过,in-distribution;是标准打包做法。 |
| `add_bos`(每条 chunk 补 BOS) | **"auto"** | **跟随 tokenizer 原生行为**:Llama/Gemma 推理首位总有 BOS→补;Qwen/gpt-oss 原生不加→不补。不能无脑"有 bos_token_id 就补"。 |

⚠️ **关键坑**:判断"是否补 BOS"**必须用探测法,不能读 `add_bos_token` 属性**。
Llama-3 的 `add_bos_token=False`,但它实际会通过 post-processor 在首位注入 BOS。
`tokenizer_adds_bos()` 用 `tokenizer("a", add_special_tokens=True)[0] == bos_id` 探测,
才能得到正确答案。

### 3.5.3 DSL 用法(和 `concat` 一样可配)

```
ultrachat_200k:concat=true                       # 默认: eos=true, bos=auto
ultrachat_200k:concat=true:eos=false             # 关闭文档间 EOS 分隔
ultrachat_200k:concat=true:bos=true              # 强制每条补 BOS
ultrachat_200k:concat=true:bos=false             # 强制不补 BOS
ultrachat_200k:concat=true:bos=false:eos=false   # 完全直连(≈旧行为)
```

编程调用:
```python
from auto_round.calib_dataset import pack_documents
packed = pack_documents(docs, seqlen, tokenizer,
                        add_eos_separator=True, add_bos="auto")
```

### 3.5.4 如何确认"Qwen/gpt-oss 推理首位不加 BOS"(实证方法)

**不要凭直觉,用两步实测**(见下方复现脚本):

1. **原生 plain 分词**:`tokenizer("Hello world", add_special_tokens=True)`,
   看首 token 是否 == `bos_token_id`。
2. **真实推理输入(chat template)**:`tokenizer.apply_chat_template(msg,
   tokenize=True, add_generation_prompt=True)`,看**真正喂给模型**的首 token 是什么。

**实测结果(2026-08,transformers 官方仓库 tokenizer)**:

| 模型 | `bos_token_id` | plain 首 token | **chat template 首 token(真实推理)** | 推理首位加 BOS? |
|---|---|---|---|---|
| Qwen3-0.6B | **None** | 9707(普通词) | `151644 <|im_start|>` | **否** |
| Llama-3.1-8B-Instruct | 128000 | `128000 <|begin_of_text|>` | `128000 <|begin_of_text|>` | **是** |
| gpt-oss-20b | 199998 `<|startoftext|>` | 13225(普通词) | `200006 <|start|>` | **否**(用 harmony `<|start|>`,不是 BOS) |
| gemma-2-9b | 2 | `2 <bos>` | (base 无 chat template) | 是(plain 即加) |
| gemma-3-4b-it | 2 | `2 <bos>` | `2 <bos>` | **是** |

**结论**:
- Qwen3 `bos_token_id` 本身就是 `None`,谈不上加 BOS;推理首位是 `<|im_start|>` 角色标记。
- gpt-oss **虽然定义了** `bos=199998 <|startoftext|>`,但**原生和 chat template 首位都不用它**
  (harmony 格式用 `<|start|>`=200006)。所以"有 bos_token_id ≠ 应该补 BOS"——
  这正是默认必须用 `auto`(探测)而非"非 None 就补"的原因。
- Llama/Gemma 推理首位确实是各自 BOS(`<|begin_of_text|>` / `<bos>`),`auto` 会正确补上。

### 3.5.5 跨模型验证(全部 PASS)

用真实 tokenizer 对 `pack_documents` 做端到端校验(每条恰好 seqlen、attention 全 1、
id 在 embedding 范围内、`auto` 补 BOS 行为 == tokenizer 原生行为):

| 模型 | bos | eos | `auto` 补 BOS | EOS 分隔生效 | max_id < emb | 结果 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | None | 151645 | 否 | ✅ | 151645<151669 | PASS |
| Llama-3.1-8B | 128000 | 128009 | 是(128000) | ✅ | 128009<128256 | PASS |
| Llama-2-7B | 1 | 2 | 是(1) | ✅ | <32000 | PASS |
| gpt-oss-20b | 199998 | 200002 | 否 | ✅ | 200002<200019 | PASS |
| gemma-2-9b | 2 | 1 | 是(2) | ✅ | <256000 | PASS |
| gemma-3-4b-it | 2 | 1 | 是(2) | ✅ | <262145 | PASS |

> 注:Llama-3.1 / gpt-oss 的特殊 token id **等于或高于 base `vocab_size`**
> (bos 恰好 = vocab_size),但都 **< 模型 embedding 尺寸 `len(tokenizer)`**,是合法输入。
> 校验/量化时 id 上界应取 `max(len(tokenizer), vocab_size)`,不是 base `vocab_size`。

### 3.5.6 复现脚本

```bash
# 用真实 tokenizer 确认推理首位 BOS 行为(需要 HF_TOKEN 访问 gated 仓库)
cd /home/hshen/lkk/calib_feat/auto-round
HF_TOKEN=<your_token> CUDA_VISIBLE_DEVICES="" python - <<'PY'
from transformers import AutoTokenizer
def to_ids(x):
    if hasattr(x,"input_ids"):
        v=x.input_ids; return list(v[0]) if v and isinstance(v[0],(list,tuple)) else list(v)
    try: return list(x["input_ids"])
    except Exception: pass
    if hasattr(x,"ids"): return list(x.ids)   # tokenizers.Encoding (gpt-oss)
    return list(x)
msg=[{"role":"user","content":"Hello, who are you?"}]
for mid in ["Qwen/Qwen3-0.6B","meta-llama/Llama-3.1-8B-Instruct",
            "openai/gpt-oss-20b","google/gemma-3-4b-it"]:
    t=AutoTokenizer.from_pretrained(mid, trust_remote_code=True); bos=t.bos_token_id
    plain=to_ids(t("Hello world", add_special_tokens=True))
    ct=to_ids(t.apply_chat_template(msg, tokenize=True, add_generation_prompt=True))
    print(mid, "bos=",bos, "plain_first=",plain[0], "chat_first=",ct[0],
          "chat_first_tok=",t.convert_ids_to_tokens([ct[0]])[0])
PY

# 用校准验证脚本看打包后 EOS 分隔符是否出现(eos_inside > 0)
cd /home/hshen/lkk/calib_feat/calib_experiments
CUDA_VISIBLE_DEVICES="" python scripts/verify_calib_datasets.py \
  --datasets "ultrachat_200k:concat=true" --nsamples 4 --preview 1 --max-rows 400
```

### 3.5.7 补充:开关行为验证矩阵 + 复现注意事项

**(a) `bos` / `eos` 开关确实生效(实测,SEQ=64,5 段合成文档)**——
`auto` 与各家原生 BOS 行为完全一致,强制开关按预期改变每条 chunk 首 token 与 EOS 分隔数:

| 模型 | natural_bos | **auto** 首 token | 强制 `bos=true` | 强制 `bos=false` | eos 默认分隔数 | `eos=false` |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | False | 785(不补) | 785(bos=None 无法补) | 785 | 3 | 0 |
| Llama-3.1-8B | True | 128000 | 128000 | 791(不补) | 3 | 0 |
| Llama-2-7B | True | 1 | 1 | 450(不补) | 4 | 0 |
| gpt-oss-20b | False | 976(不补) | 199998(强制补) | 976 | 3 | 0 |
| gemma-2-9b | True | 2 | 2 | 651(不补) | 3 | 0 |
| gemma-3-4b-it | True | 2 | 2 | 818(不补) | 3 | 0 |

断言 `auto 补 BOS == tokenizer 原生行为` 对 6 个系列**全部通过**。要点:
- `bos=false` 能把 Llama/Gemma 的强制去掉(首 token 变成真实内容 token);
- `bos=true` 能把 gpt-oss 强制补上 199998(但一般不需要,`auto` 已判为不补);
- Qwen `bos_token_id=None`,即使 `bos=true` 也无从补(保持 785),符合预期。

**(b) 复现坑:`apply_chat_template(tokenize=True)` 返回类型不统一**
- **gpt-oss**:返回 `tokenizers.Encoding` 对象 → 取 `.ids` 才是 token 列表;
- **Qwen / Llama / Gemma**:返回 `BatchEncoding`(dict) → 取 `["input_ids"]`。
- 直接 `list(...)` 会拿到 dict 的 key(`['input_ids','attention_mask']`)或报
  `TypeError`。上面复现脚本里的 `to_ids()` 已统一处理三种返回类型。

**(c) `add_bos_token` 属性不可信(再次强调)**:Llama-3.1 的 `add_bos_token=False`,
但 plain 分词与 chat template 首位**都是** `<|begin_of_text|>`(128000)。必须以
"实际分词首 token 是否 == bos_id"为准,这正是 `tokenizer_adds_bos()` 的探测逻辑。

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
| #3 concat EOS / 边界 / 计数重置 | ✅ 已修(重写 `pack_documents`,见 §3.5) |
| #3 附:EOS/BOS 可配 + 跨 6 模型系列验证 | ✅ 已验证(Qwen/Llama/gpt-oss/Gemma) |
