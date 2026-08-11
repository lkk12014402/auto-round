# AutoRound 校准数据集与 DataLoader 全面深度分析

> 适用代码库：`/home/hshen/lkk/calib_feat/auto-round`
> 覆盖范围：LLM / MLLM / Diffusion 三条数据线、注册机制、DSL、token 拼接、过滤、子进程预处理、消费端 Calibrator、校准触发场景，以及 PR #2107（代码模型自动选择校准集）的完整分析。

---

## 目录

1. [背景：量化为什么需要校准数据](#1-背景量化为什么需要校准数据)
2. [整体架构与三条数据线](#2-整体架构与三条数据线)
3. [LLM（纯文本）数据线深入](#3-llm纯文本数据线深入)
   - 3.1 注册机制
   - 3.2 数据集字符串 DSL
   - 3.3 单数据集处理流水线
   - 3.4 已注册数据集清单与差异
   - 3.5 token 级定长拼接 `concat_dataset_element`
   - 3.6 过滤 `filter_func`
   - 3.7 多数据集配额与混合
   - 3.8 子进程预处理与内存优化
   - 3.9 DataLoader 与 collate
4. [MLLM（多模态）数据线深入](#4-mllm多模态数据线深入)
5. [Diffusion（文生图/音）数据线深入](#5-diffusion文生图音数据线深入)
6. [消费端：Calibrator 如何使用校准数据](#6-消费端calibrator-如何使用校准数据)
7. [哪些场景会用到校准数据集](#7-哪些场景会用到校准数据集)
8. [PR #2107 深度分析](#8-pr-2107-深度分析)
9. [关键设计总结与工程要点](#9-关键设计总结与工程要点)

---

## 1. 背景：量化为什么需要校准数据

AutoRound 是 Intel 提出的 SOTA 权重量化算法（核心是 **SignRound**：对每个权重的舍入方向做可学习的符号扰动 `V`，并对量化区间的 min/max 做缩放 `α/β`，通过 block 级重构损失优化）。

要做这种"带优化的量化"，必须让模型跑一小批真实数据（**calibration / 校准数据**），用来：

- **统计激活分布**：为激活量化（static activation quant）确定 scale/zero-point；为 imatrix（importance matrix）加权。
- **驱动逐块（block-wise）前向**：缓存每个 transformer block 的输入张量，作为 SignRound 优化的重构目标（让量化后 block 输出尽量逼近原始输出）。
- **AutoScheme 的 delta-loss 评估**：在多种量化方案间做选择时，需要真实数据计算每种方案的损失增量。

因此"校准数据集 + DataLoader"是 AutoRound 除算法本身之外最关键的基础设施。它需要解决：数据从哪来、如何 tokenize、如何组织成定长样本、如何混合多来源、如何在有限内存下高效预处理。

---

## 2. 整体架构与三条数据线

代码按模型形态分成三条互相独立、但接口风格一致的数据线：

| 场景 | 数据文件 | 注册表 | DataLoader 入口 | 默认数据集 |
|------|----------|--------|-----------------|-----------|
| **LLM（纯文本）** | `auto_round/calib_dataset.py` (1181 行) | `CALIB_DATASETS` | `get_dataloader()` | `NeelNanda/pile-10k` |
| **MLLM（多模态）** | `auto_round/compressors/mllm/dataset.py` | `MLLM_DATASET` | `get_mllm_dataloader()` | `liuhaotian/llava_conv_58k` |
| **Diffusion（文生图/音）** | `auto_round/compressors/diffusion/dataset.py` | `DIFFUSION_DATASET` | `get_diffusion_dataloader()` | `coco2014` |

三条线共享同一套"注册装饰器 + 工厂函数"模式，但数据形态差异很大：

- LLM 产出 `{input_ids, attention_mask}` 定长 token 批。
- MLLM 产出图文混合的 `processor` 输出（含 `pixel_values` / `<image>` 占位符等）。
- Diffusion 产出 `(caption_id, caption_text)` 文本 prompt 对（图像/音频由 pipeline 内部生成）。

消费端统一在 `auto_round/calibration/` 下的三个 Calibrator：`llm.py` / `mllm.py` / `diffusion.py`，由 `_get_calibrator_kind()` 决定用哪个。

调用链（以 LLM 为例）：

```
AutoRound(...)                                  # 用户入口 (entry.py / orchestrator.py)
  └─ Compressor.__init__ (base.py)
       └─ 决定 self.dataset（默认或自动选择，见 PR #2107）
  └─ Compressor.post_init → get_calibrator("llm")(self)  → LLMCalibrator
       └─ calibration() → cache_inter_data() → calib()
            └─ get_dataloader(tokenizer, seqlen, dataset, seed, bs, nsamples)   # calib_dataset.py
                 └─ get_dataset() → _get_dataset_impl()   # 解析 DSL、加载、tokenize、拼接、过滤、混合
```

---

## 3. LLM（纯文本）数据线深入

文件：`auto_round/calib_dataset.py`。这是最核心、最复杂的一条线。

### 3.1 注册机制

```python
CALIB_DATASETS = {}

def register_dataset(name):
    """类/函数装饰器：把一个"加载函数"注册进全局表，支持别名列表。"""
    def register(dataset):
        names = name if isinstance(name, list) else [name]
        for global_name in names:
            CALIB_DATASETS[global_name] = dataset
        return dataset
    return register
```

每个数据集对应一个**加载函数**，签名统一：

```python
def get_xxx_dataset(tokenizer, seqlen, dataset_name=..., split=None,
                    seed=42, apply_chat_template=False, system_prompt=None):
    ...
    return calib_dataset   # HuggingFace Dataset 或 IterableDataset
```

这种插件式设计的好处：新增数据集只需写一个函数 + 一行装饰器，不用改动核心逻辑；别名（如 `"NeelNanda/pile-10k"` 与 `"pile-10k"`）指向同一函数。

### 3.2 数据集字符串 DSL

`dataset_name` 不是简单的字符串，而是一个**逗号分隔的 mini-DSL**，在 `_get_dataset_impl()` 中解析。示例：

```
"pile-10k,github-code-clean:num=100:concat=true,./my.json:apply_chat_template=true"
```

解析规则（每段以 `name` 开头，后接零个或多个 `:key=value`）：

| key | 含义 | 备注 |
|-----|------|------|
| `split=train+validation` | 指定 / 拼接 split | 用 `+` 连接多个 split |
| `num=100` | 该来源固定取 100 条 | 记入 `data_lens`，参与配额计算 |
| `concat=true` | 是否做 token 级定长拼接 | 见 §3.5，默认 `false` |
| `apply_chat_template=true` | 套用 chat 模板 | instruct 模型常用 |
| `system_prompt=...` | 自定义系统提示 | 设置后自动开启 chat template |

解析代码（节选）：

```python
if ":" in name:
    name, split_list = name.split(":")[0], name.split(":")[1:]
    for ele in split_list:
        key, values = ele.split("=")[0], ele.split("=")[1:]
        if key == "split":                 split = values[0].split("+")
        if key == "num":                   data_lens[name] = int(values[0])
        if key == "concat":                do_concat = ... True 除非显式 "false"
        if key == "apply_chat_template":   apply_chat_template = ... True 除非显式 "false"
        if key == "system_prompt":         system_prompt = values[0]; apply_chat_template = True
```

**名称解析容错**（重要，PR #2107 依赖它）：

```python
if is_local_path(name):
    get_dataset = CALIB_DATASETS.get("local")          # 本地文件走 local
else:
    calib_name = name
    if name not in CALIB_DATASETS:                      # 精确匹配失败
        calib_name = name.split("/")[-1]                # 退而取 "/" 后缀
        for key in CALIB_DATASETS.keys():
            if calib_name in key:                        # 再做子串匹配
                calib_name = key; break
    get_dataset = CALIB_DATASETS.get(calib_name)
```

找不到时抛出 `ValueError`，并列出所有不含 `/` 的可用短名（更友好）。

### 3.3 单数据集处理流水线

以默认 `get_pile_dataset` 为例，标准四步：

```python
split = "train"
tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_chat_template, system_prompt)
calib_dataset = load_dataset("NeelNanda/pile-10k", split=split)   # 1. 加载
calib_dataset = calib_dataset.shuffle(seed=seed)                   # 2. 打乱
calib_dataset = calib_dataset.map(                                 # 3. tokenize
    tokenizer_function, batched=True,
    new_fingerprint=_make_map_fingerprint(...))                    # 4. 稳定指纹→缓存命中
```

关键辅助函数：

- **`get_tokenizer_function`**：返回闭包，`apply_chat_template=False` 时直接 `tokenizer(examples["text"], truncation=True, max_length=seqlen)`；否则走 `apply_chat_template_to_samples`。
- **`apply_chat_template_to_samples`**：把每条文本包装成 `messages`（可选 system prompt），渲染为对话字符串。**容错**：模板渲染失败时自动剥离 system role 重试（DeepSeek 等模型不建议用 system prompt）。
- **`_make_map_fingerprint`**：为 `.map()` 计算稳定指纹，保证 HuggingFace datasets 磁盘缓存跨进程/跨运行命中（对 §3.8 的子进程优化至关重要）。

**网络错误友好提示**：`get_pile_dataset` 捕获加载异常，若识别为 proxy/SSL 错误，提示用户检查代理，或改用 `pip install modelscope` + `--dataset swift/pile-val-backup` 备份源，然后 `sys.exit(1)`。

### 3.4 已注册数据集清单与差异

| 短名 | 全名 | 领域 | 加载方式 | 文本字段 | 特殊处理 |
|------|------|------|----------|----------|----------|
| `pile-10k` | NeelNanda/pile-10k | 通用英文 | 全量 load | `text` | 默认数据集，train split |
| `pile-val-backup` | swift/pile-val-backup | 通用英文 | modelscope 流式 take(10000) | `text` | 需 `pip install modelscope`，备份源 |
| `CCI3-HQ` | BAAI/CCI3-HQ | 中文高质量 | streaming take(10000) | `text` | 流式 |
| `github-code-clean` | codeparrot/github-code-clean | 代码 | streaming，concat MIT+Apache | `code` | **需 datasets≤3.6.0**（script 数据集） |
| `opencode-instruct` | nvidia/OpenCodeInstruct | 代码指令 | streaming take(10000) | 拼 `input`+`output` | **函数内自带定长拼接** |
| `ultrachat_200k` | HuggingFaceH4/ultrachat_200k | 对话 | streaming take(20000) | `messages` | 自动检测 instruct tokenizer |
| `Ultra-FineWeb` | openbmb/Ultra-FineWeb | 通用 en/zh | streaming take(20000) | `content` | 支持 `split=en/zh` |
| `new-title-chinese` | madao33/new-title-chinese | 中文 | 全量 load | `content` | train split |
| `mbpp` | google-research-datasets/mbpp | 代码 | 全量 load | 拼 `text`+`code` | 默认合并 train+val+test |
| `audiocaps` | AudioCaps | 音频描述 | CSV 下载 | `caption` | 音频相关模型校准 |
| `local` | 本地 json/jsonl | 任意 | 读文件 | 见下 | 用户自定义 |

**几个值得注意的差异**：

- **字段名不同**：pile 用 `text`，github-code 用 `code`，Ultra-FineWeb/new-title 用 `content`，ultrachat 用 `messages`。所以每个函数各自定义 `tokenizer_function` 来读对应字段。
- **`opencode-instruct` 内置拼接**：它在函数内部就做了一遍 §3.5 的定长拼接逻辑（用 `attention_mask = torch.ones([seqlen])` 且输出 `.tolist()`），产出已经是定长块。这解释了 PR #2107 为何对它用 `concat=true`——外层再包一层 DSL 的 concat 语义一致。
- **`ultrachat` 的 apply_chat_template 逻辑**：先探测 tokenizer 是否为 instruct 型，给出日志建议，但**最后一行强制 `apply_chat_template = False`**（当前实现里对 ultrachat 实质走纯拼接分支，属于代码中的一个"保守回退"）。
- **`local`**：支持 `.json` / `.jsonl`；每条可为 `str`、`{单键}`、`{"text":...}`、`{"input_ids":...}`；统一转成 `{"text": ...}`，用 `random.Random(seed).shuffle` 打乱后走标准 map。

### 3.5 token 级定长拼接 `concat_dataset_element`

**动机**：校准要求每条样本"正好 `seqlen` 长"。真实语料长度参差不齐——短文本直接被过滤掉（浪费），长文本被截断（丢信息）。拼接把多条短文本首尾相接，切成整齐的 `seqlen` 定长块，**最大化语料利用率**。

核心算法（`do_concat=True` 时对该数据集调用）：

```python
def concat_dataset_element(dataset):
    input_ids = [eg["input_ids"] for eg in dataset]
    concat_input_ids, attention_mask_list = [], []
    attention_mask = torch.ones([1, seqlen]).to(torch.int64)   # 拼接块无 padding，全 1
    buffer_input_id = torch.Tensor().to(torch.int64)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    os_cnt, have_bos, have_eos = 0, False, False

    for input_id in input_ids:
        # 1. 剥掉每条自带的首 bos / 尾 eos（记录数量 os_cnt，稍后统一补）
        if input_id[0] == bos:  input_id = input_id[1:]; os_cnt+=1; have_bos=True
        if input_id[-1] == eos: input_id = input_id[:-1]; os_cnt+=1; have_eos=True

        # 2. 若 buffer + 当前 + 预留的 bos/eos 会超过 seqlen → 切出恰好填满的一块
        if buffer.shape[-1] + input_id.shape[-1] + os_cnt > seqlen:
            idx_keep = seqlen - buffer.shape[-1] - os_cnt
            块 = [buffer, input_id[:idx_keep]]
            if have_bos: 块 = [bos] + 块          # 块首补回 bos
            if have_eos: 块 = 块 + [eos]          # 块尾补回 eos
            concat_input_ids.append(cat(块))       # 产出一条定长样本
            attention_mask_list.append(attention_mask)
            buffer = input_id[idx_keep:]           # 余下部分留到下一块
        else:
            buffer = cat([buffer, input_id])       # 不够 seqlen，继续累积

        # 3. 边界情形：正好凑满 seqlen
        if buffer.shape[-1] + os_cnt == seqlen:
            ...同样补 bos/eos，产出一块，buffer 清空
```

**要点**：
- `attention_mask` 恒为全 1（拼接样本不含 padding）。
- bos/eos 被"剥离—累计—补回"，保证每块结构合法（首 bos、尾 eos）。
- `os_cnt` 是为 bos/eos 预留的位置数，保证补回后长度仍恰为 `seqlen`。

> 注：`_get_dataset_impl` 里的 `concat_dataset_element` 用 `torch.ones([1, seqlen])`（2D）并保留张量；而 `opencode-instruct` 函数内的等价实现用 `torch.ones([seqlen])`（1D）并转 `.tolist()`。两者语义相同，只是落地形态不同。

### 3.6 过滤 `filter_func`

拼接（或不拼接）后，对每个数据集统一 `.filter(filter_func)`：

```python
def filter_func(example):
    if isinstance(example["input_ids"], list):
        example["input_ids"] = torch.tensor(example["input_ids"])
    if example["input_ids"].shape[-1] < seqlen:          # 太短→丢
        return False
    input_ids = example["input_ids"][:seqlen]            # 截断到 seqlen
    lst = input_ids.tolist()
    # 退化文本检测：末尾 token 重复次数 > seqlen//2（多为 padding / 复读）→丢
    if len(lst) > 1 and seqlen > 2 and lst.count(lst[-1]) > seqlen // 2:
        return False
    return True
```

三条规则：**长度不足丢弃、超长截断、末尾复读退化样本丢弃**。这保证进入优化的样本都是"足够长且信息量正常"的。

### 3.7 多数据集配额与混合

`_get_dataset_impl` 尾部负责把多个来源合并到目标 `nsamples`：

1. 对每个来源：`get_dataset(...)` → 可选 `concat` → `filter` → 若指定 `num=` 则 `select_dataset(range(num))` → 若是 `IterableDataset` 转成 `Dataset` → `set_format("torch")` → cast 特征（`input_ids`→int64，`attention_mask`→int8，省内存）。
2. **配额分配**（多来源时）：
   - 按数据集长度**升序排序**（先满足小数据集，避免大数据集挤占）。
   - 已用 `num=` 固定的来源直接取该值。
   - 其余来源按 `(nsamples - 已分配) // 剩余来源数` 均分，并 `min(target, 实际长度)`。
   ```python
   target_cnt = (nsamples - cnt) // (len(datasets) - len(data_lens)) if data_lens \
                else (nsamples - cnt) // (len(datasets) - i)
   target_cnt = min(target_cnt, lens[i])
   ```
3. `concatenate_datasets` 合并 → `shuffle(seed)` → 打印 `dataset_cnt_info` 日志 → 若总数仍 > `nsamples`，最后 `select_dataset(range(nsamples))` 截断。

辅助函数 `select_dataset` / `select` / `get_dataset_len`：优先用 HF 原生 `.select`，失败则手动迭代（兼容 IterableDataset），并处理 `len()` 不可用的情形。

### 3.8 子进程预处理与内存优化

`get_dataset` 是对 `_get_dataset_impl` 的一层包装，解决 **HuggingFace datasets `.map()/.filter()` 临时内存不易释放**的问题：

```python
def get_dataset(...):
    if envs.AR_DISABLE_DATASET_SUBPROCESS:               # 环境变量可关闭
        return _get_dataset_impl(...)
    # 在子进程里跑完整个预处理（fork；macOS 用 spawn 避免线程 fork 崩溃）
    ctx = multiprocessing.get_context("spawn" if darwin else "fork")
    p = ctx.Process(target=_get_dataset_impl, args=(...)); p.start(); p.join()
    # 子进程退出→OS 回收其全部临时内存；但磁盘缓存已被预热
    return _get_dataset_impl(...)   # 主进程重跑，命中 HF 缓存→几乎瞬时
```

**原理**：`.map()` 会生成大量中间 Arrow 表和临时内存，Python/HF 在主进程里难以完全归还给 OS。做法是先在子进程完整跑一遍（副作用是把结果写进 HF 的磁盘缓存），子进程退出让内核回收所有内存；主进程再跑时因为 `_make_map_fingerprint` 保证指纹一致，直接命中缓存，几乎零成本。子进程失败时回退到进程内模式。Windows / macOS 有各自的分支处理（Windows 无 fork；macOS fork+线程会 SIGSEGV，故用 spawn）。

### 3.9 DataLoader 与 collate

```python
def get_dataloader(tokenizer, seqlen, dataset_name, seed=42, bs=8, nsamples=512):
    dataset_final = get_dataset(tokenizer, seqlen, dataset_name, seed, nsamples)
    return DataLoader(dataset_final, batch_size=bs, shuffle=False, collate_fn=collate_batch)
```

`collate_batch`（`@torch.no_grad`）：
- 逐条截断到 `seqlen`；再次剔除"末尾 token 重复 > seqlen//2"的退化样本（`continue`）。
- `torch.vstack` 堆成 `{input_ids, attention_mask}` 批张量。
- 整批为空返回 `None`（消费端会 `if data is None: continue` 跳过）。
- **`shuffle=False`**：校准必须确定性、可复现（顺序固定）。

---

## 4. MLLM（多模态）数据线深入

文件：`auto_round/compressors/mllm/dataset.py`，注册表 `MLLM_DATASET`。

**`LlavaDataset`**（`torch.utils.data.Dataset` 子类，注册名含 `liuhaotian/llava*` 系列）：

- **数据来源**：本地 json 或从 HuggingFace `liuhaotian/LLaVA-Instruct-150K` 按 URL 下载（`conversation_58k` / `instruct_80k` / `instruct_150k`），图像来自 COCO train2017（`_COCO_DATA_URL`）。
- **`MAX_SUPPORT_SEQLEN = 512`**：超过会被 `get_mllm_dataloader` 警告并重置为 512。
- **`check()`（长度自适应）**：按"词数"（空格切分）筛选落在 `[min_word_len, max_word_len)` 的对话；若样本不足，**递归**降低 `min_word_len`（每次 -128）直到凑够 `nsamples`。这是一种在有限数据下满足 seqlen 需求的兜底策略。同时把 `<image>` token 规范化到对话开头。
- **`__getitem__`**：取对话 → `covert_conversations`（把 `human/gpt` 角色映射为 `user/assistant`，并做 `template.replace_tokens` 替换）→ `template._encode` → `template.processor.get_input(text, images=image_path, padding, truncation, max_length, truncation_strategy="text")`，产出图文混合张量。
- **可选 LRU 缓存**：`cache_size`（来自环境变量 `AR_MLLM_DATASET_CACHE_SIZE`）用 `OrderedDict` 做样本级缓存。

**`get_mllm_dataloader`**：
- 若 `dataset` 是注册的 MLLM 数据集或本地文件：实例化 `dataset_cls`，用 `template.processor.data_collator` 做 collate；某些模型（`check_mllm_only_support_bs1`）强制 `bs=1`；`set_seed(seed)`、`shuffle=False`（确定性）。
- **否则回退到纯文本**：调用 LLM 线的 `get_dataloader`。但如果 `quant_nontext_module=True`（要量化视觉/非文本模块），纯文本数据无法校准这些模块，直接 `logger.error + exit(-1)`。这是文本/多模态数据兼容性的关键约束。

---

## 5. Diffusion（文生图/音）数据线深入

文件：`auto_round/compressors/diffusion/dataset.py`，注册表 `DIFFUSION_DATASET`。

数据形态最简单：只提供**文本 prompt**，图像/音频由 diffusion pipeline 内部按 prompt 生成，校准发生在 pipeline 的 UNet/Transformer 前向上。

- **`Text2ImgDataset`**（注册名 `local`）：读 TSV，要求含 `id` 和 `caption` 列，`__getitem__` 返回 `(caption_id, caption_text)`。`nsamples>0` 时截断。
- **`AudioCapsDataset`**（注册名 `audiocaps`）：读 CSV，`audiocap_id`/`id` 作为 id，`caption` 作为 prompt，跳过空 caption。
- **`get_diffusion_dataloader(dataset="coco2014", bs=1, seed=42, nsamples=128)`**：
  - `coco2014` → 从 mlcommons GitHub 下载 `captions_source.tsv`，构造 `Text2ImgDataset`。
  - `audiocaps` → `download_audiocaps_csv()`。
  - 本地 `.csv` → `AudioCapsDataset`；本地其它（tsv）→ `Text2ImgDataset`。
  - `set_seed(seed)`；**`shuffle=True`**（与 LLM/MLLM 不同——扩散校准对顺序不敏感，且 prompt 多样性有益）。

---

## 6. 消费端：Calibrator 如何使用校准数据

文件：`auto_round/calibration/llm.py`（`LLMCalibrator`）。核心方法链：

```
calibration()  →  cache_inter_data()  →  calib()
```

### 6.1 `calibration()`（原 `try_cache_inter_data_gpucpu`）

- 决定在 **GPU 还是 CPU** 上缓存中间数据：`low_gpu_mem_usage` 或"只校准 embedding 层"时走 CPU。
- 处理 **多卡 dispatch**：用 accelerate 的 `infer_auto_device_map` / `dispatch_model` / `get_balanced_memory` 把模型铺到多卡；显存不足时 offload 到 CPU 甚至磁盘（`offload_dir`）。
- **OOM 兜底**：GPU 缓存 OOM 时 `mv_module_from_gpu` 搬回 CPU 重试，并警告部分层可能回退到 `rtn` 模式（影响精度）。

### 6.2 `cache_inter_data()`

- 用 hook 替换 block 的 `forward`（`replace_forward_with_hooks`），记录需要缓存输入的层集合（`to_cached_layers`、`last_cache_name`）。
- 调 `self.calib(nsamples, bs)` 驱动前向，把每个 block 的输入张量截获缓存。
- `finally` 中恢复 forward、清理临时属性、**`del self.dataloader`** 释放已 tokenized 的样本张量（省内存）。

### 6.3 `calib()`（真正消费 DataLoader）

```python
def calib(self, nsamples, bs):
    from auto_round.calib_dataset import get_dataloader
    if isinstance(self.dataset, str):
        self.dataloader = get_dataloader(self.tokenizer, self.seqlen,
                                         self.dataset.replace(" ", ""), self.seed, bs, nsamples)
    else:
        self.dataloader = self.dataset          # 用户直接传入的 dataloader
    ...
    for data in self.dataloader:
        # 兼容多种输入形态：Tensor / str / tuple / list / dict / BatchEncoding
        ...
        if input_ids.shape[-1] < self.seqlen: continue
        # 缓存 token，并构造 next-token 监督标签
        ids_to_cache = input_ids.clone()
        if tokenizer.pad_token_id is not None:
            ids_to_cache[ids_to_cache == pad_token_id] = -100   # pad 位置忽略
        else:
            # 启发式：末尾重复 token 视为 padding，置 -100
            ...
        ids_to_cache[:, -1] = -100                              # 最后一个 token 无监督目标
        self.inputs["input_ids"].extend(torch.split(ids_to_cache.cpu(), 1, dim=0))
        # 构造 attention_mask，强制最后一位为 0（规避"全1 mask 被当成 None"的模型 bug）
        ...
```

**要点**：
- `-100` 是 PyTorch 的 `ignore_index`：pad 位置和每条最后一个 token 都不参与损失。
- 无 `pad_token_id` 时用"末尾重复 token = padding"的启发式。
- 强制 `attention_mask[:, -1] = 0`：某些模型把全 1 mask 内部替换为 `None`，导致 block 输入拼接时 shape 不匹配，这里做 workaround。
- 缓存的 `self.inputs["input_ids"]` 供后续 SignRound block 优化使用。

MLLM/Diffusion 的 Calibrator（`mllm.py` / `diffusion.py`）结构类似，但分别调用 `get_mllm_dataloader` / `get_diffusion_dataloader`，并处理图文/prompt 形态。

---

## 7. 哪些场景会用到校准数据集

是否加载校准数据由 `Compressor._check_need_calib()` → `_needs_calibration_data()` 决定（`base.py`）：

```python
def _needs_calibration_data(self) -> bool:
    # 1. 任一算法配置声明 need_calib（默认 True，如 SignRound / imatrix / opt-rtn）
    if any(getattr(config, "need_calib", True) for config in self._alg_configs):
        return True
    # 2. AutoScheme：需要真实数据做 delta-loss 方案选择
    if isinstance(self.scheme, AutoScheme):
        return True
    # 3. 静态激活量化：act_bits<=8 且 act_dynamic=False（NV FP 等）需要激活校准
    if is_act_quantize and check_need_act_calibration(act_dynamic, act_data_type, act_bits, ...):
        return True
    return False   # 纯 zero-shot RTN：不需要数据
```

**需要校准数据的场景**：

1. **SignRound / 迭代优化（`iters > 0`）**：AutoRound 的招牌算法，逐块重构损失优化 rounding，必须用数据驱动前向。
2. **imatrix / opt-RTN**：用重要性矩阵加权的 RTN，需要数据统计激活重要性。
3. **静态激活量化（static activation quant）**：`act_dynamic=False`（如 NVFP/MXFP 静态、static KV cache、static attention）需要用数据统计激活 scale。
4. **AutoScheme**：在多种量化方案间自动选择时，用数据算每种方案的 delta-loss。
5. **block 外的量化层**（`has_qlayer_outside_block and need_calib`，见 base.py:1541）。

**不需要校准数据的场景**：

- **纯 zero-shot RTN（Round-To-Nearest）**：`iters=0` 且动态激活量化，直接按最近邻取整，无需任何数据。此时 `need_calib=False`，不会构建 DataLoader。

此外，只有当 `need_calib and not dataset_was_explicitly_set and calibrator_kind == "llm"` 时才触发 PR #2107 的"代码模型自动选数据集"逻辑（见 §8）。MLLM 用 `_get_calibrator_kind()=="mllm"`、Diffusion=="diffusion"，各自有独立默认集（llava / coco2014），不参与代码模型自动选择。

---

## 8. PR #2107 深度分析

**标题**：Automatically select calibration datasets for code models（为代码模型自动选择校准数据集）
**作者**：changwangss｜**合并者**：XuehaoSun｜**状态**：已合并（2026-08-04）
**规模**：11 文件，+220 / -12，7 commits｜**关联 issue**：#1986

### 8.1 动机

用户量化一个**代码专用模型**（如 Qwen3-Coder、CodeLlama、StarCoder）时，若不显式指定 `--dataset`，此前一律用通用英文的 `NeelNanda/pile-10k` 校准。但代码模型的激活分布与自然语言差异大，**用代码语料校准更贴合真实推理分布，量化精度更好**。本 PR 让 AutoRound 在检测到代码模型且用户未指定数据集时，**自动选用代码校准集**。

### 8.2 核心改动一：把默认值从字符串改为 `None`（区分"未指定"与"显式指定"）

这是整个 PR 的关键前提。此前默认参数是 `dataset="NeelNanda/pile-10k"`，无法区分"用户没填"和"用户填了 pile-10k"。PR 把所有入口的默认值统一改为 `None`：

- `auto_round/autoround.py`、`compressors/entry.py`（两处）、`compressors/orchestrator.py`：`dataset: Optional[...] = None`
- `cli/parser.py`：`--dataset default=None`
- `compressors/base.py`：
  ```python
  dataset_was_explicitly_set = dataset is not None
  self.dataset = dataset if dataset_was_explicitly_set else "NeelNanda/pile-10k"
  ```
  用 `dataset_was_explicitly_set` 这个布尔量记住"用户到底填没填"。
- `compressors/diffusion_mixin.py`：把判断从 `== "NeelNanda/pile-10k"` 改为 `in (None, "NeelNanda/pile-10k")`，保证 diffusion 默认仍是 `coco2014`。

**测试**（`test_init.py`）：断言四个入口 `dataset` 默认都是 `None`；CLI 不填时为 `None`，显式填 `pile-10k` 时保留原值。

### 8.3 核心改动二：`is_code_model()` 模型检测（`utils/model.py`，+80 行）

**零额外 Hub 请求**，仅从模型名和已有 config 元数据判断：

```python
_CODE_MODEL_TOKENS   = {"code","coder","coding","programming","swe","devstral"}
_CODE_MODEL_FAMILIES = {"codellama","codegemma","codestral","deepseekcoder",
                        "granitecode","magicoder","opencoder","qwencoder",
                        "santacoder","stablecode","starcoder","wizardcoder"}
_CODE_MODEL_TASKS    = {"code-generation","software-engineering","text-to-code"}
```

检测流程 `_get_code_model_match`：
1. **名称匹配** `_match_code_model_name`：
   - 只取路径最后一段（`re.split(r"[/\\]", ...)[-1]`）——**避免父目录名为 `code` 的误报**（如 `/srv/code/checkpoints/Llama-3` 不算代码模型）。
   - 驼峰拆词（`Qwen3Coder`→`Qwen3 Coder`），再做**整词匹配**——`coder` 命中，但 `encoder`/`decoder`/`codec`/`notstarcoder` 因整词不等而**不误报**。
   - 家族匹配允许 `family` 或 `family\d+`（如 `starcoder2`）。
2. **config 匹配**：依次检查 `config._name_or_path`、`config.model_type`、`config.architectures`（如 `DeepseekCoderForCausalLM`）。
3. **任务匹配**：`finetuning_task` / `task` / `pipeline_tag` / `task_specific_params`，命中 `_CODE_MODEL_TASKS`。

命中即 `logger.info("Detected a code-specialized model ...")` 并返回 `True`。

**测试**（`test_utils.py`）覆盖正/负例：
- 正例：`Qwen/Qwen3-Coder-30B`、`/models/CodeLlama-7b`、`bigcode/starcoder2-15b`、`architectures=["DeepseekCoderForCausalLM"]`、`finetuning_task="code-generation"`、`task_specific_params={"text-to-code":{}}`。
- 负例（防误报）：`Qwen/Qwen3-4B`、`org/encoder-decoder`、`org/audio-codec`、`/srv/code/checkpoints/Llama-3`、`org/notstarcoder-model`、`_name_or_path="/srv/code/checkpoints/Llama-3"`、`architectures=["SomeEncoderDecoderModel"]`。

### 8.4 核心改动三：`get_code_calibration_dataset()`（`calib_dataset.py`，+28 行）

按 datasets 版本生成**精确尺寸**的代码校准 DSL 字符串：

```python
_GITHUB_CODE_CLEAN_MAX_DATASETS_VERSION = Version("3.6.0")

def get_code_calibration_dataset(nsamples, datasets_version=None):
    sources = [("opencode-instruct:concat=true", 50), ("github-code-clean", 50)]
    if parsed_version > Version("3.6.0"):
        # datasets>3.6.0 不再支持 script-based github-code-clean → 仅用 OpenCodeInstruct
        sources = [s for s in sources if s[0] != "github-code-clean"]
        logger.warning_once("datasets %s does not support ... using OpenCodeInstruct only", ...)
    # 按权重把 nsamples 分配到各来源，用"最大余数法"处理取整余数
    weights = [w for _, w in sources]
    raw_counts = [nsamples * w / sum(weights) for w in weights]
    counts = [int(c) for c in raw_counts]
    remainder = nsamples - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: (-(raw_counts[i]-counts[i]), i))
    for i in order[:remainder]: counts[i] += 1
    return ",".join(f"{name}:num={c}" for (name,_),c in zip(sources,counts) if c)
```

- **datasets ≤ 3.6.0**：`OpenCodeInstruct 50% + GitHub-Code-Clean 50%`，例如 `nsamples=128` → `"opencode-instruct:concat=true:num=64,github-code-clean:num=64"`。
- **datasets > 3.6.0**：仅 OpenCodeInstruct → `"opencode-instruct:concat=true:num=128"`（因为 github-code-clean 是 script 数据集，新版 datasets 已停止支持）。
- **`concat=true`**：让 OpenCodeInstruct 的短样本拼成定长序列（复用 §3.5 拼接逻辑）。
- **最大余数法**：保证各来源 `num` 之和精确等于 `nsamples`，且 `num=0` 的来源被过滤掉（`if c`）。

**测试**（`test_calib_dataset.py`）：
- `get_code_calibration_dataset(128, "3.6.0") == "opencode-instruct:concat=true:num=64,github-code-clean:num=64"`
- `get_code_calibration_dataset(128, "5.0.0") == "opencode-instruct:concat=true:num=128"`
- `get_code_calibration_dataset(1, "3.6.0") == "opencode-instruct:concat=true:num=1"`（小样本时 github-code 分到 0，被省略）

### 8.5 核心改动四：接入点（`compressors/base.py`，+20 行）

在 `__init__` 末尾，`need_calib` 确定后：

```python
self.need_calib = self._check_need_calib()
calibrator_kind = self._get_calibrator_kind()
# 只对"需要校准 + 用户未指定 + 纯文本 LLM"生效
if self.need_calib and not dataset_was_explicitly_set and calibrator_kind == "llm":
    from auto_round.calib_dataset import get_code_calibration_dataset
    from auto_round.utils.model import is_code_model
    detection_config = model_config or getattr(self.model_context.model, "config", None)
    if is_code_model(model, detection_config):
        self.dataset = get_code_calibration_dataset(self.calibration_context.nsamples)
        logger.info("Automatically selected code calibration dataset: %s", self.dataset)
    else:
        logger.info("No explicit code-specialization signal ...; using default %s", self.dataset)
    self.calibration_context.dataset = self.dataset
```

**三重触发条件**（缺一不可）：
1. `need_calib` —— 该量化任务真的需要数据（排除 zero-shot RTN，见 §7）。
2. `not dataset_was_explicitly_set` —— 用户没显式指定（**尊重用户选择**）。
3. `calibrator_kind == "llm"` —— 仅纯文本 LLM（MLLM/Diffusion 有各自默认集，不干预）。

### 8.6 行为矩阵（改动前后对比）

| 场景 | 改动前 | 改动后 |
|------|--------|--------|
| 普通 LLM，不指定 dataset | pile-10k | pile-10k（不变） |
| **代码模型，不指定 dataset** | pile-10k | **自动选代码校准集** |
| 任意模型，显式 `--dataset pile-10k` | pile-10k | pile-10k（尊重用户） |
| 任意模型，显式 `--dataset xxx` | xxx | xxx（尊重用户） |
| Diffusion，不指定 dataset | coco2014 | coco2014（不变） |
| zero-shot RTN 代码模型 | pile-10k（但不加载） | **不触发**（need_calib=False） |

### 8.7 设计评价

**优点**：
- **对用户完全无侵入**：显式指定的数据集绝不被覆盖；`None` 哨兵值干净地区分了两种意图。
- **零额外网络请求**：`is_code_model` 只读本地名字/config，不访问 Hub，快且离线可用。
- **防误报考究**：整词匹配 + 只取路径末段 + 排除 encoder/decoder/codec/父目录 code，负例测试充分。
- **版本自适应**：对 datasets>3.6.0 优雅降级为 OpenCodeInstruct-only，避免 script 数据集报错。
- **精确尺寸**：最大余数法保证 `num` 之和恰为 `nsamples`，小样本自动省略空来源。
- **测试覆盖全面**：默认值哨兵、CLI、检测正负例、DSL 生成、端到端自动选择 + 显式覆盖。

**潜在注意点**：
- 家族/token 词表是**硬编码枚举**，新代码模型（未含相应关键词且 config 无信号）不会被识别——但这只是"回退到 pile-10k"，不会出错，属可接受的保守设计。
- 依赖 `github-code-clean` 需 `datasets<=3.6.0`；新版环境实际只用 OpenCodeInstruct 单源，代码语料多样性略降（已通过日志告知）。
- 端到端测试用 `symlink` 到 tiny OPT 模型并命名 `Qwen3-Coder-smoke` 来伪造代码模型名——说明检测**完全基于名字/config**，与权重无关。

---

## 9. 关键设计总结与工程要点

1. **插件式注册 + 统一签名**：三条数据线都用 `register_*` 装饰器把加载器注册进全局表，新增数据集零侵入。
2. **DSL 字符串**：一行 `name:key=value,name2:...` 即可混配多来源、按 `num=` 定额、控制 `concat` / chat 模板 / system prompt，表达力强。
3. **token 级定长拼接**：`concat_dataset_element` 把碎片语料拼成 `seqlen` 定长块，最大化利用率，并正确处理 bos/eos。
4. **质量过滤**：丢弃过短样本、末尾复读退化样本；`collate` 里再兜一层。
5. **确定性**：LLM/MLLM 用 `shuffle=False` + 固定 seed 保证校准可复现；Diffusion 例外用 `shuffle=True`。
6. **子进程预处理**：绕开 HF datasets 内存泄漏，靠磁盘缓存 + 稳定指纹复用，主进程近乎零成本重载。
7. **标签构造**：pad 与每条末位 token 置 `-100`（ignore_index），配合 block 重构损失做 SignRound 优化；强制 mask 末位为 0 规避模型兼容性 bug。
8. **按需加载**：只有 `need_calib=True`（SignRound/imatrix/静态激活/AutoScheme 等）才构建 DataLoader；纯 zero-shot RTN 不用数据。
9. **PR #2107**：用 `None` 哨兵区分"未指定/显式指定"，据此对**代码模型自动切换代码校准集**，在"尊重用户 + 零额外请求 + 防误报 + 版本自适应"之间取得了很好的平衡。

---

### 附：关键文件索引

| 功能 | 文件 | 关键符号 |
|------|------|----------|
| LLM 数据集/加载 | `auto_round/calib_dataset.py` | `CALIB_DATASETS`、`register_dataset`、`_get_dataset_impl`、`get_dataset`、`get_dataloader`、`concat_dataset_element`、`get_code_calibration_dataset` |
| MLLM 数据集 | `auto_round/compressors/mllm/dataset.py` | `MLLM_DATASET`、`LlavaDataset`、`get_mllm_dataloader` |
| Diffusion 数据集 | `auto_round/compressors/diffusion/dataset.py` | `DIFFUSION_DATASET`、`Text2ImgDataset`、`AudioCapsDataset`、`get_diffusion_dataloader` |
| LLM 消费端 | `auto_round/calibration/llm.py` | `LLMCalibrator.calibration/cache_inter_data/calib` |
| 是否需要校准 | `auto_round/compressors/base.py` | `_check_need_calib`、`_needs_calibration_data`、`_get_calibration_dataset` |
| 代码模型检测 | `auto_round/utils/model.py` | `is_code_model`、`_get_code_model_match`、`_match_code_model_name` |
| 校准器选择 | `auto_round/compressors/orchestrator.py` 等 | `_get_calibrator_kind` |
