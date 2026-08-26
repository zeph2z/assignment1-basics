# A1 知识点总结（训练循环 · 生成 · 显存核算 · 架构概念）

> 基于完成 CS336 Assignment 1 全过程的对话整理。用途：A2（Triton FlashAttention-2）之前的复习速查 + 面试可复述结论。
> 目标读者：有 C++/体系结构背景、Python 较新的学习者。

---

## 1. 训练循环（已跑通，loss 9.22 → 4.8）

**一步训练的 7 件事**：
取数据 `run_get_batch` → 前向 `logits = model(x)` → 展平算交叉熵 → `loss.backward()` → 梯度裁剪 → LR schedule 写回 `param_groups[0]["lr"]` → `opt.step()`。每步开头 `zero_grad()`。

关键概念：
- **batch**：一次打包喂给 GPU 的样本数。是**并行度**概念，不是正确性要求。batch=1 推理和 batch=8 训练数学上完全一致。
- **为什么训练并行、推理串行**：训练时标签现成（teacher forcing），所有位置可同时算；生成是自回归，token 5 依赖 token 4——因果性把生成变成串行依赖链。
- **交叉熵展平**：logits `(B,S,V)` → `reshape(-1, V)`，y `(B,S)` → `reshape(-1)`，行优先对齐，`b*S+s` 下标一一对应。平均对分组不敏感，所以合并算合法。
- **初始 loss sanity check**：随机初始化 ≈ `ln(vocab_size)`（vocab=10000 时 ≈ 9.21）。不是这个值 = 前向/损失接错了。
- **LR schedule**：算出的 lr 写回 `opt.param_groups[0]["lr"]`（优化器每步现读 group），warmup 从 0 线性升，it=0 时 lr=0 是标准行为。
- **`zero_grad`**：backward 是累加到 `.grad` 的，不清会跨步累积。`zero_grad(set_to_none=True)` 更省内存带宽。

## 2. AdamW（已实现并过测试）

**算法**：每参数维护 m（一阶矩）、v（二阶矩）、t（计步器）。
- `m ← β1·m + (1-β1)·g`，`v ← β2·v + (1-β2)·g²`
- bias correction：折进 `lr_t = lr·√(1-β2ᵗ)/(1-β1ᵗ)`（t 是**每参数**的计步器，从 1 起，存 `state[param]` 里，否则 checkpoint 恢复后 bias correction 会错）
- 更新：先 `θ ← θ − lr·wd·θ`（衰减），再 `θ ← θ − lr_t·m/√(v+ε)`（动量）
- **解耦 weight decay vs L2 正则**：L2 把 `wd·w` 加进梯度、污染 m/v 统计，且被 `√v` 调制（大梯度坐标衰减被压小）；AdamW 直接作用在权重上，每坐标衰减恒为 `lr·wd·w`。SGD 里两者等价（无自适应缩放）。

**PyTorch 结构**：
- `super().__init__(params, defaults)` → 得到 `self.defaults`、`self.param_groups`（list[dict]，每项含 `'params'` + 超参）、`self.state`（以**参数对象**为 key 的 dict）。
- 基类只给容器：m/v/t 存什么是子类的事；`state_dict()` 会序列化 `self.state` 和 `param_groups`（所以 t 必须放 state 里）。
- 更新必须**原地**：`param.data` 或 `param.add_()`，不能 `param = ...`（模型持有的引用没变）。
- m/v 用 `torch.zeros_like(param)`（自动继承形状/dtype/device），惰性初始化（`"t" not in self.state[param]` 判断）。

**显存（面试一页纸）**：fp32 朴素训练，非激活部分 = **16N**（参数 4N + 梯度 4N + 优化器状态 8N）。SGD 只要 8N，AdamW 多出的 8N 是 8-bit optimizer（DeepSpeed）要砍的。推理只要 4N。**优化器状态和 batch 无关**，激活才随 batch 线性。

## 3. FLOPs 核算（transformer_accounting）

**规则**：`A(m×n) @ B(n×p)` = `2mnp` FLOPs（m·p 个输出元素 × 每个是长度 n 的点积 = 2n）。

**forward 每步 FLOPs**（B=batch, L=seq, D=d_model, F=d_ff, V=vocab, n=层数）：
```
2BLDV + n·(8BLD² + 4BL²D + 6BLDF)
```
- `8BLD²`：Q/K/V 三次投影 + output 投影（4 × 2BLD²）
- `4BL²D`：QKᵀ + scores×V（各 2BL²D，L² 项——context 拉长会暴涨）
- `6BLDF`：SwiGLU w1/w3/w2
- `2BLDV`：lm_head

**结论**：代入 F=8/3·D，FFN 项 = 16BLD² > 投影 8BLD² > attention 4BL²D（L=1024 时）。**FFN 最耗 FLOPs**。

**不计入的部分**（作业约定只数 matmul）：RMSNorm/softmax/SiLU/RoPE/门控乘法全是逐元素 O(BLD) 或 O(BLF)，比 matmul 小一个 D 因子，忽略合理。RoPE 和门控乘法是**内存受限**的逐元素算子（算力 ≈ 0，成本全在读写）——融合 kernel 要省的就是这类中间张量的读写。

**backward = 2× forward**：每个线性层反向要算两份同规模 matmul——`∂L/∂W = Xᵀ·G`（更新权重）和 `∂L/∂X = G·Wᵀ`（传给上一层）。所以训练一步 ≈ **3 × forward**。

**GPT-2 XL 关键数字**：参数 ≈ 1.64B（2·vocab·d_model + d_model + 48 层·(2·d_model + 4·d_model² + 3·d_model·d_ff)）；fp32 加载 6.6GB；真实 GPT-2 报 1.5B 是因为 **tied embeddings**（lm_head 和 embedding 共享，省 vocab·d_model 个参数）；H100 50% MFU 训 400K 步 × batch 1024 ≈ **5000 小时 ≈ 200 天**（单卡）。

## 4. 激活（显存核算的第四类）

- **激活 = 前向过程中每层的中间结果张量**（ln1 输出、Q/K/V、注意力分数……）。参数是永久账本，激活是**每步一换的临时账本**。
- **为什么要保存**：backward 的链式法则需要前向的中间值——`∂L/∂W = XᵀG` 需要 X；softmax 反向需要 softmax 输出本身。PyTorch autograd 自动扣留，backward 用完释放。
- 替代方案：**gradient checkpointing**（重算换显存）、**FlashAttention**（QKᵀ 的 L×L 矩阵不落内存，边算边用）——A2 的主题。
- 激活和 **batch_size 线性**（每个中间张量都有 B 维）。激活里最大头通常是 **logits `BLV`**（大词表时）。
- GPT-2 XL 峰值内存 ≈ `16N(≈26GB) + 激活(≈十几 GB/ batch)` → 80GB 下最大 batch 只有 2~4（真实 GPT-2 XL 必须 activation checkpointing）。

## 5. 文本生成（已跑通）

**循环**（`no_grad` 里）：forward 取最后位置 logits → 温度缩放（`logits/temp`，τ=0 直接 argmax 贪心）→ top-p（排序 → cumsum → 累计 ≥ p 的最小集合，其余置 -inf → **scatter 还原顺序**）→ softmax → `torch.multinomial(probs, 1)` 采样 → cat 拼接 → 超长截断到 context_length。

要点：
- **贪心必须确定性**（τ=0 同 prompt 两次输出一致），随机路（τ=1, p=1.0）两次不同——两个 sanity check。
- `next_id.unsqueeze(0)`：cat 要求维度数一致，`(1,)` → `(1,1)` 对齐 `(1, seq)`。
- 输入恒为 `(1, seq)`——batch 维是模型契约的一部分，batch=1 只是退化情形。
- 欠训练模型会**复读死循环**（一直输出同一个 token）——不是 bug，是分布问题，温度/top-p 能缓解。
- 朴素生成每步重算整个上下文 = **O(seq²)、decode 内存受限**——KV cache 存在的理由（A2/Lecture 10 的主题）。
- 推理的 batch = **并发请求数**（不同序列并行，序列内串行）——continuous batching 的概念。

## 6. 架构概念

- **decoder-only**：只有因果注意力的那半个 Transformer。因果 mask（`torch.triu(..., diagonal=1)`）就是 decoder 的定义性特征——**你写的 A1 模型就是 decoder-only**，今天 99% 的 LLM 都是它的放大版。
- 对照：encoder-only（BERT，双向注意力，读输入做表示）vs decoder-only（GPT，因果，逐 token 生成）vs encoder-decoder（原始 Transformer/T5，双向读 + 因果生成）。
- **MHA 是内容寻址检索**：query="要找什么"，key="标签"，value="内容"，softmax(QKᵀ)V = 按相似度加权搬信息。学到的模式可解释（前 token 头、induction head 等）。
- **FFN 是知识存储 + 逐位置处理**：attention 让 token 交流，FFN 让 token"独立思考"；事实知识多存在 FFN 里。
- **端到端/灰箱**：只有最终 loss 监督，中间层功能是梯度"挤"出来的；但组件机制清楚（灰箱不是黑箱）。中间层可以拿出来用（当 embedding、early exit、probing），但不是可插拔模块。
- **后训练（SFT/RLHF/DPO）**：用**同一个 transformer 的同一批参数**继续训练。SFT = 你 A1 的交叉熵训练换数据（指令-回答对）；RLHF 换个目标函数。**推理代码对 base/后训练模型一视同仁**——生成机制不变，变的是续写分布。

## 7. Python 陷阱速查（C++ 背景版）

- `torch.Tensor(...)`（大写，类构造器）= float32；`torch.tensor(...)`（小写）= 按数据推断 dtype。**索引张量必须 long/int**，uint16 也不行。
- **device**：组件构造时传 `device=device`，参数才会建在目标设备；`torch.arange` 等默认 CPU。
- **生成器一次性**：`yield` 的函数被遍历一次就耗尽，不能再"遍历第二遍"。
- **默认参数在定义时求值**：`def f(x=global_var)` 绑定的是定义那一刻的值，之后改全局不影响。
- **`is` vs `==`**：整数比较用 `==`。
- **`**dict` 展开要求 key 与签名精确匹配**：配置 dict 是超集时不能直接 `f(**cfg)`，要显式挑参数或单独建子集 dict。
- `nn.ModuleList([...])` 收列表；`[X()] * 4` 是同一个实例复制，`[X() for _ in range(4)]` 才是 4 个独立实例。
- `a = b = X()` 是两个名字同一个对象。
- REPL 没有 IDE 补全；API 事实靠查不靠背。

## 8. A2 预备

- **Triton 需要 NVIDIA GPU**（不支持 MPS）——本机跑不了，准备 Colab/租卡。
- 前置理解：算子"算力受限 vs 内存受限"二分；逐元素操作（norm/RoPE/门控）是内存受限；FlashAttention 省的是 `(B,h,L,L)` 中间矩阵的读写；算术强度 = 每字节访存对应的 FLOPs。
- 学习顺序建议：讲义第 6 讲 → Triton 官方 tutorial → A2 作业。