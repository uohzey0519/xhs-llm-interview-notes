# 大模型面经八股整理

生成日期：2026-05-24

## 概览

- 收藏页标称笔记数：353；本轮 DOM 盘点到可读标题/链接：315。
- 标题初筛并打开抽取：1508 条；后处理判定大模型相关：1199 条。
- 发现页新增抽取：1193 条；其中大模型相关：995 条。
- 正文抽取状态：text-ok 1325 条，title-only 151 条。
- 归一化问题簇：42 个。频率按“不同来源笔记数”统计。
- 说明：不复制小红书原文全文；答案为基于来源要点和通用面试知识的整理。抽取不足的来源在 `sources.csv` 中保留状态。

## 高频题 Top 20

| Rank | Question | Category | Frequency |
|---:|---|---|---:|
| 1 | 大模型算法/Agent 岗公司面经中最常被深挖哪些方向？ | 公司面经 | 728 |
| 2 | SFT、RLHF、DPO、PPO、GRPO、DAPO、GSPO 等后训练方法如何区分？ | 后训练与强化学习 | 301 |
| 3 | Agent 的上下文、短期记忆、长期记忆和上下文压缩怎么做？ | RAG、Agent 与工具调用 | 259 |
| 4 | Transformer 的基本架构和 Self-Attention 计算流程是什么？ | Transformer 基础 | 228 |
| 5 | 大模型训练、RAG 和上线效果如何评估？PPL 有什么意义？ | 评估与对齐 | 191 |
| 6 | LoRA、QLoRA 和全量微调有什么区别？LoRA rank 怎么选？ | 训练与优化 | 174 |
| 7 | GRPO 的核心思想、loss、优势估计和 PPO 区别是什么？ | 后训练与强化学习 | 162 |
| 8 | Function Call、MCP、Tool、Skill 和 Rules 的区别是什么？ | RAG、Agent 与工具调用 | 158 |
| 9 | 多模态大模型、VLM/VLA 和具身 Agent 面试常问什么？ | 多模态与具身智能 | 158 |
| 10 | DeepSeek R1/V3/V4、Qwen、Kimi 等技术报告常问哪些架构创新？ | 公司面经 | 156 |
| 11 | RAG 和微调的区别、优缺点和选择标准是什么？ | RAG、Agent 与工具调用 | 125 |
| 12 | 大模型算法岗常见手撕代码题有哪些？ | 手撕代码 | 118 |
| 13 | MoE 的原理、Dense 对比、路由和负载均衡难点是什么？ | MoE 与稀疏模型 | 116 |
| 14 | MHA、MQA、GQA 为什么能省显存？为什么只有 KV Cache 没有 Q Cache？ | Attention 与 KV Cache | 112 |
| 15 | Claude Code 为什么偏向 grep/ripgrep，而不是传统向量 RAG？ | RAG、Agent 与工具调用 | 109 |
| 16 | Agent Eval、可观测体系和自动评测环境如何设计？ | 评估与对齐 | 108 |
| 17 | BF16/FP8/INT8 量化和反量化的核心流程是什么？ | 推理服务与性能 | 102 |
| 18 | 为什么常说 SFT 更像正向 KL，RL/偏好优化更像反向 KL？ | 后训练与强化学习 | 91 |
| 19 | vLLM/SGLang 的加速方法、PagedAttention 和资源调度怎么理解？ | 推理服务与性能 | 81 |
| 20 | RoPE 的直觉是什么，为什么有外推能力？Qwen/LLaMA 的 RoPE 实现有何差异？ | 位置编码 | 79 |

## Transformer 基础

### Transformer 的基本架构和 Self-Attention 计算流程是什么？

- 频率：228
- 来源：xhs_6a0fc50a, xhs_6a0d5f2d, xhs_6a0efd3c, xhs_6a089599, xhs_6a0733b1, xhs_6a06c1b8, xhs_6a02019f, xhs_6a0018b9, xhs_69f9bacb, xhs_69f770dc, xhs_69e5baea, xhs_69d4f398, xhs_69de2e4c, xhs_69ba051d, xhs_69d8c35c, +213
- 难度：基础

**面试官常问：**

1. Transformer 的 Q/K/V 分别是什么？Self-Attention 的完整计算流程是什么？
2. 为什么 attention score 要除以 sqrt(d_head)？不除会发生什么？
3. Encoder-only、Decoder-only、Encoder-Decoder 的区别是什么？LLM 为什么多用 decoder-only？
4. Self-Attention 的时间复杂度和显存瓶颈在哪里？长上下文为什么难？

面试回答要先把 Transformer 讲成一个可计算的 block，而不是泛泛地说“注意力机制”。给定输入 `X in R^{B x T x d_model}`，每层通常包含 Multi-Head Self-Attention、FFN、残差连接和归一化。Decoder-only LLM 还会加 causal mask，保证第 `t` 个 token 只能看见 `<= t` 的历史。

核心公式：

$$
\begin{aligned}
Q &= X W_Q, K = X W_K, V = X W_V \\
Attention(Q, K, V) &= \operatorname{softmax}((Q K^T + mask) / \sqrt{d_{head}}) V \\
MHA(X) &= \operatorname{Concat}(head_1, ..., head_h) W_O \\
FFN(X) &= W_down act(W_up X)
\end{aligned}
$$

除以 `sqrt(d_head)` 是为了控制点积方差。若 `q_i,k_i` 方差约为 1，则 `q·k` 方差约为 `d_head`，不缩放会让 softmax 很容易饱和，梯度变小。复杂度上，自注意力的主要瓶颈是 `O(B * h * T^2 * d_head)`，也就是 `O(B * T^2 * d_model)`；FFN 通常是 `O(B * T * d_model * d_ff)`。短序列时 FFN 很重，长序列时 attention 的二次复杂度和 KV cache 更显著。

一个最小可手撕版本：

```python
import math
import torch
import torch.nn.functional as F

def causal_self_attention(x, wq, wk, wv, wo, num_heads):
    # x: [B, T, D]
    b, t, d = x.shape
    dh = d // num_heads
    q = x @ wq
    k = x @ wk
    v = x @ wv

    def split_heads(z):
        return z.view(b, t, num_heads, dh).transpose(1, 2)  # [B, H, T, Dh]

    q, k, v = map(split_heads, (q, k, v))
    score = q @ k.transpose(-1, -2) / math.sqrt(dh)         # [B, H, T, T]
    mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
    score = score.masked_fill(mask, float("-inf"))
    attn = F.softmax(score, dim=-1)
    out = attn @ v                                          # [B, H, T, Dh]
    out = out.transpose(1, 2).contiguous().view(b, t, d)
    return out @ wo
```

常见追问：Encoder-only、Decoder-only、Encoder-Decoder 区别。BERT 是双向 encoder，适合理解；GPT/LLaMA/Qwen 是 causal decoder，适合自回归生成；T5 这类 encoder-decoder 适合输入输出都较结构化的 seq2seq。现代 LLM 常用 Pre-Norm、RMSNorm、RoPE、SwiGLU、GQA/MQA/MLA、FlashAttention 等工程改造，但主干仍是“注意力混合 token 信息 + FFN 做逐 token 非线性变换”。

### 高频追问补充

#### 1. 20 多种 Transformer 改进里，为什么最后留下来的通常是 Norm、RoPE、GQA/MQA、SwiGLU、FlashAttention 这几类？

答：这题不要逐个背论文名，而要解释“为什么某些改造能活下来”。Transformer 主干长期稳定，是因为 attention 负责 token 间信息混合，FFN 负责逐 token 非线性变换；能留下来的改进通常是不破坏这条主干，同时明确改善一个工程瓶颈。

可以按四条线回答：

1. 稳定训练：Pre-Norm、RMSNorm 让深层梯度更稳，减少归一化计算和均值依赖。
2. 长度外推：RoPE、NTK scaling、YaRN 让位置信息以相对距离进入 QK 点积，并能扩展上下文。
3. 推理效率：GQA/MQA/MLA 减少 KV cache 体积，FlashAttention 减少 HBM 读写。
4. 表达能力：SwiGLU 在 FFN 中加入门控，提高通道选择能力。

面试最后补一句取舍：这些方案不是“越新越好”，而是要同时满足训练稳定、推理高效、算子可实现、和原有 checkpoint/框架兼容。

#### 2. CNN、RNN、Transformer 的归纳偏置分别是什么？为什么 LLM 更偏向 Transformer？

答：CNN 的归纳偏置是局部性和平移等变，适合图像网格；RNN 的归纳偏置是时间顺序和递归状态，适合流式序列但难并行；Transformer 的归纳偏置更弱，主要依靠 attention 学任意 token 间关系，配合位置编码表达顺序。

LLM 更偏向 Transformer 的原因有三点：第一，训练可并行，长文本上吞吐远好于 RNN；第二，attention 能直接连接远距离 token，缓解长程依赖；第三，架构可扩展性好，参数、数据和算力扩大后 scaling law 表现稳定。

但要注意 Transformer 不是所有场景都无脑最优。它的 `O(T^2)` attention 成本很高，所以长上下文、多模态和端侧部署仍会引出 sparse attention、linear attention、state space、混合架构等方向。

#### 3. Multi-Head Attention 的多头到底在学什么？如果头数变多或变少会怎样？

答：多头的直觉是把 `d_model` 切成多个子空间，让不同 head 学不同关系：有的关注局部邻近，有的关注语法依赖，有的关注特殊 token，有的在长上下文里做检索式匹配。数学上每个 head 有自己的 `W_Q/W_K/W_V`，最后 concat 后经过 `W_O` 融合。

头数变多时，每个 head 的 `d_head = d_model / num_heads` 会变小，单头表达能力下降，但关系类型更多；头数变少时单头维度变大，但多样性下降。推理阶段还要考虑 KV cache：MHA 每个 Q head 都有自己的 KV，缓存大；GQA/MQA 让多个 Q head 共享 KV，降低显存和带宽。

面试可以这样收束：head 数不是越多越好，要在表达多样性、head_dim、KV cache、吞吐和训练稳定性之间折中。

### SwiGLU 为什么常用于 LLM FFN？和普通 FFN/ReLU 有什么区别？

- 频率：29
- 来源：xhs_6a0806ae, xhs_6a02019f, xhs_689c756d, xhs_6801c82b, xhs_67b46786, xhs_6a0e63b8, xhs_6a05c60a, xhs_69fb5599, xhs_69452f79, xhs_69435d73, xhs_69b3c3d1, xhs_69a6db5f, xhs_695d0d7d, xhs_69e0fbef, xhs_697e039f, +14
- 难度：基础

**面试官常问：**

1. SwiGLU 的公式是什么？和普通 FFN/GELU FFN 有什么区别？
2. 为什么 LLaMA/Qwen 等 LLM 常用 SwiGLU？
3. SwiGLU 中 gate/up/down 三个矩阵的 shape 怎么设？
4. SwiGLU 的参数量和传统 FFN 如何对齐？

SwiGLU 是门控 FFN。普通 FFN：

$$
FFN(x) = W_down activation(W_up x)
$$

SwiGLU：

$$
\begin{aligned}
SwiGLU(x) &= W_down( SiLU(W_gate x) \cdot  (W_up x) ) \\
SiLU(a) &= a \cdot  \operatorname{sigmoid}(a)
\end{aligned}
$$

它让一支分支作为门控，控制另一支特征通过多少。现代 LLM 中 `d_ff` 常约为 `8/3 * d_model` 或其他工程配置，配合三矩阵使参数量接近传统 `4*d` FFN。

PyTorch 模块：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

面试追问：为什么比 ReLU/GELU 好？可以回答“门控提高通道选择能力，实践上在大模型中效果和稳定性好；代价是多一个投影分支，算子融合和参数布局要优化”。

### 高频追问补充

#### 1. SwiGLU 相比 GELU FFN 为什么在 LLaMA/Qwen 中常见？

答：SwiGLU 常见是因为门控分支能动态选择通道信息，实践上比 ReLU/GELU FFN 更适合大模型扩展。

SwiGLU 用门控分支控制通道信息，形式通常是 `down_proj(silu(gate_proj(x)) * up_proj(x))`。如果 hidden 是 `d`、中间维是 `d_ff`，三个矩阵大约是 `d*d_ff、d*d_ff、d_ff*d`，参数量约 `3*d*d_ff`，经常大于 attention 的 `4*d*d`。它的优势是表达能力和训练效果好，代价是多一条投影分支。

SwiGLU 常见形式：

$$
FFN(x)=W_{down}(SiLU(W_{gate}x)\odot W_{up}x)
$$

若 hidden 为 `d`，中间维为 `d_ff`，参数量约：

$$
3d\cdot d_{ff}
$$

#### 2. gate_proj、up_proj、down_proj 三个矩阵的 shape 怎么写？

答：若 hidden size 是 `d`、FFN 中间维是 `d_ff`，gate/up/down 三个矩阵通常对应 `d x d_ff`、`d x d_ff`、`d_ff x d`。

SwiGLU 用门控分支控制通道信息，形式通常是 `down_proj(silu(gate_proj(x)) * up_proj(x))`。如果 hidden 是 `d`、中间维是 `d_ff`，三个矩阵大约是 `d*d_ff、d*d_ff、d_ff*d`，参数量约 `3*d*d_ff`，经常大于 attention 的 `4*d*d`。它的优势是表达能力和训练效果好，代价是多一条投影分支。

SwiGLU 常见形式：

$$
FFN(x)=W_{down}(SiLU(W_{gate}x)\odot W_{up}x)
$$

若 hidden 为 `d`，中间维为 `d_ff`，参数量约：

$$
3d\cdot d_{ff}
$$

#### 3. FFN 参数量为什么常常超过 attention？

答：FFN 参数量常超过 attention，因为 SwiGLU 有三块大矩阵，而 attention 常见 Q/K/V/O 合计约 `4d^2`。

SwiGLU 用门控分支控制通道信息，形式通常是 `down_proj(silu(gate_proj(x)) * up_proj(x))`。如果 hidden 是 `d`、中间维是 `d_ff`，三个矩阵大约是 `d*d_ff、d*d_ff、d_ff*d`，参数量约 `3*d*d_ff`，经常大于 attention 的 `4*d*d`。它的优势是表达能力和训练效果好，代价是多一条投影分支。

SwiGLU 常见形式：

$$
FFN(x)=W_{down}(SiLU(W_{gate}x)\odot W_{up}x)
$$

若 hidden 为 `d`，中间维为 `d_ff`，参数量约：

$$
3d\cdot d_{ff}
$$


## Attention 与 KV Cache

### MHA、MQA、GQA 为什么能省显存？为什么只有 KV Cache 没有 Q Cache？

- 频率：112
- 来源：xhs_6a0fc50a, xhs_6a0d5f2d, xhs_69f017bc, xhs_6a01559b, xhs_6a089599, xhs_6a02019f, xhs_69f5f803, xhs_69f0d3a5, xhs_69d724da, xhs_698878d1, xhs_699621c7, xhs_6904cfb9, xhs_68deb9f3, xhs_6850b0b6, xhs_67b46786, +97
- 难度：基础

**面试官常问：**

1. 为什么自回归推理只缓存 K/V，不缓存 Q？
2. MHA、MQA、GQA 的 head 关系和显存差异是什么？
3. KV Cache 显存如何估算？seq_len、layers、kv_heads 分别怎么影响？
4. GQA 相比 MQA 的优势和代价是什么？

自回归解码第 `t` 步，只需要当前 token 的 Q，但要和历史所有 token 的 K/V 做 attention，所以缓存 K/V，不缓存历史 Q。第 `t+1` 步旧 Q 不再参与计算，旧 K/V 仍然需要。

KV cache 公式：

$$
bytes = batch \cdot  seq_len \cdot  layers \cdot  kv_heads \cdot  head_dim \cdot  2(K,V) \cdot  bytes_per_elem
$$

MHA、MQA、GQA 的差别：

$$
\begin{aligned}
\text{MHA} &: q_heads = kv_heads = h \\
\text{MQA} &: q_heads = h, kv_heads = 1 \\
\text{GQA} &: q_heads = h, kv_heads = g, 1 < g < h
\end{aligned}
$$

显存估算代码：

```python
def kv_cache_bytes(batch, seq_len, layers, kv_heads, head_dim, dtype_bytes=2):
    return batch * seq_len * layers * kv_heads * head_dim * 2 * dtype_bytes

for kv_heads in [32, 8, 1]:
    gb = kv_cache_bytes(1, 32768, 32, kv_heads, 128) / 1024**3
    print(kv_heads, round(gb, 2), "GB")
```

GQA 的工程价值：大幅减少 KV cache 和内存带宽，同时保持比 MQA 更好的表达能力。代价是若 kv_heads 太少，可能损失多头注意力的细粒度表示。面试追问“为什么不缓存 Q”：因为 Q 是当前 query 的投影，只服务当前步；历史 token 在未来作为 key/value 被查询，而不是作为 query 主动查询未来。

### 高频追问补充

#### 1. FP8 KV Cache 和 GQA/MQA 是互补还是替代？

答：FP8 KV cache 减少每个元素字节数，GQA/MQA 减少 KV head 数，它们作用在公式的不同因子上，是互补关系。

GQA/MQA 减少 KV head 数，FP8 KV cache 减少每个元素字节数，它们是互补的。decode 每生成一个 token 都要读历史 K/V，长上下文下读内存比算 QK 更容易成为瓶颈。公式仍是 `B * T * L * kv_heads * head_dim * 2 * bytes`，其中 `2` 是 K 和 V。面试追问时可以顺手比较 MHA、MQA、GQA、MLA 四种压缩路线。

KV cache 公式：

$$
M_{KV}=B\cdot T\cdot L\cdot H_{kv}\cdot D_{head}\cdot 2\cdot bytes
$$

MHA、GQA、MQA 的主要差别就是 `H_kv`。Q 是当前步临时算出来的，不需要缓存历史 Q；K/V 是历史上下文，decode 每步都要复用。

#### 2. 为什么 decode 阶段 KV cache 会成为吞吐瓶颈？

答：decode 每步都要读历史 K/V，序列越长读内存越多，所以吞吐常受 KV 带宽和调度限制。

GQA/MQA 减少 KV head 数，FP8 KV cache 减少每个元素字节数，它们是互补的。decode 每生成一个 token 都要读历史 K/V，长上下文下读内存比算 QK 更容易成为瓶颈。公式仍是 `B * T * L * kv_heads * head_dim * 2 * bytes`，其中 `2` 是 K 和 V。面试追问时可以顺手比较 MHA、MQA、GQA、MLA 四种压缩路线。

KV cache 公式：

$$
M_{KV}=B\cdot T\cdot L\cdot H_{kv}\cdot D_{head}\cdot 2\cdot bytes
$$

MHA、GQA、MQA 的主要差别就是 `H_kv`。Q 是当前步临时算出来的，不需要缓存历史 Q；K/V 是历史上下文，decode 每步都要复用。

#### 3. kv_heads、head_dim、seq_len 改变时，KV cache 如何线性变化？

答：KV cache 对 batch、seq_len、layers、kv_heads、head_dim 和 dtype bytes 都是线性关系，改任一项都会线性改变显存。

GQA/MQA 减少 KV head 数，FP8 KV cache 减少每个元素字节数，它们是互补的。decode 每生成一个 token 都要读历史 K/V，长上下文下读内存比算 QK 更容易成为瓶颈。公式仍是 `B * T * L * kv_heads * head_dim * 2 * bytes`，其中 `2` 是 K 和 V。面试追问时可以顺手比较 MHA、MQA、GQA、MLA 四种压缩路线。

KV cache 公式：

$$
M_{KV}=B\cdot T\cdot L\cdot H_{kv}\cdot D_{head}\cdot 2\cdot bytes
$$

MHA、GQA、MQA 的主要差别就是 `H_kv`。Q 是当前步临时算出来的，不需要缓存历史 Q；K/V 是历史上下文，decode 每步都要复用。

### DeepSeek MLA/相关架构如何压缩 KV Cache？

- 频率：57
- 来源：xhs_6a0fc50a, xhs_6a0d5f2d, xhs_69f017bc, xhs_6a01559b, xhs_6a089599, xhs_69fb2bb8, xhs_69f0d3a5, xhs_69ba051d, xhs_6904cfb9, xhs_68deb9f3, xhs_67b46786, xhs_67ab4e1b, xhs_67adfa9e, xhs_6a0f13ed, xhs_6a0c71a7, +42
- 难度：进阶

**面试官常问：**

1. MLA 和 MQA/GQA 都在压缩 KV Cache，它们压缩的维度有什么不同？
2. MLA 的 latent KV 表示大概怎么写？
3. MLA 为什么能降低长上下文显存？代价是什么？
4. MLA 和量化能不能叠加？分别省什么？

MLA 可以理解为比 GQA 更进一步的 KV 压缩思路：不是只减少 KV head 数，而是把 K/V 先压到低维 latent，再在需要时恢复或参与注意力。一个抽象写法：

$$
\begin{aligned}
c_t &= x_t W_DKV          # low-rank latent, dim = d_c \\
k_t &= c_t W_UK \\
v_t &= c_t W_UV \\
cache c_t instead of full k_t/v_t
\end{aligned}
$$

如果原始 KV cache 维度是 `kv_heads * head_dim * 2`，latent 维度 `d_c` 远小于它，就能节省显存。和 MQA/GQA 对比：

```text
MQA/GQA: 减少 KV heads
MLA:    减少每个 token 需要缓存的表示维度
```

概念代码：

```python
import torch

def mla_project(x, w_dkv, w_uk, w_uv):
    latent = x @ w_dkv       # cache this: [B, T, d_c]
    k = latent @ w_uk
    v = latent @ w_uv
    return latent, k, v
```

面试要讲 tradeoff：MLA 省 KV cache，但会增加投影/恢复复杂度，对 kernel fusion、并行实现、RoPE 处理和训练稳定性都有要求。不要把 MLA 和量化混为一谈：量化是降低 dtype bytes，MLA 是降低表示维度，两者可以叠加。

### 高频追问补充

#### 1. MLA 和 GQA/MQA 在压缩 KV cache 上的区别是什么？

答：GQA/MQA 压缩的是 KV head 数，MLA 压缩的是 K/V 缓存表示维度，二者都是减少 K/V 冗余但机制不同。

GQA/MQA 是减少 KV head 数，MLA 是把 K/V 压到低维潜表示再恢复，压的是表示维度。低秩会带来信息瓶颈，所以需要训练时共同适配，不是推理时随便替换。DeepSeek 架构题常把 MLA、DeepSeekMoE、MTP、FP8、负载均衡一起问，回答时按“省 cache、省计算、扩容量、提训练/推理效率”分层讲。

MLA 可抽象成先压缩再恢复：

$$
c_t=W_c h_t,\quad K_t=W_k c_t,\quad V_t=W_v c_t
$$

缓存低维 `c_t` 比缓存完整 K/V 更省，但信息瓶颈和算子实现更复杂，所以需要训练时联合适配。

#### 2. DeepSeek 里 MLA 为什么常和 MoE、MTP、负载均衡一起被问？

答：DeepSeek 架构题常把 MLA、MoE、MTP、FP8、负载均衡放一起问，因为它们分别对应 KV、容量、训练信号、数值效率和系统吞吐。

GQA/MQA 是减少 KV head 数，MLA 是把 K/V 压到低维潜表示再恢复，压的是表示维度。低秩会带来信息瓶颈，所以需要训练时共同适配，不是推理时随便替换。DeepSeek 架构题常把 MLA、DeepSeekMoE、MTP、FP8、负载均衡一起问，回答时按“省 cache、省计算、扩容量、提训练/推理效率”分层讲。

MLA 可抽象成先压缩再恢复：

$$
c_t=W_c h_t,\quad K_t=W_k c_t,\quad V_t=W_v c_t
$$

缓存低维 `c_t` 比缓存完整 K/V 更省，但信息瓶颈和算子实现更复杂，所以需要训练时联合适配。

#### 3. 低秩潜表示会不会影响注意力表达能力？

答：低秩潜表示会带来信息瓶颈，不能推理时硬替换，必须训练时让模型共同适配。

GQA/MQA 是减少 KV head 数，MLA 是把 K/V 压到低维潜表示再恢复，压的是表示维度。低秩会带来信息瓶颈，所以需要训练时共同适配，不是推理时随便替换。DeepSeek 架构题常把 MLA、DeepSeekMoE、MTP、FP8、负载均衡一起问，回答时按“省 cache、省计算、扩容量、提训练/推理效率”分层讲。

MLA 可抽象成先压缩再恢复：

$$
c_t=W_c h_t,\quad K_t=W_k c_t,\quad V_t=W_v c_t
$$

缓存低维 `c_t` 比缓存完整 K/V 更省，但信息瓶颈和算子实现更复杂，所以需要训练时联合适配。

### Sparse Attention、Linear Attention、Gated Attention 如何降低复杂度，各自代价是什么？

- 频率：51
- 来源：xhs_6a0efd3c, xhs_6a089599, xhs_6a015529, xhs_69f0d3a5, xhs_69d8c35c, xhs_690410e2, xhs_690415f6, xhs_68ec676c, xhs_68d8e3cf, xhs_68deb9f3, xhs_68c0d1ff, xhs_67ade40f, xhs_67adfa9e, xhs_626adc39, xhs_6275ed75, +36
- 难度：进阶

**面试官常问：**

1. Sparse Attention、Linear Attention、Gated Attention 分别如何降低复杂度？
2. Linear Attention 用核函数改写 softmax attention 的直觉是什么？
3. 滑动窗口 attention 的 mask 怎么写？复杂度是多少？
4. 这些方法相比标准 attention 的表达能力代价是什么？

标准 attention 的瓶颈是 `O(T^2)`。Sparse Attention 直接减少连接数，例如局部窗口、block sparse、global token：

$$
\begin{aligned}
Attention_i only attends to j in window(i) or selected global tokens \\
complexity \approx O(T \cdot  window \cdot  d)
\end{aligned}
$$

Linear Attention 则把 softmax attention 改写成核函数形式：

$$
\operatorname{softmax}(QK^T)V \approx \phi(Q) (\phi(K)^T V)
$$

如果先算 `S = phi(K)^T V`，复杂度可接近 `O(T d^2)` 或 `O(T d)` 的变体，但通常会损失精确 softmax 表达。Gated Attention 引入门控控制信息流：

$$
y = gate(x) \cdot  attention(x) + (1 - gate(x)) \cdot  residual
$$

滑动窗口 mask 示例：

```python
import torch

def sliding_window_mask(seq_len, window, device="cpu"):
    i = torch.arange(seq_len, device=device)[:, None]
    j = torch.arange(seq_len, device=device)[None, :]
    causal = j <= i
    local = j >= i - window + 1
    return causal & local
```

面试回答 tradeoff：Sparse 简单可控但可能漏长程依赖；Linear 长序列吞吐好但训练稳定和表达力是难点；Gated 可以过滤噪声但多了参数和调参面。不能只说“复杂度低”，要说代价是什么。

### 高频追问补充

#### 1. Attention 真被高估了吗？哪些任务可以用 sparse/linear attention 替代？

答：Attention 没有被简单高估；稀疏/线性 attention 适合局部依赖强或超长上下文任务，但会牺牲部分全局精确交互。

稀疏/线性注意力降低复杂度，但会牺牲全局精确交互或引入近似误差。对局部依赖强、超长上下文、检索式任务，它们可能很有价值；对需要任意 token 两两精细交互的任务，标准 softmax attention 仍强。Gated Attention 更像学习信息流开关，稀疏注意力更像减少连接边，面试中要讲清收益和代价，而不是只说复杂度从 `O(n^2)` 到 `O(n)`。

复杂度对比：

$$
SoftmaxAttention=O(T^2d),\quad SparseAttention\approx O(Twd),\quad LinearAttention\approx O(Td^2)
$$

其中 `w` 是窗口或稀疏连接数。回答时要说清任务是否允许牺牲任意 token 两两精确交互。

#### 2. Linear Attention 为什么不总是比 softmax attention 好？

答：Linear Attention 不总是更好，因为核近似或递推状态可能损失 softmax attention 的表达能力和训练稳定性。

稀疏/线性注意力降低复杂度，但会牺牲全局精确交互或引入近似误差。对局部依赖强、超长上下文、检索式任务，它们可能很有价值；对需要任意 token 两两精细交互的任务，标准 softmax attention 仍强。Gated Attention 更像学习信息流开关，稀疏注意力更像减少连接边，面试中要讲清收益和代价，而不是只说复杂度从 `O(n^2)` 到 `O(n)`。

复杂度对比：

$$
SoftmaxAttention=O(T^2d),\quad SparseAttention\approx O(Twd),\quad LinearAttention\approx O(Td^2)
$$

其中 `w` 是窗口或稀疏连接数。回答时要说清任务是否允许牺牲任意 token 两两精确交互。

#### 3. Gated Attention 和稀疏注意力的收益分别来自哪里？

答：Gated Attention 的收益来自学习信息流开关，稀疏注意力的收益来自减少连接边和计算量，二者不是一回事。

稀疏/线性注意力降低复杂度，但会牺牲全局精确交互或引入近似误差。对局部依赖强、超长上下文、检索式任务，它们可能很有价值；对需要任意 token 两两精细交互的任务，标准 softmax attention 仍强。Gated Attention 更像学习信息流开关，稀疏注意力更像减少连接边，面试中要讲清收益和代价，而不是只说复杂度从 `O(n^2)` 到 `O(n)`。

复杂度对比：

$$
SoftmaxAttention=O(T^2d),\quad SparseAttention\approx O(Twd),\quad LinearAttention\approx O(Td^2)
$$

其中 `w` 是窗口或稀疏连接数。回答时要说清任务是否允许牺牲任意 token 两两精确交互。

### FlashAttention V1/V2/V3/V4 的核心优化思路是什么？

- 频率：44
- 来源：xhs_68dba071, xhs_689de9c3, xhs_68079ffd, xhs_67fe01a8, xhs_67b583d0, xhs_67ab4e1b, xhs_67930856, xhs_6a10491f, xhs_6a053551, xhs_6a12af6c, xhs_69ae814d, xhs_69156c6a, xhs_69435d73, xhs_69f31b54, xhs_69d90b7c, +29
- 难度：进阶

**面试官常问：**

1. FlashAttention 是近似 attention 吗？它到底优化了什么？
2. 为什么普通 attention 的 HBM 读写是瓶颈？
3. online softmax 如何避免保存完整 attention matrix？
4. FlashAttention V1/V2/V3/V4 的演进主要在哪些方向？

FlashAttention 的核心不是近似注意力，而是 IO-aware exact attention。普通实现会显式生成 `T x T` attention matrix，写入 HBM 后再读回来；FlashAttention 按 block 计算，用 online softmax 避免保存完整矩阵。

Online softmax 合并公式。已有块统计 `(m_old, l_old, o_old)`，新块 score 最大值 `m_new`：

$$
\begin{aligned}
m &= \max(m_old, m_new) \\
l &= \exp(m_old - m) \cdot  l_old + \exp(m_new - m) \cdot  l_new \\
o &= \exp(m_old - m) \cdot  o_old + \exp(m_new - m) \cdot  o_new
\end{aligned}
$$

其中 `l` 是 softmax denominator，`o` 是未归一化输出累积。伪代码：

```python
for q_block in Q_blocks:
    m = -inf
    l = 0
    o = 0
    for k_block, v_block in KV_blocks:
        s = q_block @ k_block.T / sqrt(d)
        m_new = \max(s, dim=-1)
        p = exp(s - m_new)
        l_new = sum(p, dim=-1)
        o_new = p @ v_block
        m_all = maximum(m, m_new)
        o = exp(m - m_all) * o + exp(m_new - m_all) * o_new
        l = exp(m - m_all) * l + exp(m_new - m_all) * l_new
        m = m_all
    output_block = o / l
```

复杂度仍是 `O(T^2 d)`，但 HBM 读写显著降低，所以实际更快、更省显存。V2/V3/V4 的演进主要在并行划分、反向传播、warp/block 调度和新硬件利用率上。

### 高频追问补充

#### 1. FlashAttention 为什么说是 IO-aware，而不是 attention 近似？

答：FlashAttention 是 IO-aware exact attention，它不近似 softmax，而是减少 HBM 读写和中间矩阵 materialization。

FlashAttention 不改变精确 attention 结果，核心是分块计算并避免把完整 `T*T` attention matrix 写回 HBM。online softmax 在每个块维护当前最大值和归一化因子，合并块时重新缩放，因此数值上等价。训练时主要省 attention 中间矩阵显存和 HBM 读写；推理 prefill 也能提速，但 decode 仍常受 KV cache 读带宽影响。

FlashAttention 的 online softmax 合并块时维护最大值 `m` 和归一化项 `l`：

$$
m_{new}=\max(m_{old}, m_{block})
$$

$$
l_{new}=e^{m_{old}-m_{new}}l_{old}+e^{m_{block}-m_{new}}l_{block}
$$

它省的是 HBM IO 和中间 attention matrix，不是把 attention 变成近似。

#### 2. online softmax 如何保证分块计算结果和标准 softmax 一致？

答：online softmax 通过维护块内最大值和归一化因子，把分块 softmax 合并成与全量 softmax 等价的结果。

FlashAttention 不改变精确 attention 结果，核心是分块计算并避免把完整 `T*T` attention matrix 写回 HBM。online softmax 在每个块维护当前最大值和归一化因子，合并块时重新缩放，因此数值上等价。训练时主要省 attention 中间矩阵显存和 HBM 读写；推理 prefill 也能提速，但 decode 仍常受 KV cache 读带宽影响。

FlashAttention 的 online softmax 合并块时维护最大值 `m` 和归一化项 `l`：

$$
m_{new}=\max(m_{old}, m_{block})
$$

$$
l_{new}=e^{m_{old}-m_{new}}l_{old}+e^{m_{block}-m_{new}}l_{block}
$$

它省的是 HBM IO 和中间 attention matrix，不是把 attention 变成近似。

#### 3. FlashAttention 对长上下文训练和推理分别省什么？

答：训练时主要省 attention score 矩阵显存和 HBM 往返；推理 prefill 也受益，但 decode 往往仍被 KV cache 读带宽限制。

FlashAttention 不改变精确 attention 结果，核心是分块计算并避免把完整 `T*T` attention matrix 写回 HBM。online softmax 在每个块维护当前最大值和归一化因子，合并块时重新缩放，因此数值上等价。训练时主要省 attention 中间矩阵显存和 HBM 读写；推理 prefill 也能提速，但 decode 仍常受 KV cache 读带宽影响。

FlashAttention 的 online softmax 合并块时维护最大值 `m` 和归一化项 `l`：

$$
m_{new}=\max(m_{old}, m_{block})
$$

$$
l_{new}=e^{m_{old}-m_{new}}l_{old}+e^{m_{block}-m_{new}}l_{block}
$$

它省的是 HBM IO 和中间 attention matrix，不是把 attention 变成近似。


## 位置编码

### RoPE 的直觉是什么，为什么有外推能力？Qwen/LLaMA 的 RoPE 实现有何差异？

- 频率：79
- 来源：xhs_6a089599, xhs_69de2e4c, xhs_6904cfb9, xhs_68d73f62, xhs_689de9c3, xhs_67b583d0, xhs_67ab146f, xhs_67ab4e1b, xhs_626adc39, xhs_6a013010, xhs_6a0722df, xhs_6a0ff8e7, xhs_6a0bc06b, xhs_6a0fd626, xhs_69156c6a, +64
- 难度：进阶

**面试官常问：**

1. RoPE 的数学形式是什么？为什么说它天然包含相对位置信息？
2. RoPE 为什么比绝对位置编码更容易做长度外推？
3. NTK scaling、YaRN、位置插值分别想解决什么问题？
4. Qwen/LLaMA 的 RoPE 实现差异常体现在哪里？

RoPE 的核心直觉：不要把位置简单加到 token embedding 上，而是对 Q/K 的偶数维二维子空间做旋转。位置 `m` 的旋转矩阵是 `R_m`，注意力分数中会出现：

$$
(R_m q)^T (R_n k) = q^T R_{n-m} k
$$

所以分数天然依赖相对距离 `n-m`，这就是 RoPE 比绝对位置 embedding 更适合长度外推的原因之一。对每一对维度 `(2i, 2i+1)`：

$$
\begin{aligned}
\theta_i &= base^(-2i / d) \\
x'_2i &= x_2i \cdot  cos(m \theta_i) - x_2i+1 \cdot  sin(m \theta_i) \\
x'_{2i+1} &= x_2i \cdot  sin(m \theta_i) + x_2i+1 \cdot  cos(m \theta_i)
\end{aligned}
$$

PyTorch 版实现：

```python
import torch

def apply_rope(x, cos, sin):
    # x: [B, H, T, D], cos/sin: [T, D/2]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)

def build_rope_cache(seq_len, dim, base=10000.0, device="cpu"):
    inv_freq = base ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("t,d->td", pos, inv_freq)
    return freqs.cos(), freqs.sin()
```

长上下文扩展的常见手法包括 position interpolation、NTK scaling、YaRN 等，本质都是调整频率或位置映射，避免高频维度在远距离上相位变化过快。Qwen/LLaMA 的差异通常体现在 `rope_theta/base`、是否使用缩放、最大上下文训练长度、实现中是否对不同频段做特殊处理。面试不要死背某个版本参数，应该说清：RoPE 通过旋转让相对距离进入 QK 点积，扩长上下文时要处理频率分布和训练长度外的泛化。

### 高频追问补充

#### 1. RoPE 外推失败时通常是高频维度出问题还是低频维度出问题？

答：RoPE 外推失败本质是训练没见过更大的相位范围，高频维度更容易在长距离上相位旋转过快，导致相对位置关系失真。

RoPE 的外推本质是训练时没见过更大的相位范围。缩放位置或频率可以让长距离相位变化慢下来，但也会改变训练长度内的相对距离分布，所以可能影响短文本。多模态场景里要区分一维文本位置、二维视觉位置和跨模态拼接位置，常见做法是视觉侧用 2D position 或 patch embedding，进入 LLM 后再统一成序列位置；面试重点是说明位置编码服务于“让模型知道 token 顺序/空间关系”，而不是死背某个实现。

RoPE 可理解为把每对 hidden 维旋转一个与位置有关的角度：

$$
q_m = R(m\theta)q,\quad k_n = R(n\theta)k
$$

内积会包含相对距离：

$$
q_m^T k_n = q^T R((n-m)\theta)k
$$

长上下文缩放改变的是相位增长速度；回答时要同时说收益是外推，代价是训练长度内的位置分布被改变。

#### 2. NTK scaling、YaRN、位置插值为什么有时会牺牲短上下文效果？

答：NTK scaling、YaRN、位置插值会改变训练长度内的相对距离分布，所以长上下文变好时，短上下文可能因为位置尺度被压缩而轻微掉点。

RoPE 的外推本质是训练时没见过更大的相位范围。缩放位置或频率可以让长距离相位变化慢下来，但也会改变训练长度内的相对距离分布，所以可能影响短文本。多模态场景里要区分一维文本位置、二维视觉位置和跨模态拼接位置，常见做法是视觉侧用 2D position 或 patch embedding，进入 LLM 后再统一成序列位置；面试重点是说明位置编码服务于“让模型知道 token 顺序/空间关系”，而不是死背某个实现。

RoPE 可理解为把每对 hidden 维旋转一个与位置有关的角度：

$$
q_m = R(m\theta)q,\quad k_n = R(n\theta)k
$$

内积会包含相对距离：

$$
q_m^T k_n = q^T R((n-m)\theta)k
$$

长上下文缩放改变的是相位增长速度；回答时要同时说收益是外推，代价是训练长度内的位置分布被改变。

#### 3. 多模态/VLA/VLM 里把图像 patch 和文本 token 混在一起时，位置编码怎么处理？

答：多模态里要同时处理文本一维顺序、图像二维空间、视频时间和跨模态拼接位置，不能简单把所有 patch 当成普通文本 token 位置。

RoPE 的外推本质是训练时没见过更大的相位范围。缩放位置或频率可以让长距离相位变化慢下来，但也会改变训练长度内的相对距离分布，所以可能影响短文本。多模态场景里要区分一维文本位置、二维视觉位置和跨模态拼接位置，常见做法是视觉侧用 2D position 或 patch embedding，进入 LLM 后再统一成序列位置；面试重点是说明位置编码服务于“让模型知道 token 顺序/空间关系”，而不是死背某个实现。

RoPE 可理解为把每对 hidden 维旋转一个与位置有关的角度：

$$
q_m = R(m\theta)q,\quad k_n = R(n\theta)k
$$

内积会包含相对距离：

$$
q_m^T k_n = q^T R((n-m)\theta)k
$$

长上下文缩放改变的是相位增长速度；回答时要同时说收益是外推，代价是训练长度内的位置分布被改变。


## 归一化与 Loss

### LayerNorm、RMSNorm、Pre-Norm 和 Post-Norm 怎么比较？

- 频率：52
- 来源：xhs_6a0d5f2d, xhs_6a0efd3c, xhs_6a015609, xhs_69d8778a, xhs_690ccaa6, xhs_68da8f0d, xhs_6858d751, xhs_680635bf, xhs_6801c710, xhs_67f36f1e, xhs_67b583d0, xhs_67930e59, xhs_626adc39, xhs_6a0ea83a, xhs_6a1042c8, +37
- 难度：基础

**面试官常问：**

1. LayerNorm 和 RMSNorm 的公式有什么区别？为什么 LLM 常用 RMSNorm？
2. Pre-Norm 和 Post-Norm 的结构差异是什么？哪个更稳定？
3. LayerNorm 和 BatchNorm 为什么在 NLP/LLM 中使用场景不同？
4. 手写 RMSNorm 时 eps、dtype 和 shape 要注意什么？

LayerNorm、RMSNorm、Pre-Norm、Post-Norm 是两组概念：前两个是归一化公式，后两个是归一化放在残差分支的哪里。

LayerNorm：

$$
\begin{aligned}
mu &= \operatorname{mean}(x) \\
sigma^2 &= \operatorname{mean}((x - mu)^2) \\
LN(x) &= \gamma \cdot  (x - mu) / \sqrt{\sigma^2 + \epsilon} + \beta
\end{aligned}
$$

RMSNorm：

$$
\begin{aligned}
rms &= \sqrt{\operatorname{mean}(x^2)+\epsilon} \\
RMSNorm(x) &= \gamma \cdot  x / rms
\end{aligned}
$$

RMSNorm 不减均值，计算更省，参数也少一点；现代 decoder-only LLM 很常用 RMSNorm。Pre-Norm 和 Post-Norm 的差别：

$$
\begin{aligned}
\text{Pre-Norm} &: y = x + Sublayer(Norm(x)) \\
\text{Post-Norm} &: y = Norm(x + Sublayer(x))
\end{aligned}
$$

Pre-Norm 的梯度可以更直接沿残差路径回传，深层模型更稳定；Post-Norm 输出分布更规整，但深层训练更容易梯度不稳。

手撕 RMSNorm：

```python
import torch

def rms_norm(x, weight, eps=1e-6):
    # x: [..., D], weight: [D]
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    y = x.float() / rms
    return (y * weight).type_as(x)
```

面试易错点：BatchNorm 是跨 batch 统计，LayerNorm/RMSNorm 是对单样本 hidden 维归一化。NLP/LLM 中 batch 和 sequence 长度变化大，LayerNorm/RMSNorm 更自然；BatchNorm 在自回归生成里也不方便。

### 高频追问补充

#### 1. 为什么 Transformer 面试经常问“为什么不用 BatchNorm”？

答：BatchNorm 依赖 batch 统计，而 LLM 的变长序列、mask、padding 和自回归推理都会让 batch 统计不稳定，所以现代 LLM 更偏 LayerNorm/RMSNorm。

BatchNorm 依赖 batch 统计，NLP/LLM 的 batch、长度、mask、生成阶段都不稳定，训练和推理统计也难一致。LayerNorm/RMSNorm 按 hidden 维做单样本归一化，更适合变长序列和自回归生成。Pre-Norm 让残差分支形成更顺畅的梯度通道，深层模型更容易训；RMSNorm 不减均值，省计算，实践效果接近 LayerNorm。

LayerNorm 与 RMSNorm 的差别可以直接写公式：

$$
LayerNorm(x)=g\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+b
$$

$$
RMSNorm(x)=g\frac{x}{\sqrt{mean(x^2)+\epsilon}}
$$

Pre-Norm 子层通常是 `x + Sublayer(Norm(x))`，Post-Norm 是 `Norm(x + Sublayer(x))`。深层 LLM 偏 Pre-Norm，因为残差路径更利于梯度传播。

#### 2. DeepSeek/Qwen/LLaMA 这类 LLM 为什么偏向 RMSNorm + Pre-Norm？

答：RMSNorm + Pre-Norm 的优势是计算更省、残差梯度路径更顺，适合很深的 decoder-only 模型。

BatchNorm 依赖 batch 统计，NLP/LLM 的 batch、长度、mask、生成阶段都不稳定，训练和推理统计也难一致。LayerNorm/RMSNorm 按 hidden 维做单样本归一化，更适合变长序列和自回归生成。Pre-Norm 让残差分支形成更顺畅的梯度通道，深层模型更容易训；RMSNorm 不减均值，省计算，实践效果接近 LayerNorm。

LayerNorm 与 RMSNorm 的差别可以直接写公式：

$$
LayerNorm(x)=g\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+b
$$

$$
RMSNorm(x)=g\frac{x}{\sqrt{mean(x^2)+\epsilon}}
$$

Pre-Norm 子层通常是 `x + Sublayer(Norm(x))`，Post-Norm 是 `Norm(x + Sublayer(x))`。深层 LLM 偏 Pre-Norm，因为残差路径更利于梯度传播。

#### 3. 如果 RMSNorm 放在 attention/FFN 前后不同位置，训练稳定性会怎么变？

答：Norm 放在子层前更利于深层训练稳定；放在子层后输出分布更规整，但深层模型更容易梯度不稳或需要更仔细的 warmup。

BatchNorm 依赖 batch 统计，NLP/LLM 的 batch、长度、mask、生成阶段都不稳定，训练和推理统计也难一致。LayerNorm/RMSNorm 按 hidden 维做单样本归一化，更适合变长序列和自回归生成。Pre-Norm 让残差分支形成更顺畅的梯度通道，深层模型更容易训；RMSNorm 不减均值，省计算，实践效果接近 LayerNorm。

LayerNorm 与 RMSNorm 的差别可以直接写公式：

$$
LayerNorm(x)=g\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+b
$$

$$
RMSNorm(x)=g\frac{x}{\sqrt{mean(x^2)+\epsilon}}
$$

Pre-Norm 子层通常是 `x + Sublayer(Norm(x))`，Post-Norm 是 `Norm(x + Sublayer(x))`。深层 LLM 偏 Pre-Norm，因为残差路径更利于梯度传播。

### 手撕 CrossEntropyLoss 时前向、梯度和数值稳定怎么写？

- 频率：18
- 来源：xhs_69fff0a6, xhs_6a0d5f2d, xhs_69de2e4c, xhs_675ff0e7, xhs_67b583d0, xhs_6a034d23, xhs_69ae814d, xhs_69d61f00, xhs_69f1dc3e, xhs_69e4c3b0, xhs_6a0ee0f3, xhs_69b3c3d1, xhs_698e8aef, xhs_69226491, xhs_6a06c24f, +3
- 难度：基础

**面试官常问：**

1. CrossEntropyLoss 为什么通常直接接 logits，而不是先 softmax？
2. 请推导 CE 对 logits 的梯度。
3. log-sum-exp trick 如何保证数值稳定？
4. label smoothing 和 ignore_index 在 LLM 训练里怎么用？

交叉熵最稳的回答方式是从 log-softmax 推导。对单样本 logits `z` 和正确类别 `y`：

$$
\begin{aligned}
CE(z, y) &= -log \operatorname{softmax}(z)_y \\
 &= -z_y + \log(\sum_j \exp(z_j))
\end{aligned}
$$

为了数值稳定，用 `m = max_j z_j`：

$$
\begin{aligned}
logsum\exp(z) &= m + \log(\sum_j \exp(z_j - m)) \\
loss_i &= logsum\exp(z_i) - z_i[y_i]
\end{aligned}
$$

梯度：

$$
\begin{aligned}
d loss / d z_j &= \operatorname{softmax}(z)_j - 1[j = y] \\
batch \operatorname{mean} 时再除以 N
\end{aligned}
$$

手撕实现：

```python
import torch

def cross_entropy_from_logits(logits, target):
    # logits: [N, C], target: [N]
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_z = torch.\log(torch.exp(shifted).sum(dim=-1))
    correct = shifted[torch.arange(logits.size(0)), target]
    return (log_z - correct).mean()

def ce_grad(logits, target):
    probs = torch.softmax(logits, dim=-1)
    probs[torch.arange(logits.size(0)), target] -= 1.0
    return probs / logits.size(0)
```

Label smoothing 时目标不再是 one-hot，而是：

$$
\begin{aligned}
q_y &= 1 - \epsilon \\
q_j &= \epsilon / (C - 1), j != y \\
CE &= -\sum_j q_j log p_j
\end{aligned}
$$

LLM 训练中还要注意 ignore index。比如 padding token、prompt 部分不参与 loss，通常把 label 置为 `-100`，PyTorch CE 会跳过。

### 高频追问补充

#### 1. MTP Loss 和普通 next-token CE 的区别是什么？

答：MTP Loss 在普通 next-token CE 外增加多个未来 token 的监督，目标是给表示更密集的未来信息信号。

next-token CE 只预测下一个 token；MTP/多 token prediction 会让模型同时预测未来多个位置，给中间层或额外 head 更密集的监督，可能改善表示和推理加速适配。SFT 里通常只对 assistant answer 部分算 loss，prompt/user/system token 作为条件，不应被模型“学习复读”。label smoothing 能缓解过度自信，但会改变目标分布，评估 PPL 时要注意训练目标和评估目标是否一致。

数值稳定 CE 不要先 softmax 再 log：

$$
CE_i=\log\sum_j e^{z_{ij}}-z_{i,y_i}
$$

梯度是：

$$
\frac{\partial L}{\partial z}=p-onehot(y)
$$

SFT 中常把非 assistant token 的 label 置为 `-100`。MTP 可写为 `L = L_next + sum_k lambda_k L_{t+k}`，label smoothing 则把 one-hot 改成软标签。

#### 2. 为什么 LLM 训练里 prompt token 常常要 mask 掉？

答：SFT 中 prompt token 是条件，不是 assistant 要学习复述的目标；通常只对 assistant answer 部分算 loss。

next-token CE 只预测下一个 token；MTP/多 token prediction 会让模型同时预测未来多个位置，给中间层或额外 head 更密集的监督，可能改善表示和推理加速适配。SFT 里通常只对 assistant answer 部分算 loss，prompt/user/system token 作为条件，不应被模型“学习复读”。label smoothing 能缓解过度自信，但会改变目标分布，评估 PPL 时要注意训练目标和评估目标是否一致。

数值稳定 CE 不要先 softmax 再 log：

$$
CE_i=\log\sum_j e^{z_{ij}}-z_{i,y_i}
$$

梯度是：

$$
\frac{\partial L}{\partial z}=p-onehot(y)
$$

SFT 中常把非 assistant token 的 label 置为 `-100`。MTP 可写为 `L = L_next + sum_k lambda_k L_{t+k}`，label smoothing 则把 one-hot 改成软标签。

#### 3. label smoothing 会不会影响生成模型的校准和困惑度？

答：label smoothing 会降低过度自信、改善校准，但它改变了目标分布，因此 PPL 和 hard-label CE 的解释要分开。

next-token CE 只预测下一个 token；MTP/多 token prediction 会让模型同时预测未来多个位置，给中间层或额外 head 更密集的监督，可能改善表示和推理加速适配。SFT 里通常只对 assistant answer 部分算 loss，prompt/user/system token 作为条件，不应被模型“学习复读”。label smoothing 能缓解过度自信，但会改变目标分布，评估 PPL 时要注意训练目标和评估目标是否一致。

数值稳定 CE 不要先 softmax 再 log：

$$
CE_i=\log\sum_j e^{z_{ij}}-z_{i,y_i}
$$

梯度是：

$$
\frac{\partial L}{\partial z}=p-onehot(y)
$$

SFT 中常把非 assistant token 的 label 置为 `-100`。MTP 可写为 `L = L_next + sum_k lambda_k L_{t+k}`，label smoothing 则把 one-hot 改成软标签。

### MTP Loss、多 Token Prediction 和推理加速训练怎么理解？

- 频率：15
- 来源：xhs_69eb90ca, xhs_6a0efd3c, xhs_69408c32, xhs_690410e2, xhs_68ec676c, xhs_68c0d1ff, xhs_679f1ba0, xhs_6780ff7f, xhs_6a069e96, xhs_6a11541e, xhs_6a10491f, xhs_6a02ccc7, xhs_69c363c5, xhs_68ca6d77, xhs_6a0dd91f
- 难度：进阶

**面试官常问：**

1. MTP Loss 和普通 next-token CE 的区别是什么？
2. 多 Token Prediction 如何服务 speculative decoding 或多 token 解码？
3. 多个 future token 的 loss 如何对齐 label？
4. MTP 会不会让训练目标和推理目标不一致？

普通语言模型训练预测下一个 token：

$$
L_{NTP}=-\sum_t \log p_\theta(x_{t+1}|x_{\le t})
$$

MTP/Multi-Token Prediction 让同一位置预测多个未来 token：

$$
L_{MTP}=\sum_t \sum_{k=1}^{K} \lambda_k \cdot CE(h_t W_k, x_{t+k})
$$

其中 `K` 是预测 horizon，`W_k` 可以是多个预测头，也可以共享部分参数。它的价值是给 hidden state 更密集监督，让表示包含更多未来信息，也可能为 speculative decoding、多 token 解码或草稿头提供训练信号。

```python
import torch.nn.functional as F

def mtp_loss(logits_by_horizon, labels, weights):
    # logits_by_horizon[k]: [B, T, V], predicts labels shifted by k+1
    losses = []
    for k, logits in enumerate(logits_by_horizon, start=1):
        target = labels[:, k:].contiguous()
        pred = logits[:, :-k, :].contiguous()
        losses.append(weights[k - 1] * F.cross_entropy(
            pred.view(-1, pred.size(-1)),
            target.view(-1),
            ignore_index=-100,
        ))
    return sum(losses)
```

风险是远期 token 不确定性更大，loss 权重太高会干扰 next-token 主目标；所以通常主 loss 仍占主导，MTP loss 作为辅助。

### 高频追问补充

#### 1. MTP Loss 与普通 next-token loss 的数学目标有什么不同？

答：MTP 的目标是在一个位置预测多个未来 token，而普通 CE 只预测下一个 token。

普通 next-token 只预测下一个 token，MTP 会在同一位置或中间层同时预测多个未来 token，提供更密集监督。它可以训练 draft head、提升表示对未来信息的捕捉，并服务 speculative decoding。难点是未来步越远不确定性越大，辅助 loss 权重不能压过主目标；实现中要处理 horizon 对齐、mask、loss normalization 和推理时是否真的启用多 token head。

MTP 目标：

$$
L=\sum_t CE(h_t,x_{t+1})+\sum_{k=2}^{K}\lambda_k CE(h_t^{(k)},x_{t+k})
$$

实现时要处理 label shift、padding mask、未来 token 越界和不同 horizon 的 loss scale。

#### 2. 多 token prediction 如何服务 speculative decoding 或多 token 解码？

答：多 token prediction 可以训练 draft head 或未来 token head，为 speculative decoding 和多 token 解码提供候选。

普通 next-token 只预测下一个 token，MTP 会在同一位置或中间层同时预测多个未来 token，提供更密集监督。它可以训练 draft head、提升表示对未来信息的捕捉，并服务 speculative decoding。难点是未来步越远不确定性越大，辅助 loss 权重不能压过主目标；实现中要处理 horizon 对齐、mask、loss normalization 和推理时是否真的启用多 token head。

MTP 目标：

$$
L=\sum_t CE(h_t,x_{t+1})+\sum_{k=2}^{K}\lambda_k CE(h_t^{(k)},x_{t+k})
$$

实现时要处理 label shift、padding mask、未来 token 越界和不同 horizon 的 loss scale。

#### 3. 多个 future token 的 loss 权重、label shift 和归一化怎么处理？

答：多个 future token 要做 horizon 权重、mask 和归一化；远未来不确定性更大，辅助 loss 不能压过主 next-token loss。

普通 next-token 只预测下一个 token，MTP 会在同一位置或中间层同时预测多个未来 token，提供更密集监督。它可以训练 draft head、提升表示对未来信息的捕捉，并服务 speculative decoding。难点是未来步越远不确定性越大，辅助 loss 权重不能压过主目标；实现中要处理 horizon 对齐、mask、loss normalization 和推理时是否真的启用多 token head。

MTP 目标：

$$
L=\sum_t CE(h_t,x_{t+1})+\sum_{k=2}^{K}\lambda_k CE(h_t^{(k)},x_{t+k})
$$

实现时要处理 label shift、padding mask、未来 token 越界和不同 horizon 的 loss scale。


## 训练与优化

### LoRA、QLoRA 和全量微调有什么区别？LoRA rank 怎么选？

- 频率：174
- 来源：xhs_6a0efd3c, xhs_6a05f4c4, xhs_6a02019f, xhs_6a019beb, xhs_6a0018b9, xhs_6a004ca9, xhs_69f9bacb, xhs_69f770dc, xhs_69d8778a, xhs_69e2cfc5, xhs_693f70c3, xhs_68f60343, xhs_68d8e3cf, xhs_684f5604, xhs_681382a8, +159
- 难度：基础

**面试官常问：**

1. LoRA 的低秩分解公式是什么？alpha/r 起什么作用？
2. LoRA、QLoRA、全量微调各自适合什么场景？
3. LoRA rank 怎么选？target modules 怎么选？
4. LoRA 为什么不一定适合注入大量新知识？

LoRA 假设下游任务所需权重更新 `Delta W` 低秩。冻结原权重 `W`，只训练：

$$
\begin{aligned}
W' &= W + \Delta W \\
\Delta W &= (alpha / r) \cdot  B A \\
A in R^{r x d_in}, B in R^{d_out x r}
\end{aligned}
$$

其中 `r` 是 rank，`alpha/r` 是缩放。全量微调更新所有 `W`，容量强但显存大、过拟合风险高；LoRA 只训练 A/B，显存和可合并部署更友好；QLoRA 把基座权重量化到 4bit/8bit，再训练 LoRA adapter。

LoRA Linear：

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base_linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base_linear.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
```

Rank 怎么选：小数据/简单风格任务可从 `r=4/8` 开始；复杂推理或多领域可试 `16/32/64`；看验证集收益是否随 rank 增长饱和。target modules 常选 `q_proj,v_proj`，更强可加 `k_proj,o_proj,up/down/gate_proj`。面试要补充：LoRA 不一定适合注入大量新知识，RAG 或继续预训练可能更合适。

### 高频追问补充

#### 1. LoRA 微调项目从零做，数据、target modules、rank、alpha 怎么选？

答：LoRA 项目先定任务和数据，再选 target modules；简单风格任务可从 q/v 开始，复杂能力迁移常扩到 k/o/gate/up/down。

LoRA 的核心是把权重增量写成低秩 `Delta W = B A`。rank 控制容量，alpha 控制缩放，target modules 常从 `q_proj/v_proj` 扩到 `k/o/gate/up/down` 视任务复杂度而定。QLoRA 省基座显存；DoRA 分解方向和幅度；AdaLoRA/DyLoRA 动态或可变 rank。知识更新优先 RAG，稳定格式/风格和小能力迁移适合 LoRA，深度领域能力或大分布迁移才考虑全量。

LoRA 权重增量：

$$
\Delta W=\frac{\alpha}{r}BA
$$

推理可 merge：

```python
W_merged = W_base + (alpha / r) * (B @ A)
```

rank 决定容量，alpha 决定增量尺度，target modules 决定能力注入位置。

#### 2. QLoRA、DoRA、AdaLoRA/DyLoRA 各自解决什么问题？

答：QLoRA 省基座显存，DoRA 分离方向和幅度，AdaLoRA/DyLoRA 动态或可变 rank，核心都是改善低秩适配的容量/成本权衡。

LoRA 的核心是把权重增量写成低秩 `Delta W = B A`。rank 控制容量，alpha 控制缩放，target modules 常从 `q_proj/v_proj` 扩到 `k/o/gate/up/down` 视任务复杂度而定。QLoRA 省基座显存；DoRA 分解方向和幅度；AdaLoRA/DyLoRA 动态或可变 rank。知识更新优先 RAG，稳定格式/风格和小能力迁移适合 LoRA，深度领域能力或大分布迁移才考虑全量。

LoRA 权重增量：

$$
\Delta W=\frac{\alpha}{r}BA
$$

推理可 merge：

```python
W_merged = W_base + (alpha / r) * (B @ A)
```

rank 决定容量，alpha 决定增量尺度，target modules 决定能力注入位置。

#### 3. 什么时候应该用 RAG，什么时候应该用 LoRA，什么时候全量微调？

答：知识更新优先 RAG，稳定格式和小能力迁移适合 LoRA，深度领域能力或大分布迁移才考虑全量微调。

LoRA 的核心是把权重增量写成低秩 `Delta W = B A`。rank 控制容量，alpha 控制缩放，target modules 常从 `q_proj/v_proj` 扩到 `k/o/gate/up/down` 视任务复杂度而定。QLoRA 省基座显存；DoRA 分解方向和幅度；AdaLoRA/DyLoRA 动态或可变 rank。知识更新优先 RAG，稳定格式/风格和小能力迁移适合 LoRA，深度领域能力或大分布迁移才考虑全量。

LoRA 权重增量：

$$
\Delta W=\frac{\alpha}{r}BA
$$

推理可 merge：

```python
W_merged = W_base + (alpha / r) * (B @ A)
```

rank 决定容量，alpha 决定增量尺度，target modules 决定能力注入位置。

### 大模型蒸馏给小模型会遇到什么问题？on-policy 和 off-policy 蒸馏怎么区分？

- 频率：54
- 来源：xhs_6a0be406, xhs_6a019beb, xhs_6a004ca9, xhs_69fb2bb8, xhs_69e7b0f8, xhs_69d4f398, xhs_699696fc, xhs_67ab4058, xhs_679f1ba0, xhs_6a10562e, xhs_6a0dbd0c, xhs_6a0a80d8, xhs_6a0a890d, xhs_6a05209b, xhs_6a0decd2, +39
- 难度：进阶

**面试官常问：**

1. 大模型蒸馏给小模型有哪些蒸馏目标？logits、答案、偏好有什么区别？
2. on-policy 蒸馏和 off-policy 蒸馏如何区分？
3. 温度 T 在 logits 蒸馏中有什么作用？
4. 没有 teacher logits 时还能怎么蒸馏？

蒸馏可以蒸最终答案、logits、偏好排序，也可以蒸中间推理。经典 logits 蒸馏：

$$
\begin{aligned}
p_T &= \operatorname{softmax}(z_teacher / T) \\
p_S &= \operatorname{softmax}(z_student / T) \\
L_KD &= T^2 \cdot  KL(p_T || p_S) \\
L &= alpha \cdot  L_CE(y, p_S) + (1-alpha) \cdot  L_KD
\end{aligned}
$$

温度 `T` 让分布更平，学生能学到非正确类之间的暗知识。如果拿不到 teacher logits，只能用 teacher 生成文本、打分或偏好对。

Off-policy 蒸馏：先用 teacher 生成固定数据，再训练 student。优点是便宜、稳定；缺点是 student 部署时会遇到自己分布下的错误，固定数据覆盖不到。On-policy 蒸馏：student 当前策略生成，teacher/reward 再反馈。优点是贴近 student 分布；缺点是贵、噪声大、训练更不稳定。

伪代码：

```python
for batch in prompts:
    if on_policy:
        y = student.generate(batch)
        target = teacher.rewrite_or_score(batch, y)
    else:
        target = cached_teacher_answers[batch.id]
    loss = sft_loss(student, batch, target)
    loss.backward()
```

常见问题：小模型容量不足会学不会长链推理；teacher 错误会被继承；只蒸结果会丢过程能力；训练集和评测集污染会虚高。解决上可做拒绝采样、多 teacher ensemble、难例再生成、短答案蒸馏和过程蒸馏结合。

### 高频追问补充

#### 1. 搜索 Agent 自蒸馏、学生自采样再由教师打分，属于 on-policy 还是 off-policy？

答：学生当前策略采样、再由教师或环境打分，更接近 on-policy；固定教师数据训练学生是 off-policy。

固定教师生成数据再训练学生是 off-policy；学生当前策略采样、教师/环境再反馈更接近 on-policy。logits 蒸馏学概率分布，文本蒸馏学最终输出，隐藏层蒸馏学中间表示。小模型容量有限，容易模仿表面格式却承载不了长链推理；要提高效果，需要课程化数据、过程监督、难例筛选、拒绝采样和面向学生能力的数据再生成。

logits 蒸馏常用 KL：

$$
L_{KD}=T^2\cdot KL(softmax(z_t/T)\|softmax(z_s/T))
$$

文本蒸馏只看教师输出文本，隐藏层蒸馏对齐中间表示。on-policy 蒸馏更贴近学生部署分布，但采样和评价成本更高。

#### 2. 隐藏层蒸馏和 logits/文本蒸馏分别蒸什么？

答：logits 蒸馏学概率分布，文本蒸馏学最终输出，隐藏层蒸馏学中间表示或表示空间对齐。

固定教师生成数据再训练学生是 off-policy；学生当前策略采样、教师/环境再反馈更接近 on-policy。logits 蒸馏学概率分布，文本蒸馏学最终输出，隐藏层蒸馏学中间表示。小模型容量有限，容易模仿表面格式却承载不了长链推理；要提高效果，需要课程化数据、过程监督、难例筛选、拒绝采样和面向学生能力的数据再生成。

logits 蒸馏常用 KL：

$$
L_{KD}=T^2\cdot KL(softmax(z_t/T)\|softmax(z_s/T))
$$

文本蒸馏只看教师输出文本，隐藏层蒸馏对齐中间表示。on-policy 蒸馏更贴近学生部署分布，但采样和评价成本更高。

#### 3. 小模型蒸馏大模型时为什么容易学到格式但学不到推理能力？

答：小模型容量有限，容易学到表面格式但承载不了长链推理，因此要用课程数据、过程监督、难例筛选和拒绝采样补能力。

固定教师生成数据再训练学生是 off-policy；学生当前策略采样、教师/环境再反馈更接近 on-policy。logits 蒸馏学概率分布，文本蒸馏学最终输出，隐藏层蒸馏学中间表示。小模型容量有限，容易模仿表面格式却承载不了长链推理；要提高效果，需要课程化数据、过程监督、难例筛选、拒绝采样和面向学生能力的数据再生成。

logits 蒸馏常用 KL：

$$
L_{KD}=T^2\cdot KL(softmax(z_t/T)\|softmax(z_s/T))
$$

文本蒸馏只看教师输出文本，隐藏层蒸馏对齐中间表示。on-policy 蒸馏更贴近学生部署分布，但采样和评价成本更高。

### 预训练、SFT 和后训练的数据配比与数据工程怎么设计？

- 频率：53
- 来源：xhs_6a0733b1, xhs_6a05f4c4, xhs_69f4db22, xhs_69e4e580, xhs_69e2cfc5, xhs_6988310c, xhs_690410e2, xhs_68ff77f4, xhs_68f0ecf6, xhs_68deb9f3, xhs_67e174c5, xhs_67babdc6, xhs_6a0ab514, xhs_69f44114, xhs_6a08333a, +38
- 难度：进阶

**面试官常问：**

1. 预训练、SFT、后训练的数据配比有什么不同？
2. 如何做数据去重、质量过滤和污染检测？
3. 数据配比变化后，怎么判断模型能力是真的变好？
4. 合成数据和人工数据如何混合？

数据配比决定模型能力形状。预训练阶段关注覆盖面和分布，SFT 阶段关注指令遵循和格式，偏好/RL 阶段关注可验证反馈和难例。

一个常见采样形式是：

$$
P(d_i)=\frac{w_i \cdot n_i^\alpha}{\sum_j w_j \cdot n_j^\alpha}
$$

其中 `n_i` 是某领域数据量，`w_i` 是人为权重，`alpha < 1` 可以避免大领域完全淹没小领域。工程上要做版本化：数据源、过滤规则、去重 hash、采样权重、污染检测和 eval 结果都要可追溯。

```python
def mixture_sample(datasets, weights, temperature=0.7):
    sizes = {name: len(ds) for name, ds in datasets.items()}
    raw = {name: weights[name] * sizes[name] ** temperature for name in datasets}
    z = sum(raw.values())
    probs = {name: value / z for name, value in raw.items()}
    return probs
```

面试中最好强调 ablation：只改一个数据维度，观察通用能力、领域能力、格式遵循、安全和长尾任务，不要只看单一榜单。

### 高频追问补充

#### 1. 预训练、SFT、RL/偏好数据的配比目标分别是什么？

答：预训练配比决定基础能力和知识覆盖，SFT 配比决定指令行为，RL/偏好数据决定模型往什么偏好方向收缩。

预训练数据配比决定基础能力分布，SFT 数据决定指令行为和格式，RL/偏好数据决定模型偏好的方向和边界。数据工程要做去重、质量过滤、污染检测、来源版本、采样权重、难例挖掘和 per-domain eval。合成数据要防模板化、错误扩散和评测泄漏；难例要按能力缺口采样，并用 holdout 集验证是否只提升某类题而牺牲通用能力。

数据混合可用采样权重表达：

$$
P(domain_i)=\frac{w_i n_i^\alpha}{\sum_j w_j n_j^\alpha}
$$

`alpha < 1` 可避免大语料域完全支配训练。每次改配比都要配 per-domain eval，否则只看总分会掩盖能力迁移。

#### 2. 数据去重、污染检测和质量过滤具体在防什么？

答：去重防记忆和过拟合，污染检测防评测泄漏，质量过滤防噪声样本主导梯度。

预训练数据配比决定基础能力分布，SFT 数据决定指令行为和格式，RL/偏好数据决定模型偏好的方向和边界。数据工程要做去重、质量过滤、污染检测、来源版本、采样权重、难例挖掘和 per-domain eval。合成数据要防模板化、错误扩散和评测泄漏；难例要按能力缺口采样，并用 holdout 集验证是否只提升某类题而牺牲通用能力。

数据混合可用采样权重表达：

$$
P(domain_i)=\frac{w_i n_i^\alpha}{\sum_j w_j n_j^\alpha}
$$

`alpha < 1` 可避免大语料域完全支配训练。每次改配比都要配 per-domain eval，否则只看总分会掩盖能力迁移。

#### 3. 数据合成和难例挖掘如何避免把模型训偏？

答：合成数据和难例挖掘要做来源标记、难度分层和 holdout eval，否则很容易把模型训成模板化或偏科。

预训练数据配比决定基础能力分布，SFT 数据决定指令行为和格式，RL/偏好数据决定模型偏好的方向和边界。数据工程要做去重、质量过滤、污染检测、来源版本、采样权重、难例挖掘和 per-domain eval。合成数据要防模板化、错误扩散和评测泄漏；难例要按能力缺口采样，并用 holdout 集验证是否只提升某类题而牺牲通用能力。

数据混合可用采样权重表达：

$$
P(domain_i)=\frac{w_i n_i^\alpha}{\sum_j w_j n_j^\alpha}
$$

`alpha < 1` 可避免大语料域完全支配训练。每次改配比都要配 per-domain eval，否则只看总分会掩盖能力迁移。

### LLM 训练中 loss spike 或 loss 突然增大如何排查和处理？

- 频率：5
- 来源：xhs_69e2cfc5, xhs_675ff0e7, xhs_67b0886e, xhs_69b02ec5, xhs_6804a52a
- 难度：进阶

**面试官常问：**

1. LLM 训练中 loss spike 可能由哪些原因导致？
2. 如何区分数据坏样本、学习率问题和混合精度溢出？
3. 训练日志里应该记录哪些信息才能复现 spike？
4. 遇到 spike 后应该跳 batch、回滚 checkpoint 还是调学习率？

Loss spike 不是一句“学习率太大”就结束，要按可复现性分层排查：数据、数值、优化器、分布、并行/系统。关键监控量：

$$
\begin{aligned}
grad_norm &= \sqrt{\sum_i \lVert g_i \rVert_2^2} \\
update_norm_ratio &= ||\Delta W|| / (||W|| + eps) \\
loss_scale_overflow &= mixed_precision 是否溢出
\end{aligned}
$$

排查顺序建议：

1. 定位 spike 的 global step、数据 shard、样本 hash、batch token 长度。
2. 检查是否有坏样本：超长重复、乱码、全特殊 token、错误 label mask、空答案。
3. 检查数值：grad norm、激活最大值、BF16/FP16 overflow、loss scale、NaN/Inf。
4. 检查优化器：LR warmup/decay、beta、weight decay、梯度裁剪、batch size 变化。
5. 检查分布切换：数据混合比例、课程学习阶段、某类高难样本突然增多。

一个训练循环里的监控片段：

```python
import math
import torch

def global_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total)

loss.backward()
gn = global_grad_norm(model)
if not math.isfinite(gn) or gn > 1_000:
    print("bad step", step, "loss", loss.item(), "grad_norm", gn, "batch_id", batch["id"])
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

处理策略：短期可以跳过坏 batch、从 spike 前 checkpoint 恢复、降低 LR、加梯度裁剪；长期要修数据管道、增加 anomaly logging、保存 batch replay 能力。大模型里 spike 可能来自“当前模型状态 + 特定 batch”的组合，不保留样本和优化器状态很难复现。

### 高频追问补充

#### 1. 大模型训练 loss spike 后，是直接跳过 batch 还是从 checkpoint 回滚？

答：如果 spike 后能恢复，可以先跳 batch、裁剪梯度并记录样本；如果质量持续恶化，应从 spike 前 checkpoint 连同优化器状态一起回滚。

单次 spike 可以先定位 batch、保存样本 hash、观察 grad norm/NaN/overflow；如果后续能恢复，可能只需跳 batch 或裁剪梯度。如果 spike 后模型质量持续恶化，要从前一个稳定 checkpoint 回滚，并检查优化器状态。RL 后训练里 reward 分布突变、KL 约束过弱、格式奖励被 hack，都可能表现成 loss/ratio/KL 异常，因此要把 loss spike 和策略漂移一起看。

排查顺序可以固定成：数据样本 hash、loss scale/NaN/Inf、grad norm、学习率和 warmup、optimizer state、最近数据混合比例变化、验证集质量。若使用混合精度，还要看 overflow 和动态 loss scaling。

```python
if not torch.isfinite(loss) or grad_norm > threshold:
    save_bad_batch_ids()
    skip_or_rollback()
```

#### 2. 为什么同一个坏样本不一定每次都会触发 spike？

答：同一个坏样本不一定稳定触发 spike，因为触发条件还取决于当前参数、optimizer state、loss scale、batch 组合和随机种子。

单次 spike 可以先定位 batch、保存样本 hash、观察 grad norm/NaN/overflow；如果后续能恢复，可能只需跳 batch 或裁剪梯度。如果 spike 后模型质量持续恶化，要从前一个稳定 checkpoint 回滚，并检查优化器状态。RL 后训练里 reward 分布突变、KL 约束过弱、格式奖励被 hack，都可能表现成 loss/ratio/KL 异常，因此要把 loss spike 和策略漂移一起看。

排查顺序可以固定成：数据样本 hash、loss scale/NaN/Inf、grad norm、学习率和 warmup、optimizer state、最近数据混合比例变化、验证集质量。若使用混合精度，还要看 overflow 和动态 loss scaling。

```python
if not torch.isfinite(loss) or grad_norm > threshold:
    save_bad_batch_ids()
    skip_or_rollback()
```

#### 3. loss spike 与 reward hacking、RL 后训练崩掉有什么联系？

答：RL 后训练中的 reward hacking 会表现为 reward、KL、ratio、长度或格式指标异常，本质上也是优化目标被异常放大。

单次 spike 可以先定位 batch、保存样本 hash、观察 grad norm/NaN/overflow；如果后续能恢复，可能只需跳 batch 或裁剪梯度。如果 spike 后模型质量持续恶化，要从前一个稳定 checkpoint 回滚，并检查优化器状态。RL 后训练里 reward 分布突变、KL 约束过弱、格式奖励被 hack，都可能表现成 loss/ratio/KL 异常，因此要把 loss spike 和策略漂移一起看。

排查顺序可以固定成：数据样本 hash、loss scale/NaN/Inf、grad norm、学习率和 warmup、optimizer state、最近数据混合比例变化、验证集质量。若使用混合精度，还要看 overflow 和动态 loss scaling。

```python
if not torch.isfinite(loss) or grad_norm > threshold:
    save_bad_batch_ids()
    skip_or_rollback()
```


## 后训练与强化学习

### SFT、RLHF、DPO、PPO、GRPO、DAPO、GSPO 等后训练方法如何区分？

- 频率：301
- 来源：xhs_69fff0a6, xhs_6a0be406, xhs_6a10546e, xhs_6a0d5f2d, xhs_6a0733b1, xhs_6a05f4c4, xhs_6a004ca9, xhs_69fca9a5, xhs_69fb2bb8, xhs_69e97fab, xhs_69e18d77, xhs_69f9bacb, xhs_69f770dc, xhs_69ea8685, xhs_69e7b0f8, +286
- 难度：基础

**面试官常问：**

1. SFT、RLHF/PPO、DPO、GRPO 的数据形式和训练目标分别是什么？
2. DPO 为什么不需要显式 reward model？它的 loss 怎么写？
3. GRPO 和 PPO 最大区别是什么？为什么 GRPO 可以省 critic？
4. 什么时候选 SFT、什么时候选 DPO、什么时候需要 online RL？

后训练方法要按“数据形式、是否在线采样、是否需要 reward/critic、优化目标”来区分。

$$
\begin{aligned}
\text{SFT} &: \max_\theta \log \pi_\theta(y_{\text{good}}\mid x) \\
\text{RM} &: r_\phi(x,y_{\text{good}}) > r_\phi(x,y_{\text{bad}}) \\
\text{PPO} &: \max_\theta \mathbb{E}[R] - \beta KL(\pi_\theta\|\pi_{\text{ref}}) \\
\text{DPO} &: \text{用偏好对直接优化 policy，不显式训练 reward model} \\
\text{GRPO} &: \text{同 prompt 组内相对 reward 估计 advantage，通常不需要 critic}
\end{aligned}
$$

DPO 常见公式：

$$
\begin{aligned}
L_DPO &= -log \operatorname{sigmoid}(\beta \cdot  [ \\
log \pi_\theta(y_w|x) - log \pi_ref(y_w|x) \\
-log \pi_\theta(y_l|x) + log \pi_ref(y_l|x) \\
])
\end{aligned}
$$

PPO/GRPO 类公式核心：

$$
\begin{aligned}
r_t(\theta) &= \pi_\theta(a_t|s_t) / \pi_old(a_t|s_t) \\
L_\operatorname{clip} &= -E[min(r_t A_t, \operatorname{clip}(r_t, 1-eps, 1+eps) A_t)]
\end{aligned}
$$

方法选择：

```python
def choose_post_training(data, can_sample_online, has_reward, need_stability):
    if data == "instruction_answer":
        return "SFT"
    if data == "preference_pairs" and not can_sample_online:
        return "DPO/IPO/KTO"
    if can_sample_online and has_reward and need_stability:
        return "PPO/GRPO"
    if can_sample_online and has_reward:
        return "GRPO/DAPO family"
    return "collect better preference or rule reward data first"
```

DAPO/GSPO 等名字不要只背缩写，回答时落到它们想解决的问题：采样效率、优势估计偏差、长度偏置、MoE 负载、token-level 更新稳定性。面试中最稳的比较维度是成本、稳定性、reward 依赖、on-policy 程度和适用场景。

### 高频追问补充

#### 1. Agentic mid-training、Agentic post-training 和普通 SFT/RLHF 有什么区别？

答：Agentic mid-training 更像能力注入，通常放在预训练和后训练之间，用代码、工具轨迹、搜索轨迹、长上下文任务等数据补“会做事”的能力；Agentic post-training 更像行为对齐和任务优化，用 SFT、DPO、GRPO/RL 等方法让模型在工具环境里更可靠。

普通 SFT/RLHF 多是单轮或短对话，输入输出比较固定；Agentic 训练面对的是多步轨迹：计划、工具调用、观察、反思、重试和最终回答。因此数据结构从 `(prompt, answer)` 变成 `(state, action, observation, reward)`，评估也从答案正确率扩展到工具成功率、轨迹成本和可恢复性。

#### 2. DAPO、GSPO、GRPO 这类方法主要在改采样、优势估计还是稳定性？

答：DAPO、GSPO、GRPO 这类方法都围绕“让 RL 在大模型上更稳定、更便宜”展开，但侧重点不同。GRPO 用组内相对奖励估计 advantage，减少 critic；DAPO 更强调采样、clip、长度偏置、稳定训练等细节；GSPO 关注序列级或组级策略优化，试图让更新粒度更贴合生成任务。

回答时不要只背缩写，要按四个维度比较：是否需要 reward model/critic、是否 on-policy、advantage 怎么估计、如何控制 KL 和长度偏置。

#### 3. 偏好数据、过程数据、工具调用轨迹分别适合哪种后训练方法？

答：偏好数据适合 DPO/IPO/KTO 或 reward model 训练，因为它告诉模型两个答案哪个更好；过程数据适合 PRM、step-level reward 或 trajectory SFT，因为它能做信用分配；工具调用轨迹适合 Agent SFT、Agentic RL 或环境可验证 RL，因为它包含 action 和 observation。

面试可以举例：数学题只有最终对错时可以做 outcome reward；代码题可以跑测试得到可验证 reward；搜索 Agent 需要检查 query、点击、引用和最终答案，最好结合过程奖励。

### GRPO 的核心思想、loss、优势估计和 PPO 区别是什么？

- 频率：162
- 来源：xhs_69fff0a6, xhs_6a0be406, xhs_6a10546e, xhs_6a05f4c4, xhs_6a004ca9, xhs_69fca9a5, xhs_69f9bacb, xhs_69f770dc, xhs_69ea8685, xhs_69e7b0f8, xhs_69d37de9, xhs_69e4e580, xhs_69e4b468, xhs_69d4af50, xhs_69d5c180, +147
- 难度：进阶

**面试官常问：**

1. GRPO 的 group relative advantage 怎么算？
2. GRPO 的 loss 和 PPO clipping 有什么关系？
3. 为什么 reward 全 0/全 1 时 GRPO 学不动？
4. GRPO 训练中长度偏置和 KL 约束怎么处理？

GRPO 的关键是“同一 prompt 采样一组回答，用组内相对好坏估计 advantage”。设 prompt 为 `x`，采样 `G` 个回答 `y_1...y_G`，得到 reward `r_i`。组内优势可以写成：

$$
A_i = (r_i - \operatorname{mean}(r_1...r_G)) / (\operatorname{std}(r_1...r_G) + eps)
$$

然后对回答中的 token 做策略梯度更新，并加 ratio clipping 和 KL 约束：

$$
\begin{aligned}
\rho_t &= \pi_\theta(a_t|s_t) / \pi_old(a_t|s_t) \\
L &= - E[ \min(\rho_t A, \operatorname{clip}(\rho_t, 1-eps, 1+eps) A) ] \\
+ \beta \cdot  KL(\pi_\theta || \pi_ref)
\end{aligned}
$$

和 PPO 的主要区别：PPO 通常需要 critic/value model 估计 `V(s)`，优势是 `R - V(s)` 或 GAE；GRPO 用组内相对 reward，不训练 critic，显存更省、实现更简单。代价是每个 prompt 要多采样几个回答，且 reward 方差、长度偏置、组内样本多样性非常关键。

优势估计代码：

```python
import torch

def group_advantages(rewards, group_size, eps=1e-6):
    # rewards: [num_prompts * group_size]
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True, unbiased=False)
    adv = (r - mean) / (std + eps)
    return adv.reshape(-1)
```

常见追问：为什么 reward 全 0/全 1 会学不动？因为组内方差为 0，advantage 近似全 0；为什么长度会影响？长回答 token 更多，loss 累加方式若不归一化，长回答梯度权重可能更大。

### 高频追问补充

#### 1. GRPO 为什么可以不要 critic？代价是什么？

答：GRPO 不要 critic 的关键是“同一 prompt 采样一组回答”。这组回答共享题目，奖励可以组内归一化：

$$
A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r)+\epsilon}
$$

这样组平均就充当 baseline，避免训练 value model。代价是每个 prompt 要采多条，采样成本变高；如果组内 reward 都一样，advantage 接近 0，学不动；如果 reward 噪声大，组内排序也会不稳定。

#### 2. group size、reward std、长度归一化会怎样影响 advantage？

答：group size 越大，组内 baseline 和标准差越稳定，但 rollout 成本越高。reward std 太小，advantage 会被噪声主导或接近 0；reward std 太大，更新方差大。长度归一化也很关键，因为长回答 token 更多，如果 loss 按 token 累加，长答案可能获得更大梯度权重。

工程上要同时监控 reward mean/std、response length、KL、clip fraction 和 pass rate，不能只看 reward 均值。

#### 3. GRPO 和 DPO 都不用显式 value model，它们本质差别在哪里？

答：DPO 是离线偏好优化，它直接使用 chosen/rejected 对，把偏好学习写成分类式目标；GRPO 是从当前策略采样多条回答，根据 reward 做策略梯度式更新。二者都可以不训练 value model，但数据闭环不同。

DPO 成本低、稳定、适合已有偏好数据；GRPO 更适合有环境反馈、规则验证或在线采样的任务，比如数学、代码、工具 Agent，但训练更敏感。

### 为什么常说 SFT 更像正向 KL，RL/偏好优化更像反向 KL？

- 频率：91
- 来源：xhs_69fff0a6, xhs_6a0aa47b, xhs_6a019beb, xhs_69fca9a5, xhs_69ec3021, xhs_69f4db22, xhs_69e4e580, xhs_69d5c180, xhs_69e2cfc5, xhs_69de541d, xhs_69de2e4c, xhs_69dde219, xhs_69d8c35c, xhs_69c389db, xhs_69adb8c8, +76
- 难度：进阶

**面试官常问：**

1. 为什么常说 SFT 更像正向 KL，RL 更像反向 KL？
2. 正向 KL 和反向 KL 在 mode-covering/mode-seeking 上有什么差异？
3. SFT 和 RL 在输出多样性、保守性、奖励偏好上会有什么不同？
4. 这个 KL 类比有哪些前提和局限？

这题本质是在比较 distribution matching 的方向。KL 定义：

$$
KL(P || Q) = \sum_x P(x) \log(P(x) / Q(x))
$$

如果把数据分布当 `P_data`，模型当 `P_theta`，SFT 最大似然：

$$
\begin{aligned}
max sum_{(x,y) in D} log P_\theta(y|x) \\
等价于最小化 KL(P_data || P_\theta) 的经验近似
\end{aligned}
$$

这个方向叫 forward KL，特点是 mode-covering：数据里出现过的模式都希望模型覆盖，否则 `P_data>0, P_theta≈0` 会受到很大惩罚。RL/偏好优化更像 mode-seeking：策略从自己采样的输出里强化高 reward 样本，同时用 KL 约束不偏离参考模型：

$$
max E_{y ~ \pi_\theta(.|x)} [ r(x,y) ] - \beta \cdot  KL(\pi_\theta(.|x) || \pi_ref(.|x))
$$

它倾向把概率集中到高奖励区域，低奖励模式即使在 SFT 数据里出现过，也可能被压低。因此常说 SFT 更像“模仿人类数据分布”，RL 更像“在可接受范围内寻找高 reward 输出”。

一个直观小例子：

```python
# 数据有两个答案模式 A/B，各 50%。SFT 会希望模型都能覆盖。
data = {"A": 0.5, "B": 0.5}

# reward 只偏好 A。RL 会把概率往 A 推，但 KL beta 会限制推多远。
reward = {"A": 1.0, "B": 0.0}
```

注意：这是面试直觉，不是所有 SFT/RL 目标都严格等同某个 KL 方向。DPO、PPO、GRPO 的具体损失、采样策略、reference KL、温度都会影响行为。

### 高频追问补充

#### 1. 为什么 SFT 后模型会更“平均”，RL 后更容易变得尖锐？

答：SFT 学数据集中多种可接受答案，行为上更覆盖模式；RL/偏好优化把概率推向高奖励答案，更容易让分布变尖。

SFT 用数据集里的标准答案做行为克隆，倾向覆盖数据分布中的多种写法；偏好优化/RL 会把概率质量推向高奖励答案，若约束太弱就会过度集中、风格变窄甚至 reward hacking。DPO/PPO/GRPO 都要处理“提高奖励”和“别偏离参考模型”的张力，beta/KL 系数越强，模型越保守；越弱，越容易学到偏好但也越可能漂移。

两个 KL 方向的直觉：

$$
D_{KL}(p_{data}\|p_\theta)
$$

更怕漏掉数据模式，偏 mode-covering；

$$
D_{KL}(p_\theta\|p_{ref})
$$

在偏好/RL 约束中更强调别偏离参考策略。DPO 常见目标里 beta 控制偏离强度，beta 越大越保守，越小越激进。

#### 2. 正向 KL/反向 KL 的 mode-covering 和 mode-seeking 在回答质量上怎么体现？

答：mode-covering 体现为回答风格更平均、更保守；mode-seeking 体现为高奖励模式更强，但也更容易丢多样性。

SFT 用数据集里的标准答案做行为克隆，倾向覆盖数据分布中的多种写法；偏好优化/RL 会把概率质量推向高奖励答案，若约束太弱就会过度集中、风格变窄甚至 reward hacking。DPO/PPO/GRPO 都要处理“提高奖励”和“别偏离参考模型”的张力，beta/KL 系数越强，模型越保守；越弱，越容易学到偏好但也越可能漂移。

两个 KL 方向的直觉：

$$
D_{KL}(p_{data}\|p_\theta)
$$

更怕漏掉数据模式，偏 mode-covering；

$$
D_{KL}(p_\theta\|p_{ref})
$$

在偏好/RL 约束中更强调别偏离参考策略。DPO 常见目标里 beta 控制偏离强度，beta 越大越保守，越小越激进。

#### 3. DPO 里的 beta 或 KL 强度如何影响模型偏离 reference？

答：DPO 的 beta 越大，参考模型约束越强，更新越保守；beta 越小，偏好学习更激进，也更容易偏离 reference。

SFT 用数据集里的标准答案做行为克隆，倾向覆盖数据分布中的多种写法；偏好优化/RL 会把概率质量推向高奖励答案，若约束太弱就会过度集中、风格变窄甚至 reward hacking。DPO/PPO/GRPO 都要处理“提高奖励”和“别偏离参考模型”的张力，beta/KL 系数越强，模型越保守；越弱，越容易学到偏好但也越可能漂移。

两个 KL 方向的直觉：

$$
D_{KL}(p_{data}\|p_\theta)
$$

更怕漏掉数据模式，偏 mode-covering；

$$
D_{KL}(p_\theta\|p_{ref})
$$

在偏好/RL 约束中更强调别偏离参考策略。DPO 常见目标里 beta 控制偏离强度，beta 越大越保守，越小越激进。

### Agentic RL、自蒸馏和自进化 Agent 训练范式是什么？

- 频率：67
- 来源：xhs_6a0fb45d, xhs_69fe0629, xhs_69fb2bb8, xhs_69e97fab, xhs_69e18d77, xhs_69ea8685, xhs_69d4f398, xhs_69cc6f17, xhs_69adb8c8, xhs_68ff77f4, xhs_68f0ecf6, xhs_6a10562e, xhs_6a108186, xhs_6a12beac, xhs_6a02de7a, +52
- 难度：进阶

**面试官常问：**

1. Agentic RL 和普通 RLHF/GRPO 的区别是什么？
2. 自蒸馏、自进化 Agent、Search-E1、SDAR、OPD 这类方法在解决什么问题？
3. Agent 轨迹怎么产生、筛选和回流训练？
4. 如何避免自我训练越训越偏？

Agentic RL 的对象不是单轮回答，而是多步轨迹：

$$
\tau = (s_0, a_0, o_0, s_1, a_1, o_1, ..., y)
$$

其中 `a_t` 可能是自然语言、工具调用或环境动作，`o_t` 是工具/环境反馈。优化目标可以写成：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[R_{final}(\tau)+\sum_t \gamma^t R_{step}(s_t,a_t,o_t)-\beta KL(\pi_\theta||\pi_{ref})\right]
$$

自蒸馏/自进化的常见闭环是：当前模型采样轨迹，环境或规则验证结果，教师模型/奖励模型打分，筛出成功轨迹，再做 SFT、DPO 或 RL 更新。OPD/在线策略蒸馏强调学生在自己分布上采样，避免固定教师数据和学生部署分布不一致。

```python
for task in tasks:
    traces = [agent.rollout(task, temperature=0.8) for _ in range(k)]
    scored = [(trace, env.verify(trace), judge.score(trace)) for trace in traces]
    good = [t for t, ok, score in scored if ok and score > threshold]
    replay_buffer.extend(good)

train_on(replay_buffer, objective="sft_or_rl")
```

风险是 reward hacking、环境过窄、错误轨迹自我强化、样本多样性下降。稳定做法是保留 reference eval、人工抽检、失败样本池、KL 约束和跨任务 holdout。

### 高频追问补充

#### 1. Agentic RL 和普通 RLHF 最大区别是什么？

答：Agentic RL 的对象是多步轨迹，普通 RLHF 多优化单轮回答偏好；前者更依赖环境反馈、工具结果和过程奖励。

Agentic RL 把模型放进多步工具环境里优化轨迹，不只是优化单轮答案偏好。自蒸馏通常让当前模型采样轨迹，再用规则、环境、教师或人工筛选成功样本回流训练；核心风险是错误自我强化，所以必须有可验证任务、失败轨迹过滤、难例再采样和 holdout eval。教师反馈适合开放质量判断，环境反馈适合代码、搜索、工具任务等可验证场景，两者可以组合成 outcome reward + process reward。

Agentic RL 轨迹可写成：

$$
\tau=(s_0,a_0,o_0,\ldots,s_T,a_T),\quad R(\tau)=R_{outcome}+\sum_i R_{process,i}
$$

自蒸馏必须把成功轨迹、失败轨迹和不确定轨迹分开处理，避免把错误答案当新教师。

#### 2. 自蒸馏/自进化 Agent 如何避免错误轨迹越训越多？

答：自蒸馏要避免错误轨迹回流，必须用规则验证、教师复核、失败样本过滤和独立 holdout 集。

Agentic RL 把模型放进多步工具环境里优化轨迹，不只是优化单轮答案偏好。自蒸馏通常让当前模型采样轨迹，再用规则、环境、教师或人工筛选成功样本回流训练；核心风险是错误自我强化，所以必须有可验证任务、失败轨迹过滤、难例再采样和 holdout eval。教师反馈适合开放质量判断，环境反馈适合代码、搜索、工具任务等可验证场景，两者可以组合成 outcome reward + process reward。

Agentic RL 轨迹可写成：

$$
\tau=(s_0,a_0,o_0,\ldots,s_T,a_T),\quad R(\tau)=R_{outcome}+\sum_i R_{process,i}
$$

自蒸馏必须把成功轨迹、失败轨迹和不确定轨迹分开处理，避免把错误答案当新教师。

#### 3. 在线策略蒸馏、环境反馈和教师模型反馈如何组合？

答：教师模型适合评价开放质量，环境反馈适合可验证任务，在线策略蒸馏适合让学生覆盖自己的部署分布。

Agentic RL 把模型放进多步工具环境里优化轨迹，不只是优化单轮答案偏好。自蒸馏通常让当前模型采样轨迹，再用规则、环境、教师或人工筛选成功样本回流训练；核心风险是错误自我强化，所以必须有可验证任务、失败轨迹过滤、难例再采样和 holdout eval。教师反馈适合开放质量判断，环境反馈适合代码、搜索、工具任务等可验证场景，两者可以组合成 outcome reward + process reward。

Agentic RL 轨迹可写成：

$$
\tau=(s_0,a_0,o_0,\ldots,s_T,a_T),\quad R(\tau)=R_{outcome}+\sum_i R_{process,i}
$$

自蒸馏必须把成功轨迹、失败轨迹和不确定轨迹分开处理，避免把错误答案当新教师。

### Agentic RL 中过程奖励、PRM 和 token-level advantage 怎么设计？

- 频率：42
- 来源：xhs_6a0be406, xhs_6a0fb45d, xhs_69e7b0f8, xhs_69de2e4c, xhs_69adb8c8, xhs_6988310c, xhs_68d4fe24, xhs_68b98855, xhs_6858d751, xhs_689eeb65, xhs_6894a2fe, xhs_68159779, xhs_678e5e6b, xhs_6a10562e, xhs_6a0a890d, +27
- 难度：进阶

**面试官常问：**

1. Agentic RL 里最终奖励和过程奖励有什么区别？
2. PRM 应该给哪些中间步骤打分？如何避免 reward hacking？
3. token-level advantage 如何从 step reward 或 trajectory reward 分配？
4. 工具调用型 Agent 的 reward 应该包含哪些维度？

Agentic RL 的难点是信用分配。最终 reward 只告诉“整条轨迹是否成功”，但轨迹包含规划、工具调用、观察、反思和最终回答。可以把总 reward 写成：

$$
R = w_task R_task + w_format R_format + w_tool R_tool + w_safe R_safe - w_cost Cost
$$

过程奖励 PRM 给中间步骤打分，例如每次工具调用参数是否正确、检索证据是否支持结论、代码执行是否通过测试。Token-level advantage 则要把 step 或 trajectory reward 分配回 token：

$$
\begin{aligned}
A_token_t &= A_step_j, if token t belongs to step j \\
或 A_token_t &= discounted_future_reward_t - baseline_t
\end{aligned}
$$

一个简单分配方式：

```python
def assign_step_reward_to_tokens(step_rewards, step_token_spans, total_tokens):
    adv = [0.0] * total_tokens
    for reward, (lo, hi) in zip(step_rewards, step_token_spans):
        length = \max(hi - lo, 1)
        for t in range(lo, hi):
            adv[t] = reward / length
    return adv
```

设计 PRM 时要防 reward hacking：如果只奖励“调用了工具”，模型会乱调用；如果只奖励“答案长”，模型会变啰嗦。实践中要多维奖励、约束工具白名单、失败轨迹采样、人工抽检，并把 reward model 的输入限定为可验证证据。

### 高频追问补充

#### 1. PRM 和 ORM 的区别是什么？为什么 Agent 任务更需要过程奖励？

答：ORM 只评价最终答案，PRM 评价中间步骤；Agent 多步任务需要 PRM 来判断计划、检索、工具调用和反思哪一步贡献了成败。

ORM 只评价最终答案，PRM 评价中间步骤。Agent 任务包含搜索、工具调用、反思、重试，最终对错很难告诉哪一步贡献最大，所以需要过程奖励做信用分配。token-level advantage 可以按步骤边界均摊，也可以用价值模型/奖励模型逐步估计；同时要对长度、重复、无效工具调用加惩罚，否则模型可能通过堆步骤拿过程分。

把 step reward 分配到 token 的一个简单方式：

$$
A_t = R_{step(i)},\quad t\in tokens(step_i)
$$

更细的方法是训练 value/reward model 估计每个状态或 token 的优势。PRM 要配合最终结果校验，否则会奖励“看起来努力”的长轨迹。

#### 2. token-level advantage 如何从 step-level reward 分配回来？

答：step-level reward 可以按步骤边界均摊给该步骤 token，也可以训练 value/reward model 给每个 token 或 span 估计 advantage。

ORM 只评价最终答案，PRM 评价中间步骤。Agent 任务包含搜索、工具调用、反思、重试，最终对错很难告诉哪一步贡献最大，所以需要过程奖励做信用分配。token-level advantage 可以按步骤边界均摊，也可以用价值模型/奖励模型逐步估计；同时要对长度、重复、无效工具调用加惩罚，否则模型可能通过堆步骤拿过程分。

把 step reward 分配到 token 的一个简单方式：

$$
A_t = R_{step(i)},\quad t\in tokens(step_i)
$$

更细的方法是训练 value/reward model 估计每个状态或 token 的优势。PRM 要配合最终结果校验，否则会奖励“看起来努力”的长轨迹。

#### 3. 如何避免过程奖励鼓励模型写更多无用步骤？

答：过程奖励要惩罚无效步骤、重复搜索、无意义反思和过长轨迹，否则模型会学会堆步骤而不是解决任务。

ORM 只评价最终答案，PRM 评价中间步骤。Agent 任务包含搜索、工具调用、反思、重试，最终对错很难告诉哪一步贡献最大，所以需要过程奖励做信用分配。token-level advantage 可以按步骤边界均摊，也可以用价值模型/奖励模型逐步估计；同时要对长度、重复、无效工具调用加惩罚，否则模型可能通过堆步骤拿过程分。

把 step reward 分配到 token 的一个简单方式：

$$
A_t = R_{step(i)},\quad t\in tokens(step_i)
$$

更细的方法是训练 value/reward model 估计每个状态或 token 的优势。PRM 要配合最终结果校验，否则会奖励“看起来努力”的长轨迹。

### 如何判断 SFT 到什么程度可以开始 RL？SFT 后复读或效果差怎么办？

- 频率：17
- 来源：xhs_69fca9a5, xhs_69e4e580, xhs_69d5c180, xhs_69e2cfc5, xhs_69de541d, xhs_69dde219, xhs_68a598ff, xhs_682ab42c, xhs_67af2f3e, xhs_6a0d9b47, xhs_69c407c0, xhs_69c8d73c, xhs_6a09a480, xhs_69ddfdee, xhs_69b02ec5, +2
- 难度：进阶

**面试官常问：**

1. 怎么判断 SFT 模型已经可以开始做 RL？
2. SFT 后模型复读、格式错、泛化差分别怎么排查？
3. 为什么坏的 SFT policy 直接做 RL 容易放大奖励噪声？
4. RL 前验证集应该看哪些指标？

SFT 到 RL 的门槛不是 loss 降到某个绝对值，而是模型已经具备“可被 reward 优化”的基础行为。最低标准：

$$
\begin{aligned}
format_acc \ge  目标阈值 \\
instruction_following 稳定 \\
基础任务 pass_rate 非零且有方差 \\
采样输出有多样性 \\
bad_case 可由 reward 区分
\end{aligned}
$$

如果 SFT 模型连格式都不稳定，RL 会把 reward 噪声放大；如果输出几乎全错，组内 reward 没区分度；如果输出完全模式塌缩，探索不够。

一个上线前检查：

```python
def ready_for_rl(metrics):
    return (
        metrics["format_acc"] > 0.95
        and metrics["base_pass_rate"] > 0.15
        and metrics["reward_std"] > 0.05
        and metrics["avg_len"] < metrics["max_len_budget"]
        and metrics["dup_rate"] < 0.30
    )
```

SFT 后复读常见原因：数据重复、模板太强、训练太久、学习率太高、只有正样本没有反例、prompt/answer 边界 mask 错。处理策略：去重、降低 LR、早停、混入通用指令数据、加入偏好/负例、检查 label mask。面试里可以补一句：RL 不是修复坏 SFT 的万能药，好的 RL 往往建立在可用的 SFT policy 上。

### 高频追问补充

#### 1. SFT 没做好就上 RL 会发生什么？

答：SFT 没做好就上 RL，会在低质量策略附近探索，reward 噪声会被放大，常见结果是格式投机、复读、KL 飙升或 pass rate 不稳。

SFT 是 RL 的地基。如果模型连格式、任务理解、基础正确率都不稳，RL 会放大奖励噪声，甚至学会投机。开始 RL 前至少要看：验证集格式准确率、基础任务 pass rate、采样多样性、拒答率、工具调用成功率。复读常来自模板过强、重复数据多、学习率过大或训练过久，处理方式是去重、混入通用指令数据、降低学习率、早停和增加负例。

开始 RL 的最低门槛可以写成四个 gate：格式准确率、基础正确率、采样多样性、reward 可区分性。若 reward 对大多数样本全 0 或全 1，RL 很难学。

常见修复：去重、减少模板、混入通用指令数据、降低学习率、缩短训练轮数、加入 chosen/rejected 负例。

#### 2. 如何判断模型已经具备开始 RL 的 base capability？

答：开始 RL 前至少要确认格式准确率、基础任务正确率、采样多样性和拒答率过线；Agent 任务还要看工具调用成功率。

SFT 是 RL 的地基。如果模型连格式、任务理解、基础正确率都不稳，RL 会放大奖励噪声，甚至学会投机。开始 RL 前至少要看：验证集格式准确率、基础任务 pass rate、采样多样性、拒答率、工具调用成功率。复读常来自模板过强、重复数据多、学习率过大或训练过久，处理方式是去重、混入通用指令数据、降低学习率、早停和增加负例。

开始 RL 的最低门槛可以写成四个 gate：格式准确率、基础正确率、采样多样性、reward 可区分性。若 reward 对大多数样本全 0 或全 1，RL 很难学。

常见修复：去重、减少模板、混入通用指令数据、降低学习率、缩短训练轮数、加入 chosen/rejected 负例。

#### 3. SFT 后模型复读 prompt、格式僵硬或拒答变多怎么处理？

答：复读 prompt、格式僵硬或拒答变多，通常来自模板过强、重复数据多、学习率大或训练过久，要做去重、混入通用数据、降学习率和早停。

SFT 是 RL 的地基。如果模型连格式、任务理解、基础正确率都不稳，RL 会放大奖励噪声，甚至学会投机。开始 RL 前至少要看：验证集格式准确率、基础任务 pass rate、采样多样性、拒答率、工具调用成功率。复读常来自模板过强、重复数据多、学习率过大或训练过久，处理方式是去重、混入通用指令数据、降低学习率、早停和增加负例。

开始 RL 的最低门槛可以写成四个 gate：格式准确率、基础正确率、采样多样性、reward 可区分性。若 reward 对大多数样本全 0 或全 1，RL 很难学。

常见修复：去重、减少模板、混入通用指令数据、降低学习率、缩短训练轮数、加入 chosen/rejected 负例。

### 训练 GRPO/Agentic RL 时应该监控哪些指标？

- 频率：5
- 来源：xhs_69fca9a5, xhs_69ea8685, xhs_69e7b0f8, xhs_69c389db, xhs_69eafcb5
- 难度：进阶

**面试官常问：**

1. GRPO 训练时必须监控哪些 reward、policy 和质量指标？
2. clip fraction、KL、entropy 分别代表什么异常信号？
3. 为什么 group reward std 很重要？
4. 如何判断模型是真的变强，而不是 reward hacking？

GRPO 训练要看三层指标：reward 是否有学习信号、policy 是否漂移、生成质量是否真实提升。

Reward 层：

$$
\begin{aligned}
reward_\operatorname{mean}, reward_\operatorname{std}, reward_min/max \\
group_\operatorname{std} &= \operatorname{std}(rewards within same prompt) \\
pass_rate &= \operatorname{mean}(rule_pass) \\
format_acc &= \operatorname{mean}(format_ok)
\end{aligned}
$$

Policy 层：

$$
\begin{aligned}
KL(\pi_\theta || \pi_ref) \\
ratio &= \exp(logprob_new - logprob_old) \\
\operatorname{clip}_fraction &= \operatorname{mean}(abs(ratio - 1) > eps) \\
entropy &= -sum p log p \\
avg_response_len
\end{aligned}
$$

任务质量层：pass@k、人工胜率、tool success、答案正确率、安全拒答率、线上成本和延迟。一个监控函数：

```python
import torch

def grpo_stats(reward, logp_new, logp_old, kl, group_size, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)
    grouped = reward.view(-1, group_size)
    return {
        "reward_mean": reward.mean().item(),
        "reward_std": reward.std(unbiased=False).item(),
        "group_reward_std": grouped.std(dim=1, unbiased=False).mean().item(),
        "kl": kl.mean().item(),
        "clip_fraction": ((ratio - 1).abs() > eps).float().mean().item(),
        "ratio_mean": ratio.mean().item(),
    }
```

判断异常：`reward_std` 很低说明没区分度；`KL` 突然变大说明模型漂移；`clip_fraction` 长期过高说明学习率或 reward 尺度过猛；平均长度持续增加可能是 reward 偏好啰嗦答案。

### 高频追问补充

#### 1. 训练 Agentic RL 时只看 reward mean 为什么不够？

答：只看 reward mean 不够，因为 reward 上升可能来自输出变长、格式投机或奖励漏洞，必须同时看 reward std、KL、长度和真实 pass rate。

reward mean 上升可能来自输出变长、格式投机或奖励漏洞，所以必须同时看 reward std、KL、entropy、clip fraction、长度、格式准确率、工具成功率和人工抽检。KL 飙升说明策略偏离 reference；entropy 快速下降说明模式塌缩；clip fraction 过高说明更新过猛；工具成功率高但最终答案差，通常是规划/信息整合问题，不是单纯工具调用问题。

GRPO 组内 advantage 常写成：

$$
A_i=\frac{r_i-\mu(r)}{\sigma(r)+\epsilon}
$$

因此 reward std 过低会让优势失效。监控表至少包含 `reward_mean/std`、`kl`、`entropy`、`clip_fraction`、`response_len`、`format_acc`、`tool_success`、`pass_rate`。

#### 2. clip fraction、KL、entropy、response length 分别异常说明什么？

答：clip fraction 高说明更新过猛，KL 高说明偏离 reference，entropy 快速下降说明模式塌缩，response length 异常常提示长度奖励或格式奖励被利用。

reward mean 上升可能来自输出变长、格式投机或奖励漏洞，所以必须同时看 reward std、KL、entropy、clip fraction、长度、格式准确率、工具成功率和人工抽检。KL 飙升说明策略偏离 reference；entropy 快速下降说明模式塌缩；clip fraction 过高说明更新过猛；工具成功率高但最终答案差，通常是规划/信息整合问题，不是单纯工具调用问题。

GRPO 组内 advantage 常写成：

$$
A_i=\frac{r_i-\mu(r)}{\sigma(r)+\epsilon}
$$

因此 reward std 过低会让优势失效。监控表至少包含 `reward_mean/std`、`kl`、`entropy`、`clip_fraction`、`response_len`、`format_acc`、`tool_success`、`pass_rate`。

#### 3. tool success rate 和 final answer pass rate 冲突时听谁的？

答：tool success rate 只说明工具调用链路跑通，final answer pass rate 才说明任务完成；二者冲突时要回放轨迹看是规划、检索、工具参数还是信息整合出错。

reward mean 上升可能来自输出变长、格式投机或奖励漏洞，所以必须同时看 reward std、KL、entropy、clip fraction、长度、格式准确率、工具成功率和人工抽检。KL 飙升说明策略偏离 reference；entropy 快速下降说明模式塌缩；clip fraction 过高说明更新过猛；工具成功率高但最终答案差，通常是规划/信息整合问题，不是单纯工具调用问题。

GRPO 组内 advantage 常写成：

$$
A_i=\frac{r_i-\mu(r)}{\sigma(r)+\epsilon}
$$

因此 reward std 过低会让优势失效。监控表至少包含 `reward_mean/std`、`kl`、`entropy`、`clip_fraction`、`response_len`、`format_acc`、`tool_success`、`pass_rate`。


## MoE 与稀疏模型

### MoE 的原理、Dense 对比、路由和负载均衡难点是什么？

- 频率：116
- 来源：xhs_6a0be406, xhs_6a11a95f, xhs_6a0806ae, xhs_6a0733b1, xhs_69f770dc, xhs_69d5c180, xhs_69ddf065, xhs_69a06a56, xhs_698878d1, xhs_69962665, xhs_6904cfb9, xhs_690415f6, xhs_68d8e3cf, xhs_68deb9f3, xhs_68d78d54, +101
- 难度：进阶

**面试官常问：**

1. MoE 和 Dense FFN 的核心区别是什么？为什么 MoE 能扩大总参数？
2. Top-k router、expert、capacity factor 分别是什么？
3. MoE 的负载均衡 loss 为什么重要？
4. MoE 的通信、token dropping、专家塌缩问题怎么理解？

MoE 的核心是条件计算：总参数很多，但每个 token 只激活少数专家。设 router 输出：

$$
\begin{aligned}
g &= \operatorname{softmax}(x W_r) \\
\operatorname{TopK} &= \operatorname{topk}(g) \\
y &= sum_{e in \operatorname{TopK}} g_e \cdot  Expert_e(x)
\end{aligned}
$$

Dense FFN 每个 token 都过同一套 FFN；MoE 把 FFN 换成多个 expert，并由 router 分配 token。优点是容量大、单位 token FLOPs 可控；难点是路由不均、通信开销、专家塌缩、token dropping、训练稳定性。

简化实现：

```python
import torch

def moe_forward(x, router, experts, top_k=2):
    # x: [T, D]
    probs = torch.softmax(router(x), dim=-1)       # [T, E]
    weight, idx = torch.topk(probs, k=top_k, dim=-1)
    out = torch.zeros_like(x)
    for k in range(top_k):
        for e, expert in enumerate(experts):
            mask = idx[:, k] == e
            if mask.any():
                out[mask] += weight[mask, k:k+1] * expert(x[mask])
    return out
```

负载均衡常用思想是让不同专家接收 token 比例接近均匀：

$$
L_balance \approx E \cdot  sum_e fraction_tokens_e \cdot  \operatorname{mean}_router_prob_e
$$

面试中要强调：MoE 不是免费午餐。它降低的是“每 token 激活计算”，但引入了 all-to-all 通信、路由训练、容量因子、专家并行部署和显存放置问题。

### 高频追问补充

#### 1. MoE 为什么总参数多但单 token FLOPs 不一定多？

答：MoE 总参数大是因为有很多专家，但每个 token 只激活 top-k 专家，所以单 token FLOPs 不一定随总参数线性增长。

MoE 每个 token 只激活少量专家，所以参数容量变大而计算量受控。shared expert 提供通用能力，细粒度专家提高专家组合灵活性，top-k routing 在表达和成本之间折中。系统瓶颈在 all-to-all 通信、专家负载不均、token dropping 和跨卡调度；所以除了模型效果，还要监控每专家 token 数、routing entropy、通信时间和容量因子。

MoE 单 token 计算近似由激活专家决定：

$$
FLOPs_{token}\approx FLOPs_{attn}+k\cdot FLOPs_{expert}
$$

不是由总专家数直接决定。负载均衡常看每个专家 token 数、routing entropy、capacity overflow、token drop 和 all-to-all 时间。

#### 2. shared expert、fine-grained expert、top-k routing 各解决什么问题？

答：shared expert 提供通用能力，fine-grained expert 提高组合粒度，top-k routing 在表达能力、计算量和通信之间折中。

MoE 每个 token 只激活少量专家，所以参数容量变大而计算量受控。shared expert 提供通用能力，细粒度专家提高专家组合灵活性，top-k routing 在表达和成本之间折中。系统瓶颈在 all-to-all 通信、专家负载不均、token dropping 和跨卡调度；所以除了模型效果，还要监控每专家 token 数、routing entropy、通信时间和容量因子。

MoE 单 token 计算近似由激活专家决定：

$$
FLOPs_{token}\approx FLOPs_{attn}+k\cdot FLOPs_{expert}
$$

不是由总专家数直接决定。负载均衡常看每个专家 token 数、routing entropy、capacity overflow、token drop 和 all-to-all 时间。

#### 3. MoE 推理为什么常被通信和负载均衡卡住？

答：MoE 推理常被 all-to-all 通信、专家负载不均、capacity overflow 和 token dropping 卡住，系统指标和模型指标要一起看。

MoE 每个 token 只激活少量专家，所以参数容量变大而计算量受控。shared expert 提供通用能力，细粒度专家提高专家组合灵活性，top-k routing 在表达和成本之间折中。系统瓶颈在 all-to-all 通信、专家负载不均、token dropping 和跨卡调度；所以除了模型效果，还要监控每专家 token 数、routing entropy、通信时间和容量因子。

MoE 单 token 计算近似由激活专家决定：

$$
FLOPs_{token}\approx FLOPs_{attn}+k\cdot FLOPs_{expert}
$$

不是由总专家数直接决定。负载均衡常看每个专家 token 数、routing entropy、capacity overflow、token drop 和 all-to-all 时间。

### 为什么 DAPO/GRPO 等 RL 方法在 MoE 上容易踩坑？

- 频率：40
- 来源：xhs_6a0be406, xhs_6a0733b1, xhs_6a05f4c4, xhs_69f770dc, xhs_69d5c180, xhs_69de2e4c, xhs_69ddf065, xhs_69a240a0, xhs_69a06a56, xhs_6904cfb9, xhs_68d78d54, xhs_68d4fe24, xhs_68c40708, xhs_68bbc021, xhs_68b64f44, +25
- 难度：进阶

**面试官常问：**

1. 为什么 RL 后训练会让 MoE 更容易出现专家负载不均？
2. MoE + GRPO/DAPO 需要额外监控哪些路由指标？
3. 长度偏置和 reward hacking 如何影响专家分配？
4. 如何缓解 RL 阶段的 MoE 路由塌缩？

RL 会改变 token 分布、回答长度和策略偏好，这些都会影响 MoE router。常见坑：

1. 高 reward 样本集中走少数专家，导致专家利用率塌缩。
2. 组内采样带来更大方差，某些专家梯度更新非常不均。
3. 长答案获得更多 token-level loss 权重，专家负载随长度偏置变化。
4. reward hacking 会把某类模板 token 放大，router 也跟着偏。

要监控：

$$
\begin{aligned}
\expert_load_e &= tokens_routed_to_e / total_tokens \\
router_entropy &= -sum_e p_e log p_e \\
drop_rate &= dropped_tokens / total_tokens \\
\expert_grad_norm_e
\end{aligned}
$$

代码片段：

```python
import torch

def moe_routing_stats(router_probs, top1_idx, num_experts):
    load = torch.bincount(top1_idx.flatten(), minlength=num_experts).float()
    load = load / load.sum().clamp_min(1)
    entropy = -(router_probs * router_probs.clamp_min(1e-9).\log()).sum(dim=-1).mean()
    return {
        "load_min": load.min().item(),
        "load_max": load.max().item(),
        "load_cv": (load.std() / load.mean().clamp_min(1e-9)).item(),
        "router_entropy": entropy.item(),
    }
```

缓解方式：保留 MoE load-balance loss、控制 KL、做长度归一化、限制 reward 尺度、监控每专家梯度，必要时冻结部分 router 或降低 RL 学习率。

### 高频追问补充

#### 1. 为什么 RL 会让 MoE 专家利用率变差？

答：RL 会改变输出 token 分布，也改变路由器看到的数据分布，所以专家利用率可能向少数高奖励路径集中。

RL 改变输出分布，也改变 token 到专家的路由分布。如果高奖励样本集中在少数模式，路由器可能越来越偏向少数专家，造成专家塌缩、负载不均和泛化下降。训练时要同时看 reward/KL 和专家利用率，必要时加负载均衡损失、路由正则、容量控制，或者对采样数据做多样性约束。

路由熵可作为专家利用多样性的信号：

$$
H(router)=-\sum_e p(e)\log p(e)
$$

RL 后如果 reward 上升但 routing entropy 下降、少数专家 token 数暴涨，就要警惕专家塌缩。

#### 2. GRPO/DAPO 在 MoE 上为什么要特别关注 routing entropy？

答：routing entropy 下降说明专家选择变窄，可能出现专家塌缩；GRPO/DAPO 的采样和 reward 方差会放大这种风险。

RL 改变输出分布，也改变 token 到专家的路由分布。如果高奖励样本集中在少数模式，路由器可能越来越偏向少数专家，造成专家塌缩、负载不均和泛化下降。训练时要同时看 reward/KL 和专家利用率，必要时加负载均衡损失、路由正则、容量控制，或者对采样数据做多样性约束。

路由熵可作为专家利用多样性的信号：

$$
H(router)=-\sum_e p(e)\log p(e)
$$

RL 后如果 reward 上升但 routing entropy 下降、少数专家 token 数暴涨，就要警惕专家塌缩。

#### 3. reward 偏向某类输出时，是否会把梯度集中到少数专家？

答：如果 reward 偏向某类输出，梯度会更多流向处理这类 token 的专家，需要负载均衡、路由正则和数据多样性约束。

RL 改变输出分布，也改变 token 到专家的路由分布。如果高奖励样本集中在少数模式，路由器可能越来越偏向少数专家，造成专家塌缩、负载不均和泛化下降。训练时要同时看 reward/KL 和专家利用率，必要时加负载均衡损失、路由正则、容量控制，或者对采样数据做多样性约束。

路由熵可作为专家利用多样性的信号：

$$
H(router)=-\sum_e p(e)\log p(e)
$$

RL 后如果 reward 上升但 routing entropy 下降、少数专家 token 数暴涨，就要警惕专家塌缩。


## 推理服务与性能

### BF16/FP8/INT8 量化和反量化的核心流程是什么？

- 频率：102
- 来源：xhs_6a0806ae, xhs_69f770dc, xhs_69e2cfc5, xhs_69ddf065, xhs_69408c32, xhs_68870e9f, xhs_689eeb65, xhs_68079ffd, xhs_67ced398, xhs_67c26620, xhs_67b6fad0, xhs_679f1ba0, xhs_6780ff7f, xhs_676d618e, xhs_6a10653e, +87
- 难度：基础

**面试官常问：**

1. 对称量化和非对称量化的公式是什么？
2. BF16、FP8、INT8 的动态范围和使用场景有什么区别？
3. per-tensor、per-channel、per-group scale 有什么差异？
4. weight-only quant 和 activation quant 哪个更难？

量化把浮点数映射到低 bit 表示。对称 INT8：

$$
\begin{aligned}
scale &= \max(abs(x)) / 127 \\
q &= round(x / scale) \\
x_hat &= q \cdot  scale
\end{aligned}
$$

非对称量化：

$$
\begin{aligned}
scale &= (x_max - x_min) / (q_max - q_min) \\
zero_point &= round(q_min - x_min / scale) \\
q &= round(x / scale + zero_point) \\
x_hat &= (q - zero_point) \cdot  scale
\end{aligned}
$$

代码：

```python
import torch

def quantize_symmetric(x, qmax=127):
    scale = x.abs().max() / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)
    return q, scale

def dequantize_symmetric(q, scale):
    return q.float() * scale
```

BF16 不是传统整数型量化，它保留 FP32 的 8 位指数，动态范围大，训练稳定；FP8/INT8 更省显存和带宽，但需要校准、per-channel scale、异常值处理和 kernel 支持。Weight-only quant 只量化权重，activation quant 还量化激活，后者更难但收益更大。面试中要说清 scale 粒度：per-tensor 简单，per-channel/per-group 更准。

### 高频追问补充

#### 1. BF16、FP16、FP8、INT8/INT4 的差别不只是 bit 数，动态范围有什么区别？

答：BF16 指数范围接近 FP32，训练更稳；FP16 范围小；FP8/INT8/INT4 更省显存和带宽，但更依赖 scale、校准和 kernel。

BF16 保留 FP32 的指数位，动态范围大；FP16 精度更细但范围小；FP8 需要更谨慎的 scale；INT8/INT4 通常要量化 scale。weight-only 量化主要省参数显存和带宽，KV cache 量化省长上下文缓存。对称量化常写作 `x_q = round(x / s)`，非对称量化加 zero-point；group size 越小越精细但 scale 开销越大。

对称量化：

$$
x_q=round(x/s),\quad x\approx s\cdot x_q
$$

非对称量化：

$$
x_q=round(x/s)+z,\quad x\approx s(x_q-z)
$$

group size 越小，每组 scale 越贴合局部分布，但 scale/metadata 开销越大。

#### 2. KV cache 量化和 weight-only 量化分别影响什么？

答：weight-only 量化主要省权重显存和加载带宽，KV cache 量化主要省长上下文 decode 时的缓存容量和读带宽。

BF16 保留 FP32 的指数位，动态范围大；FP16 精度更细但范围小；FP8 需要更谨慎的 scale；INT8/INT4 通常要量化 scale。weight-only 量化主要省参数显存和带宽，KV cache 量化省长上下文缓存。对称量化常写作 `x_q = round(x / s)`，非对称量化加 zero-point；group size 越小越精细但 scale 开销越大。

对称量化：

$$
x_q=round(x/s),\quad x\approx s\cdot x_q
$$

非对称量化：

$$
x_q=round(x/s)+z,\quad x\approx s(x_q-z)
$$

group size 越小，每组 scale 越贴合局部分布，但 scale/metadata 开销越大。

#### 3. calibration、scale、zero-point、group size 怎么解释？

答：scale 决定浮点到整数的比例，zero-point 处理非对称零点，group size 越小越精细但 scale 开销越大。

BF16 保留 FP32 的指数位，动态范围大；FP16 精度更细但范围小；FP8 需要更谨慎的 scale；INT8/INT4 通常要量化 scale。weight-only 量化主要省参数显存和带宽，KV cache 量化省长上下文缓存。对称量化常写作 `x_q = round(x / s)`，非对称量化加 zero-point；group size 越小越精细但 scale 开销越大。

对称量化：

$$
x_q=round(x/s),\quad x\approx s\cdot x_q
$$

非对称量化：

$$
x_q=round(x/s)+z,\quad x\approx s(x_q-z)
$$

group size 越小，每组 scale 越贴合局部分布，但 scale/metadata 开销越大。

### vLLM/SGLang 的加速方法、PagedAttention 和资源调度怎么理解？

- 频率：81
- 来源：xhs_69f70429, xhs_69f5f803, xhs_69d37de9, xhs_69d724da, xhs_69962665, xhs_6964dd92, xhs_69408c32, xhs_694b8e81, xhs_694444e9, xhs_690410e2, xhs_690415f6, xhs_683fb3d0, xhs_67b46786, xhs_67836d81, xhs_6a10653e, +66
- 难度：进阶

**面试官常问：**

1. vLLM 的 PagedAttention 解决了什么显存问题？
2. prefill 和 decode 阶段在调度上有什么差异？
3. continuous batching 为什么能提升吞吐？
4. TTFT、TPOT、吞吐和 KV cache 之间怎么权衡？

推理服务瓶颈通常在 KV cache、动态 batch 和调度。vLLM 的 PagedAttention 把 KV cache 切成 block，像虚拟内存分页，避免为每个请求预留连续大块显存。

```text
request -> logical token positions -> physical KV blocks
```

优势：减少显存碎片，支持 continuous batching，提高吞吐。调度器要在延迟和吞吐间权衡：prefill 阶段计算重，decode 阶段每步小但有 KV 读写；多个请求可以混合调度。

简化 scheduler：

```python
def schedule(waiting, running, free_blocks):
    # 先接纳能放下 KV blocks 的请求
    for req in list(waiting):
        need = estimate_blocks(req.prompt_len + req.max_new_tokens)
        if need <= free_blocks:
            running.append(req)
            waiting.remove(req)
            free_blocks -= need
    # decode 阶段 continuous batching
    batch = [req for req in running if not req.finished]
    return batch
```

SGLang 更强调结构化生成、prefix cache、programmatic runtime 和复杂 Agent/RAG 场景的执行效率。面试答题主线：KV cache 管理决定并发上限，continuous batching 提吞吐，prefix cache 复用公共前缀，调度策略决定 TTFT、TPOT 和总成本。

### 高频追问补充

#### 1. vLLM 的 PagedAttention 到底解决了什么碎片问题？

答：PagedAttention 把 KV cache 分块管理，避免为每个请求分配大段连续显存，减少碎片和浪费。

PagedAttention 把 KV cache 像虚拟内存一样分页，减少连续显存分配和碎片，提高 batch 调度效率。SGLang/RadixAttention 更强调前缀复用，Agent 多轮和工具调用常有共享 prompt/prefix，缓存命中能显著省 prefill。HiCache/分层缓存把 KV 放在 GPU/CPU/SSD 等层级之间权衡容量和带宽；FP8 KV cache 则减少单元素字节。

PagedAttention 把 KV 切成 block，避免连续大块显存分配；prefix/Radix cache 复用公共前缀；分层 KV 则在容量和带宽间权衡。

面试中可用一句话区分：PagedAttention 管碎片，Radix/prefix 管重复 prefill，HiCache 管容量层级，FP8 KV 管单元素字节数。

#### 2. SGLang 的 RadixAttention/prefix cache 为什么对 Agent 有价值？

答：RadixAttention/prefix cache 对 Agent 有价值，因为 system prompt、工具说明和历史前缀常重复，可直接省 prefill。

PagedAttention 把 KV cache 像虚拟内存一样分页，减少连续显存分配和碎片，提高 batch 调度效率。SGLang/RadixAttention 更强调前缀复用，Agent 多轮和工具调用常有共享 prompt/prefix，缓存命中能显著省 prefill。HiCache/分层缓存把 KV 放在 GPU/CPU/SSD 等层级之间权衡容量和带宽；FP8 KV cache 则减少单元素字节。

PagedAttention 把 KV 切成 block，避免连续大块显存分配；prefix/Radix cache 复用公共前缀；分层 KV 则在容量和带宽间权衡。

面试中可用一句话区分：PagedAttention 管碎片，Radix/prefix 管重复 prefill，HiCache 管容量层级，FP8 KV 管单元素字节数。

#### 3. HiCache、分层 KV cache、FP8 KV cache 分别优化哪一层？

答：HiCache 优化 KV 容量层级，FP8 KV cache 优化单元素字节，分层 KV 在 GPU/CPU/SSD 间用带宽换容量。

PagedAttention 把 KV cache 像虚拟内存一样分页，减少连续显存分配和碎片，提高 batch 调度效率。SGLang/RadixAttention 更强调前缀复用，Agent 多轮和工具调用常有共享 prompt/prefix，缓存命中能显著省 prefill。HiCache/分层缓存把 KV 放在 GPU/CPU/SSD 等层级之间权衡容量和带宽；FP8 KV cache 则减少单元素字节。

PagedAttention 把 KV 切成 block，避免连续大块显存分配；prefix/Radix cache 复用公共前缀；分层 KV 则在容量和带宽间权衡。

面试中可用一句话区分：PagedAttention 管碎片，Radix/prefix 管重复 prefill，HiCache 管容量层级，FP8 KV 管单元素字节数。

### 如何估算 LLM 参数量、一次前向计算量和训练显存？

- 频率：66
- 来源：xhs_69f017bc, xhs_6a05f4c4, xhs_69d4f398, xhs_69e2cfc5, xhs_68d8e3cf, xhs_689c756d, xhs_68787f48, xhs_683bc16f, xhs_68079ffd, xhs_67c26620, xhs_67b6fad0, xhs_67babdc6, xhs_67b583d0, xhs_67b295cf, xhs_67ab4e1b, +51
- 难度：进阶

**面试官常问：**

1. 给一个 7B/13B 模型，如何估算参数显存、训练显存和 KV Cache？
2. 一次 forward 的 FLOPs 如何粗略估算？prefill 和 decode 阶段有什么不同？
3. Adam 训练时为什么显存远大于模型参数本身？
4. batch、seq_len、hidden size、layer 数分别影响哪部分显存？

估算题要先声明口径：参数量、推理显存、训练显存、一次 forward FLOPs 不是一个东西。以 decoder-only、`L` 层、隐藏维度 `d`、词表 `V`、FFN 中间维度 `d_ff`、序列长度 `T` 为例，粗略参数量：

$$
\begin{aligned}
\text{Embedding} &: V \cdot  d \\
\text{每层 Attention} &: Wq/Wk/Wv/Wo \approx 4 \cdot  d \cdot  d \\
\text{每层 SwiGLU FFN} &: W_gate/W_up/W_down \approx 3 \cdot  d \cdot  d_ff \\
\text{每层 Norm} &: O(d), 通常可忽略 \\
\text{LM Head} &: V \cdot  d, 若 tied embedding 可共享 \\
Total \approx V\cdot d + L\cdot (4\cdot d^2 + 3\cdot d\cdot d_ff) + optional_lm_head
\end{aligned}
$$

FLOPs 粗估时，矩阵乘 `A[m,n] @ B[n,p]` 约 `2*m*n*p` FLOPs。单 token 解码时 attention score 与历史长度相关，prefill 阶段是 `T^2`，decode 阶段每步是 `T`。训练显存通常包括参数、梯度、优化器状态、激活。Adam 混合精度常见粗估：

$$
\begin{aligned}
\text{参数 BF16} &: 2\ \text{bytes/param} \\
\text{梯度 BF16} &: 2\ \text{bytes/param} \\
\text{Adam }m/v\text{ FP32} &: 8\ \text{bytes/param} \\
\text{Master weight FP32} &: 4\ \text{bytes/param} \\
\text{参数相关合计} &\approx 16\ \text{bytes/param}
\end{aligned}
$$

KV cache 估算公式：

$$
KV bytes = batch \cdot  seq_len \cdot  layers \cdot  kv_heads \cdot  head_dim \cdot  2(K,V) \cdot  bytes_per_elem
$$

可手撕一个估算函数：

```python
def estimate_decoder_params(vocab, layers, d_model, d_ff, tied_lm_head=True):
    embedding = vocab * d_model
    attn = layers * 4 * d_model * d_model
    ffn = layers * 3 * d_model * d_ff  # SwiGLU
    norms = layers * 2 * d_model
    lm_head = 0 if tied_lm_head else vocab * d_model
    return embedding + attn + ffn + norms + lm_head

def kv_cache_gb(batch, seq_len, layers, kv_heads, head_dim, bytes_per_elem=2):
    nbytes = batch * seq_len * layers * kv_heads * head_dim * 2 * bytes_per_elem
    return nbytes / 1024**3

print(round(estimate_decoder_params(151936, 32, 4096, 11008) / 1e9, 2), "B params")
print(round(kv_cache_gb(batch=1, seq_len=32768, layers=32, kv_heads=8, head_dim=128), 2), "GB")
```

面试里最容易丢分的是把“加载模型的参数显存”和“训练所需显存”混在一起，或者忘记 KV cache 随 `seq_len * layers * kv_heads * head_dim` 线性增长。

### 高频追问补充

#### 1. 训练 7B/13B 时，参数、梯度、优化器状态、激活、KV cache 哪一项最容易爆显存？

答：训练显存里最容易爆的通常不是参数本身，而是 Adam 优化器状态和长序列激活；长上下文训练时激活随 batch、序列长度和层数放大，很容易超过参数显存。

参数显存只和参数量、dtype 有关；训练显存还要加梯度、Adam 状态和激活，激活通常随 `batch * seq_len * layers * hidden` 增长，长序列训练时很容易超过参数本身。prefill 是大矩阵乘和 `T^2` attention，更像吞吐问题；decode 每步只有新 token，但要读历史 KV，更容易受内存带宽、KV cache 和 batch 调度影响。回答估算题时先说明口径，再给公式，最后说 ZeRO/checkpoint/tensor parallel 分别切哪一块。

常用估算式：

$$
M_{param}=N_{param}\cdot bytes
$$

$$
M_{KV}=B\cdot T\cdot L\cdot H_{kv}\cdot D_{head}\cdot 2\cdot bytes
$$

Adam 训练还要考虑参数、梯度、一阶/二阶动量和可能的 FP32 master weight。激活粗略随 `B * T * L * hidden` 增长，checkpointing 省激活，ZeRO 切 optimizer/grad/param，tensor parallel 切层内矩阵。

#### 2. 为什么 prefill 和 decode 的瓶颈不同？吞吐该分别看什么？

答：prefill 一次吃完整 prompt，瓶颈更偏大矩阵乘和 attention 的并行吞吐；decode 每步只生成一个 token，但要反复读历史 KV，所以更容易被内存带宽、KV cache 布局和调度卡住。

参数显存只和参数量、dtype 有关；训练显存还要加梯度、Adam 状态和激活，激活通常随 `batch * seq_len * layers * hidden` 增长，长序列训练时很容易超过参数本身。prefill 是大矩阵乘和 `T^2` attention，更像吞吐问题；decode 每步只有新 token，但要读历史 KV，更容易受内存带宽、KV cache 和 batch 调度影响。回答估算题时先说明口径，再给公式，最后说 ZeRO/checkpoint/tensor parallel 分别切哪一块。

常用估算式：

$$
M_{param}=N_{param}\cdot bytes
$$

$$
M_{KV}=B\cdot T\cdot L\cdot H_{kv}\cdot D_{head}\cdot 2\cdot bytes
$$

Adam 训练还要考虑参数、梯度、一阶/二阶动量和可能的 FP32 master weight。激活粗略随 `B * T * L * hidden` 增长，checkpointing 省激活，ZeRO 切 optimizer/grad/param，tensor parallel 切层内矩阵。

#### 3. MTP loss、多 token prediction 或 speculative decoding 会怎样改变计算量估算？

答：MTP 和 speculative decoding 会额外引入 draft/head/验证开销，但目标是减少串行 decode 步数；估算时要把训练额外 loss head 的 FLOPs 和推理少跑 target model 的收益分开算。

参数显存只和参数量、dtype 有关；训练显存还要加梯度、Adam 状态和激活，激活通常随 `batch * seq_len * layers * hidden` 增长，长序列训练时很容易超过参数本身。prefill 是大矩阵乘和 `T^2` attention，更像吞吐问题；decode 每步只有新 token，但要读历史 KV，更容易受内存带宽、KV cache 和 batch 调度影响。回答估算题时先说明口径，再给公式，最后说 ZeRO/checkpoint/tensor parallel 分别切哪一块。

常用估算式：

$$
M_{param}=N_{param}\cdot bytes
$$

$$
M_{KV}=B\cdot T\cdot L\cdot H_{kv}\cdot D_{head}\cdot 2\cdot bytes
$$

Adam 训练还要考虑参数、梯度、一阶/二阶动量和可能的 FP32 master weight。激活粗略随 `B * T * L * hidden` 增长，checkpointing 省激活，ZeRO 切 optimizer/grad/param，tensor parallel 切层内矩阵。

### ZeRO、张量并行、流水线并行和梯度检查点分别省什么？

- 频率：54
- 来源：xhs_6a0733b1, xhs_6a05f4c4, xhs_69f770dc, xhs_69ddf065, xhs_68ff77f4, xhs_68ec676c, xhs_68afbc7a, xhs_67ced398, xhs_67c26620, xhs_67b6fad0, xhs_67ab4e1b, xhs_6780ff7f, xhs_6a10653e, xhs_6a0fc900, xhs_6a13001c, +39
- 难度：进阶

**面试官常问：**

1. ZeRO-1/2/3 分别切分了哪些模型状态？
2. 张量并行、流水线并行、数据并行分别适合解决什么瓶颈？
3. gradient checkpointing 为什么能省显存？代价是什么？
4. 梯度累积和 checkpointing 的区别是什么？

显存组成：

```text
Model states = parameters + gradients + optimizer states
Activations = forward 中为 backward 保存的中间结果
KV cache = 推理时历史 K/V
```

ZeRO 分片：

```text
ZeRO-1: shard optimizer states
ZeRO-2: shard optimizer states + gradients
ZeRO-3: shard optimizer states + gradients + parameters
```

若数据并行度为 `N`，理想情况下 ZeRO-3 的模型状态可近似除以 `N`，但通信更多。张量并行把单层矩阵切开，流水线并行按层切开，梯度检查点用重算换激活显存。

估算器：

```python
def zero_state_memory(params, bytes_param=2, bytes_grad=2, bytes_optim=8, dp=8, stage=2):
    p = params * bytes_param
    g = params * bytes_grad
    o = params * bytes_optim
    if stage == 1:
        return p + g + o / dp
    if stage == 2:
        return p + (g + o) / dp
    if stage == 3:
        return (p + g + o) / dp
    raise ValueError(stage)
```

面试易错点：梯度累积不直接省单步激活显存，它是用多个 micro-batch 模拟大 batch；gradient checkpointing 才是用计算换激活显存。

### 高频追问补充

#### 1. ZeRO-1/2/3 分别切分 optimizer、grad、param 的哪一块？

答：ZeRO-1 切 optimizer states，ZeRO-2 再切 gradients，ZeRO-3 连 parameters 也切；越往后省显存越多但通信越复杂。

ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 连参数也切；tensor parallel 把单层矩阵切到多卡，层内通信多；pipeline parallel 把不同层放不同卡，难点是流水线气泡和 micro-batch 调度。checkpointing 省激活显存，但反向要重算 forward，所以用计算换显存。

显存切分对象：

$$
ZeRO1: optimizer,\quad ZeRO2: optimizer+grad,\quad ZeRO3: optimizer+grad+param
$$

checkpointing 则不切参数，而是少存激活；tensor parallel 切矩阵，pipeline parallel 切层。

#### 2. tensor parallel 和 pipeline parallel 的通信模式有什么不同？

答：Tensor parallel 切单层矩阵，通信多发生在层内 all-reduce/all-gather；pipeline parallel 按层切，难点是流水线气泡。

ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 连参数也切；tensor parallel 把单层矩阵切到多卡，层内通信多；pipeline parallel 把不同层放不同卡，难点是流水线气泡和 micro-batch 调度。checkpointing 省激活显存，但反向要重算 forward，所以用计算换显存。

显存切分对象：

$$
ZeRO1: optimizer,\quad ZeRO2: optimizer+grad,\quad ZeRO3: optimizer+grad+param
$$

checkpointing 则不切参数，而是少存激活；tensor parallel 切矩阵，pipeline parallel 切层。

#### 3. gradient checkpointing 是省显存还是省计算？

答：Gradient checkpointing 是省激活显存，不是省计算；反向时要重算一部分 forward，用算力换显存。

ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 连参数也切；tensor parallel 把单层矩阵切到多卡，层内通信多；pipeline parallel 把不同层放不同卡，难点是流水线气泡和 micro-batch 调度。checkpointing 省激活显存，但反向要重算 forward，所以用计算换显存。

显存切分对象：

$$
ZeRO1: optimizer,\quad ZeRO2: optimizer+grad,\quad ZeRO3: optimizer+grad+param
$$

checkpointing 则不切参数，而是少存激活；tensor parallel 切矩阵，pipeline parallel 切层。

### Temperature、Top-k、Top-p 等采样参数顺序和作用是什么？

- 频率：31
- 来源：xhs_6a0806ae, xhs_6a089599, xhs_6a0018b9, xhs_69f49925, xhs_69f9bacb, xhs_69a240a0, xhs_69c389db, xhs_69adb8c8, xhs_68deb9f3, xhs_68b983a2, xhs_689eeb65, xhs_6a03c1b5, xhs_6a0a75e9, xhs_6a0f3fda, xhs_69de34c8, +16
- 难度：基础

**面试官常问：**

1. Temperature、Top-k、Top-p 分别如何改变采样分布？
2. 采样参数的常见执行顺序是什么？
3. 为什么 temperature=0 有时仍可能不完全 deterministic？
4. RL rollout 中采样温度会如何影响 reward 方差？

采样参数控制的是从 logits 到 token 的分布。Temperature：

$$
\begin{aligned}
p_i &= \operatorname{softmax}(z_i / T) \\
\text{T -> 0} &: 更贪心 \\
\text{T > 1} &: 更随机
\end{aligned}
$$

Top-k 保留概率最高的 `k` 个 token；Top-p 保留累计概率达到 `p` 的最小集合。典型顺序：logits processor -> temperature -> top-k/top-p filter -> softmax -> sample。

实现：

```python
import torch

def sample_token(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / \max(temperature, 1e-6)
    if top_k:
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[-1]] = -float("inf")
    if top_p:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_idx[remove]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()
```

面试追问：为什么 temperature=0 仍可能不完全稳定？因为实际系统可能有并行非确定性、浮点误差、服务端 fallback、相同概率 tie-breaking、不同 kernel 或 prefix cache 状态差异。

### 高频追问补充

#### 1. temperature、top-p、top-k、repeat penalty 的执行顺序怎么理解？

答：常见顺序是 logits 先经 temperature 缩放，再做 top-k/top-p 截断，最后重新归一化并采样；repeat penalty 会先调整已出现 token 的 logits。

采样通常先得到 logits，做 temperature 缩放，再按 top-k/top-p 截断候选，最后采样。temperature 越高分布越平，随机性越强；top-p 保留累计概率质量，top-k 保留固定数量候选。评测要可复现，所以低温或 greedy；创作需要多样性，所以适当升温。线上不确定可能来自并行 kernel、浮点误差、tie-breaking、服务 fallback 和缓存状态。

temperature 缩放：

$$
p_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

top-k 保留前 k 个 token，top-p 保留累计概率超过 p 的最小集合；过滤后要重新归一化。repeat penalty 通常在过滤前调整已出现 token 的 logits。

#### 2. 为什么面试/评测常用低温，创作常用高温？

答：评测低温是为了可复现和低方差，创作高温是为了多样性；RL rollout 温度还会影响 reward 方差和探索范围。

采样通常先得到 logits，做 temperature 缩放，再按 top-k/top-p 截断候选，最后采样。temperature 越高分布越平，随机性越强；top-p 保留累计概率质量，top-k 保留固定数量候选。评测要可复现，所以低温或 greedy；创作需要多样性，所以适当升温。线上不确定可能来自并行 kernel、浮点误差、tie-breaking、服务 fallback 和缓存状态。

temperature 缩放：

$$
p_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

top-k 保留前 k 个 token，top-p 保留累计概率超过 p 的最小集合；过滤后要重新归一化。repeat penalty 通常在过滤前调整已出现 token 的 logits。

#### 3. temperature=0 为什么线上仍可能不完全确定？

答：temperature=0 仍可能因浮点误差、并行 kernel、tie-breaking、服务 fallback 或缓存状态出现细微不确定。

采样通常先得到 logits，做 temperature 缩放，再按 top-k/top-p 截断候选，最后采样。temperature 越高分布越平，随机性越强；top-p 保留累计概率质量，top-k 保留固定数量候选。评测要可复现，所以低温或 greedy；创作需要多样性，所以适当升温。线上不确定可能来自并行 kernel、浮点误差、tie-breaking、服务 fallback 和缓存状态。

temperature 缩放：

$$
p_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

top-k 保留前 k 个 token，top-p 保留累计概率超过 p 的最小集合；过滤后要重新归一化。repeat penalty 通常在过滤前调整已出现 token 的 logits。

### Prefix Cache、RadixAttention、HiCache 和 Speculative Decoding 如何优化推理？

- 频率：21
- 来源：xhs_69f5f803, xhs_69d724da, xhs_698878d1, xhs_694b8e81, xhs_67836d81, xhs_6a10653e, xhs_6a0e7a3c, xhs_6a087059, xhs_6a0d6fe9, xhs_6a0dda05, xhs_6a10491f, xhs_6a119f9b, xhs_6a0710db, xhs_6a032759, xhs_6a12b1b3, +6
- 难度：进阶

**面试官常问：**

1. Prefix Cache、RadixAttention 和 PagedAttention 分别解决什么问题？
2. HiCache/分层 KV cache 为什么对长上下文有价值？
3. Speculative Decoding 为什么能加速？什么时候不划算？
4. FP8 KV cache 和 GQA/MLA 是替代还是互补？

推理优化要拆 prefill、decode 和 KV 管理。Prefill 阶段主要是大矩阵乘和 attention；decode 阶段每次生成一个 token，但要频繁读取历史 KV，容易受内存带宽限制。

$$
KVBytes = B \cdot T \cdot L \cdot H_{kv} \cdot D_{head} \cdot 2 \cdot bytes
$$

Prefix Cache 复用相同 prompt 前缀的 KV；RadixAttention 用前缀树组织共享前缀，适合多轮 Agent 和结构化 prompt；PagedAttention 把 KV cache 分块管理，减少显存碎片；HiCache/分层缓存把 KV 放到 GPU/CPU/SSD 等层级，换取更大上下文容量。

Speculative Decoding 用小模型先草拟多个 token，再由大模型一次验证：

$$
speedup \approx \frac{\text{accepted draft tokens}}{\text{extra draft cost} + \text{verify cost}}
$$

如果草稿模型接受率低，或者大模型验证成本没有被摊薄，加速就不明显。FP8 KV cache、GQA/MLA、Prefix Cache 是互补关系：一个降 bytes，一个降 KV heads/维度，一个减少重复 prefill。

### 高频追问补充

#### 1. Prefix Cache 和 RadixAttention 为什么特别适合 Agent 多轮任务？

答：Prefix/Radix cache 适合 Agent 多轮，是因为多轮共享 system prompt、工具 schema 和历史前缀，缓存命中能显著省 prefill。

Prefix/Radix cache 复用共享 system prompt、历史轮次和工具模板，主要省 prefill；HiCache/分层 KV cache 用容量更大的低速层承接长上下文 KV，牺牲部分带宽换可服务长度；Speculative Decoding 用 draft model 先猜多个 token，再由 target model 并行验证，减少串行 decode 次数。三者优化对象不同：前缀复用、省 KV 容量、减少 target model 调用步数。

Speculative decoding 的基本流程是 draft 先生成 `k` 个 token，target 一次性验证：

$$
speedup \approx \frac{k}{1 + verify\_cost}
$$

实际收益取决于 draft 接受率、target batch 效率和验证开销。prefix cache 省 prefill，spec decoding 省 decode 串行步数，二者可叠加。

#### 2. HiCache/分层 KV cache 在 GPU、CPU、SSD 间怎么取舍？

答：HiCache 把 KV 放到 GPU/CPU/SSD 等层级，GPU 快但贵，CPU/SSD 容量大但带宽和延迟差。

Prefix/Radix cache 复用共享 system prompt、历史轮次和工具模板，主要省 prefill；HiCache/分层 KV cache 用容量更大的低速层承接长上下文 KV，牺牲部分带宽换可服务长度；Speculative Decoding 用 draft model 先猜多个 token，再由 target model 并行验证，减少串行 decode 次数。三者优化对象不同：前缀复用、省 KV 容量、减少 target model 调用步数。

Speculative decoding 的基本流程是 draft 先生成 `k` 个 token，target 一次性验证：

$$
speedup \approx \frac{k}{1 + verify\_cost}
$$

实际收益取决于 draft 接受率、target batch 效率和验证开销。prefix cache 省 prefill，spec decoding 省 decode 串行步数，二者可叠加。

#### 3. Speculative Decoding 为什么能减少 decode 延迟，什么时候不划算？

答：Speculative Decoding 当 draft model 准确且便宜时划算；如果草稿接受率低或验证开销大，收益会被抵消。

Prefix/Radix cache 复用共享 system prompt、历史轮次和工具模板，主要省 prefill；HiCache/分层 KV cache 用容量更大的低速层承接长上下文 KV，牺牲部分带宽换可服务长度；Speculative Decoding 用 draft model 先猜多个 token，再由 target model 并行验证，减少串行 decode 次数。三者优化对象不同：前缀复用、省 KV 容量、减少 target model 调用步数。

Speculative decoding 的基本流程是 draft 先生成 `k` 个 token，target 一次性验证：

$$
speedup \approx \frac{k}{1 + verify\_cost}
$$

实际收益取决于 draft 接受率、target batch 效率和验证开销。prefix cache 省 prefill，spec decoding 省 decode 串行步数，二者可叠加。


## RAG、Agent 与工具调用

### Agent 的上下文、短期记忆、长期记忆和上下文压缩怎么做？

- 频率：259
- 来源：xhs_6a0efd3c, xhs_69f017bc, xhs_69ec3021, xhs_69f5f803, xhs_69f753a5, xhs_69f0d3a5, xhs_69e74f65, xhs_69e6254c, xhs_69e5baea, xhs_69e572f1, xhs_69e39db6, xhs_69e391f1, xhs_69d724da, xhs_69ce972e, xhs_69aa9ff2, +244
- 难度：进阶

**面试官常问：**

1. Agent 的短期记忆、长期记忆、任务记忆怎么区分？
2. 上下文压缩时哪些信息必须保留？
3. 如何给上下文片段打分并装入有限 context window？
4. 为什么长期记忆必须带来源、时间和置信度？

Agent 记忆要分层，否则上下文会失控：

```text
Working context: 当前窗口内必须立刻用的信息
Short-term memory: 当前任务轨迹、已读文件、工具结果
Long-term memory: 用户偏好、项目约定、历史决策
Task memory: 某个项目内可复用事实和产物
```

上下文压缩不是简单摘要，而是保留约束、证据和可恢复指针。常用打分：

$$
score = relevance \cdot  recency \cdot  authority \cdot  risk_weight
$$

打包上下文伪代码：

```python
def pack_context(items, budget):
    # item: {text, tokens, relevance, recency, authority, risk}
    for item in items:
        item["score"] = item["relevance"] * item["recency"] * item["authority"] * item["risk"]
    selected = []
    used = 0
    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        if used + item["tokens"] <= budget:
            selected.append(item)
            used += item["tokens"]
    return selected
```

长期记忆必须有来源、时间和置信度，避免把旧摘要当当前事实。对于代码 Agent，最重要的是“可恢复”：摘要里要有文件路径、函数名、命令输出位置，而不是只写“修好了某模块”。

### 高频追问补充

#### 1. Agent Memory 的工作记忆、短期记忆、长期记忆、情景记忆分别是什么？

答：工作记忆是当前任务正在使用的信息，例如当前计划、临时变量、刚读过的文件片段；短期记忆是本轮会话/任务状态，例如已做步骤、失败尝试、用户最新约束；长期记忆是跨任务保留的偏好、事实和经验；情景记忆是历史任务轨迹，可按相似任务检索。

面试中要强调：这些不是必须用四个数据库实现，而是四种生命周期和读写策略。工作记忆常在上下文里，长期/情景记忆更适合外部存储和检索。

#### 2. 长上下文和外部记忆是替代关系吗？

答：长上下文和外部记忆不是替代关系。长上下文降低了短期检索成本，但上下文越长，成本越高、噪声越多、注意力越分散；外部记忆可以长期保存、结构化检索、跨会话复用。

比较好的架构是：当前高优先级约束直接 pin 在上下文；大段历史和知识放外部记忆；每轮按任务目标检索，再把少量证据放进上下文。

#### 3. 上下文压缩如何避免丢掉关键约束？

答：上下文压缩最怕把约束压没或把不确定信息压成事实。压缩时要保留：目标、用户硬约束、已验证事实、文件路径/来源、失败尝试、当前计划、未决问题。可以用结构化摘要而不是纯自然语言。

例如压缩成：`Goal / Constraints / Evidence / Decisions / Open Issues / Next Actions`。每条 evidence 最好带 source id，后续需要时可回读原文。

### Function Call、MCP、Tool、Skill 和 Rules 的区别是什么？

- 频率：158
- 来源：xhs_6a02019f, xhs_69f49925, xhs_69ea8685, xhs_69e6254c, xhs_69e5baea, xhs_69e572f1, xhs_69ce972e, xhs_69aa9ff2, xhs_69962665, xhs_69408c32, xhs_68f60343, xhs_689eeb65, xhs_6a0e7134, xhs_6a0717a4, xhs_6a0ea22f, +143
- 难度：基础

**面试官常问：**

1. Function Call、Tool、MCP、Skill、Rules 分别是什么？
2. 为什么有 Tool/MCP 之后还需要 Skill？
3. 一个工具调用系统如何做 schema 校验、权限和审计？
4. MCP 的 resources/prompts/tools 分别适合暴露什么？

这些词可以按层次理解：

```text
Function Call: 模型输出结构化函数名和参数的能力
Tool: 可执行能力，例如 search/read_file/run_tests
MCP: 暴露 tools/resources/prompts 的协议
Skill: 面向 Agent 的工作流说明 + 脚本 + 模板
Rules: 长期行为约束或项目规范
```

Function call 示例：

```json
{
  "name": "search_code",
  "arguments": {
    "query": "def apply_rope",
    "path": "src/"
  }
}
```

Tool schema 负责约束参数，MCP 负责让不同工具服务以统一方式被发现和调用，Skill 则告诉 Agent“遇到某类任务应该按什么步骤做”。例如“小红书收藏夹整理”这个工作，不只是一个 tool，它需要登录态检查、滚动盘点、抽取、分类、验证、提交，这就适合封装成 Skill。

面试一句话：Function Call 解决“模型如何结构化调用”，Tool 解决“调用什么能力”，MCP 解决“能力如何标准化接入”，Skill 解决“复杂任务如何流程化复用”，Rules 解决“长期边界和风格约束”。

更工程化地说，function call 是模型输出层协议，tool 是运行时能力，MCP 是跨进程/跨应用的工具注册与调用协议，skill 是 Agent 的操作手册。一个可落地的工具调用系统通常还要有参数校验、权限、幂等、超时和观测：

```python
def run_tool_call(call, registry, user_ctx):
    tool = registry.get(call["name"])
    if tool is None:
        raise ValueError("unknown tool")
    if not user_ctx.can_use(tool.name):
        raise PermissionError(tool.name)
    args = tool.schema.validate(call["arguments"])
    with timeout(tool.timeout_seconds):
        result = tool.fn(**args)
    audit_\log(user=user_ctx.user_id, tool=tool.name, args=args, result=result.summary())
    return result
```

面试追问“为什么有 MCP 还要 Skill”：MCP 暴露能力，但不知道业务流程；Skill 把多个 MCP/tool 调用编排成可复用工作流，并记录哪些文件该读、哪些中间产物不能提交、失败时怎么降级。Rules 则更像全局约束，例如“不要提交 cookie”“先验证再回答”。这几个层次一起工作，Agent 才既能调用外部能力，又不至于每次从零摸索流程。

### 高频追问补充

#### 1. 面试官问 MCP 还是 CLI，怎么比较？

答：MCP 和 CLI 不是同层概念。CLI 是人或脚本调用程序的命令行接口，简单直接；MCP 是给 Agent 暴露 tools/resources/prompts 的协议，让模型能发现能力、读取上下文并用结构化方式调用。

面试可以说：CLI 适合稳定脚本和本地自动化；MCP 适合多工具、多资源、跨应用的 Agent 生态。很多 MCP server 底层也可能调用 CLI。

#### 2. Tool、Function Call、MCP Server、Skill、Rules 在 Agent 系统里各管什么？

答：Function Call 是模型输出结构化参数的能力；Tool 是被调用的外部能力；MCP Server 是标准化承载 tool/resource 的服务；Skill 是把工具、流程、经验、边界条件组织成可复用工作流；Rules 是长期约束。

一句话：Function Call 解决“怎么调”，MCP 解决“调什么和从哪拿上下文”，Skill 解决“按什么流程完成任务”，Rules 解决“永远不能越过哪些边界”。

#### 3. 写好一个 Agent Skill 为什么比单独暴露工具更重要？

答：单独暴露工具只告诉 Agent 有锤子，但不告诉什么时候敲、敲哪里、失败怎么处理。Skill 把任务流程写下来，包括该读哪些文件、如何验证、哪些 raw 文件不能提交、出错如何降级。

对复杂工作流，Skill 能减少每次从零探索的成本，也能降低误操作。比如小红书采集 skill 里会记录 signed URL、风控冷却、raw 文件 gitignore，这些不是单个 tool 能表达完整的。

### RAG 和微调的区别、优缺点和选择标准是什么？

- 频率：125
- 来源：xhs_6a05f4c4, xhs_6a02019f, xhs_69ec3021, xhs_69e6254c, xhs_69aa9ff2, xhs_69aaedf4, xhs_68f60343, xhs_68abec41, xhs_680e5d9b, xhs_67c50837, xhs_67c26620, xhs_67c1b190, xhs_67c01094, xhs_67ac9541, xhs_6a1178d6, +110
- 难度：基础

**面试官常问：**

1. RAG 和微调分别解决什么问题？如何做选型？
2. RAG 的召回、rerank、faithfulness 如何评估？
3. 知识频繁更新时为什么通常优先 RAG？
4. 什么时候需要 RAG + SFT/偏好优化组合？

RAG 和微调解决的问题不同。RAG 是“把外部知识放进上下文”，微调是“改变模型参数中的行为/能力/风格”。

RAG 流程：

```text
query -> rewrite -> embedding -> retrieve topK -> rerank -> pack context -> generate with citations
```

RAG 适合知识动态、需要引用、企业文档问答、可追溯场景。微调适合固定格式输出、领域术语风格、工具调用习惯、特定任务能力。评估也不同：

```text
RAG: recall@k, MRR, nDCG, context precision, answer faithfulness
FT: task accuracy, win rate, format acc, safety, general capability regression
```

选择函数：

```python
def choose_rag_or_ft(knowledge_changes, need_citation, behavior_gap, stable_task):
    if knowledge_changes or need_citation:
        return "RAG first"
    if behavior_gap and stable_task:
        return "SFT/Preference tuning"
    if behavior_gap and knowledge_changes:
        return "RAG + light SFT for format/tool use"
    return "prompt engineering baseline"
```

面试里要避免说“RAG 比微调便宜所以都用 RAG”。RAG 的难点是切 chunk、召回漏、rerank、上下文冲突、引用幻觉；微调的难点是数据质量、灾难性遗忘、成本和更新慢。很多生产系统是 RAG + 小规模 SFT/偏好优化组合。

### 高频追问补充

#### 1. 本地知识库最后没用 RAG，可能是哪些原因？

答：本地知识库没用 RAG 可能是因为知识量小、结构强、更新慢，规则、SQL、grep 或直接上下文更稳定。

RAG 不是默认最优。如果知识规模小、结构强、更新慢，规则/SQL/grep 可能更稳定；如果问题需要多跳推理，单次向量召回也不够。优化顺序建议：先做数据清洗和 chunk 设计，再看 embedding/召回，再加 rerank 和 query rewrite，最后做答案生成约束与引用校验。Agent memory 更偏个体历史和任务状态，RAG 更偏外部知识库；两者可以共享检索基础设施，但评估指标不同。

RAG 链路可拆成：

$$
Query \rightarrow Retrieve \rightarrow Rerank \rightarrow Compress \rightarrow Generate \rightarrow Verify
$$

指标也要拆：召回率、命中率、NDCG、引用正确率、答案忠实性和端到端准确率。微调则看任务泛化、遗忘、风格稳定和成本。

#### 2. RAG 准确率提升应该从切分、召回、rerank、生成哪一层入手？

答：RAG 提升要按链路定位：切分清洗、召回、rerank、上下文压缩、生成约束和引用校验，不能只换 embedding。

RAG 不是默认最优。如果知识规模小、结构强、更新慢，规则/SQL/grep 可能更稳定；如果问题需要多跳推理，单次向量召回也不够。优化顺序建议：先做数据清洗和 chunk 设计，再看 embedding/召回，再加 rerank 和 query rewrite，最后做答案生成约束与引用校验。Agent memory 更偏个体历史和任务状态，RAG 更偏外部知识库；两者可以共享检索基础设施，但评估指标不同。

RAG 链路可拆成：

$$
Query \rightarrow Retrieve \rightarrow Rerank \rightarrow Compress \rightarrow Generate \rightarrow Verify
$$

指标也要拆：召回率、命中率、NDCG、引用正确率、答案忠实性和端到端准确率。微调则看任务泛化、遗忘、风格稳定和成本。

#### 3. RAG 和 Agent memory 的边界在哪里？

答：RAG 管外部知识，Agent memory 管个体历史和任务状态；二者都可检索，但评估指标不同。

RAG 不是默认最优。如果知识规模小、结构强、更新慢，规则/SQL/grep 可能更稳定；如果问题需要多跳推理，单次向量召回也不够。优化顺序建议：先做数据清洗和 chunk 设计，再看 embedding/召回，再加 rerank 和 query rewrite，最后做答案生成约束与引用校验。Agent memory 更偏个体历史和任务状态，RAG 更偏外部知识库；两者可以共享检索基础设施，但评估指标不同。

RAG 链路可拆成：

$$
Query \rightarrow Retrieve \rightarrow Rerank \rightarrow Compress \rightarrow Generate \rightarrow Verify
$$

指标也要拆：召回率、命中率、NDCG、引用正确率、答案忠实性和端到端准确率。微调则看任务泛化、遗忘、风格稳定和成本。

### Claude Code 为什么偏向 grep/ripgrep，而不是传统向量 RAG？

- 频率：109
- 来源：xhs_69eb90ca, xhs_6a02019f, xhs_69e572f1, xhs_69e35ff7, xhs_69e39db6, xhs_69ce972e, xhs_6a0717a4, xhs_6a0ea22f, xhs_6a03338d, xhs_6a019cba, xhs_6a0f13ed, xhs_6a0f6145, xhs_6a11541e, xhs_6a0299c7, xhs_6a02ae37, +94
- 难度：进阶

**面试官常问：**

1. Claude Code 类工具为什么常用 grep/ripgrep，而不是只用向量 RAG？
2. 代码检索中 exact match、调用链、路径信息为什么重要？
3. Agentic search 和一次性 topK RAG 的区别是什么？
4. 什么场景下代码库仍然适合向量检索？

代码库检索和知识库 RAG 不一样。代码问题经常需要精确标识符、路径、调用链和最新文件状态，向量语义召回容易把“意思相近但不是同一个函数”的片段召回来。`grep/ripgrep` 的优势：

```text
实时：不依赖过期索引
精确：函数名、变量名、错误信息可 exact match
可解释：命中的文件和行号明确
可组合：先搜符号，再读文件，再搜调用方
```

Agentic code search 伪代码：

```python
def investigate(repo, user_question):
    terms = plan_search_terms(user_question)
    evidence = []
    for term in terms:
        hits = ripgrep(repo, term)
        for path, line in rank_hits(hits):
            text = read_window(path, line, before=40, after=80)
            evidence.append((path, line, text))
            terms.extend(extract_new_symbols(text))
    return synthesize_answer(evidence)
```

传统向量 RAG 仍有用，例如查设计文档、README、自然语言 issue；但在代码编辑任务中，`rg` + AST/LSP + 测试反馈通常更可靠。面试可以总结：Claude Code 这类工具更偏“agentic search”，让模型动态决定下一步搜什么，而不是一次 embedding topK 决定全部上下文。

### 高频追问补充

#### 1. 为什么 Claude Code/Codex 类工具常用 grep/ripgrep 搜代码？

答：代码检索需要精确符号、路径、错误文本和调用点，grep/ripgrep 实时、可解释、不会索引过期。

代码问题经常需要精确符号、路径、调用点和错误文本，`rg` 实时、可解释、不会索引过期。向量索引适合概念相似、文档说明、跨语言语义搜索，但对精确变量名和新改文件不如 grep 靠谱。Agentic Search 的关键是多轮：先宽搜，再读文件，再根据证据收窄查询，而不是一次 embedding 召回后直接回答。

代码搜索常先用精确检索缩小范围：

```bash
rg -n "class Agent|def run|error message" .
```

再读调用点和测试。向量索引可作为补充，但要解决索引新鲜度、权限和符号精确度问题。

#### 2. 代码库检索什么时候适合向量索引？

答：向量索引适合概念相似、跨语言说明和文档语义检索，但对刚改代码、变量名和精确错误不如 grep。

代码问题经常需要精确符号、路径、调用点和错误文本，`rg` 实时、可解释、不会索引过期。向量索引适合概念相似、文档说明、跨语言语义搜索，但对精确变量名和新改文件不如 grep 靠谱。Agentic Search 的关键是多轮：先宽搜，再读文件，再根据证据收窄查询，而不是一次 embedding 召回后直接回答。

代码搜索常先用精确检索缩小范围：

```bash
rg -n "class Agent|def run|error message" .
```

再读调用点和测试。向量索引可作为补充，但要解决索引新鲜度、权限和符号精确度问题。

#### 3. Agentic Search 和传统 RAG 的最大区别是什么？

答：Agentic Search 是多轮搜读改写查询的过程，传统 RAG 更像一次召回后生成，代码库任务更需要前者。

代码问题经常需要精确符号、路径、调用点和错误文本，`rg` 实时、可解释、不会索引过期。向量索引适合概念相似、文档说明、跨语言语义搜索，但对精确变量名和新改文件不如 grep 靠谱。Agentic Search 的关键是多轮：先宽搜，再读文件，再根据证据收窄查询，而不是一次 embedding 召回后直接回答。

代码搜索常先用精确检索缩小范围：

```bash
rg -n "class Agent|def run|error message" .
```

再读调用点和测试。向量索引可作为补充，但要解决索引新鲜度、权限和符号精确度问题。

### 企业级 Agent 架构如何设计，Agent 与 Workflow 有何区别？

- 频率：47
- 来源：xhs_6a02019f, xhs_69e6254c, xhs_69e572f1, xhs_67aa9bb4, xhs_6a09af6a, xhs_6a075431, xhs_6a056eef, xhs_6a0d7b07, xhs_6a0863cb, xhs_6a0fc900, xhs_6a0adad3, xhs_6a0891b2, xhs_6a0e74d2, xhs_6a0e848b, xhs_6a0d4483, +32
- 难度：进阶

**面试官常问：**

1. Agent 和 Workflow 的本质区别是什么？
2. 企业级 Agent 架构应该包含哪些模块？
3. Planner、Executor、Tool Server、Memory 分别负责什么？
4. 如何做权限、审计、超时、重试和人工确认？

Workflow 是固定流程，Agent 是模型驱动的动态决策。企业级 Agent 不能只画一个 LLM 框，要拆成：

```text
Client/BFF -> Planner -> Executor -> Tool Server -> Worker
                     -> Memory/RAG
                     -> Policy/Auth/Audit
                     -> Observability/Eval
```

Agent loop：

```python
def agent_loop(task, tools, max_steps=20):
    state = {"task": task, "history": []}
    for _ in range(max_steps):
        action = planner(state)
        if action.name == "final":
            return action.content
        if action.name not in tools:
            state["history"].append(("error", "tool not allowed"))
            continue
        result = tools[action.name](**validate(action.args))
        state["history"].append((action, result))
    return "need human review"
```

企业落地关键点：工具白名单、参数 schema、权限隔离、超时重试、幂等、审计日志、人类确认、成本预算、eval 和回放。Workflow 稳定可控，适合审批、固定 ETL；Agent 灵活，适合探索、代码修复、多步问答。多 Agent 只有在角色分工确实降低复杂度时才值得，否则协调成本会很高。

### 高频追问补充

#### 1. 企业级 Agent 为什么不能只有一个大模型 prompt？

答：企业级 Agent 不能只有 prompt，因为生产系统还需要权限、工具参数校验、状态管理、超时重试、观测和人工确认。

企业级 Agent 至少要有 planner/executor/tool/memory/evaluator/observability/safety 几层。Workflow 是确定流程，适合稳定业务；Agent 是动态决策，适合开放任务；Harness/Runtime 负责把模型、工具、状态、权限、日志、重试连接起来。多 Agent 容易踩通信成本、职责重叠、状态不一致、循环调用和难评估的问题，所以生产系统常用“少 Agent + 明确工具 + 强观测”而不是盲目堆 Agent。

企业级 Agent 轨迹最好结构化：

```json
{"state":"plan","action":"tool_call","tool":"search","args":{},"observation":"...","cost_ms":123}
```

有了结构化轨迹，才能做权限审计、失败 replay、成本统计、自动评测和人工接管。

#### 2. Agent、Workflow、Harness、Runtime 分别是什么层？

答：Workflow 是固定流程，Agent 是动态决策，Harness/Runtime 负责把模型、工具、状态、日志和调度连接起来。

企业级 Agent 至少要有 planner/executor/tool/memory/evaluator/observability/safety 几层。Workflow 是确定流程，适合稳定业务；Agent 是动态决策，适合开放任务；Harness/Runtime 负责把模型、工具、状态、权限、日志、重试连接起来。多 Agent 容易踩通信成本、职责重叠、状态不一致、循环调用和难评估的问题，所以生产系统常用“少 Agent + 明确工具 + 强观测”而不是盲目堆 Agent。

企业级 Agent 轨迹最好结构化：

```json
{"state":"plan","action":"tool_call","tool":"search","args":{},"observation":"...","cost_ms":123}
```

有了结构化轨迹，才能做权限审计、失败 replay、成本统计、自动评测和人工接管。

#### 3. 多 Agent 集群协作最容易踩哪些坑？

答：多 Agent 最容易踩职责重叠、循环调用、状态不一致、通信成本和难评估问题，要先证明单 Agent + 工具不够。

企业级 Agent 至少要有 planner/executor/tool/memory/evaluator/observability/safety 几层。Workflow 是确定流程，适合稳定业务；Agent 是动态决策，适合开放任务；Harness/Runtime 负责把模型、工具、状态、权限、日志、重试连接起来。多 Agent 容易踩通信成本、职责重叠、状态不一致、循环调用和难评估的问题，所以生产系统常用“少 Agent + 明确工具 + 强观测”而不是盲目堆 Agent。

企业级 Agent 轨迹最好结构化：

```json
{"state":"plan","action":"tool_call","tool":"search","args":{},"observation":"...","cost_ms":123}
```

有了结构化轨迹，才能做权限审计、失败 replay、成本统计、自动评测和人工接管。


## 评估与对齐

### 大模型训练、RAG 和上线效果如何评估？PPL 有什么意义？

- 频率：191
- 来源：xhs_6a0be406, xhs_6a0733b1, xhs_6a0018b9, xhs_69fca9a5, xhs_69f31d0a, xhs_69e9a5e3, xhs_69dde219, xhs_69ce972e, xhs_69c4f43e, xhs_69aa9ff2, xhs_69aaedf4, xhs_698878d1, xhs_690410e2, xhs_68f60343, xhs_68d8e3cf, +176
- 难度：基础

**面试官常问：**

1. PPL 的定义是什么？为什么 PPL 不等于指令效果？
2. SFT、RLHF/DPO/GRPO、RAG 分别应该看哪些评估指标？
3. RAG 的 recall@k、MRR、faithfulness 怎么理解？
4. 离线评测和线上指标如何闭环？

PPL 是语言建模指标：

$$
\begin{aligned}
NLL &= -1/N \cdot  \sum_t log p(x_t | x_<t) \\
PPL &= \exp(NLL)
\end{aligned}
$$

PPL 低说明模型更会预测验证文本，但不等于指令遵循、事实正确或用户满意。SFT/RL/RAG/线上评估要拆开：

```text
SFT: format_acc, instruction_following, task_accuracy
RLHF/DPO/GRPO: win_rate, reward_score, KL, safety
RAG: recall@k, MRR, context_precision, faithfulness, citation_correctness
Online: latency, cost, fallback_rate, user_success, complaint_rate
```

RAG 召回评估：

```python
def recall_at_k(retrieved_ids, gold_ids, k):
    hit = set(retrieved_ids[:k]) & set(gold_ids)
    return len(hit) / \max(len(gold_ids), 1)

def mrr(retrieved_ids, gold_ids):
    gold = set(gold_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0
```

面试回答要强调：离线评测用于快速迭代，线上指标验证真实用户价值；单一指标会误导，例如 PPL 降低但安全性下降，RAG recall 提升但上下文噪声变多导致答案变差。

### 高频追问补充

#### 1. Agent Eval 如何设计，为什么不能只看最终答案？

答：Agent Eval 不能只看最终答案，因为同样答对可能有完全不同的工具成本、风险和可恢复性。

评估要拆层：基础模型看 PPL/benchmark，RAG 看召回率、命中率、rerank NDCG、引用正确性和答案忠实性，Agent 看 task success、tool success、步骤数、成本、恢复能力和安全。PPL 是语言建模指标，能反映分布拟合，但不等于指令遵循、推理、工具调用或业务成功。面试中最好给一个离线集 + 在线观测 + 人工抽检的组合方案。

PPL：

$$
PPL=\exp\left(-\frac{1}{N}\sum_i \log p(x_i|x_{<i})\right)
$$

RAG 指标拆检索和生成；Agent 指标拆 task success、tool success、轨迹步数、cost、latency、安全和可恢复性。

#### 2. RAG 评估要分开看检索、rerank、生成还是端到端？

答：RAG 要拆检索、rerank、上下文构造和生成忠实性；端到端分数只能说明整体结果，不能定位问题。

评估要拆层：基础模型看 PPL/benchmark，RAG 看召回率、命中率、rerank NDCG、引用正确性和答案忠实性，Agent 看 task success、tool success、步骤数、成本、恢复能力和安全。PPL 是语言建模指标，能反映分布拟合，但不等于指令遵循、推理、工具调用或业务成功。面试中最好给一个离线集 + 在线观测 + 人工抽检的组合方案。

PPL：

$$
PPL=\exp\left(-\frac{1}{N}\sum_i \log p(x_i|x_{<i})\right)
$$

RAG 指标拆检索和生成；Agent 指标拆 task success、tool success、轨迹步数、cost、latency、安全和可恢复性。

#### 3. PPL 降低一定代表下游能力变好吗？

答：PPL 降低说明语言建模拟合更好，但不等于指令遵循、推理、工具调用或业务成功一定提升。

评估要拆层：基础模型看 PPL/benchmark，RAG 看召回率、命中率、rerank NDCG、引用正确性和答案忠实性，Agent 看 task success、tool success、步骤数、成本、恢复能力和安全。PPL 是语言建模指标，能反映分布拟合，但不等于指令遵循、推理、工具调用或业务成功。面试中最好给一个离线集 + 在线观测 + 人工抽检的组合方案。

PPL：

$$
PPL=\exp\left(-\frac{1}{N}\sum_i \log p(x_i|x_{<i})\right)
$$

RAG 指标拆检索和生成；Agent 指标拆 task success、tool success、轨迹步数、cost、latency、安全和可恢复性。

### Agent Eval、可观测体系和自动评测环境如何设计？

- 频率：108
- 来源：xhs_69ea8685, xhs_69ce972e, xhs_69aaedf4, xhs_690410e2, xhs_68abec41, xhs_6a044cb4, xhs_6a0f13ed, xhs_69fffb4a, xhs_6a0e6864, xhs_6a02ae37, xhs_6a125c27, xhs_6a09e5da, xhs_6a13143b, xhs_6a086305, xhs_6a02ec36, +93
- 难度：进阶

**面试官常问：**

1. Agent Eval 和普通 LLM benchmark 有什么区别？
2. Agent 可观测体系应该记录哪些字段？
3. 如何自动生成或构造 Agent 评测环境？
4. tool success rate、task success rate、final answer accuracy 冲突时怎么分析？

Agent 评估不能只看最终答案，因为 Agent 的失败可能发生在规划、工具选择、参数生成、工具执行、结果整合、重试恢复任一环节。一个可落地的评估体系至少拆成：

$$
\begin{aligned}
TaskSuccess &= 1[\text{最终任务完成}] \\
ToolSuccess &= \frac{\#\text{成功工具调用}}{\#\text{总工具调用}} \\
Cost &= \text{tokens} + \lambda_1 \cdot \text{latency} + \lambda_2 \cdot \#\text{tool_calls} \\
TrajectoryScore &= f(\text{plan}, \text{tool_args}, \text{observations}, \text{recovery})
\end{aligned}
$$

可观测日志要记录：用户目标、系统约束、每轮 prompt、模型原始输出、tool name、tool args、tool result、状态迁移、异常、重试、人工接管点、token/latency/cost。没有这些字段，失败只能靠猜。

自动评测环境的关键是“可验证”。例如检索/网页 Agent 可以有标准答案和引用检查；代码 Agent 可以跑测试；数据库 Agent 可以比对 SQL 结果；工具调用 Agent 可以用 mock server 检查参数。

```python
def evaluate_agent_trace(trace, judge):
    tool_ok = sum(s["ok"] for s in trace.tool_calls) / max(len(trace.tool_calls), 1)
    final_ok = judge.check_final_answer(trace.final_answer)
    cost = trace.tokens + 0.1 * trace.latency_ms + 5 * len(trace.tool_calls)
    recovered = any(s.get("retry_after_error") for s in trace.steps)
    return {
        "task_success": int(final_ok),
        "tool_success": tool_ok,
        "cost": cost,
        "recovered": recovered,
    }
```

面试追问“为什么 task success 高但 tool success 低”：可能是模型乱试但最后碰对，成本高且不稳定；“tool success 高但 final answer 差”：可能是规划或结果整合差。

### 高频追问补充

#### 1. Agent Eval 为什么不能只看最终答案对不对？

答：Agent Eval 要看轨迹，不只看答案；答案对但工具乱调、成本过高或靠错误证据生成，都不能算稳定能力。

Agent 评估要拆最终结果、工具轨迹、成本延迟、恢复能力和安全边界。只看最终答案会掩盖错误工具调用、无效重试、幻觉引用和高成本路径。可观测体系要记录 prompt、模型输出、工具调用参数、工具返回、状态迁移、重试、人工接管和评测标签。自动评测环境要能构造任务、给出可验证反馈、支持轨迹 replay，并把 task success、tool success、step quality、cost、latency 分开统计。

Agent 评测记录建议至少包含：

```json
{"task_id":"...","step":3,"action":"tool","tool_success":true,"task_success":false,"cost":0.12}
```

最终看聚合指标：

$$
SuccessRate=\frac{passed\ tasks}{all\ tasks}
$$

并分层分析工具失败、规划失败、信息整合失败和安全失败。

#### 2. Agent 可观测体系应该记录哪些字段，如何支持 replay？

答：可观测字段要支持 replay，所以必须记录输入、模型动作、工具参数、工具结果、状态迁移、错误和重试。

Agent 评估要拆最终结果、工具轨迹、成本延迟、恢复能力和安全边界。只看最终答案会掩盖错误工具调用、无效重试、幻觉引用和高成本路径。可观测体系要记录 prompt、模型输出、工具调用参数、工具返回、状态迁移、重试、人工接管和评测标签。自动评测环境要能构造任务、给出可验证反馈、支持轨迹 replay，并把 task success、tool success、step quality、cost、latency 分开统计。

Agent 评测记录建议至少包含：

```json
{"task_id":"...","step":3,"action":"tool","tool_success":true,"task_success":false,"cost":0.12}
```

最终看聚合指标：

$$
SuccessRate=\frac{passed\ tasks}{all\ tasks}
$$

并分层分析工具失败、规划失败、信息整合失败和安全失败。

#### 3. 自动评测环境如何设计 task success、tool success 和安全指标？

答：自动评测环境要把 task success、tool success、安全、成本和延迟拆开算，并保留失败轨迹用于回归。

Agent 评估要拆最终结果、工具轨迹、成本延迟、恢复能力和安全边界。只看最终答案会掩盖错误工具调用、无效重试、幻觉引用和高成本路径。可观测体系要记录 prompt、模型输出、工具调用参数、工具返回、状态迁移、重试、人工接管和评测标签。自动评测环境要能构造任务、给出可验证反馈、支持轨迹 replay，并把 task success、tool success、step quality、cost、latency 分开统计。

Agent 评测记录建议至少包含：

```json
{"task_id":"...","step":3,"action":"tool","tool_success":true,"task_success":false,"cost":0.12}
```

最终看聚合指标：

$$
SuccessRate=\frac{passed\ tasks}{all\ tasks}
$$

并分层分析工具失败、规划失败、信息整合失败和安全失败。

### 训练时让模型输出 CoT、推理时只输出答案如何做？

- 频率：24
- 来源：xhs_6a0be406, xhs_69e9a5e3, xhs_69aa9ff2, xhs_68ec676c, xhs_67c50837, xhs_67ab4058, xhs_678e5e6b, xhs_6a0e74d2, xhs_6a125bba, xhs_6a0b163c, xhs_6a0ff8e7, xhs_6a0521f6, xhs_6a058ddc, xhs_69435d73, xhs_697c6a08, +9
- 难度：进阶

**面试官常问：**

1. 训练时让模型输出 CoT，推理时只输出最终答案，怎么实现？
2. reasoning/answer 结构化模板如何设计？
3. 如何避免部署时泄露中间推理？
4. 如果隐藏 CoT 后性能下降，怎么评估和处理？

训练时让模型学习推理，推理时只输出答案，常见做法是结构化模板：

```text
<reasoning> ... chain of thought ... </reasoning>
<answer> final answer </answer>
```

训练时可以对 reasoning 和 answer 都算 loss，也可以只对 answer 加强权重；部署时通过 system prompt、stop sequence 或后处理只返回 `<answer>`。如果业务不能暴露 CoT，可以做蒸馏：teacher 生成推理，student 学短答案。

后处理：

```python
import re

def extract_final_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[-1]
```

风险：模型可能依赖显式 CoT 才能做对；隐藏 reasoning 后性能下降；或把不该展示的推理泄露出来。评估要有两套：带 reasoning 的开发集和只输出答案的部署集。RL 阶段可以用格式奖励：

$$
R = R_correct + 0.2 \cdot  R_format - 0.1 \cdot  R_leak_reasoning
$$

### 高频追问补充

#### 1. 训练时用 CoT，推理时不展示 CoT，会不会能力下降？

答：训练用 CoT、推理隐藏 CoT 不一定掉能力，但要用无 CoT 验证集确认模型不是强依赖显式长推理文本。

训练 CoT 是给模型中间推理监督，推理时可以通过格式约束只输出答案，但模型内部仍可能形成推理过程。hidden reasoning 是系统内部保留推理，short answer 是对用户隐藏过程，distilled rationale 是把长推理压成可控短解释。生产系统隐藏 CoT 常出于安全、隐私、可读性和避免暴露不稳定中间过程；评估时则要看最终正确率和必要解释质量。

训练时可用结构化字段：

```json
{"reasoning":"...","answer":"..."}
```

部署时只展示 `answer`。关键是验证无显式 CoT 时最终正确率，以及防止模型把隐藏推理泄露到用户可见输出。

#### 2. hidden reasoning、short answer、distilled rationale 三者怎么区分？

答：hidden reasoning 是系统内部推理，short answer 是只展示结论，distilled rationale 是把长推理蒸馏成短解释。

训练 CoT 是给模型中间推理监督，推理时可以通过格式约束只输出答案，但模型内部仍可能形成推理过程。hidden reasoning 是系统内部保留推理，short answer 是对用户隐藏过程，distilled rationale 是把长推理压成可控短解释。生产系统隐藏 CoT 常出于安全、隐私、可读性和避免暴露不稳定中间过程；评估时则要看最终正确率和必要解释质量。

训练时可用结构化字段：

```json
{"reasoning":"...","answer":"..."}
```

部署时只展示 `answer`。关键是验证无显式 CoT 时最终正确率，以及防止模型把隐藏推理泄露到用户可见输出。

#### 3. 为什么有些系统要求只输出 final answer？

答：系统只输出 final answer 通常是为了安全、隐私、稳定性、可读性和避免暴露不可靠的中间推理。

训练 CoT 是给模型中间推理监督，推理时可以通过格式约束只输出答案，但模型内部仍可能形成推理过程。hidden reasoning 是系统内部保留推理，short answer 是对用户隐藏过程，distilled rationale 是把长推理压成可控短解释。生产系统隐藏 CoT 常出于安全、隐私、可读性和避免暴露不稳定中间过程；评估时则要看最终正确率和必要解释质量。

训练时可用结构化字段：

```json
{"reasoning":"...","answer":"..."}
```

部署时只展示 `answer`。关键是验证无显式 CoT 时最终正确率，以及防止模型把隐藏推理泄露到用户可见输出。


## 多模态与具身智能

### 多模态大模型、VLM/VLA 和具身 Agent 面试常问什么？

- 频率：158
- 来源：xhs_6a0733b1, xhs_69fca9a5, xhs_69fb2bb8, xhs_69ea8685, xhs_69f0d3a5, xhs_69e4e580, xhs_69e391f1, xhs_69a240a0, xhs_69adb8c8, xhs_6988310c, xhs_69408c32, xhs_694b8e81, xhs_69023cf9, xhs_68ff77f4, xhs_68ee705e, +143
- 难度：进阶

**面试官常问：**

1. VLM/MLLM 的基本架构是什么？图像 token 如何接入 LLM？
2. CLIP、ViT、Q-Former、投影层分别起什么作用？
3. VLA/具身 Agent 和普通 VLM 的区别是什么？
4. 多模态模型的训练目标、数据和评估怎么设计？

典型 VLM 可以拆成视觉编码器、跨模态投影和 LLM：

$$
\begin{aligned}
Z_v &= VisionEncoder(I) \\
E_v &= Projector(Z_v) \in \mathbb{R}^{N_v \times d_{llm}} \\
Y &= LLM([E_v; E_{text}])
\end{aligned}
$$

CLIP 偏图文对齐，ViT 把图像切成 patch token，Q-Former/Projector 负责把视觉表示压缩或映射到 LLM hidden size。VLM 主要输出文本理解结果；VLA/具身 Agent 还要把语言规划变成动作，形成感知-规划-控制闭环。

多模态训练常见目标包括图文对比、caption、VQA、指令微调、动作克隆和环境反馈：

$$
L = L_{caption} + \lambda_1 L_{contrastive} + \lambda_2 L_{action} + \lambda_3 L_{preference}
$$

评估不能只看 VQA accuracy。具身/VLA 要看任务成功率、轨迹长度、碰撞/安全、泛化场景和 sim-to-real。面试回答要明确多模态的核心难点是跨模态对齐、视觉 token 成本、数据质量和闭环评估。

### 高频追问补充

#### 1. VLM 里视觉 token 如何接入 LLM，常见 projector 有哪些？

答：视觉 token 通常由 ViT/视觉编码器产生，再经 projector 对齐到 LLM hidden space，常见 projector 有 linear、MLP、Q-Former 和 cross-attention。

VLM 通常用视觉编码器把图像切成 patch/token，再通过 linear projector、MLP projector、Q-Former 或 cross-attention 接入 LLM。VLM 偏图文理解和生成，VLA/具身 Agent 还要把语言规划映射到动作控制，评估从 QA accuracy 扩展到闭环任务成功率、动作安全、恢复能力和真实环境泛化。位置编码要区分文本一维位置、图像二维空间位置、视频时间位置和动作序列位置。

典型 VLM 接入：

$$
image \rightarrow ViT \rightarrow projector \rightarrow LLM\ tokens
$$

VLA 进一步把输出映射到动作：

$$
(vision, instruction, state)\rightarrow action
$$

评估不能只看图文 QA，还要看闭环 task success、碰撞/安全、恢复能力和跨环境泛化。

#### 2. VLM 和 VLA/具身 Agent 的训练目标、评估指标有什么不同？

答：VLM 偏图文理解，VLA/具身 Agent 还要输出动作并在环境中闭环验证，所以评估更看任务成功和安全。

VLM 通常用视觉编码器把图像切成 patch/token，再通过 linear projector、MLP projector、Q-Former 或 cross-attention 接入 LLM。VLM 偏图文理解和生成，VLA/具身 Agent 还要把语言规划映射到动作控制，评估从 QA accuracy 扩展到闭环任务成功率、动作安全、恢复能力和真实环境泛化。位置编码要区分文本一维位置、图像二维空间位置、视频时间位置和动作序列位置。

典型 VLM 接入：

$$
image \rightarrow ViT \rightarrow projector \rightarrow LLM\ tokens
$$

VLA 进一步把输出映射到动作：

$$
(vision, instruction, state)\rightarrow action
$$

评估不能只看图文 QA，还要看闭环 task success、碰撞/安全、恢复能力和跨环境泛化。

#### 3. 多模态面试里如何回答图像位置编码、视频帧和动作 token？

答：图像用二维位置，视频还要时间位置，动作 token 有动作序列位置；拼接进 LLM 时要清楚模态边界和位置体系。

VLM 通常用视觉编码器把图像切成 patch/token，再通过 linear projector、MLP projector、Q-Former 或 cross-attention 接入 LLM。VLM 偏图文理解和生成，VLA/具身 Agent 还要把语言规划映射到动作控制，评估从 QA accuracy 扩展到闭环任务成功率、动作安全、恢复能力和真实环境泛化。位置编码要区分文本一维位置、图像二维空间位置、视频时间位置和动作序列位置。

典型 VLM 接入：

$$
image \rightarrow ViT \rightarrow projector \rightarrow LLM\ tokens
$$

VLA 进一步把输出映射到动作：

$$
(vision, instruction, state)\rightarrow action
$$

评估不能只看图文 QA，还要看闭环 task success、碰撞/安全、恢复能力和跨环境泛化。


## 手撕代码

### 大模型算法岗常见手撕代码题有哪些？

- 频率：118
- 来源：xhs_6a0fc50a, xhs_6a0d5f2d, xhs_6a004ca9, xhs_69e7b0f8, xhs_69e6254c, xhs_69e5baea, xhs_69d8778a, xhs_69e4e580, xhs_69d4af50, xhs_69d5c180, xhs_69ddf065, xhs_69dcbb63, xhs_69d8c35c, xhs_69c389db, xhs_69aaedf4, +103
- 难度：基础

**面试官常问：**

1. 大模型算法岗常见手撕代码题有哪些？
2. 三数之和、NMS、CrossEntropy、Attention 这些题如何写到可运行？
3. 手撕模型相关代码时 shape 和数值稳定怎么检查？
4. 代码题回答时如何说明复杂度和边界条件？

大模型算法岗手撕分两类：通用算法和模型相关实现。通用题要能讲复杂度；模型题要能讲 shape、数值稳定和梯度。

三数之和：

```python
def three_sum(nums):
    nums.sort()
    ans = []
    for i, x in enumerate(nums):
        if i and x == nums[i - 1]:
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = x + nums[l] + nums[r]
            if s == 0:
                ans.append([x, nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
                while l < r and nums[r] == nums[r + 1]:
                    r -= 1
            elif s < 0:
                l += 1
            else:
                r -= 1
    return ans
```

NMS：

```python
import torch

def nms(boxes, scores, iou_thr=0.5):
    keep = []
    idx = scores.argsort(descending=True)
    while idx.numel() > 0:
        i = idx[0]
        keep.append(i.item())
        if idx.numel() == 1:
            break
        cur = boxes[i]
        rest = boxes[idx[1:]]
        x1 = torch.maximum(cur[0], rest[:, 0])
        y1 = torch.maximum(cur[1], rest[:, 1])
        x2 = torch.minimum(cur[2], rest[:, 2])
        y2 = torch.minimum(cur[3], rest[:, 3])
        inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
        area_cur = (cur[2] - cur[0]) * (cur[3] - cur[1])
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = inter / (area_cur + area_rest - inter).clamp_min(1e-9)
        idx = idx[1:][iou <= iou_thr]
    return keep
```

准备顺序：CrossEntropy、LayerNorm/RMSNorm、Self-Attention、RoPE、LoRA、Top-p sampling、NMS、二分、双指针、BFS/DFS。每题都要能说时间复杂度、空间复杂度和边界条件。

### 高频追问补充

#### 1. 大模型算法岗手撕除了 Attention/CE，还会问哪些非 Hot100 题？

答：非 Hot100 题越来越偏模型组件，如 Attention、CE、RMSNorm、RoPE、LoRA merge、top-p、KV cache、beam search。

新增面经里手撕题更偏“模型组件 + 训练细节”，包括 Attention、CrossEntropy、LayerNorm/RMSNorm、RoPE、LoRA merge、top-p sampling、KV cache、两层 MLP、梯度裁剪、简单 beam search。准备方式是每个模块都按四步：写 shape、写公式、写 numpy/PyTorch、说数值稳定和边界条件。

手撕模型组件时固定四步：

```text
shape -> formula -> code -> stability/boundary
```

例如 RMSNorm：

```python
y = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight
```

#### 2. MTP Loss、RMSNorm、RoPE、KV cache 这些能不能手写？

答：MTP、RMSNorm、RoPE、KV cache 都能手写，关键是先写 shape 和公式，再落成 PyTorch/Numpy。

新增面经里手撕题更偏“模型组件 + 训练细节”，包括 Attention、CrossEntropy、LayerNorm/RMSNorm、RoPE、LoRA merge、top-p sampling、KV cache、两层 MLP、梯度裁剪、简单 beam search。准备方式是每个模块都按四步：写 shape、写公式、写 numpy/PyTorch、说数值稳定和边界条件。

手撕模型组件时固定四步：

```text
shape -> formula -> code -> stability/boundary
```

例如 RMSNorm：

```python
y = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight
```

#### 3. 面试里代码题如何从公式快速落到 PyTorch？

答：从公式落代码要按输入 shape、前向、mask/稳定性、反向或复杂度四步讲，别直接写黑盒调用。

新增面经里手撕题更偏“模型组件 + 训练细节”，包括 Attention、CrossEntropy、LayerNorm/RMSNorm、RoPE、LoRA merge、top-p sampling、KV cache、两层 MLP、梯度裁剪、简单 beam search。准备方式是每个模块都按四步：写 shape、写公式、写 numpy/PyTorch、说数值稳定和边界条件。

手撕模型组件时固定四步：

```text
shape -> formula -> code -> stability/boundary
```

例如 RMSNorm：

```python
y = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight
```

### 手撕两层 MLP 单步训练应该怎么推导？

- 频率：4
- 来源：xhs_6a0fc50a, xhs_67c26620, xhs_6a0ea83a, xhs_69f305ea
- 难度：基础

**面试官常问：**

1. 请手推两层 MLP 的 forward/backward/update。
2. MSE loss 下 W1、W2 的梯度 shape 分别是什么？
3. ReLU 的反向传播怎么写？如果换成 sigmoid 怎么改？
4. 手写训练 step 时为什么要特别注意 batch mean 的缩放？

手撕两层 MLP 的核心是 shape 和链式法则。设：

$$
\begin{aligned}
\text{X} &: [N, D] \\
\text{W1} &: [D, H], b1: [H] \\
\text{W2} &: [H, C], b2: [C] \\
Z1 &= X W1 + b1 \\
A1 &= ReLU(Z1) \\
Y_hat &= A1 W2 + b2 \\
L &= \operatorname{mean}((Y_hat - Y)^2)
\end{aligned}
$$

若 MSE 对所有元素取均值，则：

$$
\begin{aligned}
dY_hat &= 2 \cdot  (Y_hat - Y) / (N \cdot  C) \\
dW2 &= A1^T dY_hat \\
db2 &= sum_N dY_hat \\
dA1 &= dY_hat W2^T \\
dZ1 &= dA1 \cdot  1[Z1 > 0] \\
dW1 &= X^T dZ1 \\
db1 &= sum_N dZ1
\end{aligned}
$$

可运行手撕版：

```python
import torch

def train_step(X, Y, W1, b1, W2, b2, lr=1e-2):
    Z1 = X @ W1 + b1
    A1 = torch.clamp(Z1, min=0)
    Y_hat = A1 @ W2 + b2
    loss = ((Y_hat - Y) ** 2).mean()

    dY = 2 * (Y_hat - Y) / Y.numel()
    dW2 = A1.T @ dY
    db2 = dY.sum(dim=0)
    dA1 = dY @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(dim=0)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return loss
```

面试官常追问：如果激活是 sigmoid/tanh 怎么改？只需要替换 `dZ1 = dA1 * activation'(Z1)`。如果 loss 是 CrossEntropy，最后一层梯度通常是 `softmax(logits)-one_hot`。

### 高频追问补充

#### 1. 手撕两层 MLP 时，如果加 ReLU，反向传播在哪一步乘 mask？

答：加 ReLU 后，反向传播在从激活值回到 pre-activation 时乘 mask，也就是 `dZ1 = dA1 * (Z1 > 0)`。

最稳的方法是每一层都写出 shape：`X[N,D] -> Z1[N,H] -> A1[N,H] -> Y[N,C]`。ReLU 的反向是 `dZ1 = dA1 * (Z1 > 0)`；如果 loss 从 mean 变成 sum，梯度大约放大 `N` 倍，学习率要相应缩小。面试中 shape 对了通常就成功一半，别急着写代码，先在纸上标注每个梯度和参数同形。

两层 MLP 的最小反向推导：

$$
Z_1=XW_1+b_1,\quad A_1=ReLU(Z_1),\quad \hat{Y}=A_1W_2+b_2
$$

$$
dW_2=A_1^T d\hat{Y},\quad dZ_1=(d\hat{Y}W_2^T)\odot 1[Z_1>0],\quad dW_1=X^T dZ_1
$$

```python
dlogits = 2 * (y_hat - y) / y.numel()
dW2 = A1.T @ dlogits
dZ1 = (dlogits @ W2.T) * (Z1 > 0)
dW1 = X.T @ dZ1
```

#### 2. 如果 batch mean 写错成 sum，学习率需要怎么变？

答：loss 从 mean 误写成 sum 时，梯度大约放大 batch size 倍，相当于学习率被放大，训练会更不稳定。

最稳的方法是每一层都写出 shape：`X[N,D] -> Z1[N,H] -> A1[N,H] -> Y[N,C]`。ReLU 的反向是 `dZ1 = dA1 * (Z1 > 0)`；如果 loss 从 mean 变成 sum，梯度大约放大 `N` 倍，学习率要相应缩小。面试中 shape 对了通常就成功一半，别急着写代码，先在纸上标注每个梯度和参数同形。

两层 MLP 的最小反向推导：

$$
Z_1=XW_1+b_1,\quad A_1=ReLU(Z_1),\quad \hat{Y}=A_1W_2+b_2
$$

$$
dW_2=A_1^T d\hat{Y},\quad dZ_1=(d\hat{Y}W_2^T)\odot 1[Z_1>0],\quad dW_1=X^T dZ_1
$$

```python
dlogits = 2 * (y_hat - y) / y.numel()
dW2 = A1.T @ dlogits
dZ1 = (dlogits @ W2.T) * (Z1 > 0)
dW1 = X.T @ dZ1
```

#### 3. 手撕时如何快速检查矩阵 shape 没错？

答：手撕时先写每个张量 shape，再写梯度公式；梯度矩阵必须和对应参数同形。

最稳的方法是每一层都写出 shape：`X[N,D] -> Z1[N,H] -> A1[N,H] -> Y[N,C]`。ReLU 的反向是 `dZ1 = dA1 * (Z1 > 0)`；如果 loss 从 mean 变成 sum，梯度大约放大 `N` 倍，学习率要相应缩小。面试中 shape 对了通常就成功一半，别急着写代码，先在纸上标注每个梯度和参数同形。

两层 MLP 的最小反向推导：

$$
Z_1=XW_1+b_1,\quad A_1=ReLU(Z_1),\quad \hat{Y}=A_1W_2+b_2
$$

$$
dW_2=A_1^T d\hat{Y},\quad dZ_1=(d\hat{Y}W_2^T)\odot 1[Z_1>0],\quad dW_1=X^T dZ_1
$$

```python
dlogits = 2 * (y_hat - y) / y.numel()
dW2 = A1.T @ dlogits
dZ1 = (dlogits @ W2.T) * (Z1 > 0)
dW1 = X.T @ dZ1
```


## 公司面经

### 大模型算法/Agent 岗公司面经中最常被深挖哪些方向？

- 频率：728
- 来源：xhs_69eb90ca, xhs_69fff0a6, xhs_6a0fc50a, xhs_6a0d5f2d, xhs_6a01559b, xhs_6a015609, xhs_6a015529, xhs_6a0018b9, xhs_6a004ca9, xhs_69f49925, xhs_69e18d77, xhs_69f9bacb, xhs_69f4db22, xhs_69f70429, xhs_69f705e6, +713
- 难度：基础

**面试官常问：**

1. 大模型算法/Agent 岗面经里项目通常会被怎么深挖？
2. 如何把 RAG/SFT/RL 项目讲成一个完整工程闭环？
3. 面试官问失败 case、指标和替代方案时怎么回答？
4. 算法岗手撕和项目追问应该如何一起准备？

公司面经高频不是单个知识点，而是“项目闭环 + 基础机制 + 手撕实现”。回答项目时建议用固定结构：

```text
背景: 业务目标/问题规模
方案: 数据、模型、训练、推理、评估
难点: 遇到的失败 case 和定位过程
指标: 离线指标 + 线上指标 + 成本/延迟
取舍: 为什么不用其他方案
复盘: 如果重做会改什么
```

项目追问示例：

```text
RAG: chunk 怎么切？召回漏了怎么发现？rerank 为什么有效？
SFT: 数据怎么清洗？label mask 是否正确？为什么不用全参？
RL: reward 怎么设计？如何防 reward hacking？KL 怎么控？
Serving: QPS/延迟/显存怎么估？KV cache 怎么管理？
```

可以准备一个“项目回答卡片”：

```yaml
project:
  goal: "把客服知识问答准确率从 72% 提到 85%"
  data: "10w query-log + 3k 人工标注 + hard negative"
  method: "hybrid retrieval + rerank + answer verifier"
  metrics: ["recall@20", "faithfulness", "latency p95", "manual win rate"]
  failures: ["召回同义词漏", "长文档噪声", "答案引用错段落"]
  tradeoff: "RAG 更新快；SFT 只用于格式和工具调用"
```

面试官更看重你能不能解释为什么这样做、失败时怎么定位，而不是只报模型名字。

### 高频追问补充

#### 1. Agent 岗、RAG 工程岗、大模型算法岗的面试侧重点有什么不同？

答：Agent 岗更重系统设计、工具、记忆和评估；RAG 工程岗更重检索链路和忠实性；算法岗更重架构、训练、RL、MoE、推理优化和手撕。

Agent 岗重点是系统设计、工具调用、记忆、评估和工程落地；RAG 岗重点是切分、召回、rerank、忠实性和线上指标；算法岗更重 Transformer、后训练、RL、MoE、推理优化和手撕。项目追问通常按“为什么做、数据怎么来、模型/系统怎么选、指标怎么评估、失败 case 怎么处理、成本如何优化”推进。准备时每个项目都要有一张链路图、一组指标、一套 ablation 和三个失败案例。

项目回答模板：

```text
背景目标 -> 数据来源 -> 方法选择 -> 指标设计 -> 失败案例 -> 取舍与优化 -> 线上验证
```

每个项目至少准备一张链路图、三组 ablation、三个失败 case 和一组成本/延迟指标。

#### 2. 简历项目里写 Agent/RAG/LoRA，面试官最容易追哪几层？

答：项目追问最容易追数据来源、为什么选这个方案、指标怎么定义、失败 case、ablation 和线上成本。

Agent 岗重点是系统设计、工具调用、记忆、评估和工程落地；RAG 岗重点是切分、召回、rerank、忠实性和线上指标；算法岗更重 Transformer、后训练、RL、MoE、推理优化和手撕。项目追问通常按“为什么做、数据怎么来、模型/系统怎么选、指标怎么评估、失败 case 怎么处理、成本如何优化”推进。准备时每个项目都要有一张链路图、一组指标、一套 ablation 和三个失败案例。

项目回答模板：

```text
背景目标 -> 数据来源 -> 方法选择 -> 指标设计 -> 失败案例 -> 取舍与优化 -> 线上验证
```

每个项目至少准备一张链路图、三组 ablation、三个失败 case 和一组成本/延迟指标。

#### 3. 公司面经里频繁出现的“业务一面/二面/主管面”该怎么准备？

答：业务一面看项目细节，二面看技术深度和取舍，主管面看 owner 意识、协作、影响面和风险控制。

Agent 岗重点是系统设计、工具调用、记忆、评估和工程落地；RAG 岗重点是切分、召回、rerank、忠实性和线上指标；算法岗更重 Transformer、后训练、RL、MoE、推理优化和手撕。项目追问通常按“为什么做、数据怎么来、模型/系统怎么选、指标怎么评估、失败 case 怎么处理、成本如何优化”推进。准备时每个项目都要有一张链路图、一组指标、一套 ablation 和三个失败案例。

项目回答模板：

```text
背景目标 -> 数据来源 -> 方法选择 -> 指标设计 -> 失败案例 -> 取舍与优化 -> 线上验证
```

每个项目至少准备一张链路图、三组 ablation、三个失败 case 和一组成本/延迟指标。

### DeepSeek R1/V3/V4、Qwen、Kimi 等技术报告常问哪些架构创新？

- 频率：156
- 来源：xhs_6a11a95f, xhs_6a0efd3c, xhs_6a0806ae, xhs_69f017bc, xhs_6a089599, xhs_6a06c1b8, xhs_69fca9a5, xhs_69fb2bb8, xhs_69f31d0a, xhs_69d8778a, xhs_69d4f398, xhs_69ba051d, xhs_69a06a56, xhs_69cc6f17, xhs_69aaedf4, +141
- 难度：进阶

**面试官常问：**

1. DeepSeek/Qwen/Kimi 技术报告里最常被问的架构创新有哪些？
2. MLA、MoE、GRPO、FP8 分别解决什么瓶颈？
3. 如何从显存、算力、通信、数据成本四个角度解释架构创新？
4. 回答模型技术报告题时如何避免编造未公开细节？

这类题不要背成“某模型用了某模块”，要按模块回答。公开技术报告和面经里常见考点：

```text
Attention/KV: MQA/GQA/MLA, RoPE, 长上下文扩展
FFN/MoE: expert routing, shared expert, load balance, fine-grained experts
Training: 数据配比、FP8/BF16、并行训练、稳定性
Post-training: SFT、DPO/PPO/GRPO、rule reward、R1-style reasoning
Serving: KV cache、算子融合、量化、batching
Eval: reasoning、code、math、tool use、安全
```

回答模板：

```text
1. 这个创新解决什么瓶颈？
2. 公式或机制是什么？
3. 省的是显存、算力、通信还是数据成本？
4. 代价是什么？
5. 和 LLaMA/Qwen/Kimi/DeepSeek 其他方案怎么对比？
```

例如 MLA 重点是压缩 KV cache；MoE 重点是增加总容量但控制激活计算；GRPO/R1 类重点是用可验证 reward 和组内相对优势提升推理能力。面试要避免编造具体未公开参数，能说清机制和 tradeoff 比背版本号更重要。

可以把几个模块用统一公式串起来：

$$
\begin{aligned}
\text{KV 优化} &: bytes \approx L \cdot  T \cdot  kv_heads \cdot  head_dim \cdot  2 \cdot  dtype_bytes \\
\text{MoE 激活计算} &: active_params_per_token \ll  total_params \\
\text{RL 目标} &: max E[R(y)] - \beta KL(\pi_\theta || \pi_ref) \\
\text{量化收益} &: memory \approx params \cdot  bits / 8
\end{aligned}
$$

回答示例：如果被问 DeepSeek/Qwen/Kimi 的架构创新，可以先说“我按瓶颈拆”。长上下文瓶颈在 KV cache 和 attention IO，所以会出现 GQA/MLA、FlashAttention、prefix cache；模型容量瓶颈在 Dense FFN 计算，所以会出现 MoE、共享专家、负载均衡；推理能力瓶颈在后训练信号，所以会出现 SFT 到 RL、rule reward、GRPO 类方法；部署瓶颈在显存和吞吐，所以会出现 FP8/INT8、算子融合、continuous batching。最后补 tradeoff：省显存可能增加实现复杂度，MoE 提容量但加通信和路由难度，RL 提推理但带 reward hacking 与 KL 控制问题。

```python
def answer_arch_question(module):
    mapping = {
        "MLA": "压缩 KV 表示维度，降低长上下文 cache 显存",
        "MoE": "提高总参数容量，但每 token 只激活少量专家",
        "GRPO": "用组内相对奖励估优势，减少 critic 依赖",
        "FP8": "降低训练/推理显存和带宽，但要求校准与 kernel 支持",
    }
    return mapping.get(module, "先说明瓶颈，再讲机制、收益和代价")
```

### 高频追问补充

#### 1. DeepSeek R1/V3/V4 常被问的关键词如何串起来？

答：DeepSeek 技术报告题可以把 MLA、MoE、MTP、FP8、GRPO 串成 KV 压缩、容量扩展、训练信号、数值效率和后训练五条线。

技术报告题要按目标组织：MoE 提升容量/计算比，MLA 降低 KV cache，MTP 增强训练信号或推理适配，FP8 提升训练/推理效率，GRPO/RL 提升推理和对齐能力。不要只报名词，应该说“它解决什么瓶颈、用什么机制、代价是什么、怎么验证”。Qwen/Kimi/DeepSeek 的具体版本会变，但这套回答框架稳定。

回答技术报告题用固定框架：

$$
Module \rightarrow Bottleneck \rightarrow Mechanism \rightarrow Tradeoff \rightarrow Metric
$$

例如 MLA: KV cache 瓶颈 -> 低维潜表示 -> 省显存但实现复杂 -> 看长上下文吞吐和效果。

#### 2. Qwen/Kimi/DeepSeek 技术报告题，回答时是背参数还是讲设计取舍？

答：不要背参数表，面试官更想听设计解决什么瓶颈、带来什么代价、如何用指标验证。

技术报告题要按目标组织：MoE 提升容量/计算比，MLA 降低 KV cache，MTP 增强训练信号或推理适配，FP8 提升训练/推理效率，GRPO/RL 提升推理和对齐能力。不要只报名词，应该说“它解决什么瓶颈、用什么机制、代价是什么、怎么验证”。Qwen/Kimi/DeepSeek 的具体版本会变，但这套回答框架稳定。

回答技术报告题用固定框架：

$$
Module \rightarrow Bottleneck \rightarrow Mechanism \rightarrow Tradeoff \rightarrow Metric
$$

例如 MLA: KV cache 瓶颈 -> 低维潜表示 -> 省显存但实现复杂 -> 看长上下文吞吐和效果。

#### 3. MLA、MoE、MTP、FP8、GRPO 分别服务哪个目标？

答：MLA 服务 KV cache，MoE 服务容量/计算比，MTP 服务训练信号或推理适配，FP8 服务效率，GRPO 服务推理和对齐能力。

技术报告题要按目标组织：MoE 提升容量/计算比，MLA 降低 KV cache，MTP 增强训练信号或推理适配，FP8 提升训练/推理效率，GRPO/RL 提升推理和对齐能力。不要只报名词，应该说“它解决什么瓶颈、用什么机制、代价是什么、怎么验证”。Qwen/Kimi/DeepSeek 的具体版本会变，但这套回答框架稳定。

回答技术报告题用固定框架：

$$
Module \rightarrow Bottleneck \rightarrow Mechanism \rightarrow Tradeoff \rightarrow Metric
$$

例如 MLA: KV cache 瓶颈 -> 低维潜表示 -> 省显存但实现复杂 -> 看长上下文吞吐和效果。

## 待复核来源

- xhs_6a0aa47b: 从 KL 的方向看 SFT 与 RL：模型到底是在“ (title-only)
- xhs_69d67834: 小红书 ai 应用开发一面 (title-only)
- xhs_69d9d3c8: 腾讯 ai 应用开发 一面 (title-only)
- xhs_68159779: 做RL？先搞清楚Reward Model怎么搞 (title-only)
- xhs_682ab42c: 为什么你简历上的大模型SFT经历很水？ (title-only)
- xhs_67ad8f07: 美团大模型算法面经 (title-only)
