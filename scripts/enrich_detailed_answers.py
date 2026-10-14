#!/usr/bin/env python3
"""Enrich detailed answers with discovery/search follow-up questions."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import generate_notebook
from research_updates import curated_answers, sync_curated_answers


ROOT = Path(__file__).resolve().parents[1]
DETAIL_PATH = ROOT / "data" / "processed" / "detailed_answers.md"
SOURCES_PATH = ROOT / "data" / "processed" / "sources.csv"
START = "### 高频追问补充"
OLD_START = "### Discovery 995 条新增补充"


SUPPLEMENTS: dict[str, str] = {
    "transformer_arch": """新增常问问题：

1. 20 多种 Transformer 改进里，为什么最后留下来的通常是 Norm、RoPE、GQA/MQA、SwiGLU、FlashAttention 这几类？
2. CNN、RNN、Transformer 的归纳偏置分别是什么？为什么 LLM 更偏向 Transformer？
3. Multi-Head Attention 的多头到底在学什么？如果头数变多或变少会怎样？

补充答法：面试里不要把所有改进都背成名词，要按“稳定训练、长度外推、推理效率、表达能力”四条主线归类。Pre-Norm/RMSNorm 解决深层训练稳定性；RoPE/YaRN/NTK scaling 解决位置和长上下文；GQA/MQA/MLA 解决 KV cache；SwiGLU 改善 FFN 表达；FlashAttention 解决 IO 瓶颈。能留下来的改进通常满足两个条件：数学行为不破坏主干 attention/FFN，且能在训练或推理系统里稳定落地。""",
    "param_flops": """新增常问问题：

1. 训练 7B/13B 时，参数、梯度、优化器状态、激活、KV cache 哪一项最容易爆显存？
2. 为什么 prefill 和 decode 的瓶颈不同？吞吐该分别看什么？
3. MTP loss、多 token prediction 或 speculative decoding 会怎样改变计算量估算？

补充答法：参数显存只和参数量、dtype 有关；训练显存还要加梯度、Adam 状态和激活，激活通常随 `batch * seq_len * layers * hidden` 增长，长序列训练时很容易超过参数本身。prefill 是大矩阵乘和 `T^2` attention，更像吞吐问题；decode 每步只有新 token，但要读历史 KV，更容易受内存带宽、KV cache 和 batch 调度影响。回答估算题时先说明口径，再给公式，最后说 ZeRO/checkpoint/tensor parallel 分别切哪一块。""",
    "rope_position": """新增常问问题：

1. RoPE 外推失败时通常是高频维度出问题还是低频维度出问题？
2. NTK scaling、YaRN、位置插值为什么有时会牺牲短上下文效果？
3. 多模态/VLA/VLM 里把图像 patch 和文本 token 混在一起时，位置编码怎么处理？

补充答法：RoPE 的外推本质是训练时没见过更大的相位范围。缩放位置或频率可以让长距离相位变化慢下来，但也会改变训练长度内的相对距离分布，所以可能影响短文本。多模态场景里要区分一维文本位置、二维视觉位置和跨模态拼接位置，常见做法是视觉侧用 2D position 或 patch embedding，进入 LLM 后再统一成序列位置；面试重点是说明位置编码服务于“让模型知道 token 顺序/空间关系”，而不是死背某个实现。""",
    "norms": """新增常问问题：

1. 为什么 Transformer 面试经常问“为什么不用 BatchNorm”？
2. DeepSeek/Qwen/LLaMA 这类 LLM 为什么偏向 RMSNorm + Pre-Norm？
3. 如果 RMSNorm 放在 attention/FFN 前后不同位置，训练稳定性会怎么变？

补充答法：BatchNorm 依赖 batch 统计，NLP/LLM 的 batch、长度、mask、生成阶段都不稳定，训练和推理统计也难一致。LayerNorm/RMSNorm 按 hidden 维做单样本归一化，更适合变长序列和自回归生成。Pre-Norm 让残差分支形成更顺畅的梯度通道，深层模型更容易训；RMSNorm 不减均值，省计算，实践效果接近 LayerNorm。""",
    "cross_entropy": """新增常问问题：

1. MTP Loss 和普通 next-token CE 的区别是什么？
2. 为什么 LLM 训练里 prompt token 常常要 mask 掉？
3. label smoothing 会不会影响生成模型的校准和困惑度？

补充答法：next-token CE 只预测下一个 token；MTP/多 token prediction 会让模型同时预测未来多个位置，给中间层或额外 head 更密集的监督，可能改善表示和推理加速适配。SFT 里通常只对 assistant answer 部分算 loss，prompt/user/system token 作为条件，不应被模型“学习复读”。label smoothing 能缓解过度自信，但会改变目标分布，评估 PPL 时要注意训练目标和评估目标是否一致。""",
    "loss_spike": """新增常问问题：

1. 大模型训练 loss spike 后，是直接跳过 batch 还是从 checkpoint 回滚？
2. 为什么同一个坏样本不一定每次都会触发 spike？
3. loss spike 与 reward hacking、RL 后训练崩掉有什么联系？

补充答法：单次 spike 可以先定位 batch、保存样本 hash、观察 grad norm/NaN/overflow；如果后续能恢复，可能只需跳 batch 或裁剪梯度。如果 spike 后模型质量持续恶化，要从前一个稳定 checkpoint 回滚，并检查优化器状态。RL 后训练里 reward 分布突变、KL 约束过弱、格式奖励被 hack，都可能表现成 loss/ratio/KL 异常，因此要把 loss spike 和策略漂移一起看。""",
    "mlp_training": """新增常问问题：

1. 手撕两层 MLP 时，如果加 ReLU，反向传播在哪一步乘 mask？
2. 如果 batch mean 写错成 sum，学习率需要怎么变？
3. 手撕时如何快速检查矩阵 shape 没错？

补充答法：最稳的方法是每一层都写出 shape：`X[N,D] -> Z1[N,H] -> A1[N,H] -> Y[N,C]`。ReLU 的反向是 `dZ1 = dA1 * (Z1 > 0)`；如果 loss 从 mean 变成 sum，梯度大约放大 `N` 倍，学习率要相应缩小。面试中 shape 对了通常就成功一半，别急着写代码，先在纸上标注每个梯度和参数同形。""",
    "sft_rl_kl": """新增常问问题：

1. 为什么 SFT 后模型会更“平均”，RL 后更容易变得尖锐？
2. 正向 KL/反向 KL 的 mode-covering 和 mode-seeking 在回答质量上怎么体现？
3. DPO 里的 beta 或 KL 强度如何影响模型偏离 reference？

补充答法：SFT 用数据集里的标准答案做行为克隆，倾向覆盖数据分布中的多种写法；偏好优化/RL 会把概率质量推向高奖励答案，若约束太弱就会过度集中、风格变窄甚至 reward hacking。DPO/PPO/GRPO 都要处理“提高奖励”和“别偏离参考模型”的张力，beta/KL 系数越强，模型越保守；越弱，越容易学到偏好但也越可能漂移。""",
    "post_training_methods": """新增常问问题：

1. Agentic mid-training、Agentic post-training 和普通 SFT/RLHF 有什么区别？
2. DAPO、GSPO、GRPO 这类方法主要在改采样、优势估计还是稳定性？
3. 偏好数据、过程数据、工具调用轨迹分别适合哪种后训练方法？

补充答法：mid-training 更像在预训练和指令微调之间注入领域/能力数据，目标是补能力；post-training 更关注指令遵循、对齐、工具使用和偏好。SFT 需要高质量示范轨迹；DPO 需要 chosen/rejected 偏好对；PPO/GRPO/DAPO/GSPO 更偏在线或半在线采样，需要 reward、规则验证或环境反馈。Agentic 场景里，轨迹质量、工具结果、格式奖励和过程奖励比单轮问答更重要。""",
    "grpo_core": """新增常问问题：

1. GRPO 为什么可以不要 critic？代价是什么？
2. group size、reward std、长度归一化会怎样影响 advantage？
3. GRPO 和 DPO 都不用显式 value model，它们本质差别在哪里？

补充答法：GRPO 对同一 prompt 采样一组回答，用组内均值/标准差做 baseline，优势来自“比同组其他回答好多少”，所以省掉 critic。代价是需要多样采样和稳定 reward；如果组内 reward 全一样，advantage 近似为 0。DPO 是离线偏好分类目标，直接提升 chosen 相对 rejected；GRPO 是对当前策略采样结果做 RL 更新，更接近 on-policy，能利用环境/规则奖励，但训练更敏感。""",
    "grpo_metrics": """新增常问问题：

1. 训练 Agentic RL 时只看 reward mean 为什么不够？
2. clip fraction、KL、entropy、response length 分别异常说明什么？
3. tool success rate 和 final answer pass rate 冲突时听谁的？

补充答法：reward mean 上升可能来自输出变长、格式投机或奖励漏洞，所以必须同时看 reward std、KL、entropy、clip fraction、长度、格式准确率、工具成功率和人工抽检。KL 飙升说明策略偏离 reference；entropy 快速下降说明模式塌缩；clip fraction 过高说明更新过猛；工具成功率高但最终答案差，通常是规划/信息整合问题，不是单纯工具调用问题。""",
    "reward_advantage_prm": """新增常问问题：

1. PRM 和 ORM 的区别是什么？为什么 Agent 任务更需要过程奖励？
2. token-level advantage 如何从 step-level reward 分配回来？
3. 如何避免过程奖励鼓励模型写更多无用步骤？

补充答法：ORM 只评价最终答案，PRM 评价中间步骤。Agent 任务包含搜索、工具调用、反思、重试，最终对错很难告诉哪一步贡献最大，所以需要过程奖励做信用分配。token-level advantage 可以按步骤边界均摊，也可以用价值模型/奖励模型逐步估计；同时要对长度、重复、无效工具调用加惩罚，否则模型可能通过堆步骤拿过程分。""",
    "sft_to_rl": """新增常问问题：

1. SFT 没做好就上 RL 会发生什么？
2. 如何判断模型已经具备开始 RL 的 base capability？
3. SFT 后模型复读 prompt、格式僵硬或拒答变多怎么处理？

补充答法：SFT 是 RL 的地基。如果模型连格式、任务理解、基础正确率都不稳，RL 会放大奖励噪声，甚至学会投机。开始 RL 前至少要看：验证集格式准确率、基础任务 pass rate、采样多样性、拒答率、工具调用成功率。复读常来自模板过强、重复数据多、学习率过大或训练过久，处理方式是去重、混入通用指令数据、降低学习率、早停和增加负例。""",
    "distillation": """新增常问问题：

1. 搜索 Agent 自蒸馏、学生自采样再由教师打分，属于 on-policy 还是 off-policy？
2. 隐藏层蒸馏和 logits/文本蒸馏分别蒸什么？
3. 小模型蒸馏大模型时为什么容易学到格式但学不到推理能力？

补充答法：固定教师生成数据再训练学生是 off-policy；学生当前策略采样、教师/环境再反馈更接近 on-policy。logits 蒸馏学概率分布，文本蒸馏学最终输出，隐藏层蒸馏学中间表示。小模型容量有限，容易模仿表面格式却承载不了长链推理；要提高效果，需要课程化数据、过程监督、难例筛选、拒绝采样和面向学生能力的数据再生成。""",
    "moe": """新增常问问题：

1. MoE 为什么总参数多但单 token FLOPs 不一定多？
2. shared expert、fine-grained expert、top-k routing 各解决什么问题？
3. MoE 推理为什么常被通信和负载均衡卡住？

补充答法：MoE 每个 token 只激活少量专家，所以参数容量变大而计算量受控。shared expert 提供通用能力，细粒度专家提高专家组合灵活性，top-k routing 在表达和成本之间折中。系统瓶颈在 all-to-all 通信、专家负载不均、token dropping 和跨卡调度；所以除了模型效果，还要监控每专家 token 数、routing entropy、通信时间和容量因子。""",
    "moe_rl": """新增常问问题：

1. 为什么 RL 会让 MoE 专家利用率变差？
2. GRPO/DAPO 在 MoE 上为什么要特别关注 routing entropy？
3. reward 偏向某类输出时，是否会把梯度集中到少数专家？

补充答法：RL 改变输出分布，也改变 token 到专家的路由分布。如果高奖励样本集中在少数模式，路由器可能越来越偏向少数专家，造成专家塌缩、负载不均和泛化下降。训练时要同时看 reward/KL 和专家利用率，必要时加负载均衡损失、路由正则、容量控制，或者对采样数据做多样性约束。""",
    "gqa_kv_cache": """新增常问问题：

1. FP8 KV Cache 和 GQA/MQA 是互补还是替代？
2. 为什么 decode 阶段 KV cache 会成为吞吐瓶颈？
3. kv_heads、head_dim、seq_len 改变时，KV cache 如何线性变化？

补充答法：GQA/MQA 减少 KV head 数，FP8 KV cache 减少每个元素字节数，它们是互补的。decode 每生成一个 token 都要读历史 K/V，长上下文下读内存比算 QK 更容易成为瓶颈。公式仍是 `B * T * L * kv_heads * head_dim * 2 * bytes`，其中 `2` 是 K 和 V。面试追问时可以顺手比较 MHA、MQA、GQA、MLA 四种压缩路线。""",
    "mla_deepseek": """新增常问问题：

1. MLA 和 GQA/MQA 在压缩 KV cache 上的区别是什么？
2. DeepSeek 里 MLA 为什么常和 MoE、MTP、负载均衡一起被问？
3. 低秩潜表示会不会影响注意力表达能力？

补充答法：GQA/MQA 是减少 KV head 数，MLA 是把 K/V 压到低维潜表示再恢复，压的是表示维度。低秩会带来信息瓶颈，所以需要训练时共同适配，不是推理时随便替换。DeepSeek 架构题常把 MLA、DeepSeekMoE、MTP、FP8、负载均衡一起问，回答时按“省 cache、省计算、扩容量、提训练/推理效率”分层讲。""",
    "flash_attention": """新增常问问题：

1. FlashAttention 为什么说是 IO-aware，而不是 attention 近似？
2. online softmax 如何保证分块计算结果和标准 softmax 一致？
3. FlashAttention 对长上下文训练和推理分别省什么？

补充答法：FlashAttention 不改变精确 attention 结果，核心是分块计算并避免把完整 `T*T` attention matrix 写回 HBM。online softmax 在每个块维护当前最大值和归一化因子，合并块时重新缩放，因此数值上等价。训练时主要省 attention 中间矩阵显存和 HBM 读写；推理 prefill 也能提速，但 decode 仍常受 KV cache 读带宽影响。""",
    "sparse_linear_attention": """新增常问问题：

1. Attention 真被高估了吗？哪些任务可以用 sparse/linear attention 替代？
2. Linear Attention 为什么不总是比 softmax attention 好？
3. Gated Attention 和稀疏注意力的收益分别来自哪里？

补充答法：稀疏/线性注意力降低复杂度，但会牺牲全局精确交互或引入近似误差。对局部依赖强、超长上下文、检索式任务，它们可能很有价值；对需要任意 token 两两精细交互的任务，标准 softmax attention 仍强。Gated Attention 更像学习信息流开关，稀疏注意力更像减少连接边，面试中要讲清收益和代价，而不是只说复杂度从 `O(n^2)` 到 `O(n)`。""",
    "swiglu_ffn": """新增常问问题：

1. SwiGLU 相比 GELU FFN 为什么在 LLaMA/Qwen 中常见？
2. gate_proj、up_proj、down_proj 三个矩阵的 shape 怎么写？
3. FFN 参数量为什么常常超过 attention？

补充答法：SwiGLU 用门控分支控制通道信息，形式通常是 `down_proj(silu(gate_proj(x)) * up_proj(x))`。如果 hidden 是 `d`、中间维是 `d_ff`，三个矩阵大约是 `d*d_ff、d*d_ff、d_ff*d`，参数量约 `3*d*d_ff`，经常大于 attention 的 `4*d*d`。它的优势是表达能力和训练效果好，代价是多一条投影分支。""",
    "lora": """新增常问问题：

1. LoRA 微调项目从零做，数据、target modules、rank、alpha 怎么选？
2. QLoRA、DoRA、AdaLoRA/DyLoRA 各自解决什么问题？
3. 什么时候应该用 RAG，什么时候应该用 LoRA，什么时候全量微调？

补充答法：LoRA 的核心是把权重增量写成低秩 `Delta W = B A`。rank 控制容量，alpha 控制缩放，target modules 常从 `q_proj/v_proj` 扩到 `k/o/gate/up/down` 视任务复杂度而定。QLoRA 省基座显存；DoRA 分解方向和幅度；AdaLoRA/DyLoRA 动态或可变 rank。知识更新优先 RAG，稳定格式/风格和小能力迁移适合 LoRA，深度领域能力或大分布迁移才考虑全量。""",
    "rag_vs_finetune": """新增常问问题：

1. 本地知识库最后没用 RAG，可能是哪些原因？
2. RAG 准确率提升应该从切分、召回、rerank、生成哪一层入手？
3. RAG 和 Agent memory 的边界在哪里？

补充答法：RAG 不是默认最优。如果知识规模小、结构强、更新慢，规则/SQL/grep 可能更稳定；如果问题需要多跳推理，单次向量召回也不够。优化顺序建议：先做数据清洗和 chunk 设计，再看 embedding/召回，再加 rerank 和 query rewrite，最后做答案生成约束与引用校验。Agent memory 更偏个体历史和任务状态，RAG 更偏外部知识库；两者可以共享检索基础设施，但评估指标不同。""",
    "code_search_rag": """新增常问问题：

1. 为什么 Claude Code/Codex 类工具常用 grep/ripgrep 搜代码？
2. 代码库检索什么时候适合向量索引？
3. Agentic Search 和传统 RAG 的最大区别是什么？

补充答法：代码问题经常需要精确符号、路径、调用点和错误文本，`rg` 实时、可解释、不会索引过期。向量索引适合概念相似、文档说明、跨语言语义搜索，但对精确变量名和新改文件不如 grep 靠谱。Agentic Search 的关键是多轮：先宽搜，再读文件，再根据证据收窄查询，而不是一次 embedding 召回后直接回答。""",
    "agent_arch": """新增常问问题：

1. 企业级 Agent 为什么不能只有一个大模型 prompt？
2. Agent、Workflow、Harness、Runtime 分别是什么层？
3. 多 Agent 集群协作最容易踩哪些坑？

补充答法：企业级 Agent 至少要有 planner/executor/tool/memory/evaluator/observability/safety 几层。Workflow 是确定流程，适合稳定业务；Agent 是动态决策，适合开放任务；Harness/Runtime 负责把模型、工具、状态、权限、日志、重试连接起来。多 Agent 容易踩通信成本、职责重叠、状态不一致、循环调用和难评估的问题，所以生产系统常用“少 Agent + 明确工具 + 强观测”而不是盲目堆 Agent。""",
    "mcp_function_skill": """新增常问问题：

1. 面试官问 MCP 还是 CLI，怎么比较？
2. Tool、Function Call、MCP Server、Skill、Rules 在 Agent 系统里各管什么？
3. 写好一个 Agent Skill 为什么比单独暴露工具更重要？

补充答法：Function Call 是模型选择结构化函数参数；Tool 是可执行能力；MCP 是标准化暴露工具/资源/上下文的协议；CLI 是传统命令入口，更适合人和脚本；Skill 是把一组工具、文件、经验和失败处理固化成工作流；Rules 是全局约束。MCP 解决“能调用什么”，Skill 解决“什么时候、按什么顺序、用什么口径调用”。""",
    "agent_context_memory": """新增常问问题：

1. Agent Memory 的工作记忆、短期记忆、长期记忆、情景记忆分别是什么？
2. 长上下文和外部记忆是替代关系吗？
3. 上下文压缩如何避免丢掉关键约束？

补充答法：工作记忆是当前任务 scratchpad，短期记忆是本轮会话状态，长期记忆是跨任务偏好/事实，情景记忆是可检索的历史轨迹。长上下文能减少检索，但不是无限记忆；成本、噪声和注意力稀释仍存在。压缩时要保留目标、约束、已验证事实、未决问题、文件路径和失败记录，最好结构化成状态表，而不是随意摘要。""",
    "sampling": """新增常问问题：

1. temperature、top-p、top-k、repeat penalty 的执行顺序怎么理解？
2. 为什么面试/评测常用低温，创作常用高温？
3. temperature=0 为什么线上仍可能不完全确定？

补充答法：采样通常先得到 logits，做 temperature 缩放，再按 top-k/top-p 截断候选，最后采样。temperature 越高分布越平，随机性越强；top-p 保留累计概率质量，top-k 保留固定数量候选。评测要可复现，所以低温或 greedy；创作需要多样性，所以适当升温。线上不确定可能来自并行 kernel、浮点误差、tie-breaking、服务 fallback 和缓存状态。""",
    "serving_vllm": """新增常问问题：

1. vLLM 的 PagedAttention 到底解决了什么碎片问题？
2. SGLang 的 RadixAttention/prefix cache 为什么对 Agent 有价值？
3. HiCache、分层 KV cache、FP8 KV cache 分别优化哪一层？

补充答法：PagedAttention 把 KV cache 像虚拟内存一样分页，减少连续显存分配和碎片，提高 batch 调度效率。SGLang/RadixAttention 更强调前缀复用，Agent 多轮和工具调用常有共享 prompt/prefix，缓存命中能显著省 prefill。HiCache/分层缓存把 KV 放在 GPU/CPU/SSD 等层级之间权衡容量和带宽；FP8 KV cache 则减少单元素字节。""",
    "zero_parallel": """新增常问问题：

1. ZeRO-1/2/3 分别切分 optimizer、grad、param 的哪一块？
2. tensor parallel 和 pipeline parallel 的通信模式有什么不同？
3. gradient checkpointing 是省显存还是省计算？

补充答法：ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 连参数也切；tensor parallel 把单层矩阵切到多卡，层内通信多；pipeline parallel 把不同层放不同卡，难点是流水线气泡和 micro-batch 调度。checkpointing 省激活显存，但反向要重算 forward，所以用计算换显存。""",
    "quantization": """新增常问问题：

1. BF16、FP16、FP8、INT8/INT4 的差别不只是 bit 数，动态范围有什么区别？
2. KV cache 量化和 weight-only 量化分别影响什么？
3. calibration、scale、zero-point、group size 怎么解释？

补充答法：BF16 保留 FP32 的指数位，动态范围大；FP16 精度更细但范围小；FP8 需要更谨慎的 scale；INT8/INT4 通常要量化 scale。weight-only 量化主要省参数显存和带宽，KV cache 量化省长上下文缓存。对称量化常写作 `x_q = round(x / s)`，非对称量化加 zero-point；group size 越小越精细但 scale 开销越大。""",
    "evaluation": """新增常问问题：

1. Agent Eval 如何设计，为什么不能只看最终答案？
2. RAG 评估要分开看检索、rerank、生成还是端到端？
3. PPL 降低一定代表下游能力变好吗？

补充答法：评估要拆层：基础模型看 PPL/benchmark，RAG 看召回率、命中率、rerank NDCG、引用正确性和答案忠实性，Agent 看 task success、tool success、步骤数、成本、恢复能力和安全。PPL 是语言建模指标，能反映分布拟合，但不等于指令遵循、推理、工具调用或业务成功。面试中最好给一个离线集 + 在线观测 + 人工抽检的组合方案。""",
    "cot_train_infer": """新增常问问题：

1. 训练时用 CoT，推理时不展示 CoT，会不会能力下降？
2. hidden reasoning、short answer、distilled rationale 三者怎么区分？
3. 为什么有些系统要求只输出 final answer？

补充答法：训练 CoT 是给模型中间推理监督，推理时可以通过格式约束只输出答案，但模型内部仍可能形成推理过程。hidden reasoning 是系统内部保留推理，short answer 是对用户隐藏过程，distilled rationale 是把长推理压成可控短解释。生产系统隐藏 CoT 常出于安全、隐私、可读性和避免暴露不稳定中间过程；评估时则要看最终正确率和必要解释质量。""",
    "agent_eval_observability": """新增常问问题：

1. Agent Eval 为什么不能只看最终答案对不对？
2. Agent 可观测体系应该记录哪些字段，如何支持 replay？
3. 自动评测环境如何设计 task success、tool success 和安全指标？

补充答法：Agent 评估要拆最终结果、工具轨迹、成本延迟、恢复能力和安全边界。只看最终答案会掩盖错误工具调用、无效重试、幻觉引用和高成本路径。可观测体系要记录 prompt、模型输出、工具调用参数、工具返回、状态迁移、重试、人工接管和评测标签。自动评测环境要能构造任务、给出可验证反馈、支持轨迹 replay，并把 task success、tool success、step quality、cost、latency 分开统计。""",
    "agentic_rl_self_evolution": """新增常问问题：

1. Agentic RL 和普通 RLHF 最大区别是什么？
2. 自蒸馏/自进化 Agent 如何避免错误轨迹越训越多？
3. 在线策略蒸馏、环境反馈和教师模型反馈如何组合？

补充答法：Agentic RL 把模型放进多步工具环境里优化轨迹，不只是优化单轮答案偏好。自蒸馏通常让当前模型采样轨迹，再用规则、环境、教师或人工筛选成功样本回流训练；核心风险是错误自我强化，所以必须有可验证任务、失败轨迹过滤、难例再采样和 holdout eval。教师反馈适合开放质量判断，环境反馈适合代码、搜索、工具任务等可验证场景，两者可以组合成 outcome reward + process reward。""",
    "inference_cache_speculative": """新增常问问题：

1. Prefix Cache 和 RadixAttention 为什么特别适合 Agent 多轮任务？
2. HiCache/分层 KV cache 在 GPU、CPU、SSD 间怎么取舍？
3. Speculative Decoding 为什么能减少 decode 延迟，什么时候不划算？

补充答法：Prefix/Radix cache 复用共享 system prompt、历史轮次和工具模板，主要省 prefill；HiCache/分层 KV cache 用容量更大的低速层承接长上下文 KV，牺牲部分带宽换可服务长度；Speculative Decoding 用 draft model 先猜多个 token，再由 target model 并行验证，减少串行 decode 次数。三者优化对象不同：前缀复用、省 KV 容量、减少 target model 调用步数。""",
    "data_mixture_engineering": """新增常问问题：

1. 预训练、SFT、RL/偏好数据的配比目标分别是什么？
2. 数据去重、污染检测和质量过滤具体在防什么？
3. 数据合成和难例挖掘如何避免把模型训偏？

补充答法：预训练数据配比决定基础能力分布，SFT 数据决定指令行为和格式，RL/偏好数据决定模型偏好的方向和边界。数据工程要做去重、质量过滤、污染检测、来源版本、采样权重、难例挖掘和 per-domain eval。合成数据要防模板化、错误扩散和评测泄漏；难例要按能力缺口采样，并用 holdout 集验证是否只提升某类题而牺牲通用能力。""",
    "mtp_loss_speculative_training": """新增常问问题：

1. MTP Loss 与普通 next-token loss 的数学目标有什么不同？
2. 多 token prediction 如何服务 speculative decoding 或多 token 解码？
3. 多个 future token 的 loss 权重、label shift 和归一化怎么处理？

补充答法：普通 next-token 只预测下一个 token，MTP 会在同一位置或中间层同时预测多个未来 token，提供更密集监督。它可以训练 draft head、提升表示对未来信息的捕捉，并服务 speculative decoding。难点是未来步越远不确定性越大，辅助 loss 权重不能压过主目标；实现中要处理 horizon 对齐、mask、loss normalization 和推理时是否真的启用多 token head。""",
    "multimodal_vlm_vla": """新增常问问题：

1. VLM 里视觉 token 如何接入 LLM，常见 projector 有哪些？
2. VLM 和 VLA/具身 Agent 的训练目标、评估指标有什么不同？
3. 多模态面试里如何回答图像位置编码、视频帧和动作 token？

补充答法：VLM 通常用视觉编码器把图像切成 patch/token，再通过 linear projector、MLP projector、Q-Former 或 cross-attention 接入 LLM。VLM 偏图文理解和生成，VLA/具身 Agent 还要把语言规划映射到动作控制，评估从 QA accuracy 扩展到闭环任务成功率、动作安全、恢复能力和真实环境泛化。位置编码要区分文本一维位置、图像二维空间位置、视频时间位置和动作序列位置。""",
    "deepseek_arch": """新增常问问题：

1. DeepSeek R1/V3/V4 常被问的关键词如何串起来？
2. Qwen/Kimi/DeepSeek 技术报告题，回答时是背参数还是讲设计取舍？
3. MLA、MoE、MTP、FP8、GRPO 分别服务哪个目标？

补充答法：技术报告题要按目标组织：MoE 提升容量/计算比，MLA 降低 KV cache，MTP 增强训练信号或推理适配，FP8 提升训练/推理效率，GRPO/RL 提升推理和对齐能力。不要只报名词，应该说“它解决什么瓶颈、用什么机制、代价是什么、怎么验证”。Qwen/Kimi/DeepSeek 的具体版本会变，但这套回答框架稳定。""",
    "company_interview": """新增常问问题：

1. Agent 岗、RAG 工程岗、大模型算法岗的面试侧重点有什么不同？
2. 简历项目里写 Agent/RAG/LoRA，面试官最容易追哪几层？
3. 公司面经里频繁出现的“业务一面/二面/主管面”该怎么准备？

补充答法：Agent 岗重点是系统设计、工具调用、记忆、评估和工程落地；RAG 岗重点是切分、召回、rerank、忠实性和线上指标；算法岗更重 Transformer、后训练、RL、MoE、推理优化和手撕。项目追问通常按“为什么做、数据怎么来、模型/系统怎么选、指标怎么评估、失败 case 怎么处理、成本如何优化”推进。准备时每个项目都要有一张链路图、一组指标、一套 ablation 和三个失败案例。""",
    "coding_common": """新增常问问题：

1. 大模型算法岗手撕除了 Attention/CE，还会问哪些非 Hot100 题？
2. MTP Loss、RMSNorm、RoPE、KV cache 这些能不能手写？
3. 面试里代码题如何从公式快速落到 PyTorch？

补充答法：新增面经里手撕题更偏“模型组件 + 训练细节”，包括 Attention、CrossEntropy、LayerNorm/RMSNorm、RoPE、LoRA merge、top-p sampling、KV cache、两层 MLP、梯度裁剪、简单 beam search。准备方式是每个模块都按四步：写 shape、写公式、写 numpy/PyTorch、说数值稳定和边界条件。""",
}


QUESTION_ANSWER_OVERRIDES: dict[str, dict[int, str]] = {
    "transformer_arch": {
        1: """答：这题不要逐个背论文名，而要解释“为什么某些改造能活下来”。Transformer 主干长期稳定，是因为 attention 负责 token 间信息混合，FFN 负责逐 token 非线性变换；能留下来的改进通常是不破坏这条主干，同时明确改善一个工程瓶颈。

可以按四条线回答：

1. 稳定训练：Pre-Norm、RMSNorm 让深层梯度更稳，减少归一化计算和均值依赖。
2. 长度外推：RoPE、NTK scaling、YaRN 让位置信息以相对距离进入 QK 点积，并能扩展上下文。
3. 推理效率：GQA/MQA/MLA 减少 KV cache 体积，FlashAttention 减少 HBM 读写。
4. 表达能力：SwiGLU 在 FFN 中加入门控，提高通道选择能力。

面试最后补一句取舍：这些方案不是“越新越好”，而是要同时满足训练稳定、推理高效、算子可实现、和原有 checkpoint/框架兼容。""",
        2: """答：CNN 的归纳偏置是局部性和平移等变，适合图像网格；RNN 的归纳偏置是时间顺序和递归状态，适合流式序列但难并行；Transformer 的归纳偏置更弱，主要依靠 attention 学任意 token 间关系，配合位置编码表达顺序。

LLM 更偏向 Transformer 的原因有三点：第一，训练可并行，长文本上吞吐远好于 RNN；第二，attention 能直接连接远距离 token，缓解长程依赖；第三，架构可扩展性好，参数、数据和算力扩大后 scaling law 表现稳定。

但要注意 Transformer 不是所有场景都无脑最优。它的 `O(T^2)` attention 成本很高，所以长上下文、多模态和端侧部署仍会引出 sparse attention、linear attention、state space、混合架构等方向。""",
        3: """答：多头的直觉是把 `d_model` 切成多个子空间，让不同 head 学不同关系：有的关注局部邻近，有的关注语法依赖，有的关注特殊 token，有的在长上下文里做检索式匹配。数学上每个 head 有自己的 `W_Q/W_K/W_V`，最后 concat 后经过 `W_O` 融合。

头数变多时，每个 head 的 `d_head = d_model / num_heads` 会变小，单头表达能力下降，但关系类型更多；头数变少时单头维度变大，但多样性下降。推理阶段还要考虑 KV cache：MHA 每个 Q head 都有自己的 KV，缓存大；GQA/MQA 让多个 Q head 共享 KV，降低显存和带宽。

面试可以这样收束：head 数不是越多越好，要在表达多样性、head_dim、KV cache、吞吐和训练稳定性之间折中。""",
    },
    "post_training_methods": {
        1: """答：Agentic mid-training 更像能力注入，通常放在预训练和后训练之间，用代码、工具轨迹、搜索轨迹、长上下文任务等数据补“会做事”的能力；Agentic post-training 更像行为对齐和任务优化，用 SFT、DPO、GRPO/RL 等方法让模型在工具环境里更可靠。

普通 SFT/RLHF 多是单轮或短对话，输入输出比较固定；Agentic 训练面对的是多步轨迹：计划、工具调用、观察、反思、重试和最终回答。因此数据结构从 `(prompt, answer)` 变成 `(state, action, observation, reward)`，评估也从答案正确率扩展到工具成功率、轨迹成本和可恢复性。""",
        2: """答：DAPO、GSPO、GRPO 这类方法都围绕“让 RL 在大模型上更稳定、更便宜”展开，但侧重点不同。GRPO 用组内相对奖励估计 advantage，减少 critic；DAPO 更强调采样、clip、长度偏置、稳定训练等细节；GSPO 关注序列级或组级策略优化，试图让更新粒度更贴合生成任务。

回答时不要只背缩写，要按四个维度比较：是否需要 reward model/critic、是否 on-policy、advantage 怎么估计、如何控制 KL 和长度偏置。""",
        3: """答：偏好数据适合 DPO/IPO/KTO 或 reward model 训练，因为它告诉模型两个答案哪个更好；过程数据适合 PRM、step-level reward 或 trajectory SFT，因为它能做信用分配；工具调用轨迹适合 Agent SFT、Agentic RL 或环境可验证 RL，因为它包含 action 和 observation。

面试可以举例：数学题只有最终对错时可以做 outcome reward；代码题可以跑测试得到可验证 reward；搜索 Agent 需要检查 query、点击、引用和最终答案，最好结合过程奖励。""",
    },
    "grpo_core": {
        1: """答：GRPO 不要 critic 的关键是“同一 prompt 采样一组回答”。这组回答共享题目，奖励可以组内归一化：

$$
A_i = \\frac{r_i - \\operatorname{mean}(r)}{\\operatorname{std}(r)+\\epsilon}
$$

这样组平均就充当 baseline，避免训练 value model。代价是每个 prompt 要采多条，采样成本变高；如果组内 reward 都一样，advantage 接近 0，学不动；如果 reward 噪声大，组内排序也会不稳定。""",
        2: """答：group size 越大，组内 baseline 和标准差越稳定，但 rollout 成本越高。reward std 太小，advantage 会被噪声主导或接近 0；reward std 太大，更新方差大。长度归一化也很关键，因为长回答 token 更多，如果 loss 按 token 累加，长答案可能获得更大梯度权重。

工程上要同时监控 reward mean/std、response length、KL、clip fraction 和 pass rate，不能只看 reward 均值。""",
        3: """答：DPO 是离线偏好优化，它直接使用 chosen/rejected 对，把偏好学习写成分类式目标；GRPO 是从当前策略采样多条回答，根据 reward 做策略梯度式更新。二者都可以不训练 value model，但数据闭环不同。

DPO 成本低、稳定、适合已有偏好数据；GRPO 更适合有环境反馈、规则验证或在线采样的任务，比如数学、代码、工具 Agent，但训练更敏感。""",
    },
    "agent_context_memory": {
        1: """答：工作记忆是当前任务正在使用的信息，例如当前计划、临时变量、刚读过的文件片段；短期记忆是本轮会话/任务状态，例如已做步骤、失败尝试、用户最新约束；长期记忆是跨任务保留的偏好、事实和经验；情景记忆是历史任务轨迹，可按相似任务检索。

面试中要强调：这些不是必须用四个数据库实现，而是四种生命周期和读写策略。工作记忆常在上下文里，长期/情景记忆更适合外部存储和检索。""",
        2: """答：长上下文和外部记忆不是替代关系。长上下文降低了短期检索成本，但上下文越长，成本越高、噪声越多、注意力越分散；外部记忆可以长期保存、结构化检索、跨会话复用。

比较好的架构是：当前高优先级约束直接 pin 在上下文；大段历史和知识放外部记忆；每轮按任务目标检索，再把少量证据放进上下文。""",
        3: """答：上下文压缩最怕把约束压没或把不确定信息压成事实。压缩时要保留：目标、用户硬约束、已验证事实、文件路径/来源、失败尝试、当前计划、未决问题。可以用结构化摘要而不是纯自然语言。

例如压缩成：`Goal / Constraints / Evidence / Decisions / Open Issues / Next Actions`。每条 evidence 最好带 source id，后续需要时可回读原文。""",
    },
    "mcp_function_skill": {
        1: """答：MCP 和 CLI 不是同层概念。CLI 是人或脚本调用程序的命令行接口，简单直接；MCP 是给 Agent 暴露 tools/resources/prompts 的协议，让模型能发现能力、读取上下文并用结构化方式调用。

面试可以说：CLI 适合稳定脚本和本地自动化；MCP 适合多工具、多资源、跨应用的 Agent 生态。很多 MCP server 底层也可能调用 CLI。""",
        2: """答：Function Call 是模型输出结构化参数的能力；Tool 是被调用的外部能力；MCP Server 是标准化承载 tool/resource 的服务；Skill 是把工具、流程、经验、边界条件组织成可复用工作流；Rules 是长期约束。

一句话：Function Call 解决“怎么调”，MCP 解决“调什么和从哪拿上下文”，Skill 解决“按什么流程完成任务”，Rules 解决“永远不能越过哪些边界”。""",
        3: """答：单独暴露工具只告诉 Agent 有锤子，但不告诉什么时候敲、敲哪里、失败怎么处理。Skill 把任务流程写下来，包括该读哪些文件、如何验证、哪些 raw 文件不能提交、出错如何降级。

对复杂工作流，Skill 能减少每次从零探索的成本，也能降低误操作。比如小红书采集 skill 里会记录 signed URL、风控冷却、raw 文件 gitignore，这些不是单个 tool 能表达完整的。""",
    },
}


TOPIC_QUESTION_POINTS: dict[str, list[str]] = {
    "param_flops": [
        "训练显存里最容易爆的通常不是参数本身，而是 Adam 优化器状态和长序列激活；长上下文训练时激活随 batch、序列长度和层数放大，很容易超过参数显存。",
        "prefill 一次吃完整 prompt，瓶颈更偏大矩阵乘和 attention 的并行吞吐；decode 每步只生成一个 token，但要反复读历史 KV，所以更容易被内存带宽、KV cache 布局和调度卡住。",
        "MTP 和 speculative decoding 会额外引入 draft/head/验证开销，但目标是减少串行 decode 步数；估算时要把训练额外 loss head 的 FLOPs 和推理少跑 target model 的收益分开算。",
    ],
    "rope_position": [
        "RoPE 外推失败本质是训练没见过更大的相位范围，高频维度更容易在长距离上相位旋转过快，导致相对位置关系失真。",
        "NTK scaling、YaRN、位置插值会改变训练长度内的相对距离分布，所以长上下文变好时，短上下文可能因为位置尺度被压缩而轻微掉点。",
        "多模态里要同时处理文本一维顺序、图像二维空间、视频时间和跨模态拼接位置，不能简单把所有 patch 当成普通文本 token 位置。",
    ],
    "norms": [
        "BatchNorm 依赖 batch 统计，而 LLM 的变长序列、mask、padding 和自回归推理都会让 batch 统计不稳定，所以现代 LLM 更偏 LayerNorm/RMSNorm。",
        "RMSNorm + Pre-Norm 的优势是计算更省、残差梯度路径更顺，适合很深的 decoder-only 模型。",
        "Norm 放在子层前更利于深层训练稳定；放在子层后输出分布更规整，但深层模型更容易梯度不稳或需要更仔细的 warmup。",
    ],
    "cross_entropy": [
        "MTP Loss 在普通 next-token CE 外增加多个未来 token 的监督，目标是给表示更密集的未来信息信号。",
        "SFT 中 prompt token 是条件，不是 assistant 要学习复述的目标；通常只对 assistant answer 部分算 loss。",
        "label smoothing 会降低过度自信、改善校准，但它改变了目标分布，因此 PPL 和 hard-label CE 的解释要分开。",
    ],
    "loss_spike": [
        "如果 spike 后能恢复，可以先跳 batch、裁剪梯度并记录样本；如果质量持续恶化，应从 spike 前 checkpoint 连同优化器状态一起回滚。",
        "同一个坏样本不一定稳定触发 spike，因为触发条件还取决于当前参数、optimizer state、loss scale、batch 组合和随机种子。",
        "RL 后训练中的 reward hacking 会表现为 reward、KL、ratio、长度或格式指标异常，本质上也是优化目标被异常放大。",
    ],
    "mlp_training": [
        "加 ReLU 后，反向传播在从激活值回到 pre-activation 时乘 mask，也就是 `dZ1 = dA1 * (Z1 > 0)`。",
        "loss 从 mean 误写成 sum 时，梯度大约放大 batch size 倍，相当于学习率被放大，训练会更不稳定。",
        "手撕时先写每个张量 shape，再写梯度公式；梯度矩阵必须和对应参数同形。",
    ],
    "sft_rl_kl": [
        "SFT 学数据集中多种可接受答案，行为上更覆盖模式；RL/偏好优化把概率推向高奖励答案，更容易让分布变尖。",
        "mode-covering 体现为回答风格更平均、更保守；mode-seeking 体现为高奖励模式更强，但也更容易丢多样性。",
        "DPO 的 beta 越大，参考模型约束越强，更新越保守；beta 越小，偏好学习更激进，也更容易偏离 reference。",
    ],
    "grpo_metrics": [
        "只看 reward mean 不够，因为 reward 上升可能来自输出变长、格式投机或奖励漏洞，必须同时看 reward std、KL、长度和真实 pass rate。",
        "clip fraction 高说明更新过猛，KL 高说明偏离 reference，entropy 快速下降说明模式塌缩，response length 异常常提示长度奖励或格式奖励被利用。",
        "tool success rate 只说明工具调用链路跑通，final answer pass rate 才说明任务完成；二者冲突时要回放轨迹看是规划、检索、工具参数还是信息整合出错。",
    ],
    "reward_advantage_prm": [
        "ORM 只评价最终答案，PRM 评价中间步骤；Agent 多步任务需要 PRM 来判断计划、检索、工具调用和反思哪一步贡献了成败。",
        "step-level reward 可以按步骤边界均摊给该步骤 token，也可以训练 value/reward model 给每个 token 或 span 估计 advantage。",
        "过程奖励要惩罚无效步骤、重复搜索、无意义反思和过长轨迹，否则模型会学会堆步骤而不是解决任务。",
    ],
    "sft_to_rl": [
        "SFT 没做好就上 RL，会在低质量策略附近探索，reward 噪声会被放大，常见结果是格式投机、复读、KL 飙升或 pass rate 不稳。",
        "开始 RL 前至少要确认格式准确率、基础任务正确率、采样多样性和拒答率过线；Agent 任务还要看工具调用成功率。",
        "复读 prompt、格式僵硬或拒答变多，通常来自模板过强、重复数据多、学习率大或训练过久，要做去重、混入通用数据、降学习率和早停。",
    ],
    "distillation": [
        "学生当前策略采样、再由教师或环境打分，更接近 on-policy；固定教师数据训练学生是 off-policy。",
        "logits 蒸馏学概率分布，文本蒸馏学最终输出，隐藏层蒸馏学中间表示或表示空间对齐。",
        "小模型容量有限，容易学到表面格式但承载不了长链推理，因此要用课程数据、过程监督、难例筛选和拒绝采样补能力。",
    ],
    "moe": [
        "MoE 总参数大是因为有很多专家，但每个 token 只激活 top-k 专家，所以单 token FLOPs 不一定随总参数线性增长。",
        "shared expert 提供通用能力，fine-grained expert 提高组合粒度，top-k routing 在表达能力、计算量和通信之间折中。",
        "MoE 推理常被 all-to-all 通信、专家负载不均、capacity overflow 和 token dropping 卡住，系统指标和模型指标要一起看。",
    ],
    "moe_rl": [
        "RL 会改变输出 token 分布，也改变路由器看到的数据分布，所以专家利用率可能向少数高奖励路径集中。",
        "routing entropy 下降说明专家选择变窄，可能出现专家塌缩；GRPO/DAPO 的采样和 reward 方差会放大这种风险。",
        "如果 reward 偏向某类输出，梯度会更多流向处理这类 token 的专家，需要负载均衡、路由正则和数据多样性约束。",
    ],
    "gqa_kv_cache": [
        "FP8 KV cache 减少每个元素字节数，GQA/MQA 减少 KV head 数，它们作用在公式的不同因子上，是互补关系。",
        "decode 每步都要读历史 K/V，序列越长读内存越多，所以吞吐常受 KV 带宽和调度限制。",
        "KV cache 对 batch、seq_len、layers、kv_heads、head_dim 和 dtype bytes 都是线性关系，改任一项都会线性改变显存。",
    ],
    "mla_deepseek": [
        "GQA/MQA 压缩的是 KV head 数，MLA 压缩的是 K/V 缓存表示维度，二者都是减少 K/V 冗余但机制不同。",
        "DeepSeek 架构题常把 MLA、MoE、MTP、FP8、负载均衡放一起问，因为它们分别对应 KV、容量、训练信号、数值效率和系统吞吐。",
        "低秩潜表示会带来信息瓶颈，不能推理时硬替换，必须训练时让模型共同适配。",
    ],
    "flash_attention": [
        "FlashAttention 是 IO-aware exact attention，它不近似 softmax，而是减少 HBM 读写和中间矩阵 materialization。",
        "online softmax 通过维护块内最大值和归一化因子，把分块 softmax 合并成与全量 softmax 等价的结果。",
        "训练时主要省 attention score 矩阵显存和 HBM 往返；推理 prefill 也受益，但 decode 往往仍被 KV cache 读带宽限制。",
    ],
    "sparse_linear_attention": [
        "Attention 没有被简单高估；稀疏/线性 attention 适合局部依赖强或超长上下文任务，但会牺牲部分全局精确交互。",
        "Linear Attention 不总是更好，因为核近似或递推状态可能损失 softmax attention 的表达能力和训练稳定性。",
        "Gated Attention 的收益来自学习信息流开关，稀疏注意力的收益来自减少连接边和计算量，二者不是一回事。",
    ],
    "swiglu_ffn": [
        "SwiGLU 常见是因为门控分支能动态选择通道信息，实践上比 ReLU/GELU FFN 更适合大模型扩展。",
        "若 hidden size 是 `d`、FFN 中间维是 `d_ff`，gate/up/down 三个矩阵通常对应 `d x d_ff`、`d x d_ff`、`d_ff x d`。",
        "FFN 参数量常超过 attention，因为 SwiGLU 有三块大矩阵，而 attention 常见 Q/K/V/O 合计约 `4d^2`。",
    ],
    "lora": [
        "LoRA 项目先定任务和数据，再选 target modules；简单风格任务可从 q/v 开始，复杂能力迁移常扩到 k/o/gate/up/down。",
        "QLoRA 省基座显存，DoRA 分离方向和幅度，AdaLoRA/DyLoRA 动态或可变 rank，核心都是改善低秩适配的容量/成本权衡。",
        "知识更新优先 RAG，稳定格式和小能力迁移适合 LoRA，深度领域能力或大分布迁移才考虑全量微调。",
    ],
    "rag_vs_finetune": [
        "本地知识库没用 RAG 可能是因为知识量小、结构强、更新慢，规则、SQL、grep 或直接上下文更稳定。",
        "RAG 提升要按链路定位：切分清洗、召回、rerank、上下文压缩、生成约束和引用校验，不能只换 embedding。",
        "RAG 管外部知识，Agent memory 管个体历史和任务状态；二者都可检索，但评估指标不同。",
    ],
    "code_search_rag": [
        "代码检索需要精确符号、路径、错误文本和调用点，grep/ripgrep 实时、可解释、不会索引过期。",
        "向量索引适合概念相似、跨语言说明和文档语义检索，但对刚改代码、变量名和精确错误不如 grep。",
        "Agentic Search 是多轮搜读改写查询的过程，传统 RAG 更像一次召回后生成，代码库任务更需要前者。",
    ],
    "agent_arch": [
        "企业级 Agent 不能只有 prompt，因为生产系统还需要权限、工具参数校验、状态管理、超时重试、观测和人工确认。",
        "Workflow 是固定流程，Agent 是动态决策，Harness/Runtime 负责把模型、工具、状态、日志和调度连接起来。",
        "多 Agent 最容易踩职责重叠、循环调用、状态不一致、通信成本和难评估问题，要先证明单 Agent + 工具不够。",
    ],
    "sampling": [
        "常见顺序是 logits 先经 temperature 缩放，再做 top-k/top-p 截断，最后重新归一化并采样；repeat penalty 会先调整已出现 token 的 logits。",
        "评测低温是为了可复现和低方差，创作高温是为了多样性；RL rollout 温度还会影响 reward 方差和探索范围。",
        "temperature=0 仍可能因浮点误差、并行 kernel、tie-breaking、服务 fallback 或缓存状态出现细微不确定。",
    ],
    "serving_vllm": [
        "PagedAttention 把 KV cache 分块管理，避免为每个请求分配大段连续显存，减少碎片和浪费。",
        "RadixAttention/prefix cache 对 Agent 有价值，因为 system prompt、工具说明和历史前缀常重复，可直接省 prefill。",
        "HiCache 优化 KV 容量层级，FP8 KV cache 优化单元素字节，分层 KV 在 GPU/CPU/SSD 间用带宽换容量。",
    ],
    "zero_parallel": [
        "ZeRO-1 切 optimizer states，ZeRO-2 再切 gradients，ZeRO-3 连 parameters 也切；越往后省显存越多但通信越复杂。",
        "Tensor parallel 切单层矩阵，通信多发生在层内 all-reduce/all-gather；pipeline parallel 按层切，难点是流水线气泡。",
        "Gradient checkpointing 是省激活显存，不是省计算；反向时要重算一部分 forward，用算力换显存。",
    ],
    "quantization": [
        "BF16 指数范围接近 FP32，训练更稳；FP16 范围小；FP8/INT8/INT4 更省显存和带宽，但更依赖 scale、校准和 kernel。",
        "weight-only 量化主要省权重显存和加载带宽，KV cache 量化主要省长上下文 decode 时的缓存容量和读带宽。",
        "scale 决定浮点到整数的比例，zero-point 处理非对称零点，group size 越小越精细但 scale 开销越大。",
    ],
    "evaluation": [
        "Agent Eval 不能只看最终答案，因为同样答对可能有完全不同的工具成本、风险和可恢复性。",
        "RAG 要拆检索、rerank、上下文构造和生成忠实性；端到端分数只能说明整体结果，不能定位问题。",
        "PPL 降低说明语言建模拟合更好，但不等于指令遵循、推理、工具调用或业务成功一定提升。",
    ],
    "cot_train_infer": [
        "训练用 CoT、推理隐藏 CoT 不一定掉能力，但要用无 CoT 验证集确认模型不是强依赖显式长推理文本。",
        "hidden reasoning 是系统内部推理，short answer 是只展示结论，distilled rationale 是把长推理蒸馏成短解释。",
        "系统只输出 final answer 通常是为了安全、隐私、稳定性、可读性和避免暴露不可靠的中间推理。",
    ],
    "agent_eval_observability": [
        "Agent Eval 要看轨迹，不只看答案；答案对但工具乱调、成本过高或靠错误证据生成，都不能算稳定能力。",
        "可观测字段要支持 replay，所以必须记录输入、模型动作、工具参数、工具结果、状态迁移、错误和重试。",
        "自动评测环境要把 task success、tool success、安全、成本和延迟拆开算，并保留失败轨迹用于回归。",
    ],
    "agentic_rl_self_evolution": [
        "Agentic RL 的对象是多步轨迹，普通 RLHF 多优化单轮回答偏好；前者更依赖环境反馈、工具结果和过程奖励。",
        "自蒸馏要避免错误轨迹回流，必须用规则验证、教师复核、失败样本过滤和独立 holdout 集。",
        "教师模型适合评价开放质量，环境反馈适合可验证任务，在线策略蒸馏适合让学生覆盖自己的部署分布。",
    ],
    "inference_cache_speculative": [
        "Prefix/Radix cache 适合 Agent 多轮，是因为多轮共享 system prompt、工具 schema 和历史前缀，缓存命中能显著省 prefill。",
        "HiCache 把 KV 放到 GPU/CPU/SSD 等层级，GPU 快但贵，CPU/SSD 容量大但带宽和延迟差。",
        "Speculative Decoding 当 draft model 准确且便宜时划算；如果草稿接受率低或验证开销大，收益会被抵消。",
    ],
    "data_mixture_engineering": [
        "预训练配比决定基础能力和知识覆盖，SFT 配比决定指令行为，RL/偏好数据决定模型往什么偏好方向收缩。",
        "去重防记忆和过拟合，污染检测防评测泄漏，质量过滤防噪声样本主导梯度。",
        "合成数据和难例挖掘要做来源标记、难度分层和 holdout eval，否则很容易把模型训成模板化或偏科。",
    ],
    "mtp_loss_speculative_training": [
        "MTP 的目标是在一个位置预测多个未来 token，而普通 CE 只预测下一个 token。",
        "多 token prediction 可以训练 draft head 或未来 token head，为 speculative decoding 和多 token 解码提供候选。",
        "多个 future token 要做 horizon 权重、mask 和归一化；远未来不确定性更大，辅助 loss 不能压过主 next-token loss。",
    ],
    "multimodal_vlm_vla": [
        "视觉 token 通常由 ViT/视觉编码器产生，再经 projector 对齐到 LLM hidden space，常见 projector 有 linear、MLP、Q-Former 和 cross-attention。",
        "VLM 偏图文理解，VLA/具身 Agent 还要输出动作并在环境中闭环验证，所以评估更看任务成功和安全。",
        "图像用二维位置，视频还要时间位置，动作 token 有动作序列位置；拼接进 LLM 时要清楚模态边界和位置体系。",
    ],
    "deepseek_arch": [
        "DeepSeek 技术报告题可以把 MLA、MoE、MTP、FP8、GRPO 串成 KV 压缩、容量扩展、训练信号、数值效率和后训练五条线。",
        "不要背参数表，面试官更想听设计解决什么瓶颈、带来什么代价、如何用指标验证。",
        "MLA 服务 KV cache，MoE 服务容量/计算比，MTP 服务训练信号或推理适配，FP8 服务效率，GRPO 服务推理和对齐能力。",
    ],
    "company_interview": [
        "Agent 岗更重系统设计、工具、记忆和评估；RAG 工程岗更重检索链路和忠实性；算法岗更重架构、训练、RL、MoE、推理优化和手撕。",
        "项目追问最容易追数据来源、为什么选这个方案、指标怎么定义、失败 case、ablation 和线上成本。",
        "业务一面看项目细节，二面看技术深度和取舍，主管面看 owner 意识、协作、影响面和风险控制。",
    ],
    "coding_common": [
        "非 Hot100 题越来越偏模型组件，如 Attention、CE、RMSNorm、RoPE、LoRA merge、top-p、KV cache、beam search。",
        "MTP、RMSNorm、RoPE、KV cache 都能手写，关键是先写 shape 和公式，再落成 PyTorch/Numpy。",
        "从公式落代码要按输入 shape、前向、mask/稳定性、反向或复杂度四步讲，别直接写黑盒调用。",
    ],
}


TOPIC_TECH_DETAILS: dict[str, str] = {
    "param_flops": r"""常用估算式：

$$
M_{param}=N_{param}\cdot bytes
$$

$$
M_{KV}=B\cdot T\cdot L\cdot H_{kv}\cdot D_{head}\cdot 2\cdot bytes
$$

Adam 训练还要考虑参数、梯度、一阶/二阶动量和可能的 FP32 master weight。激活粗略随 `B * T * L * hidden` 增长，checkpointing 省激活，ZeRO 切 optimizer/grad/param，tensor parallel 切层内矩阵。""",
    "rope_position": r"""RoPE 可理解为把每对 hidden 维旋转一个与位置有关的角度：

$$
q_m = R(m\theta)q,\quad k_n = R(n\theta)k
$$

内积会包含相对距离：

$$
q_m^T k_n = q^T R((n-m)\theta)k
$$

长上下文缩放改变的是相位增长速度；回答时要同时说收益是外推，代价是训练长度内的位置分布被改变。""",
    "norms": r"""LayerNorm 与 RMSNorm 的差别可以直接写公式：

$$
LayerNorm(x)=g\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+b
$$

$$
RMSNorm(x)=g\frac{x}{\sqrt{mean(x^2)+\epsilon}}
$$

Pre-Norm 子层通常是 `x + Sublayer(Norm(x))`，Post-Norm 是 `Norm(x + Sublayer(x))`。深层 LLM 偏 Pre-Norm，因为残差路径更利于梯度传播。""",
    "cross_entropy": r"""数值稳定 CE 不要先 softmax 再 log：

$$
CE_i=\log\sum_j e^{z_{ij}}-z_{i,y_i}
$$

梯度是：

$$
\frac{\partial L}{\partial z}=p-onehot(y)
$$

SFT 中常把非 assistant token 的 label 置为 `-100`。MTP 可写为 `L = L_next + sum_k lambda_k L_{t+k}`，label smoothing 则把 one-hot 改成软标签。""",
    "loss_spike": r"""排查顺序可以固定成：数据样本 hash、loss scale/NaN/Inf、grad norm、学习率和 warmup、optimizer state、最近数据混合比例变化、验证集质量。若使用混合精度，还要看 overflow 和动态 loss scaling。

```python
if not torch.isfinite(loss) or grad_norm > threshold:
    save_bad_batch_ids()
    skip_or_rollback()
```
""",
    "mlp_training": r"""两层 MLP 的最小反向推导：

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
""",
    "sft_rl_kl": r"""两个 KL 方向的直觉：

$$
D_{KL}(p_{data}\|p_\theta)
$$

更怕漏掉数据模式，偏 mode-covering；

$$
D_{KL}(p_\theta\|p_{ref})
$$

在偏好/RL 约束中更强调别偏离参考策略。DPO 常见目标里 beta 控制偏离强度，beta 越大越保守，越小越激进。""",
    "grpo_metrics": r"""GRPO 组内 advantage 常写成：

$$
A_i=\frac{r_i-\mu(r)}{\sigma(r)+\epsilon}
$$

因此 reward std 过低会让优势失效。监控表至少包含 `reward_mean/std`、`kl`、`entropy`、`clip_fraction`、`response_len`、`format_acc`、`tool_success`、`pass_rate`。""",
    "reward_advantage_prm": r"""把 step reward 分配到 token 的一个简单方式：

$$
A_t = R_{step(i)},\quad t\in tokens(step_i)
$$

更细的方法是训练 value/reward model 估计每个状态或 token 的优势。PRM 要配合最终结果校验，否则会奖励“看起来努力”的长轨迹。""",
    "sft_to_rl": r"""开始 RL 的最低门槛可以写成四个 gate：格式准确率、基础正确率、采样多样性、reward 可区分性。若 reward 对大多数样本全 0 或全 1，RL 很难学。

常见修复：去重、减少模板、混入通用指令数据、降低学习率、缩短训练轮数、加入 chosen/rejected 负例。""",
    "distillation": r"""logits 蒸馏常用 KL：

$$
L_{KD}=T^2\cdot KL(softmax(z_t/T)\|softmax(z_s/T))
$$

文本蒸馏只看教师输出文本，隐藏层蒸馏对齐中间表示。on-policy 蒸馏更贴近学生部署分布，但采样和评价成本更高。""",
    "moe": r"""MoE 单 token 计算近似由激活专家决定：

$$
FLOPs_{token}\approx FLOPs_{attn}+k\cdot FLOPs_{expert}
$$

不是由总专家数直接决定。负载均衡常看每个专家 token 数、routing entropy、capacity overflow、token drop 和 all-to-all 时间。""",
    "moe_rl": r"""路由熵可作为专家利用多样性的信号：

$$
H(router)=-\sum_e p(e)\log p(e)
$$

RL 后如果 reward 上升但 routing entropy 下降、少数专家 token 数暴涨，就要警惕专家塌缩。""",
    "gqa_kv_cache": r"""KV cache 公式：

$$
M_{KV}=B\cdot T\cdot L\cdot H_{kv}\cdot D_{head}\cdot 2\cdot bytes
$$

MHA、GQA、MQA 的主要差别就是 `H_kv`。Q 是当前步临时算出来的，不需要缓存历史 Q；K/V 是历史上下文，decode 每步都要复用。""",
    "mla_deepseek": r"""MLA 可抽象成先压缩再恢复：

$$
c_t=W_c h_t,\quad K_t=W_k c_t,\quad V_t=W_v c_t
$$

缓存低维 `c_t` 比缓存完整 K/V 更省，但信息瓶颈和算子实现更复杂，所以需要训练时联合适配。""",
    "flash_attention": r"""FlashAttention 的 online softmax 合并块时维护最大值 `m` 和归一化项 `l`：

$$
m_{new}=\max(m_{old}, m_{block})
$$

$$
l_{new}=e^{m_{old}-m_{new}}l_{old}+e^{m_{block}-m_{new}}l_{block}
$$

它省的是 HBM IO 和中间 attention matrix，不是把 attention 变成近似。""",
    "sparse_linear_attention": r"""复杂度对比：

$$
SoftmaxAttention=O(T^2d),\quad SparseAttention\approx O(Twd),\quad LinearAttention\approx O(Td^2)
$$

其中 `w` 是窗口或稀疏连接数。回答时要说清任务是否允许牺牲任意 token 两两精确交互。""",
    "swiglu_ffn": r"""SwiGLU 常见形式：

$$
FFN(x)=W_{down}(SiLU(W_{gate}x)\odot W_{up}x)
$$

若 hidden 为 `d`，中间维为 `d_ff`，参数量约：

$$
3d\cdot d_{ff}
$$
""",
    "lora": r"""LoRA 权重增量：

$$
\Delta W=\frac{\alpha}{r}BA
$$

推理可 merge：

```python
W_merged = W_base + (alpha / r) * (B @ A)
```

rank 决定容量，alpha 决定增量尺度，target modules 决定能力注入位置。""",
    "rag_vs_finetune": r"""RAG 链路可拆成：

$$
Query \rightarrow Retrieve \rightarrow Rerank \rightarrow Compress \rightarrow Generate \rightarrow Verify
$$

指标也要拆：召回率、命中率、NDCG、引用正确率、答案忠实性和端到端准确率。微调则看任务泛化、遗忘、风格稳定和成本。""",
    "code_search_rag": r"""代码搜索常先用精确检索缩小范围：

```bash
rg -n "class Agent|def run|error message" .
```

再读调用点和测试。向量索引可作为补充，但要解决索引新鲜度、权限和符号精确度问题。""",
    "agent_arch": r"""企业级 Agent 轨迹最好结构化：

```json
{"state":"plan","action":"tool_call","tool":"search","args":{},"observation":"...","cost_ms":123}
```

有了结构化轨迹，才能做权限审计、失败 replay、成本统计、自动评测和人工接管。""",
    "sampling": r"""temperature 缩放：

$$
p_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

top-k 保留前 k 个 token，top-p 保留累计概率超过 p 的最小集合；过滤后要重新归一化。repeat penalty 通常在过滤前调整已出现 token 的 logits。""",
    "serving_vllm": r"""PagedAttention 把 KV 切成 block，避免连续大块显存分配；prefix/Radix cache 复用公共前缀；分层 KV 则在容量和带宽间权衡。

面试中可用一句话区分：PagedAttention 管碎片，Radix/prefix 管重复 prefill，HiCache 管容量层级，FP8 KV 管单元素字节数。""",
    "zero_parallel": r"""显存切分对象：

$$
ZeRO1: optimizer,\quad ZeRO2: optimizer+grad,\quad ZeRO3: optimizer+grad+param
$$

checkpointing 则不切参数，而是少存激活；tensor parallel 切矩阵，pipeline parallel 切层。""",
    "quantization": r"""对称量化：

$$
x_q=round(x/s),\quad x\approx s\cdot x_q
$$

非对称量化：

$$
x_q=round(x/s)+z,\quad x\approx s(x_q-z)
$$

group size 越小，每组 scale 越贴合局部分布，但 scale/metadata 开销越大。""",
    "evaluation": r"""PPL：

$$
PPL=\exp\left(-\frac{1}{N}\sum_i \log p(x_i|x_{<i})\right)
$$

RAG 指标拆检索和生成；Agent 指标拆 task success、tool success、轨迹步数、cost、latency、安全和可恢复性。""",
    "cot_train_infer": r"""训练时可用结构化字段：

```json
{"reasoning":"...","answer":"..."}
```

部署时只展示 `answer`。关键是验证无显式 CoT 时最终正确率，以及防止模型把隐藏推理泄露到用户可见输出。""",
    "agent_eval_observability": r"""Agent 评测记录建议至少包含：

```json
{"task_id":"...","step":3,"action":"tool","tool_success":true,"task_success":false,"cost":0.12}
```

最终看聚合指标：

$$
SuccessRate=\frac{passed\ tasks}{all\ tasks}
$$

并分层分析工具失败、规划失败、信息整合失败和安全失败。""",
    "agentic_rl_self_evolution": r"""Agentic RL 轨迹可写成：

$$
\tau=(s_0,a_0,o_0,\ldots,s_T,a_T),\quad R(\tau)=R_{outcome}+\sum_i R_{process,i}
$$

自蒸馏必须把成功轨迹、失败轨迹和不确定轨迹分开处理，避免把错误答案当新教师。""",
    "inference_cache_speculative": r"""Speculative decoding 的基本流程是 draft 先生成 `k` 个 token，target 一次性验证：

$$
speedup \approx \frac{k}{1 + verify\_cost}
$$

实际收益取决于 draft 接受率、target batch 效率和验证开销。prefix cache 省 prefill，spec decoding 省 decode 串行步数，二者可叠加。""",
    "data_mixture_engineering": r"""数据混合可用采样权重表达：

$$
P(domain_i)=\frac{w_i n_i^\alpha}{\sum_j w_j n_j^\alpha}
$$

`alpha < 1` 可避免大语料域完全支配训练。每次改配比都要配 per-domain eval，否则只看总分会掩盖能力迁移。""",
    "mtp_loss_speculative_training": r"""MTP 目标：

$$
L=\sum_t CE(h_t,x_{t+1})+\sum_{k=2}^{K}\lambda_k CE(h_t^{(k)},x_{t+k})
$$

实现时要处理 label shift、padding mask、未来 token 越界和不同 horizon 的 loss scale。""",
    "multimodal_vlm_vla": r"""典型 VLM 接入：

$$
image \rightarrow ViT \rightarrow projector \rightarrow LLM\ tokens
$$

VLA 进一步把输出映射到动作：

$$
(vision, instruction, state)\rightarrow action
$$

评估不能只看图文 QA，还要看闭环 task success、碰撞/安全、恢复能力和跨环境泛化。""",
    "deepseek_arch": r"""回答技术报告题用固定框架：

$$
Module \rightarrow Bottleneck \rightarrow Mechanism \rightarrow Tradeoff \rightarrow Metric
$$

例如 MLA: KV cache 瓶颈 -> 低维潜表示 -> 省显存但实现复杂 -> 看长上下文吞吐和效果。""",
    "company_interview": r"""项目回答模板：

```text
背景目标 -> 数据来源 -> 方法选择 -> 指标设计 -> 失败案例 -> 取舍与优化 -> 线上验证
```

每个项目至少准备一张链路图、三组 ablation、三个失败 case 和一组成本/延迟指标。""",
    "coding_common": r"""手撕模型组件时固定四步：

```text
shape -> formula -> code -> stability/boundary
```

例如 RMSNorm：

```python
y = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight
```
""",
}


def topic_counts() -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {topic.key: Counter() for topic in generate_notebook.TOPICS}
    with SOURCES_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("is_llm_related") != "yes":
                continue
            origin = row.get("source_origin") or "favorites"
            for key in [part.strip() for part in row.get("matched_topics", "").split(";") if part.strip()]:
                counts.setdefault(key, Counter())[origin] += 1
    return counts


def strip_existing_supplement(section: str) -> str:
    cleaned = section
    for marker in (OLD_START, START):
        pattern = re.compile(rf"\n{re.escape(marker)}\n.*?(?=\n### |\Z)", re.S)
        cleaned = pattern.sub("", cleaned)
    return cleaned.rstrip()


def parse_supplement(raw: str) -> tuple[list[str], str]:
    questions_part = raw
    answer = ""
    if "补充答法：" in raw:
        questions_part, answer = raw.split("补充答法：", 1)
    questions = []
    for line in questions_part.splitlines():
        match = re.match(r"\s*\d+\.\s*(.+?)\s*$", line)
        if match:
            questions.append(match.group(1))
    return questions, answer.strip()


def fallback_answer(question: str, summary: str) -> str:
    q = question.lower()
    if "prefill" in q or "decode" in q:
        return """答：prefill 和 decode 的瓶颈不同，因为计算形态不同。Prefill 一次处理整段 prompt，attention 需要处理 `T x T` 的上下文矩阵，大量 GEMM 可以并行，通常看 TTFT、prefill tokens/s 和显存峰值；decode 每步只生成一个新 token，但每步都要读取历史 KV cache，串行依赖强，常受内存带宽、KV cache 布局和 batch 调度限制。

面试可以这样说：prefill 更像“算得快不快”，decode 更像“读 KV 和调度能不能跟上”。所以 vLLM/SGLang 这类系统会同时做 continuous batching、prefix cache、paged KV、speculative decoding 等优化。"""
    if "mtp" in q or "multi" in q or "多 token" in question or "多token" in question:
        return """答：MTP/multi-token prediction 会让一个位置同时预测多个未来 token，因此训练 FLOPs 会增加额外 prediction heads 或额外 logits 计算；但它可能让 hidden state 学到更远未来的信息，并为 speculative decoding、多 token 解码或 draft head 提供训练信号。

普通 next-token loss 是：

$$
L=-\sum_t \log p(x_{t+1}|x_{\le t})
$$

MTP 可写成：

$$
L=\sum_t \sum_{k=1}^{K}\lambda_k CE(\hat{x}_{t+k}, x_{t+k})
$$

关键取舍是：未来 token 越远不确定性越大，辅助 loss 权重不能压过主 next-token 目标。"""
    if "batchnorm" in q or "bn" in q:
        return """答：Transformer/LLM 通常不用 BatchNorm，是因为 BatchNorm 依赖 batch 统计，而 NLP batch 的序列长度、mask、padding、分布都不稳定；自回归推理时 batch 统计也和训练不一致。LayerNorm/RMSNorm 是对单个 token 的 hidden 维归一化，不依赖 batch，更适合变长序列和逐 token 生成。

面试里可以补一句：BatchNorm 的归纳偏置更适合视觉 CNN 中稳定的 batch/channel 统计；LLM 更关心每个 token 表示在 hidden 维上的尺度稳定。"""
    if "rmsnorm" in q or "pre-norm" in q or "llama" in q or "qwen" in q:
        return """答：RMSNorm + Pre-Norm 是现代 decoder-only LLM 的常见稳定性组合。RMSNorm 不减均值，只按均方根缩放，计算更省；Pre-Norm 把归一化放在 attention/FFN 之前，让残差路径更像一条直接的梯度高速路，深层训练更稳定。

公式上：

$$
RMSNorm(x)=g\cdot \frac{x}{\sqrt{mean(x^2)+\epsilon}}
$$

如果把 Norm 放到残差之后，输出更规整，但深层模型更容易梯度不稳。"""
    if "prompt" in q and "mask" in q:
        return """答：SFT 里 prompt token 通常是条件，不是要模型学习输出的目标。如果把 user/system prompt 也算 loss，模型会被训练去复读输入，尤其在多轮对话里会污染 assistant 行为。常见做法是把 prompt 部分 label 置为 `-100`，只对 assistant answer 算 CE。

面试可以补充：预训练是所有 token next-token prediction；SFT/指令微调则常按角色 mask loss，这两者训练目标不同。"""
    if "label smoothing" in q:
        return """答：label smoothing 把 one-hot 目标变成软标签，降低模型过度自信：

$$
q_y=1-\epsilon,\quad q_j=\epsilon/(C-1)
$$

好处是提升校准、缓解过拟合；代价是训练目标不再追求正确类概率为 1，可能影响 PPL 解释。生成模型里要小心：如果评估仍按 hard label PPL，label smoothing 训练出的模型未必在 PPL 上最优。"""
    if "checkpoint" in q or "回滚" in q or "跳过" in q:
        return """答：先看 spike 是否可恢复。单个 batch spike 后如果 loss 很快回落，可以记录样本 hash、跳过 batch 或加梯度裁剪；如果 spike 后 KL、grad norm、验证集质量持续异常，就应该从 spike 前 checkpoint 回滚，并保留优化器状态一起恢复。

判断依据包括：grad norm 是否爆炸、是否出现 NaN/Inf、loss scale 是否 overflow、后续 eval 是否掉点。不要只看训练 loss 一条曲线。"""
    if "坏样本" in question:
        return """答：同一个坏样本不一定每次触发 spike，因为 spike 取决于“样本 + 当前参数 + optimizer state + loss scale + batch 组合”。模型在不同训练阶段对同一异常样本的敏感度不同，混合精度和梯度累积也会改变数值状态。

因此排查必须保存 global step、样本 id/hash、batch 组成、随机种子、optimizer state 和 loss scale，否则很难复现。"""
    if "reward hacking" in q:
        return """答：loss spike 和 reward hacking 都可能体现为优化目标被异常放大。RL 后训练中，如果 reward 设计有漏洞，模型可能学会输出更长、更模板化或格式投机的答案，导致 reward 上升但 KL、长度、entropy、人工质量异常。

所以 RL 监控不能只看 reward mean，还要看 KL、clip fraction、response length、format accuracy、pass rate 和人工抽检。"""
    if "relu" in q:
        return """答：ReLU 的反向传播在从 `A1` 回到 `Z1` 时乘 mask：

$$
dZ_1=dA_1\cdot 1[Z_1>0]
$$

如果是两层 MLP：`X -> Z1=XW1+b1 -> A1=ReLU(Z1) -> Y=A1W2+b2`，反向顺序是先算 `dW2=A1^T dY`，再算 `dA1=dY W2^T`，最后乘 ReLU mask 得到 `dZ1`。"""
    if "batch mean" in q or "sum" in q:
        return """答：如果 loss 用 mean，梯度会除以 batch size；如果误写成 sum，梯度大约放大 `N` 倍，相当于学习率也放大 `N` 倍，容易不稳定。修法是统一 loss reduction 口径，或者把学习率按 batch size 缩放。

面试里要说清：梯度缩放不仅影响数值大小，还会影响不同 batch size 下实验能否复现。"""
    if "shape" in q:
        return """答：检查 shape 的方法是每一步都写出张量维度，并确认梯度和对应参数同形。例如 `X[N,D]`、`W1[D,H]`、`Z1[N,H]`、`W2[H,C]`、`Y[N,C]`；那么 `dW2` 必须是 `[H,C]`，`dW1` 必须是 `[D,H]`。

写代码前先写 shape，比写完再 debug 更快，也更像面试官想看的思路。"""
    if "正向 kl" in q or "反向 kl" in q or "mode" in q:
        return """答：正向 KL 更偏 mode-covering，会惩罚模型漏掉数据分布覆盖的模式；反向 KL 更偏 mode-seeking，会鼓励模型集中到高概率/高奖励模式。SFT 行为上更像覆盖示范数据分布，RL/偏好优化更容易把概率推向高奖励答案。

面试要强调这是直觉类比，不同算法的实际 KL 项和采样策略不同。真正工程上要看 beta/KL 系数、采样温度和 reward 质量。"""
    if "dapo" in q or "gspo" in q:
        return """答：这类方法主要是在大模型 RL 的稳定性和效率上做改进，常见改动包括采样策略、advantage 估计、clip 范围、长度归一化、KL 控制和 MoE 场景稳定性。GRPO 的代表性思路是组内相对优势，DAPO/GSPO 则进一步围绕训练稳定和序列级优化做调整。

答题不要只背 acronym，要说清：它们都在解决“没有稳定 value、reward 噪声大、生成长度不一、策略容易漂移”的问题。"""
    if "偏好数据" in q or "过程数据" in q or "轨迹" in q:
        return """答：偏好数据适合 DPO 或 reward model，因为它提供 chosen/rejected；过程数据适合 PRM 或 step-level reward，因为它能判断中间步骤；工具调用轨迹适合 Agent SFT/Agentic RL，因为它包含 state、action、observation 和 final outcome。

选择方法时看反馈粒度：只有最终好坏就做 outcome reward；有步骤评分就做过程监督；有环境可验证反馈就可以做在线/半在线 RL。"""
    if "prm" in q or "orm" in q:
        return """答：ORM 只评价最终答案，PRM 评价中间步骤。Agent 任务更需要 PRM，是因为多步工具调用里最终失败不告诉你哪一步错了，必须给规划、检索、工具参数、观察整合等过程分配信用。

风险是过程奖励可能鼓励模型写更多“看起来合理”的步骤，所以要配合长度惩罚、无效工具调用惩罚和最终结果校验。"""
    if "sft" in q and "rl" in q:
        return """答：SFT 没做好就上 RL，模型会在低质量策略附近探索，reward 噪声会被放大，常见问题是格式投机、拒答异常、复读、KL 飙升或 pass rate 不稳定。开始 RL 前至少要保证格式准确率、基础任务正确率和采样多样性过线。

可以把 SFT 看成“会做”，RL 看成“做得更符合偏好/奖励”。不会做时直接 RL，通常只是学会钻奖励空子。"""
    if "蒸馏" in q or "on-policy" in q or "off-policy" in q:
        return """答：固定教师数据训练学生是 off-policy 蒸馏，成本低但学生分布偏离后覆盖不足；学生当前策略采样，再由教师/环境打分，是 on-policy 或接近 on-policy，成本高但更贴近部署分布。

搜索 Agent 自蒸馏通常是：学生生成搜索轨迹，环境验证或教师评价，筛选成功轨迹回流训练。难点是筛选标准和防止错误自我强化。"""
    if "moe" in q:
        return """答：MoE 的关键是每个 token 只激活少量专家，所以总参数可以很大，但单 token FLOPs 由 top-k 专家决定。shared expert 提供通用能力，细粒度专家提高组合灵活性，负载均衡损失避免少数专家过载。

如果问题问 MoE + RL，要强调 RL 会改变 token 分布和路由分布，可能造成专家塌缩、负载不均和 routing entropy 下降。"""
    if "kv" in q or "gqa" in q or "mqa" in q:
        return """答：KV cache 的核心公式是：

$$
KVBytes=B\cdot T\cdot L\cdot H_{kv}\cdot D_{head}\cdot 2\cdot bytes
$$

GQA/MQA 减少 `H_kv`，FP8 KV 减少 `bytes`，MLA 减少缓存表示维度，三者是互补思路。decode 阶段每步都要读历史 KV，所以长上下文下 KV 带宽常是瓶颈。"""
    if "flashattention" in q or "online softmax" in q:
        return """答：FlashAttention 是精确 attention 的 IO 优化，不是近似 attention。它分块计算 attention，不把完整 `T*T` 分数矩阵写回 HBM，并用 online softmax 维护每个块合并后的最大值和归一化因子。

所以它省的是显存读写和中间矩阵，不改变 softmax attention 的数学结果。训练长序列时收益尤其明显。"""
    if "sparse" in q or "linear attention" in q or "gated" in q:
        return """答：Sparse Attention 通过局部窗口、块稀疏或选择重要 token 减少连接；Linear Attention 用核技巧或递推状态把复杂度降到近似线性；Gated Attention 用门控控制信息流。它们的收益是长上下文效率，代价是表达能力、稳定性或实现复杂度。

标准 softmax attention 仍强，是因为它保留任意 token 两两交互。替代方案要看任务是否真的需要全局精确交互。"""
    if "swiglu" in q or "ffn" in q:
        return """答：SwiGLU 的典型形式是：

$$
FFN(x)=W_{down}(SiLU(W_{gate}x)\odot W_{up}x)
$$

相比 GELU/ReLU FFN，它多一个门控分支，能动态选择通道信息。FFN 参数量常约 `3*d*d_ff`，在 LLM 中经常超过 attention 的 `4*d^2`，所以也是计算和参数大头。"""
    if "lora" in q or "qlora" in q or "dora" in q:
        return """答：LoRA 训练低秩增量：

$$
\Delta W = B A,\quad rank(A)=rank(B)=r
$$

rank 越大容量越强但更耗显存；alpha 控制增量缩放；target modules 决定改哪些层。QLoRA 量化基座省显存，DoRA 分解方向和幅度，AdaLoRA/DyLoRA 做动态或可变 rank。知识更新优先 RAG，稳定风格/格式迁移适合 LoRA，深度能力迁移才考虑全量。"""
    if "rag" in q:
        return """答：RAG 不是默认最优。若知识很小、结构强、更新慢，规则、SQL 或 grep 可能更稳定；若需要动态知识、引用溯源和低成本更新，RAG 更合适。优化 RAG 要分层：chunk 清洗、query rewrite、embedding 召回、rerank、上下文压缩、答案忠实性和引用校验。

Agent memory 更偏个体历史和任务状态，RAG 更偏外部知识库，两者可以共用检索，但评估目标不同。"""
    if "grep" in q or "ripgrep" in q or "代码" in question and "检索" in question:
        return """答：代码检索常需要精确符号、路径、错误文本和调用关系，`rg` 实时、可解释、不会索引过期。向量检索适合概念相似和文档语义，但对刚修改的代码、变量名和精确错误不如 grep。

Claude Code/Codex 类工具更像 Agentic Search：先宽搜，再读文件，再根据证据继续搜，而不是一次向量召回后直接回答。"""
    if "agent" in q and ("workflow" in q or "harness" in q or "runtime" in q):
        return """答：Workflow 是预定义流程，Agent 是动态决策；Harness/Runtime 是把模型、工具、状态、权限、日志、重试接起来的执行框架。企业级 Agent 不能只有 prompt，因为生产环境需要权限控制、参数校验、超时重试、观测、成本控制和人工确认。

多 Agent 不是越多越好，容易带来职责重叠、循环调用和难评估；很多生产系统更适合少量 Agent 加明确工具。"""
    if "mcp" in q or "skill" in q or "function" in q or "tool" in q:
        return """答：Function Call 是模型输出结构化调用参数；Tool 是实际能力；MCP 是把工具/资源标准化暴露给 Agent 的协议；Skill 是把工具和经验组织成可复用流程；Rules 是长期约束。

MCP 解决“能力怎么接入”，Skill 解决“任务该怎么做”。复杂项目里只有工具不够，还需要 Skill 记录顺序、验证方式和安全边界。"""
    if "memory" in q or "记忆" in question or "上下文压缩" in question:
        return """答：工作记忆保存当前任务临时状态，短期记忆保存本轮会话，长期记忆保存跨任务事实和偏好，情景记忆保存历史轨迹。长上下文不能完全替代外部记忆，因为成本高、噪声多，也不适合长期复用。

上下文压缩要保留目标、硬约束、已验证事实、来源、失败尝试和下一步计划，最好结构化，而不是随意摘要。"""
    if "temperature" in q or "top-p" in q or "top-k" in q:
        return """答：常见顺序是先对 logits 做 temperature 缩放，再做 top-k/top-p 截断，最后重新归一化采样。temperature 越高分布越平，多样性越强；top-k 固定候选数量，top-p 固定累计概率质量。

评测通常用低温或 greedy 保证稳定；创作用高温增加多样性。temperature=0 仍可能因浮点、并行 kernel、tie-breaking 或服务 fallback 出现细微不确定。"""
    if "vllm" in q or "sglang" in q or "pagedattention" in q or "radix" in q or "hicache" in q:
        return """答：PagedAttention 把 KV cache 分页管理，减少连续显存需求和碎片；continuous batching 动态合批提升吞吐；RadixAttention/prefix cache 复用共享前缀，特别适合 Agent 多轮和长 system prompt；HiCache/分层 KV 在 GPU/CPU/SSD 间换容量和带宽。

回答时要区分：PagedAttention 解决 KV 管理，Radix/prefix 解决重复 prefill，HiCache 解决容量层级，SGLang 还强调结构化生成和调度。"""
    if "zero" in q or "tensor" in q or "pipeline" in q or "checkpoint" in q:
        return """答：ZeRO-1 切 optimizer states，ZeRO-2 再切 gradients，ZeRO-3 连 parameters 也切；tensor parallel 切单层矩阵，层内通信多；pipeline parallel 按层切，难点是 bubble；gradient checkpointing 不存部分激活，反向重算，用计算换显存。

面试要说清“省的是哪块显存”，不要只说“并行能省显存”。"""
    if "bf16" in q or "fp8" in q or "int" in q or "量化" in question:
        return """答：BF16 的指数位多，动态范围接近 FP32，适合训练稳定；FP16 精度更细但范围小；FP8/INT8/INT4 更省显存和带宽，但需要 scale、校准和量化 kernel。weight-only 量化省权重，KV cache 量化省长上下文缓存。

对称量化可写成：

$$
x_q=round(x/s),\quad x\approx s\cdot x_q
$$

非对称量化还会有 zero-point。"""
    if "eval" in q or "评估" in question or "ppl" in q:
        return """答：评估要分层。基础模型看 PPL 和 benchmark；RAG 看召回率、rerank、引用正确性和答案忠实性；Agent 看 task success、tool success、轨迹成本、恢复能力和安全。PPL 降低说明语言建模拟合更好，但不必然代表指令遵循、推理或工具调用更好。

生产上要组合离线集、线上 A/B、人工抽检和失败 case 复盘。"""
    if "cot" in q or "reasoning" in q:
        return """答：训练时用 CoT 可以给模型中间推理监督，推理时可以通过模板或后处理只展示 final answer。关键是训练数据结构要区分 reasoning 和 answer 字段，部署时隐藏或压缩 reasoning。

风险是模型依赖显式长 CoT 才能做对，或者暴露不稳定中间过程；所以要用无 CoT 验证集检查最终正确率。"""
    if "deepseek" in q or "qwen" in q or "kimi" in q:
        return """答：技术报告题不要只背参数，要按“解决什么瓶颈”组织。MLA 解决 KV cache，MoE 提升容量/计算比，MTP 增加训练信号或推理适配，FP8 提升训练/推理效率，GRPO/RL 提升推理和偏好对齐。

回答结构：先说模块机制，再说收益，再说代价，最后说如何验证。"""
    if "算法岗" in question or "工程岗" in question or "主管面" in question or "项目" in question:
        return """答：Agent 岗重点是系统设计、工具调用、记忆、评估和工程落地；RAG 工程岗重点是切分、召回、rerank、忠实性和线上指标；大模型算法岗更重 Transformer、后训练、RL、MoE、推理优化和手撕。

项目追问按六层准备：动机、数据、方法、指标、失败 case、成本优化。主管面更看取舍和 owner 意识，不只是技术名词。"""
    if "手撕" in question or "pytorch" in q:
        return """答：大模型算法岗手撕越来越偏模型组件，包括 Attention、CrossEntropy、RMSNorm、RoPE、LoRA merge、top-p sampling、KV cache、梯度裁剪、beam search。准备方式是每题都按四步：写 shape、写公式、写 PyTorch、说数值稳定和边界条件。

如果面试官问非 Hot100，本质也是看你能否把数学对象落成可运行代码，而不是背题库。"""
    lead = "答：这题可以先正面回答问题，再补一层工程取舍。"
    if "为什么" in question:
        lead = "答：这题重点不是背现象，而是解释背后的约束和取舍。"
    elif "如何" in question or "怎么" in question:
        lead = "答：这题建议按步骤回答：先定义目标，再拆关键模块，最后说验证和风险。"
    elif "区别" in question or "什么不同" in question:
        lead = "答：这题建议用对比表述，分别说明目标、输入输出、训练/推理成本和适用场景。"
    return f"{lead}{summary}"


def topic_fallback_answer(key: str, idx: int, question: str, summary: str) -> str:
    points = TOPIC_QUESTION_POINTS.get(key, [])
    point = points[idx - 1] if idx <= len(points) else f"这道题问的是“{question}”，回答时先给结论，再解释机制、代价和验证方式。"
    topic = next((item for item in generate_notebook.TOPICS if item.key == key), None)
    base = summary or (topic.answer if topic else "")
    detail = TOPIC_TECH_DETAILS.get(key, "")
    paragraphs = [f"答：{point}"]
    if base:
        paragraphs.append(base)
    if detail:
        paragraphs.append(detail.strip())
    return "\n\n".join(paragraphs).strip()


def render_question_answers(key: str, body: str) -> str:
    questions, summary = parse_supplement(body)
    if not questions:
        return body.strip()
    overrides = QUESTION_ANSWER_OVERRIDES.get(key, {})
    lines = [START, ""]
    for idx, question in enumerate(questions, start=1):
        answer = overrides.get(idx) or topic_fallback_answer(key, idx, question, summary)
        lines.extend([f"#### {idx}. {question}", "", answer.strip(), ""])
    return "\n".join(lines).rstrip()


def enrich_section(key: str, section: str, counts: Counter[str]) -> str:
    body = SUPPLEMENTS.get(key, "").strip()
    if not body:
        return strip_existing_supplement(section).rstrip() + "\n"
    supplement = render_question_answers(key, body)
    return strip_existing_supplement(section) + "\n\n" + supplement + "\n"


def main() -> None:
    sync_curated_answers()
    curated = curated_answers()
    text = DETAIL_PATH.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## ", text)
    header = parts[0].rstrip()
    section_map: dict[str, str] = {}
    for raw in parts[1:]:
        key, _, rest = raw.partition("\n")
        section_map[key.strip()] = rest.rstrip()

    counts = topic_counts()
    lines = [header, ""]
    for topic in generate_notebook.TOPICS:
        section = section_map.get(topic.key)
        if section is None:
            raise SystemExit(f"missing detailed answer section: {topic.key}")
        lines.append(f"## {topic.key}")
        lines.append(curated[topic.key] if topic.key in curated else
                     enrich_section(topic.key, section, counts.get(topic.key, Counter())).rstrip())
        lines.append("")
    DETAIL_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"enriched {len(generate_notebook.TOPICS)} detailed answer sections")


if __name__ == "__main__":
    main()
