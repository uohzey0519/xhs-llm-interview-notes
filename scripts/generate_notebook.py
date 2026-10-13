#!/usr/bin/env python3
"""Generate the LLM interview notebook from extracted Xiaohongshu notes."""

from __future__ import annotations

import csv
import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from research_updates import MANIFEST, UPDATES, SOURCE_MAP, sync_curated_answers


REPO_ROOT = Path(__file__).resolve().parents[1]
TODAY = "2026-05-24"


@dataclass(frozen=True)
class Topic:
    key: str
    question: str
    category: str
    difficulty: str
    patterns: tuple[str, ...]
    answer: str


TOPICS: list[Topic] = [
    Topic(
        "transformer_arch",
        "Transformer 的基本架构和 Self-Attention 计算流程是什么？",
        "Transformer 基础",
        "基础",
        (r"transformer", r"self[- ]?attention", r"多头注意力", r"encoder", r"decoder"),
        "Transformer 用注意力替代循环结构，核心由多头自注意力、前馈网络、残差连接和归一化组成。Self-Attention 先把输入映射成 Q/K/V，用 QK^T/sqrt(d_k) 得到注意力分数，softmax 后加权 V；除以 sqrt(d_k) 是为了避免点积随维度变大导致 softmax 饱和。Multi-Head Attention 让不同头在不同子空间学习关系，最后拼接投影。面试里要能说清复杂度 O(n^2d)、为什么适合并行、以及长上下文时的显存瓶颈。",
    ),
    Topic(
        "param_flops",
        "如何估算 LLM 参数量、一次前向计算量和训练显存？",
        "推理服务与性能",
        "进阶",
        (r"参数量", r"计算量", r"7b模型", r"显存占用", r"显存开销", r"训练7b", r"加载一个模型"),
        "估算时先拆 Embedding、每层 Attention、FFN 和 LM Head。显存主要包括参数、梯度、优化器状态、激活、KV Cache 和碎片；训练时 Adam 通常比推理多出梯度和一二阶动量，混合精度下也常保留 FP32 master weight。粗略面试口径：推理参数显存约参数量乘 dtype 字节数；训练还要加梯度、优化器状态和激活，ZeRO、checkpointing、并行切分会改变各部分归属。回答时最好先列公式，再说明 batch、seq_len、hidden size、层数如何影响激活和 KV。",
    ),
    Topic(
        "rope_position",
        "RoPE 的直觉是什么，为什么有外推能力？Qwen/LLaMA 的 RoPE 实现有何差异？",
        "位置编码",
        "进阶",
        (r"rope", r"旋转位置", r"位置编码", r"长度外推", r"yarn", r"qwen和llama"),
        "RoPE 把绝对位置编码为对 Q/K 向量的二维旋转，使注意力分数天然包含相对位置信息。两个位置的内积会变成与相对距离有关的函数，因此比简单绝对位置嵌入更利于外推。长上下文扩展常用 NTK scaling、YaRN、位置插值等方法，本质是在频率尺度上减缓远距离相位变化。Qwen/LLaMA 的差异常体现在 base、缩放策略、实现细节和不同层/维度上的频率配置，面试中强调机制和工程取舍即可。",
    ),
    Topic(
        "norms",
        "LayerNorm、RMSNorm、Pre-Norm 和 Post-Norm 怎么比较？",
        "归一化与 Loss",
        "基础",
        (r"rms ?norm", r"layer ?norm", r"pre[- ]?norm", r"post[- ]?norm", r"归一化位置", r"归一化"),
        "LayerNorm 对特征做去均值和方差归一化，RMSNorm 只按均方根缩放，不减均值，所以计算更省且在 LLM 中效果通常接近。Pre-Norm 把归一化放在子层前，梯度路径更稳定，适合深层大模型；Post-Norm 输出更规整但深层训练更容易不稳定。回答时可以补一句：现代 decoder-only LLM 常用 RMSNorm + Pre-Norm，是稳定性和效率的折中。",
    ),
    Topic(
        "cross_entropy",
        "手撕 CrossEntropyLoss 时前向、梯度和数值稳定怎么写？",
        "归一化与 Loss",
        "基础",
        (r"crossentropy", r"交叉熵", r"ce loss", r"loss的那些事", r"分类用ce"),
        "前向不要先算 softmax 再 log，而要用 log-sum-exp：loss_i = logsumexp(logits_i) - logits_i[y_i]，再对 batch 求均值。梯度是 (softmax(logits) - one_hot(y)) / N，正确类别项为 p_y - 1。数值稳定的关键是每行减去 max logit，避免 exp 溢出；面试追问常包括为什么 CE 比 MSE 更适合分类、label smoothing 怎么改目标分布。",
    ),
    Topic(
        "loss_spike",
        "LLM 训练中 loss spike 或 loss 突然增大如何排查和处理？",
        "训练与优化",
        "进阶",
        (r"loss突然", r"loss spike", r"loss.*增大", r"训练中出现"),
        "先区分数据异常、学习率/优化器不稳定、梯度爆炸、混合精度溢出和分布切换。工程上可回滚到 spike 前 checkpoint、跳过附近 batch、降低学习率、检查 grad norm、开启/调整梯度裁剪、检查 loss scale 和数据清洗。大模型里 spike 不一定是单个坏样本造成的，可能是特定 batch 与当前参数状态相互作用，因此要保留日志、样本 hash、优化器状态和复现实验。",
    ),
    Topic(
        "mlp_training",
        "手撕两层 MLP 单步训练应该怎么推导？",
        "手撕代码",
        "基础",
        (r"手撕mlp", r"mlp 训练", r"梯度下降", r"mse损失"),
        "写清 forward、loss、backward、update 四步。若 h = XW1，y_hat = hW2，MSE = mean((y_hat-y)^2)，则 dL/dy_hat = 2(y_hat-y)/N，grad_W2 = h^T dL/dy_hat，grad_W1 = X^T((dL/dy_hat)W2^T)。面试重点是 shape 不错、链式法则方向正确、更新时 W -= lr * grad。",
    ),
    Topic(
        "sft_rl_kl",
        "为什么常说 SFT 更像正向 KL，RL/偏好优化更像反向 KL？",
        "后训练与强化学习",
        "进阶",
        (r"正向kl", r"反向kl", r"sft.*rl", r"从 kl", r"kl散度"),
        "SFT 用专家答案做监督，鼓励模型覆盖数据分布中出现的行为，更接近最小化数据分布到模型分布的正向 KL，倾向 mode-covering。RL 或偏好优化根据奖励强化高收益输出，同时通过 KL 约束别偏离参考模型，行为上更偏 mode-seeking，容易把概率集中到高奖励模式。面试要补充这只是直觉类比，具体目标还取决于损失形式、采样策略和 KL 方向。",
    ),
    Topic(
        "post_training_methods",
        "SFT、RLHF、DPO、PPO、GRPO、DAPO、GSPO 等后训练方法如何区分？",
        "后训练与强化学习",
        "基础",
        (r"post[- ]?training", r"后训练", r"rlhf", r"dpo", r"ppo", r"grpo", r"dapo", r"gspo", r"gxpo", r"偏好"),
        "SFT 学会按指令回答，是行为克隆；RLHF/PPO 先训练奖励模型再用策略优化；DPO 把偏好数据直接变成分类式损失，少掉显式 reward model 和在线采样；GRPO 用组内相对奖励估计优势，减少 critic 依赖；DAPO/GSPO 等继续围绕采样、优势估计、稳定性和 MoE 训练问题做改进。面试中按数据需求、是否 on-policy、是否需要 reward/critic、稳定性和成本来比较。",
    ),
    Topic(
        "grpo_core",
        "GRPO 的核心思想、loss、优势估计和 PPO 区别是什么？",
        "后训练与强化学习",
        "进阶",
        (r"grpo", r"group relative", r"组内", r"advantages?", r"rollout"),
        "GRPO 对同一个 prompt 采样一组回答，用组内奖励的相对值估计优势，省掉 PPO 中的 critic/value model。典型目标仍有 ratio clipping 和 KL 约束，只是 advantage 来自组内归一化奖励。相比 PPO，GRPO 实现更轻、显存和训练复杂度更低，但对组内采样多样性、奖励尺度、长度偏置和数据质量更敏感。",
    ),
    Topic(
        "grpo_metrics",
        "训练 GRPO/Agentic RL 时应该监控哪些指标？",
        "后训练与强化学习",
        "进阶",
        (r"grpo训练.*指标", r"训练监控指标", r"观测哪些指标", r"reward/mean", r"reward/std"),
        "核心看三组指标：奖励分布、策略漂移和生成质量。奖励分布包括 reward mean/std、规则准确率、组内方差；策略侧看 policy loss、ratio、clip fraction、KL、entropy、平均长度；质量侧看 pass rate、format accuracy、tool success、人工抽检和安全指标。若 reward 全 0/全 1 或方差过低，优势估计会失效；KL 过大说明模型漂移，KL 过小可能学不动。",
    ),
    Topic(
        "reward_advantage_prm",
        "Agentic RL 中过程奖励、PRM 和 token-level advantage 怎么设计？",
        "后训练与强化学习",
        "进阶",
        (r"过程奖励", r"process reward", r"prm", r"reward model", r"token[- ]?level", r"信用分配", r"优势计算"),
        "最终奖励只告诉整条轨迹好坏，Agent 多步工具调用需要过程奖励来做信用分配。PRM 可以给关键步骤、工具调用、反思或中间结论打分；token-level advantage 通常要把 step/trajectory 奖励分配回生成 token，可用均摊、按步骤边界、价值估计或奖励模型逐步评分。设计时要避免 reward hacking，保留格式、正确性、安全、成本等多维奖励，并用人工样本校准。",
    ),
    Topic(
        "sft_to_rl",
        "如何判断 SFT 到什么程度可以开始 RL？SFT 后复读或效果差怎么办？",
        "后训练与强化学习",
        "进阶",
        (r"sft.*程度", r"开始做rl", r"sft后", r"复读机", r"先sft后rl", r"sft经历"),
        "SFT 至少要让模型稳定遵循格式、理解任务、具备基本可采样质量，否则 RL 只会放大奖励噪声。判断信号包括验证集指令遵循率、格式准确率、基础任务分数、拒答/幻觉率和采样多样性。SFT 后复读或效果差通常来自数据重复、模板过强、学习率过大、训练过久或数据分布窄，可通过去重、混合通用数据、降低学习率、早停、增加负例和偏好数据缓解。",
    ),
    Topic(
        "distillation",
        "大模型蒸馏给小模型会遇到什么问题？on-policy 和 off-policy 蒸馏怎么区分？",
        "训练与优化",
        "进阶",
        (r"蒸馏", r"distill", r"teacher", r"onpolicy", r"offpolicy", r"教师模型概率"),
        "Off-policy 蒸馏通常用教师生成的固定数据训练学生，便宜稳定但学生分布一旦偏离，训练样本不能覆盖自身错误。On-policy 蒸馏让学生在当前策略下采样，再由教师或奖励信号纠偏，更贴近部署分布但成本更高、噪声更大。如果拿不到教师概率，可用教师打分、偏好排序、拒绝采样、答案重写或只蒸馏最终文本。常见风险是学生容量不足、错误继承、数据单一和评价泄漏。",
    ),
    Topic(
        "moe",
        "MoE 的原理、Dense 对比、路由和负载均衡难点是什么？",
        "MoE 与稀疏模型",
        "进阶",
        (r"moe", r"稀疏模型", r"dense", r"专家", r"路由", r"load balance", r"负载均衡"),
        "MoE 用路由器为每个 token 选择少量专家激活，使总参数变大但单 token 计算量可控。相比 Dense，MoE 容量更高、训练/推理系统更复杂，难点在路由偏置、专家负载不均、通信开销、token dropping 和稳定性。负载均衡损失通常作用在 token 分配/专家利用率上，既要让专家分工，又不能让少数专家过载。面试还常问 DeepSeek 的专家设计和共享专家、细粒度专家等改进。",
    ),
    Topic(
        "moe_rl",
        "为什么 DAPO/GRPO 等 RL 方法在 MoE 上容易踩坑？",
        "MoE 与稀疏模型",
        "进阶",
        (r"moe.*rl", r"dapo.*grpo", r"grpo.*moe", r"moe会有什么问题", r"坑了 moe"),
        "RL 会改变输出分布和 token 访问模式，MoE 路由随之变化，可能出现专家塌缩、负载不均、奖励集中到少数专家路径、训练不稳定。组内采样还会放大奖励方差和长度偏置，使某些专家获得更密集梯度。处理上要监控专家利用率、routing entropy、token drop、每专家梯度/吞吐，配合负载均衡损失、路由正则、容量因子和稳定的采样策略。",
    ),
    Topic(
        "gqa_kv_cache",
        "MHA、MQA、GQA 为什么能省显存？为什么只有 KV Cache 没有 Q Cache？",
        "Attention 与 KV Cache",
        "基础",
        (r"gqa", r"mqa", r"mha", r"kv ?cache", r"q cache", r"为什么只有kv"),
        "自回归推理第 t 步只需要当前 token 的 Q，但需要历史所有 token 的 K/V 来做注意力，所以缓存 K/V 而不缓存 Q。MHA 每个 query head 都有独立 K/V head，KV Cache 大；MQA 让所有 Q head 共享一组 K/V，最省但表达力可能下降；GQA 在两者之间，让一组 Q head 共享一组 K/V，兼顾显存和效果。显存估算可按 layers * seq_len * kv_heads * head_dim * 2(KV) * bytes。",
    ),
    Topic(
        "mla_deepseek",
        "DeepSeek MLA/相关架构如何压缩 KV Cache？",
        "Attention 与 KV Cache",
        "进阶",
        (r"mla", r"kv cahce", r"kv cache.*压缩", r"mha", r"deepseek.*kv", r"mhc"),
        "MLA 的核心是把 K/V 投影到低维潜空间缓存，推理时再恢复到注意力需要的表示，从而降低 KV Cache 随上下文增长的显存。它和 MQA/GQA 都是在减少 K/V 侧冗余，但 MLA 更强调低秩/潜表示压缩。回答时可以对比：MQA/GQA 减少 KV head 数，MLA 减少缓存维度；代价是实现复杂、算子融合和训练适配要求更高。",
    ),
    Topic(
        "flash_attention",
        "FlashAttention V1/V2/V3/V4 的核心优化思路是什么？",
        "Attention 与 KV Cache",
        "进阶",
        (r"flash ?attention", r"flashattention", r"softmax加速"),
        "FlashAttention 不是改变数学结果，而是改变计算组织：分块计算 attention，避免把完整 n*n attention 矩阵写回 HBM，使用 online softmax 保持数值稳定，从而降低显存读写。V2/V3/V4 主要继续优化并行划分、warp/block 调度、反向传播、硬件适配和吞吐。面试要抓住 IO-aware attention 这个主线：瓶颈通常不是 FLOPs，而是 HBM 读写。",
    ),
    Topic(
        "sparse_linear_attention",
        "Sparse Attention、Linear Attention、Gated Attention 如何降低复杂度，各自代价是什么？",
        "Attention 与 KV Cache",
        "进阶",
        (r"sparse attention", r"linear attention", r"gated attention", r"长文本处理", r"注意力机制"),
        "标准注意力是 O(n^2)，Sparse Attention 通过局部窗口、块稀疏或重要 token 选择减少连接；Linear Attention 用核技巧或状态递推把复杂度降到近似 O(n)，适合长序列但可能损失精确 softmax 表达；Gated Attention 引入门控控制信息流，缓解无效注意力或长程噪声。面试中要强调：降复杂度通常用表达能力、稳定性或实现复杂度换吞吐和上下文长度。",
    ),
    Topic(
        "swiglu_ffn",
        "SwiGLU 为什么常用于 LLM FFN？和普通 FFN/ReLU 有什么区别？",
        "Transformer 基础",
        "基础",
        (r"swiglu", r"ffn", r"feed forward"),
        "SwiGLU 把 FFN 的激活写成门控形式，一支经过 SiLU，另一支做线性变换，两者相乘后再投影。相比 ReLU/GELU FFN，门控结构能更灵活控制通道信息，在 LLaMA/Qwen 等模型中常见。代价是参数/计算形式略复杂，但实践效果好。手写时注意维度通常是 up_proj、gate_proj、down_proj 三个矩阵。",
    ),
    Topic(
        "lora",
        "LoRA、QLoRA 和全量微调有什么区别？LoRA rank 怎么选？",
        "训练与优化",
        "基础",
        (r"lora", r"qlora", r"peft", r"全量微调", r"低秩", r"rank", r"adalora", r"dylora", r"dora"),
        "LoRA 冻结原权重，只训练低秩增量 BA，假设下游任务所需权重更新具有低内在秩；全量微调更新所有参数，效果上限高但显存和过拟合风险更大。QLoRA 进一步把基座量化，配合 LoRA 训练以节省显存。rank 越大容量越强但更耗显存，实际按任务复杂度、数据规模和验证集效果调；AdaLoRA/DyLoRA/DoRA 等变体分别处理动态秩、可变秩和方向/幅度分解等问题。",
    ),
    Topic(
        "rag_vs_finetune",
        "RAG 和微调的区别、优缺点和选择标准是什么？",
        "RAG、Agent 与工具调用",
        "基础",
        (r"rag和微调", r"rag.*区别", r"知识库", r"向量数据库", r"embedding模型", r"rerank", r"召回策略"),
        "RAG 把外部知识通过检索放进上下文，适合知识频繁变化、需要可追溯来源的场景；微调把能力或风格写进参数，适合稳定任务格式、特定行为习惯和领域表达。RAG 的难点是切分、召回、rerank、上下文压缩和评估；微调的难点是数据质量、遗忘、幻觉和成本。选择时看知识是否动态、是否要求引用、任务是否可由 prompt+检索解决，以及是否需要模型内化能力。",
    ),
    Topic(
        "code_search_rag",
        "Claude Code 为什么偏向 grep/ripgrep，而不是传统向量 RAG？",
        "RAG、Agent 与工具调用",
        "进阶",
        (r"claude code", r"ripgrep", r"grep", r"代码检索", r"agentic search"),
        "代码检索常需要精确匹配函数名、变量名、调用链和文件路径，grep/ripgrep 实时、零索引、可解释，能避免向量索引过期和语义误召回。Claude Code 这类工具通常把 ripgrep 作为底层搜索能力，再由 Agent 多轮决定搜什么、读什么、下一步怎么收窄范围，这更像 Agentic Search。RAG 仍适合文档和概念检索，但代码库高频变化和安全边界会让索引方案成本更高。",
    ),
    Topic(
        "agent_arch",
        "企业级 Agent 架构如何设计，Agent 与 Workflow 有何区别？",
        "RAG、Agent 与工具调用",
        "进阶",
        (r"agent与workflow", r"企业级agent", r"agent架构", r"单智能体", r"多智能体", r"bff", r"worker", r"tool server"),
        "Workflow 是预定义流程，稳定可控；Agent 是模型根据目标动态规划和调用工具，更灵活但更难控。企业级架构通常拆 BFF/API 层、Planner/Executor、Tool Server、Worker、Memory/RAG、权限审计和观测系统。设计重点是工具白名单、参数校验、超时重试、日志追踪、成本控制和人类确认节点。单 Agent 简单一致，多 Agent 适合复杂分工但会增加协调成本。",
    ),
    Topic(
        "mcp_function_skill",
        "Function Call、MCP、Tool、Skill 和 Rules 的区别是什么？",
        "RAG、Agent 与工具调用",
        "基础",
        (r"function call", r"mcp", r"skill", r"rules", r"tool"),
        "Function Call 是模型输出结构化调用参数的能力；Tool 是可执行外部能力；MCP 是标准化工具/资源/提示暴露协议，解决工具接入和上下文分发；Skill 是面向 Agent 的流程知识和本地脚本封装；Rules 更像长期约束或编码规范。面试中可用一条线回答：Function Call 解决怎么调用，MCP 解决怎么接入和发现，Skill 解决怎么把可重复工作流程化，Rules 解决长期行为边界。",
    ),
    Topic(
        "agent_context_memory",
        "Agent 的上下文、短期记忆、长期记忆和上下文压缩怎么做？",
        "RAG、Agent 与工具调用",
        "进阶",
        (r"context", r"上下文", r"memory", r"记忆", r"上下文压缩", r"short-term", r"long-term", r"task memory"),
        "短期记忆通常是当前对话和任务轨迹；长期记忆保存跨任务偏好、事实和经验；Task memory 保存项目内状态、文件和决策。上下文过长时可做摘要、分层压缩、检索式读取、重要片段 pin、按工具结果结构化存储和 rerank。关键是区分可丢信息和不可丢约束，保留 source trace，避免把不确定摘要当事实。",
    ),
    Topic(
        "sampling",
        "Temperature、Top-k、Top-p 等采样参数顺序和作用是什么？",
        "推理服务与性能",
        "基础",
        (r"temperature", r"topk", r"top-k", r"top_p", r"top-p", r"采样三参数", r"采样温度"),
        "Temperature 先缩放 logits，温度高分布更平，低温更确定；Top-k 保留概率最高的 k 个 token；Top-p 保留累计概率达到 p 的最小集合。常见实现顺序是温度缩放后再过滤并重新归一化，但具体框架可能略有差异。面试要能说明它们控制多样性、稳定性和探索，RL rollout 温度还会影响奖励方差和组内优势有效性。",
    ),
    Topic(
        "serving_vllm",
        "vLLM/SGLang 的加速方法、PagedAttention 和资源调度怎么理解？",
        "推理服务与性能",
        "进阶",
        (r"vllm", r"sglang", r"pagedattention", r"page attention", r"资源管理", r"抢占", r"kv block", r"推理框架"),
        "vLLM 的 PagedAttention 把 KV Cache 管理成块，像虚拟内存一样按需分配，减少连续显存需求和碎片；continuous batching 把不同请求动态合批，提高吞吐。资源不足时可能通过抢占、swap/recompute 等方式腾出 GPU KV block。SGLang 更强调结构化生成、前缀缓存、调度和程序化接口。回答时要围绕 KV 管理、动态 batching、调度策略和延迟/吞吐权衡。",
    ),
    Topic(
        "zero_parallel",
        "ZeRO、张量并行、流水线并行和梯度检查点分别省什么？",
        "推理服务与性能",
        "进阶",
        (r"zero", r"deepspeed", r"张量并行", r"流水线并行", r"pipeline", r"gradient checkpoint", r"gradient_accumulation"),
        "ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 进一步切参数；越往后显存越省，通信和实现复杂度越高。张量并行把单层矩阵乘切到多卡，适合大层但有 all-reduce/all-gather；流水线并行按层切，适合模型深但有 pipeline bubble；梯度检查点不存部分激活，反向时重算，用算力换显存；梯度累积用小 micro-batch 模拟大 batch。",
    ),
    Topic(
        "quantization",
        "BF16/FP8/INT8 量化和反量化的核心流程是什么？",
        "推理服务与性能",
        "基础",
        (r"量化", r"bf16", r"fp8", r"int8", r"scale", r"zero[-_ ]?point", r"混合精度"),
        "量化把高精度权重或激活映射到低精度表示，核心是确定 scale，有符号非对称量化还会有 zero-point。推理时按 scale/zero-point 反量化或直接用量化 kernel 计算。BF16 保留 FP32 的指数位范围，训练稳定性好；INT8/FP8 更省显存和带宽，但要处理校准、异常值、精度损失和算子支持。面试可补充 per-tensor/per-channel、weight-only 和 activation quant 的差异。",
    ),
    Topic(
        "evaluation",
        "大模型训练、RAG 和上线效果如何评估？PPL 有什么意义？",
        "评估与对齐",
        "基础",
        (r"评估", r"ppl", r"困惑度", r"recall", r"precision", r"准确率", r"线上指标", r"harness"),
        "PPL 衡量模型对验证文本的平均预测困惑度，适合预训练语言建模，但不等价于指令效果。SFT/RL 要看任务集、格式遵循、偏好胜率、安全和人工评估；RAG 要拆召回率、命中率、rerank、答案正确性和引用可追溯；上线还看延迟、成本、用户反馈和失败率。回答时强调离线评测与线上指标要闭环，避免只看单一分数。",
    ),
    Topic(
        "cot_train_infer",
        "训练时让模型输出 CoT、推理时只输出答案如何做？",
        "评估与对齐",
        "进阶",
        (r"cot", r"训练时.*思考", r"推理只回答", r"只输出答案", r"以训代思"),
        "可以在训练数据中保留推理过程作为隐式能力学习，同时在推理 prompt 或解码后处理要求只输出最终答案。更稳的做法是使用带 reasoning/answer 字段的结构化模板，训练时学习中间推理，部署时隐藏或蒸馏成短答案。风险是模型依赖显式 CoT 才能做对，或泄露不该展示的推理，因此要用格式奖励、答案抽取评估和无 CoT 验证集检查。",
    ),
    Topic(
        "agent_eval_observability",
        "Agent Eval、可观测体系和自动评测环境如何设计？",
        "评估与对齐",
        "进阶",
        (r"agent eval", r"agent.*评估", r"评估.*agent", r"可观测", r"observability", r"评测环境", r"harness", r"task success", r"tool success"),
        "Agent 评估要拆成最终任务成功率、工具调用成功率、轨迹质量、成本延迟、安全和可恢复性。可观测体系要记录 prompt、模型输出、tool call、tool result、状态迁移、错误重试和人工接管点。自动评测环境负责构造任务、注入可验证反馈、判断成功条件，并支持 replay，避免只靠人工看最终答案。",
    ),
    Topic(
        "agentic_rl_self_evolution",
        "Agentic RL、自蒸馏和自进化 Agent 训练范式是什么？",
        "后训练与强化学习",
        "进阶",
        (r"agentic ?rl", r"自进化", r"自蒸馏", r"self[- ]?evolving", r"search[- ]?e1", r"sdar", r"opd", r"online policy distillation", r"continual harness"),
        "Agentic RL 把模型放进多步工具环境，用最终奖励和过程奖励优化轨迹。自蒸馏/自进化通常让当前模型生成轨迹，再用规则、环境、教师模型或反思机制筛选高质量样本回流训练。核心难点是 credit assignment、环境可验证性、分布漂移、reward hacking 和训练成本。",
    ),
    Topic(
        "inference_cache_speculative",
        "Prefix Cache、RadixAttention、HiCache 和 Speculative Decoding 如何优化推理？",
        "推理服务与性能",
        "进阶",
        (r"prefix cache", r"radixattention", r"hicache", r"speculative", r"投机", r"分层.*kv", r"kv cache.*分层", r"fp8 kvcache", r"oscar"),
        "推理优化除了算子加速，还要复用前缀、管理 KV cache 和减少 decode 步数。Prefix/Radix cache 复用共享 prompt 前缀，HiCache/分层 KV cache 在 GPU/CPU/SSD 间权衡容量和带宽，Speculative Decoding 用小模型草稿加大模型验证来减少串行等待。面试回答要分别说明它们优化的是 prefill、decode 还是内存层级。",
    ),
    Topic(
        "data_mixture_engineering",
        "预训练、SFT 和后训练的数据配比与数据工程怎么设计？",
        "训练与优化",
        "进阶",
        (r"数据配比", r"data mixture", r"数据工程", r"数据清洗", r"数据.*配比", r"预训练.*数据", r"后训练.*数据", r"数据合成"),
        "数据配比决定模型能力形状。预训练看领域覆盖、质量过滤、去重和混合比例；SFT 看指令多样性、格式一致性和示范质量；RL/偏好优化看任务可验证性、奖励信号和难例分布。工程上要做数据版本、去重、污染检测、采样权重、ablation 和 per-domain eval。",
    ),
    Topic(
        "mtp_loss_speculative_training",
        "MTP Loss、多 Token Prediction 和推理加速训练怎么理解？",
        "归一化与 Loss",
        "进阶",
        (r"mtp", r"multi[- ]?token", r"多 ?token", r"多步预测", r"多token prediction", r"next.*n"),
        "MTP/Multi-Token Prediction 让模型在一个位置预测多个未来 token，为中间表示提供更密集监督，也可能服务 speculative decoding 或多 token 解码。它通常在主 next-token loss 外增加多个未来步预测 head 或共享 head，训练时要处理不同 horizon 的权重、label shift 和 loss 归一化。",
    ),
    Topic(
        "multimodal_vlm_vla",
        "多模态大模型、VLM/VLA 和具身 Agent 面试常问什么？",
        "多模态与具身智能",
        "进阶",
        (r"多模态", r"mllm", r"ml?lm", r"vlm", r"vla", r"openvla", r"具身", r"机器人", r"clip", r"omni", r"视觉语言", r"vision[- ]?language"),
        "多模态面试常问视觉编码器、投影层、跨模态对齐、图文/视频数据、推理成本和评估。VLM 偏理解图像/视频与文本，VLA/具身 Agent 进一步把感知、语言规划和动作控制连接起来。回答时要区分视觉 token 如何进入 LLM、训练目标如何对齐模态、以及真实机器人/工具环境中如何评估闭环成功率。",
    ),
    Topic(
        "deepseek_arch",
        "DeepSeek R1/V3/V4、Qwen、Kimi 等技术报告常问哪些架构创新？",
        "公司面经",
        "进阶",
        (r"deepseek", r"r1", r"v3", r"v4", r"qwen", r"kimi", r"模型训练技术报告", r"架构创新"),
        "这类问题不要背结论，要按模块拆：注意力/KV 压缩、MoE 路由与专家、训练数据和后训练、推理优化、量化/并行和评测。DeepSeek 系列常被问 MLA、MoE、GRPO/R1-Zero、FP8、算子融合；Qwen/LLaMA 常问 RoPE、RMSNorm、GQA、SwiGLU；Kimi 常问长上下文、强化学习和训练报告。答题时先说机制，再说为什么省显存/提稳定/提吞吐。",
    ),
    Topic(
        "company_interview",
        "大模型算法/Agent 岗公司面经中最常被深挖哪些方向？",
        "公司面经",
        "基础",
        (r"一面", r"二面", r"三面", r"面经", r"算法岗", r"大模型.*面", r"agent开发", r"实习一面"),
        "高频深挖集中在项目动机、数据构造、SFT/RL/DPO/GRPO 选型、reward 设计、RAG 召回与评估、LoRA/全参微调、显存估算、并行训练、MoE/Attention 架构和手撕代码。公司面经的答题策略是把项目讲成可复盘的工程闭环：为什么做、数据怎么来、指标怎么定、失败案例是什么、替代方案比较、线上如何验证。",
    ),
    Topic(
        "coding_common",
        "大模型算法岗常见手撕代码题有哪些？",
        "手撕代码",
        "基础",
        (r"手撕", r"算法题", r"leetcode", r"岛屿", r"三数之和", r"合并区间", r"平方根", r"链表", r"nms", r"代码题"),
        "除了模型相关 loss/norm/attention，通用题常见三数之和、岛屿数量/面积、合并区间、整数平方根、链表删除倒数第 N 个、NMS、时间戳/字符串处理等。准备方式是每题掌握边界条件、复杂度和可运行代码。模型岗手撕更看重你能否把数学公式落到 shape 正确、数值稳定的实现上。",
    ),
]


# Research topics have no XHS matching patterns: primary sources are not note frequency.
BASE_TOPIC_KEYS = {topic.key for topic in TOPICS}
TOPICS = [
    replace(topic, question=UPDATES[topic.key]["question"])
    if topic.key in UPDATES else topic for topic in TOPICS
]
TOPICS.extend(
    Topic(item["key"], item["question"], item["category"], item["difficulty"], (), "")
    for item in MANIFEST["topics"] if item["kind"] == "new"
)


LLM_KEYWORDS = re.compile(
    r"(?i)(大模型|llm|大语言|transformer|attention|gqa|mqa|mla|kv ?cache|flash ?attention|"
    r"deepseek|qwen|llama|kimi|claude|codex|agent|rag|mcp|function call|sft|rlhf|vla|vlm|openvla|(?<![a-z])(dpo|ppo|grpo|dapo|gspo)(?![a-z])|"
    r"reward|lora|qlora|moe|vllm|sglang|rmsnorm|layernorm|rope|量化|微调|后训练|强化学习|推理优化|"
    r"多模态|embedding|rerank|向量数据库|cot|prm|deepspeed|zero)"
)

DISCOVER_TECH_HINTS = re.compile(
    r"(?i)(面试|面经|八股|手撕|校招|社招|算法岗|常问|被问|transformer|attention|kv ?cache|rope|norm|loss|"
    r"训练|微调|推理|部署|框架|架构|agent|rag|mcp|function call|tool|sft|rlhf|(?<![a-z])(dpo|ppo|grpo)(?![a-z])|reward|"
    r"prm|lora|moe|vllm|sglang|蒸馏|量化|评估|检索|召回|rerank|embedding|上下文|代码|(?<![a-z])code(?![a-z])|算法)"
)

STRONG_DISCOVER_INTERVIEW = re.compile(r"(面试|面经|八股|手撕|校招|社招|算法岗|面试题|常问|被问)")

IRRELEVANT_HINTS = re.compile(
    r"(?i)(前端|vue|javascript|java面试|spring|redis|kafka|微服务|cookie|session|get与post|get和post|"
    r"hashmap|hashtable|产品经理|数据分析|星穹铁道|崩坏|windows电脑|vscode快捷键|http协议|浏览器输入)"
)

FINANCE_QUANT_HINTS = re.compile(r"(?i)(空仓|股票|基金|交易|投研|投资|收益|卖点|量化研究员|量化交易|量化比赛|因子|回测|alpha|cta)")
MARKET_HINTS = re.compile(r"(?i)(股票|股价|a股|港股|美股|英伟达|三星电子|海力士|美光|长鑫|摩根士丹利|高盛|基本面)")
AI_PRODUCTIVITY_HINTS = re.compile(r"(?i)(ppt|figma|答辩|论文排版|操作浏览器|创业手册|创业公司|提示词模板|个人开发者|aicoding)")
GENERIC_AI_TOOL_HINTS = re.compile(r"(?i)(claude|codex|gpt)")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def source_id(note_id: str) -> str:
    return f"xhs_{note_id[:8]}"


def public_explore_url(note_id: str) -> str:
    return f"https://www.xiaohongshu.com/explore/{note_id}"


def public_profile_note_url(note_id: str) -> str:
    profile_id = os.environ.get("XHS_PROFILE_ID", "").strip()
    if not profile_id:
        return ""
    return f"https://www.xiaohongshu.com/user/profile/{profile_id}/{note_id}"


def is_relevant(row: dict) -> bool:
    if row.get("source_origin") == "discover" and row.get("extract_status") != "text-ok":
        return False
    if row.get("source_origin") == "discover":
        text = clean(" ".join([row.get("title", ""), row.get("desc", "")]))
    else:
        text = clean(" ".join([row.get("source_title", ""), row.get("title", ""), row.get("desc", "")]))
    if not LLM_KEYWORDS.search(text):
        return False
    if row.get("source_origin") == "discover" and not DISCOVER_TECH_HINTS.search(text):
        return False
    if row.get("source_origin") == "discover" and FINANCE_QUANT_HINTS.search(text) and not re.search(r"(?i)(大模型|llm|transformer|agent|rag|deepseek|qwen|lora|moe|sft|grpo|dpo)", text):
        return False
    if row.get("source_origin") == "discover" and MARKET_HINTS.search(text) and not re.search(r"(面试|面经|八股|校招|社招|一面|二面|三面)", text):
        return False
    if row.get("source_origin") == "discover" and GENERIC_AI_TOOL_HINTS.search(text) and AI_PRODUCTIVITY_HINTS.search(text) and not (STRONG_DISCOVER_INTERVIEW.search(text) or re.search(r"(?i)(大模型|llm|transformer|agent|rag|deepseek|qwen|lora|moe|sft|grpo|dpo)", text)):
        return False
    if IRRELEVANT_HINTS.search(text) and not re.search(r"(?i)(大模型|llm|transformer|agent|rag|deepseek|qwen|lora|moe|sft|grpo|dpo)", text):
        return False
    return True


def match_topics(row: dict) -> list[Topic]:
    text = clean(" ".join([row.get("source_title", ""), row.get("title", ""), row.get("desc", "")]))
    hits: list[Topic] = []
    for topic in TOPICS:
        if any(re.search(pat, text, flags=re.I) for pat in topic.patterns):
            hits.append(topic)
    if is_relevant(row) and not hits:
        hits.append(next(t for t in TOPICS if t.key == "company_interview"))
    # Keep a source from exploding into every broad topic.
    priority = {topic.key: i for i, topic in enumerate(TOPICS)}
    return sorted({t.key: t for t in hits}.values(), key=lambda t: priority[t.key])[:8]


def read_extracts() -> list[dict]:
    rows: list[dict] = []
    favorite_path = REPO_ROOT / "data/raw/xhs_note_extracts_2026-05-24.jsonl"
    if favorite_path.exists():
        for line in favorite_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row.setdefault("source_origin", "favorites")
                rows.append(row)

    for path in sorted((REPO_ROOT / "data/raw").glob("xhs_discover_note_extracts_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row.setdefault("source_origin", "discover")
                rows.append(row)

    deduped: dict[str, dict] = {}
    for row in rows:
        note_id = row.get("note_id")
        if not note_id:
            continue
        existing = deduped.get(note_id)
        if existing is None:
            deduped[note_id] = row
            continue
        if existing.get("source_origin") == "favorites":
            continue
        if row.get("source_origin") == "favorites" or len(row.get("desc") or "") > len(existing.get("desc") or ""):
            deduped[note_id] = row
    return list(deduped.values())


def read_inventory_order() -> dict[str, int]:
    path = REPO_ROOT / "data/raw/xhs_inventory_2026-05-24.json"
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    return {item.get("note_id", ""): int(item.get("favorite_order") or idx) for idx, item in enumerate(items, start=1)}


def read_discover_order() -> dict[str, int]:
    order: dict[str, int] = {}
    for path in sorted((REPO_ROOT / "data/raw").glob("xhs_discover_inventory_*.json")):
        if not path.exists():
            continue
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(items, dict):
            items = items.get("items", [])
        for idx, item in enumerate(items, start=1):
            note_id = item.get("note_id", "")
            if note_id and note_id not in order:
                order[note_id] = int(item.get("discover_order") or idx)
    return order


def read_detailed_answers() -> dict[str, str]:
    path = REPO_ROOT / "data/processed/detailed_answers.md"
    if not path.exists():
        return {}
    answers: dict[str, str] = {}
    current_key = ""
    current_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if current_key:
                answers[current_key] = "\n".join(current_lines).strip()
            current_key = line.removeprefix("## ").strip()
            current_lines = []
        elif current_key:
            current_lines.append(line)
    if current_key:
        answers[current_key] = "\n".join(current_lines).strip()
    return answers


def update_detailed_answers_header(summary: dict) -> None:
    path = REPO_ROOT / "data/processed/detailed_answers.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    first_section = text.find("\n## ")
    if first_section == -1:
        body = ""
    else:
        body = text[first_section + 1 :].strip()
    version_updates = [
        "### Version Updates",
        "",
        "| Version | Date | Data Added | Related Notes Added | Topics Added | Total Related | Total Topics | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        "| v1.0 | 2026-05-24 | Favorites accessible set: 315 extracted | 204 | 36 | 204 | 36 | Initial favorite-note notebook with detailed answers. |",
        "| v1.1 | 2026-05-25 | Discovery/search: 1193 extracted | 995 | 6 | 1199 | 42 | Added discovery/search evidence, promoted recurring themes to topics, and expanded per-question detailed answers. |",
    ]
    header = [
        "# Detailed Answers",
        "",
        "This file stores long-form answer overrides for `scripts/generate_notebook.py`.",
        "Each `## <topic_key>` section is injected into `notes/llm-interview-master.md`.",
        "",
        *version_updates,
        "",
        f"Last corpus refresh: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "Current processed corpus:",
        f"- Extracted detail rows: {summary['extracted']}",
        f"- Strictly related notes: {summary['relevant']}",
        f"- Favorites related: {summary['favorites_relevant']}",
        f"- Discovery/search related: {summary['discover_relevant']}",
        f"- Normalized topic sections: {summary['topics']}",
        "",
        "Frequency counts, source IDs, and discovery deltas are regenerated in `notes/llm-interview-master.md`, `data/processed/question_bank.md`, `data/processed/sources.csv`, and `docs/discovery_delta_report_2026-05-25.md`.",
        "The answer bodies below are intentionally preserved across runs unless a topic's question wording or explanation needs to change.",
    ]
    path.write_text("\n".join(header).rstrip() + "\n\n" + body.rstrip() + "\n", encoding="utf-8")


def write_sources(rows: list[dict], row_topics: dict[str, list[Topic]], relevance: dict[str, bool]) -> None:
    path = REPO_ROOT / "data/processed/sources.csv"
    favorite_order = read_inventory_order()
    discover_order = read_discover_order()
    include_personal_order = os.environ.get("XHS_INCLUDE_PERSONAL_ORDER") == "1"
    include_collect_url = os.environ.get("XHS_INCLUDE_COLLECT_URL") == "1"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id",
                "note_id",
                "source_origin",
                "favorite_order",
                "discover_order",
                "title",
                "author",
                "url",
                "collect_url",
                "extraction_status",
                "is_llm_related",
                "categories",
                "matched_topics",
                "question_count",
                "processed_at",
                "notes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            sid = source_id(row["note_id"])
            topics = row_topics.get(row["note_id"], [])
            writer.writerow(
                {
                    "source_id": sid,
                    "note_id": row["note_id"],
                    "source_origin": row.get("source_origin", "favorites"),
                    "favorite_order": favorite_order.get(row["note_id"], "") if include_personal_order else "",
                    "discover_order": discover_order.get(row["note_id"], ""),
                    "title": row.get("source_title") or row.get("title", ""),
                    "author": row.get("source_author") or row.get("author", ""),
                    "url": public_explore_url(row["note_id"]),
                    "collect_url": public_profile_note_url(row["note_id"]) if include_collect_url else "",
                    "extraction_status": row.get("extract_status", ""),
                    "is_llm_related": "yes" if relevance.get(row["note_id"]) else "no",
                    "categories": ";".join(sorted({t.category for t in topics})),
                    "matched_topics": ";".join(t.key for t in topics),
                    "question_count": len(topics),
                    "processed_at": TODAY,
                    "notes": f"desc_len={len(row.get('desc') or '')}; image_count={row.get('image_count', '')}",
                }
            )


def write_question_bank(topic_sources: dict[str, list[dict]]) -> None:
    topic_sources = with_research_topics(topic_sources)
    topics_by_key = {topic.key: topic for topic in TOPICS}
    rows = []
    for key, sources in topic_sources.items():
        topic = topics_by_key[key]
        rows.append((topic.category, -len(sources), topic.question, topic, sources))
    rows.sort()

    lines = [
        "# Question Bank",
        "",
        f"Generated: {TODAY}",
        f"Research reviewed: {MANIFEST['reviewed_at']}; N/A means no XHS frequency measured.",
        "",
        "| Normalized Question | Category | Difficulty | Frequency | Sources | Status |",
        "|---|---|---|---:|---|---|",
    ]
    for _, _, _, topic, sources in rows:
        sids = ", ".join(source_id(s["note_id"]) for s in sources[:12])
        if len(sources) > 12:
            sids += f", +{len(sources)-12}"
        if topic.key in UPDATES:
            refs = research_refs(topic.key)
            sids = f"{sids}; {refs}" if sids else refs
        frequency = str(len(sources)) if sources else "N/A"
        status = "research-reviewed" if topic.key in UPDATES else "ready"
        lines.append(f"| {topic.question} | {topic.category} | {topic.difficulty} | {frequency} | {sids} | {status} |")
    lines.append("")
    (REPO_ROOT / "data/processed/question_bank.md").write_text("\n".join(lines), encoding="utf-8")


def write_master(topic_sources: dict[str, list[dict]], relevant_rows: list[dict], extracted_rows: list[dict]) -> None:
    topic_sources = with_research_topics(topic_sources)
    topics_by_key = {topic.key: topic for topic in TOPICS}
    detailed_answers = read_detailed_answers()
    categories = [
        "Transformer 基础",
        "Attention 与 KV Cache",
        "位置编码",
        "归一化与 Loss",
        "训练与优化",
        "后训练与强化学习",
        "MoE 与稀疏模型",
        "推理服务与性能",
        "RAG、Agent 与工具调用",
        "评估与对齐",
        "多模态与具身智能",
        "手撕代码",
        "公司面经",
    ]
    status_counts = defaultdict(int)
    for row in extracted_rows:
        status_counts[row.get("extract_status", "")] += 1
    discovery_rows = [row for row in extracted_rows if row.get("source_origin") == "discover"]
    discovery_relevant = [row for row in discovery_rows if row in relevant_rows]

    top = sorted(topic_sources.items(), key=lambda kv: (-len(kv[1]), topics_by_key[kv[0]].category, topics_by_key[kv[0]].question))[:20]
    lines = [
        "# 大模型面经八股整理",
        "",
        f"生成日期：{TODAY}",
        f"研究增量核对：{MANIFEST['reviewed_at']}（小红书语料日期不变）",
        "",
        "## 概览",
        "",
        f"- 收藏页标称笔记数：353；本轮 DOM 盘点到可读标题/链接：315。",
        f"- 标题初筛并打开抽取：{len(extracted_rows)} 条；后处理判定大模型相关：{len(relevant_rows)} 条。",
        f"- 发现页新增抽取：{len(discovery_rows)} 条；其中大模型相关：{len(discovery_relevant)} 条。",
        f"- 正文抽取状态：text-ok {status_counts['text-ok']} 条，title-only {status_counts['title-only']} 条。",
        f"- 归一化问题簇：{len(topic_sources)} 个，含 3 个独立研究专题；深化 6 个既有专题。",
        "- 频率只统计不同小红书来源笔记数；新增研究专题标为 N/A，不混入原语料频率。",
        "- 覆盖对照与去重说明：[研究增量审查](../docs/research_update_2026-10-13.md)。",
        "- 说明：不复制小红书原文全文；答案为基于来源要点和通用面试知识的整理。抽取不足的来源在 `sources.csv` 中保留状态。",
        "",
        "## 高频题 Top 20",
        "",
        "| Rank | Question | Category | Frequency |",
        "|---:|---|---|---:|",
    ]
    for idx, (key, sources) in enumerate(top, start=1):
        topic = topics_by_key[key]
        lines.append(f"| {idx} | {topic.question} | {topic.category} | {len(sources)} |")

    for category in categories:
        category_topics = [topics_by_key[k] for k in topic_sources if topics_by_key[k].category == category]
        category_topics.sort(key=lambda t: (-len(topic_sources[t.key]), t.question))
        if not category_topics:
            continue
        lines.extend(["", f"## {category}", ""])
        for topic in category_topics:
            sources = topic_sources[topic.key]
            sids = ", ".join(source_id(s["note_id"]) for s in sources[:15])
            if len(sources) > 15:
                sids += f", +{len(sources)-15}"
            if topic.key in UPDATES:
                refs = research_refs(topic.key)
                sids = f"{sids}; 研究：{refs}" if sids else f"研究：{refs}"
            frequency = str(len(sources)) if sources else "N/A（研究来源增补）"
            lines.extend(
                [
                    f'<a id="{topic.key}"></a>',
                    f"### {topic.question}",
                    "",
                    f"- 频率：{frequency}",
                    f"- 来源：{sids}",
                    f"- 难度：{topic.difficulty}",
                    "",
                    detailed_answers.get(topic.key, topic.answer),
                    "",
                ]
            )

    partial = [r for r in relevant_rows if r.get("extract_status") != "text-ok"]
    if partial:
        lines.extend(["## 待复核来源", ""])
        for row in partial:
            lines.append(f"- {source_id(row['note_id'])}: {row.get('source_title', '')} ({row.get('extract_status')})")
        lines.append("")

    (REPO_ROOT / "notes/llm-interview-master.md").write_text("\n".join(lines), encoding="utf-8")


def research_refs(key: str) -> str:
    return ", ".join(f"[{sid}]({SOURCE_MAP[sid]['url']})" for sid in UPDATES[key]["sources"])


def with_research_topics(topic_sources: dict[str, list[dict]]) -> dict[str, list[dict]]:
    result = dict(topic_sources)
    for item in MANIFEST["topics"]:
        if item["kind"] == "new":
            result.setdefault(item["key"], [])
    return result


def regenerate_from_sources() -> None:
    """Re-render public artifacts without raw text, reclassification, or new counts."""
    with (REPO_ROOT / "data/processed/sources.csv").open(encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    if not saved:
        raise SystemExit("sources.csv is empty; refusing to overwrite outputs")
    rows, relevant_rows = [], []
    topic_sources: dict[str, list[dict]] = defaultdict(list)
    for saved_row in saved:
        row = dict(saved_row)
        row["extract_status"] = row["extraction_status"]
        row["source_title"] = row["title"]
        rows.append(row)
        if row["is_llm_related"] != "yes":
            continue
        relevant_rows.append(row)
        for key in row["matched_topics"].split(";"):
            if key not in BASE_TOPIC_KEYS:
                raise ValueError(f"unknown XHS topic key: {key}")
            topic_sources[key].append(row)
    summary = json.loads((REPO_ROOT / "data/processed/run_summary_2026-05-24.json").read_text(encoding="utf-8"))
    if (len(rows), len(relevant_rows), len(topic_sources)) != (
        summary["extracted"], summary["relevant"], summary["topics"]
    ):
        raise ValueError("public sources disagree with corpus snapshot")
    sync_curated_answers()
    write_question_bank(topic_sources)
    write_master(topic_sources, relevant_rows, rows)
    print(json.dumps({"mode": "from-sources", "xhs_rows": len(rows),
                      "xhs_topics": len(topic_sources),
                      "rendered_topics": len(with_research_topics(topic_sources))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-sources", action="store_true",
                        help="Re-render from public sources.csv without changing XHS counts.")
    args = parser.parse_args()
    if args.from_sources:
        regenerate_from_sources()
        return
    rows = read_extracts()
    if not rows:
        raise SystemExit("No raw extracts. Use --from-sources for the public checkout; no outputs changed.")
    # Incomplete private downloads must not silently delete the published corpus.
    sources_path = REPO_ROOT / "data/processed/sources.csv"
    if sources_path.exists():
        with sources_path.open(encoding="utf-8", newline="") as f:
            existing_ids = {row["note_id"] for row in csv.DictReader(f)}
        if not existing_ids.issubset({row["note_id"] for row in rows}):
            raise SystemExit("Raw extracts omit existing notes; no outputs changed. Use --from-sources.")
    sync_curated_answers()
    relevance = {row["note_id"]: is_relevant(row) for row in rows}
    row_topics = {row["note_id"]: match_topics(row) if relevance[row["note_id"]] else [] for row in rows}
    topic_sources: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not relevance[row["note_id"]]:
            continue
        for topic in row_topics[row["note_id"]]:
            topic_sources[topic.key].append(row)

    relevant_rows = [row for row in rows if relevance[row["note_id"]]]
    write_sources(rows, row_topics, relevance)
    write_question_bank(topic_sources)
    write_master(topic_sources, relevant_rows, rows)

    summary = {
        "extracted": len(rows),
        "relevant": len(relevant_rows),
        "favorites_extracted": sum(row.get("source_origin") == "favorites" for row in rows),
        "favorites_relevant": sum(row.get("source_origin") == "favorites" for row in relevant_rows),
        "discover_extracted": sum(row.get("source_origin") == "discover" for row in rows),
        "discover_relevant": sum(row.get("source_origin") == "discover" for row in relevant_rows),
        "topics": len(topic_sources),
        "top_topics": [
            {
                "question": next(t for t in TOPICS if t.key == key).question,
                "frequency": len(sources),
            }
            for key, sources in sorted(topic_sources.items(), key=lambda kv: -len(kv[1]))[:10]
        ],
    }
    (REPO_ROOT / "data/processed/run_summary_2026-05-24.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_detailed_answers_header(summary)
    sync_curated_answers()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
