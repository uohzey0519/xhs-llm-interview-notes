# 近期研究增量审查与合并说明

审查日期：2026-10-13。基线提交：b2925e12a4b1c79dfe467be1c3c74883de603c3f。

本次按现有 42 个 topic key、主问题、详细答案、追问和代码块逐簇检查与指定研究主题的覆盖关系。深化 6 簇，新增 3 簇，总计 45 簇；9 个重点专题均包含 4 个面试问题、原理、公式、代码或伪代码、取舍以及 3 个有回答的深入追问。其余专题保留，不宣称已对全部历史技术陈述做逐项论文复现。

1,508 条已抽取笔记、1,199 条相关笔记以及历史频率保持不变。本次没有新增小红书抓取。16 项论文/官方资料另存 [研究来源](../data/research/sources.json)，有 source_id、公开 URL、版本、章节和证据范围。新增研究专题的频率是 N/A，不以论文数量冒充小红书出现频率。

## 优先阅读

| 专题 | 本次处理 | 关键补充 |
|---|---|---|
| [GLM SAO](../notes/llm-interview-master.md#glm_sao) | 新增 | 正确全称、DIS 双侧 mask、行为概率、critic、异步取舍与归因边界 |
| [长程 Agent RL](../notes/llm-interview-master.md#agentic_rl_self_evolution) | 深化 | 部分可观测轨迹、compaction 条件一致性、回流/蒸馏/RL/搜索、受控消融 |
| [信用分配](../notes/llm-interview-master.md#reward_advantage_prm) | 深化 | 终奖 0 的不同梯度情形、GAE、跳过观测、终止与截断 |
| [MLA](../notes/llm-interview-master.md#mla_deepseek) | 深化 | 列向量维度、吸收推导、低秩不等于无损、cache 字节核算 |
| [RoPE / NoPE](../notes/llm-interview-master.md#rope_position) | 深化 | 相对旋转恒等式、因果 NoPE、解耦位置分支及不交换性 |
| [Attention Residual](../notes/llm-interview-master.md#attention_residual) | 新增 | 深度轴、learned query、Block 部分和、平均与累加的区别 |
| [稀疏 / 线性 / Delta](../notes/llm-interview-master.md#sparse_linear_attention) | 深化 | 归一化分母、状态递推、非正交 key 干扰、gate 不保证降复杂度 |
| [V4.1 CED / CSA2](../notes/llm-interview-master.md#deepseek_v41_ced_csa2) | 新增 | 三种共享模式、分层候选池、首次扫描成本、近似 SWA replay |
| [Agent 评测](../notes/llm-interview-master.md#agent_eval_observability) | 深化 | 终态/约束 rubric、pass@k 与 pass^k、judge 校准、泄漏与因果归因 |

## 42 簇覆盖对照

“保留”表示已有基础内容是先修或不属于本次优先范围，并非认定其涵盖所有最新进展。键名保持稳定，六个扩展簇的主问题措辞更精确；原来的来源映射不改动。

| # | 原簇 key | 基线已覆盖 | 对指定近期主题的缺口 / 本次处理 |
|---:|---|---|---|
| 1 | transformer_arch | QKV、架构、序列复杂度 | 缺深度聚合；保留基础，另设 attention_residual |
| 2 | param_flops | 参数、训练显存、prefill/decode | 保留；MLA 与 CED 专题注明新增成本口径 |
| 3 | rope_position | RoPE、缩放、实现差异 | 深化：NoPE、解耦及矩阵不交换 |
| 4 | norms | LN/RMSNorm、Pre/Post-Norm | 保留；AttnRes 单独解释 key norm 与值幅度 |
| 5 | cross_entropy | CE 梯度、稳定实现 | 保留；修复代码中的误转义 |
| 6 | loss_spike | 数据/优化器/数值排查 | 保留；异步 lag 与样本保留率放 SAO |
| 7 | mlp_training | 两层 MLP 前反向 | 保留，不另设重复手撕题 |
| 8 | sft_rl_kl | KL 方向与奖励正则 | 保留；长程问题不重新复制 KL 入门 |
| 9 | post_training_methods | SFT/DPO/PPO/GRPO 比较 | 缺 SAO 独立机制；保留基础，新增 glm_sao |
| 10 | grpo_core | 组优势、clip、critic 对比 | 保留；SAO 明确 group=1 不等于单轨迹算法 |
| 11 | grpo_metrics | reward、KL、entropy、质量 | 保留；SAO 加队列滞后、保留率等指标 |
| 12 | reward_advantage_prm | ORM/PRM、step-to-token 分配 | 深化：零奖励误解、GAE、观测 mask、bootstrap |
| 13 | sft_to_rl | 开始 RL 的基本能力门槛 | 保留；长程探索能力放 Agent RL |
| 14 | distillation | on/off-policy 蒸馏 | 保留；长程回流只讨论边界和闭环 |
| 15 | moe | 路由、共享专家、负载均衡 | 保留；不把 MLA/CSA2 与专家稀疏混淆 |
| 16 | moe_rl | 专家利用率与训练监控 | 保留；修复代码中的误转义 |
| 17 | gqa_kv_cache | MHA/MQA/GQA 与 KV 估算 | 保留；MLA 提供明确数值口径和低秩差异 |
| 18 | mla_deepseek | latent 投影和概念 tradeoff | 深化：联合秩、吸收、无损边界及大矩阵原因 |
| 19 | flash_attention | IO-aware、online softmax | 保留；旧伪代码明确标为 pseudocode，并去误转义 |
| 20 | sparse_linear_attention | sparse/linear/gated 概览 | 深化：因果归一化、Delta 更新、索引成本与干扰 |
| 21 | swiglu_ffn | FFN 门控与参数维度 | 保留；不与 residual/depth attention 的门控混用 |
| 22 | lora | rank、量化微调与全参 | 保留；LoRA 更新低秩不同于 MLA cache 低秩 |
| 23 | rag_vs_finetune | 检索与训练的选择 | 保留；Agent 评测新增检索/harness 控制变量 |
| 24 | code_search_rag | 代码检索与向量检索 | 保留；不再建另一检索题 |
| 25 | agent_arch | 工具、状态、权限、runtime | 保留；训练/评测新内容放相应原簇 |
| 26 | mcp_function_skill | Tool/MCP/Skill 等边界 | 保留；修复 audit_log 代码误转义 |
| 27 | agent_context_memory | 上下文、摘要、恢复指针 | 保留；训练时 compaction 概率一致性放长程 RL |
| 28 | sampling | temperature/top-k/top-p | 保留；修复代码误转义；SAO 补概率口径问题 |
| 29 | serving_vllm | 分页、连续批处理、prefix cache | 保留；CED 是架构专题，避免混成调度优化 |
| 30 | zero_parallel | ZeRO/TP/PP/checkpoint | 保留；AttnRes 说明跨 stage 来源状态开销 |
| 31 | quantization | 精度、scale、KV/weight 量化 | 保留；CSA2 不把精度收益与层共享混报 |
| 32 | evaluation | PPL、RAG、离线与线上 | 保留；修复代码误转义，Agent 细节归原 Agent Eval |
| 33 | cot_train_infer | 训练推理目标与输出 | 保留；不把可见反思长度用作过程奖励 |
| 34 | agent_eval_observability | trace、task/tool success | 深化：终态、重复可靠性、judge、隔离及消融 |
| 35 | agentic_rl_self_evolution | 轨迹采样、筛选、回流 | 深化：POMDP、长程风险、compaction、系统成本 |
| 36 | inference_cache_speculative | prefix/radix/cache、推测解码 | 保留；CED/CSA2 作为不同层次的新专题 |
| 37 | data_mixture_engineering | 数据配比、去重、污染 | 保留；Agent Eval 落实 golden case/holdout 隔离 |
| 38 | mtp_loss_speculative_training | 多 token loss 与加速 | 保留；不将 MTP 效果归因给 attention |
| 39 | multimodal_vlm_vla | 视觉接入、位置、动作 | 保留；V4.1 限定在指定 attention/serving 机制 |
| 40 | deepseek_arch | 各架构创新的回答框架 | 保留；具体 V4.1 机制与版本证据独立成簇 |
| 41 | company_interview | 项目闭环、失败案例、取舍 | 保留；新的深入追问可作为项目复盘素材 |
| 42 | coding_common | 通用算法及模型手撕 | 保留；新增数值例直接随专题，不另设同义问题 |

## 去重决策与纠错

- MLA、RoPE、信用分配、Agent RL、Agent Eval 和 sparse/linear 已有对应簇，因此替换深化原答案，没有为这些主题再创建同义簇。删去这六簇旧答案中反复复用的泛化段落。
- SAO、AttnRes、CED/CSA2 各有独立数学对象、执行路径及取舍，在原 42 簇中没有专门问答，故新增。SAO 不重写 GRPO 入门；AttnRes 不重写 QKV 基础；CED/CSA2 不重写 MLA 基础。
- 自动检查覆盖规范化主问题完全重复、topic/source ID、来源 URL、章节重复及生成一致性。语义去重依据上表人工归属，不声称字符串检查能解决所有语义重复。
- SAO 正确全称为 Single-Rollout Asynchronous Optimization。论文明确提及 GLM-5.2，GLM-5.3 沿用 base 的事实并不证明所有后训练收益来自 SAO。未把此前的个人讨论作为外部证据，未公开私人研究文件。
- 区分“MLA 内部计算顺序等价”和“把任意 MHA 低秩替换无损”；“失败终奖为零”和“每步梯度必为负”；“NoPE 内容分支”和“整个模型无位置编码”；“SWA 近似 replay”和“精确缓存重建”。
- 原文 7 个 Python fence 存在误转义。其中奖励代码被新 GAE 示例替代，FlashAttention 代码按原文的伪代码意图改标签，其余 5 个直接修复；没有宣称全部历史 PyTorch 片段均已运行。

## 可复现更新路径

公开 checkout 无原始抓取数据，使用以下路径：

```bash
python3 scripts/enrich_detailed_answers.py
python3 scripts/generate_notebook.py --from-sources
python3 scripts/verify_outputs.py
python3 -m unittest discover -s tests -v
```

数值测试需要 NumPy（例如先运行 python3 -m pip install numpy）；结构验证和公开生成只使用 Python 标准库。正式生成器在 raw 缺失或不完整时提前退出，不再覆盖 sources.csv 和历史统计。原始完整私有抓取工作流仍保留。

编辑研究内容时修改 data/research/answers.md、topic_updates.json 和 sources.json，再执行上述流程。enrich 脚本保护 curated 覆盖，不会把更新后的答案替换回旧模板。历史统计文件保留 42 个 XHS 簇；渲染文档有 45 簇，两个数字口径不同。

## 验证结果与边界

- 原 verify_outputs.py 基线通过；扩展验证通过，输出 45 个详细答案、16 项研究来源、52 个 Python 代码块的语法检查。
- 12 项 CPU 测试通过：MLA 非方阵吸收、RoPE 相对恒等式及不交换性、GAE 正负优势/终止/截断、DIS 掩码及冻结权重导数、深度归一化轴、Delta 干扰、线性递推、可靠性指标穷举、终态 grader、公开生成幂等及 raw 缺失/不完整保护。
- 两次 enrich + public generate 后产物逐字节一致；sources.csv 与 2026-05-24 的统计文件未变。
- 未运行大模型训练、GPU kernel benchmark、真实 Agent 环境评测或性能复现。SAO 和 CED 系统示例明确标为教学/伪代码；通过数值示例不等于复现模型指标。
- 旧 enrich 脚本部分未使用的备用答案字符串会触发 Python invalid-escape SyntaxWarning；不影响本次输出与测试。本次未扩展修改这些历史模板。
