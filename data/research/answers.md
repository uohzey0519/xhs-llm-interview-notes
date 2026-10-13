# Research Answer Overrides

Reviewed: 2026-10-13. Canonical curated bodies for six existing clusters and three new clusters.
Source metadata and evidence boundaries: sources.json. Code is original teaching code,
not official training or inference code. NumPy examples use float64 on CPU.

## mla_deepseek

**面试官常问：**

1. MLA 如何联合压缩 K/V，为什么不是分别压缩两个独立矩阵？
2. 低秩是否无损？缓存变小以后为什么仍需要大投影矩阵？
3. 怎样通过矩阵吸收，在 decode 时避免恢复全部历史 K/V？
4. 与 MHA、GQA、KV 量化如何比较，RoPE 会引入什么障碍？

### 原理与关键公式

答：MLA 学习一个同时支撑各头 K/V 的共享潜向量。以下固定使用**列向量**，忽略 batch；$h_t\in\mathbb R^d$、$c_t\in\mathbb R^r$、每头 key/value 维度为 $d_k,d_v$：

$$
c_t=D h_t,\qquad
k_{t,i}^{C}=U_{K,i}c_t,\qquad v_{t,i}=U_{V,i}c_t.
$$

$D\in\mathbb R^{r\times d}$，$U_{K,i}\in\mathbb R^{d_k\times r}$，$U_{V,i}\in\mathbb R^{d_v\times r}$。联合压缩让所有头的 K/V 共享缓存 $c_t$；它不要求 K 与 V 相同。低秩约束约束的是联合线性映射的秩，并非“把任意训练好的 MHA 无损压成 MLA”。只有原映射存在相应秩的因子分解等条件满足时，替换才可能精确；通常应联合训练并评估能力。

吸收是**同一个 MLA 模型内部的代数等价变形**：

$$
(q_{t,i}^{C})^\top U_{K,i}c_j
 = (U_{K,i}^{\top}q_{t,i}^{C})^\top c_j,\qquad
\sum_{j\le t}a_{tj,i}U_{V,i}c_j
 =U_{V,i}\sum_{j\le t}a_{tj,i}c_j.
$$

令 $z_{t,i}=\sum_j a_{tj,i}c_j$，输出为
$o_t=\sum_i O_i U_{V,i}z_{t,i}$。key 上投影可移到当前 query，value 上投影可移到聚合之后并与输出投影结合。**等价计算不意味着低秩建模无损。** prefill 可能因 GEMM 效率采用展开路径，decode 则偏好节省 KV 读取；实际实现应分别 benchmark。

忽略 allocator、量化 scale 和对齐，仅计算单层单 token 缓存元素数：

$$
MHA:2Hd_k,\quad GQA:2H_{KV}d_k,\quad
MLA:r+d_R\quad(d_v=d_k).
$$

$d_R$ 是另存的共享 RoPE key 维度。总字节还乘 batch、层数、历史长度和 dtype bytes。教学例：$H=32,d_k=128,r=512,d_R=64$，MLA/MHA 为 $576/8192=7.03\%$；这**不是某模型实测值**。GQA 若 $H_{KV}=8$，比值为 $576/2048=28.13\%$。权重矩阵是模型级存储，cache 随 $T$ 增长；大矩阵仍负责跨维度映射，所以 cache 变小不等于参数量同比缩小。[R03]

### 代码或伪代码

下面验证单头、无位置分支下两种执行顺序等价；不模拟低秩替换 MHA，也不模拟 fused kernel。

```python
import numpy as np

def mla_content_paths(c, q, uk, uv):
    # c:[T,r], q:[dk], uk:[dk,r], uv:[dv,r]
    k = c @ uk.T
    v = c @ uv.T
    logits = k @ q / np.sqrt(q.size)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    explicit = weights @ v
    absorbed_logits = c @ (uk.T @ q) / np.sqrt(q.size)
    a = np.exp(absorbed_logits - absorbed_logits.max())
    a /= a.sum()
    absorbed = uv @ (a @ c)
    return explicit, absorbed, logits, absorbed_logits
```

### 技术取舍

GQA 改 KV 头数，MLA 改缓存表示维度，量化改每个元素的字节数。三者的收益不能在没有配置说明时直接相乘。吸收后的 latent dot-product 维度 $r$ 可能大于 $d_k$，还受内存布局、batch 和 kernel 限制；省 HBM 不保证所有长度下更快。RoPE 障碍及解耦分支统一见 rope_position。

### 深入追问

#### 1. 为什么不能缓存低秩 c，却每步把所有历史 KV 展开到 HBM？

答：能算对，但恢复和写回整段历史会抵消缓存带宽收益。要区分逻辑缓存容量与每步物化中间量；可比较 peak memory、HBM bytes 和 decode latency。

#### 2. rank 足够大是不是就一定比 MHA 好？

答：足够的 rank 只是表达某些映射的必要容量条件；优化、位置分支、训练数据和 kernel 路径仍会改变结果。rank 增大会降低压缩率，需画质量—内存—延迟曲线。

#### 3. MLA、MoE、MTP 应该怎样一起回答？

答：分别对应缓存表示、条件计算容量、训练信号与推测解码。可以组合，但不能把系统总收益全部归因于 MLA，也不能把 mHC 当作 KV 压缩。

## rope_position

**面试官常问：**

1. RoPE 为什么让注意力包含相对位置，但不保证无限长度外推？
2. NoPE 是没有显式位置编码，还是完全没有顺序信息？
3. 为什么对完整 MLA key 施加 RoPE 会破坏简单矩阵吸收？
4. 解耦 RoPE 的内容分支与位置分支分别缓存什么？

### 原理与关键公式

答：RoPE 对 Q/K 的二维子空间施加位置相关旋转；对列向量有：

$$
R_t=
\begin{bmatrix}
\cos(t\omega)&-\sin(t\omega)\\
\sin(t\omega)&\cos(t\omega)
\end{bmatrix},\qquad
(R_tq)^\top(R_jk)=q^\top R_{j-t}k.
$$

它把相对位移写入打分结构，但训练长度、频率谱、数据和任务仍限制泛化。位置插值缩放位置，NTK 类方法调整频率尺度，YaRN 组合频率与 attention scaling；不要把某个模型家族固定等同于某一种维度排列。交错配对与前后半维配对在权重也相应重排时可等价，直接混用 checkpoint 则不等价。多模态位置还需说明轴、patch 网格及文本位置如何衔接。[R04,R10]

**NoPE** 表示不加显式位置向量/旋转，并不删除 causal mask。双向、无位置的标准 self-attention 具有置换等变性；因果可见集合、边界 token 和多层组合会打破这种简单对称，因此 decoder-only NoPE 可以学习位置相关行为。不能推断任意一层、任意重复 token 都能恢复绝对序号，也不能把特定合成任务上的外推结果推广为所有任务优于 RoPE。[R05]

在 MLA 内容路径上，若完整 key 被旋转：

$$
(R_tq)^\top R_jU_Kc_j
=q^\top R_{j-t}U_Kc_j.
$$

投影前出现随历史位置 $j$ 变化的矩阵，不能再用一个与 $j$ 无关的 query 吸收它。旋转和一般的上投影不满足所需交换关系。解耦方法把打分拆成两部分：

$$
\ell_{tj,i}=
\frac{(U_{K,i}^{\top}q_{t,i}^{C})^\top c_j+
(q_{t,i}^{R})^\top k_j^{R}}{\sqrt{d_k+d_R}},
\quad
q_{t,i}^{R}=R_t\bar q_{t,i}^{R},\quad
k_j^{R}=R_j\bar k_j^{R}.
$$

只缓存内容 latent $c_j$ 和共享位置 key $k_j^R$。内容子空间不加旋转，可保留吸收；额外的位置分支承担相对位置。这里的“内容 NoPE 分支”**不是宣称整个模型为 NoPE**。[R03]

### 代码或伪代码

```python
import numpy as np

def rotate_pairs(x, position, frequencies):
    # x:[...,2m], frequencies:[m]; adjacent-pair convention
    x = np.asarray(x)
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    angles = position * np.asarray(frequencies)
    co, si = np.cos(angles), np.sin(angles)
    a, b = pairs[..., 0], pairs[..., 1]
    return np.stack((a * co - b * si, a * si + b * co),
                    axis=-1).reshape(x.shape)
```

### 技术取舍

完整 RoPE 提供强位置结构但使简单 latent 吸收困难；解耦保留小型显式位置通道，并引入额外 cache。NoPE 去掉显式机制，顺序学习与长度泛化更依赖数据及结构。比较时同时测原长度、目标长上下文、精确检索、顺序敏感任务和 KV 成本，不能只测一个 needle。

### 深入追问

#### 1. 直接把旋转作用到 r 维 latent，是否就解决问题？

答：这是另一个模型参数化。只有存在满足 $R_jU_K=U_K\widetilde R_j$ 等关系的特殊结构，才可与原完整旋转等价；一般矩阵不满足。需要训练验证，不能靠矩阵乘法换序宣称等价。

#### 2. 外推失败一定是高频维度的问题吗？

答：不一定。频率、相位覆盖与 attention 分布都有关，低频未充分训练同样可能出问题。应对频段、长度和任务做消融，而不是给出单一频段的通用归因。

#### 3. MLA 内容路径无 RoPE，怎么还能区分两个内容相同的位置？

答：总分还包括显式位置分支，且 hidden state 本身受历史影响。区分能力来自联合表示；位置分支过窄是否损害顺序任务需实测。

## reward_advantage_prm

**面试官常问：**

1. 80 步任务在第 79 步失败、终奖为零，是否会把前 78 步全都“惩罚”？
2. PPO 的 critic、GRPO 的组基线和 PRM 分别解决什么？
3. 多轮轨迹中工具返回 token 是否参与 actor loss 和 GAE？
4. 过程奖励怎样避免奖励冗长、重复试错与投机？

### 原理与关键公式

答：终奖只有一个数，**梯度符号由优势决定**，不是由“失败”这个词决定。忽略其他正则项，策略梯度为：

$$
\nabla_\theta J
=\mathbb E\!\left[\sum_{t\in\mathcal A}
\nabla_\theta\log\pi_\theta(a_t\mid h_t)\widehat A_t\right].
$$

$\mathcal A$ 只包括模型动作 token，$h_t$ 是含观测的历史。无 baseline 的 REINFORCE 若整条回报为 0，任务梯度可为 0；有 critic 时 $G_t-V(h_t)$ 可能为负；组归一化 GRPO 若同组有成功轨迹，失败轨迹优势通常为负，若全组同分则任务优势为 0。KL、熵等项仍可能产生梯度。不能说所有 PPO/GRPO 都等强度惩罚每一步。

GAE 用局部 TD 残差组合估计：

$$
\delta_t=r_t+\gamma(1-d_t)V(h_{t+1})-V(h_t),\qquad
\widehat A_t=\delta_t+\gamma\lambda(1-d_t)\widehat A_{t+1}.
$$

$d_t$ 表示真实终止。取 $\lambda=1$、精确终止 bootstrap、$\gamma=1$ 时可退化为 $G_t-V(h_t)$；critic 并不自动知道“究竟哪一步错了”，它只估计条件期望，错误价值函数也会错误归因。[R08,R09]

**教学反例**：三个动作的 critic 预测分别为 $0.2,0.8,0.1$，最终奖励为 0。令 $\gamma=1,\lambda=0$，优势为 $[0.6,-0.7,-0.1]$；失败轨迹上的第一个动作仍可具有正估计优势。这不是证明动作一定正确，而是说明“终奖 0 ⇒ 每步必负”不成立。

Skip-observation GAE 把递推索引移到相邻**动作 token**。工具输出仍进入下一动作的上下文；只是不给外部文字计算 actor logprob loss，也不沿其每个 token 传播 GAE。观测越长，不应凭空引入更多折扣步。真正终止置 bootstrap 为 0；训练分片/时间截断若代表可继续的任务，通常保留价值 bootstrap。若预算耗尽本身就是任务定义的失败终点，则按终止处理。[R01]

### 代码或伪代码

```python
import numpy as np

def action_gae(rewards, values, terminated, bootstrap=0.0,
               gamma=1.0, lam=0.95):
    # Arrays contain action-token positions only, for one trajectory.
    # values[i] conditions on all prior observations.
    r, v = np.asarray(rewards, float), np.asarray(values, float)
    done = np.asarray(terminated, bool)
    if r.ndim != 1 or r.shape != v.shape or r.shape != done.shape:
        raise ValueError("one aligned action-token trajectory is required")
    advantages = np.zeros_like(r)
    carry = 0.0
    for i in range(len(r) - 1, -1, -1):
        next_value = float(bootstrap) if i == len(r) - 1 else v[i + 1]
        keep = float(not done[i])
        delta = r[i] + gamma * keep * next_value - v[i]
        carry = delta + gamma * lam * keep * carry
        advantages[i] = carry
    return advantages, advantages + v
```

### 技术取舍

ORM 给终局反馈，PRM 给中间步骤反馈，critic 预测未来回报，三者并非同一对象。仅终奖简单但方差大；PRM 更密集但容易产生偏好噪声和 reward hacking。将一个 step reward 除以 token 数后复制，不等于 GAE，且会改变长度加权。应明确 token mean、sequence mean、episode sum 的目标区别。

工程建议：优先使用可执行的中间证据，如验证测试、数据库约束及状态检查；把成本、违规率与成功率分开报告。不要奖励“说自己验证了”，也不要因多写反思、多调工具就给正分。训练用 grader 与封存评测 grader 应隔离；持有集持续退化时，即使训练奖励上升也应停查。

### 深入追问

#### 1. 为什么不只保留前 78 步作为成功训练样本？

答：无法仅凭“错误发生得晚”确认前缀正确；早期动作可能埋下问题。可在可恢复环境中从前缀重采样、验证不同后续，估计条件成功率；这增加环境和 rollout 成本，也不是无代价的真实反事实。

#### 2. 更小的 lambda 一定更适合长任务吗？

答：小 lambda 更依赖 critic、偏差可能增大；大 lambda 更接近蒙特卡洛、方差和长跨度噪声更大。需同时检查 value error、优势分布及长任务成功率。

#### 3. SAO 是否彻底解决长程信用分配？

答：没有。它提供适合异步、多轮轨迹的估计与稳定化设计；价值模型误差、探索不足和奖励缺陷仍在，相关系统问题见 agentic_rl_self_evolution。

## agentic_rl_self_evolution

**面试官常问：**

1. 长程 Agent RL 相比单轮 RLHF 多了哪些困难？
2. 增加上下文窗口、增加 rollout 长度与增加训练能力是一回事吗？
3. 上下文 compaction 后如何计算正确的行为概率与价值？
4. 成功轨迹回流、自蒸馏、在线 RL 和 AlphaZero 式搜索如何区分？

### 原理与关键公式

答：Agent 的动作改变环境，下一轮观测又影响动作，优化对象是闭环轨迹。可观察历史不一定是环境的完整 Markov 状态，更严谨的表述是部分可观测决策过程。给定初始任务 $x$：

$$
p_\theta(\tau\mid x)
=\prod_t \pi_\theta(a_t\mid h_t)\,P(o_{t+1}\mid h_t,a_t),\qquad
J(\theta)=\mathbb E_\tau\!\left[\sum_{t=0}^{T-1}\gamma^tr_t\right].
$$

环境转移不由 actor 直接生成。长任务的主要困难包括：奖励稀疏、策略滞后、工具结果噪声、可变时延、环境重置、上下文丢失及分片边界。若作简化假设，每步独立且都必须成功，单步成功率为 $p$，则整链成功率为 $p^T$；例如 $0.99^{80}\approx0.448$。这是**风险累积的教学模型**，真实 Agent 可恢复、步骤不独立，不能拿它估计真实 benchmark。

长窗口只放宽可见历史上限；长输出预算只放宽行动机会。模型必须在对应的多轮环境中学会持续推进、检测错误和恢复，更多 tokens 不保证更好。GLM-5.3 官方卡强调同 base 下的后训练改进；不能据此断言所有增益由 SAO 或更长轨迹单独带来。[R02]

### 代码或伪代码

下列是系统设计伪代码，函数名代表接口，**不可直接运行，也不是 GLM 训练配方**：

```pseudocode
for task in sample_tasks_with_strata():
    env = reset_from_versioned_snapshot(task)
    history, trace = task.initial_context, []
    while not terminal and budget_remaining:
        context = compact_if_needed(history, pinned_constraints, artifact_refs)
        action, token_logp, policy_version = rollout_engine.sample(context)
        observation, reward, terminal = env.step(action)
        trace.append(context_hash, action, token_logp, policy_version,
                     observation, reward, terminal, compaction_version)
        history = append(history, action, observation)
    verify_terminal_state_in_isolated_grader(trace, env)
    enqueue_complete_or_explicitly_truncated_trace(trace)

for batch in learner_queue:
    validate_episode_boundaries_and_behavior_logp(batch)
    update_critic_then_actor(batch)  # e.g. SAO; objective explicitly configured
    run_frozen_holdout_and_cost_checks()
```

**Compaction 关键约束（工程推导）**：行为概率是基于采样时真实使用的 $c_t=C(h_t)$：

$$
\rho_t=\frac{\pi_\theta(a_t\mid c_t)}
{\pi_{\mathrm{behavior}}(a_t\mid c_t)}.
$$

训练时若改用事后重写的摘要 $\widetilde c_t$，这已不是同一条件下的比率。应保存实际输入或可重建的版本及哈希，保留动作 mask、tool/schema 版本、环境快照、截断原因和逐 token 行为 logprob。观测要参与 conditioning，不能因 loss mask 为 0 就从上下文删除。若摘要本身由待训练策略生成，是否纳入动作与奖励范围也要明确定义。

### 技术取舍

| 方案 | 更新依据 | 主要收益 | 局限 |
|---|---|---|---|
| 成功轨迹 SFT 回流 | 经验证的行动序列 | 简单、可离线 | 筛选偏差，失败状态覆盖不足 |
| 在线策略蒸馏 | 学生采样，教师提供监督 | 覆盖学生实际访问状态 | 教师成本与错误继承 |
| 在线 Agent RL | 环境回报及优势 | 优化闭环任务 | rollout 昂贵、奖励与环境工程复杂 |
| 搜索/自博弈 | 可分支模拟、价值及终局反馈 | 提高探索和前缀评估 | 真实工具未必可撤销或精确模拟 |

把这些方法串成一条流程不等于它们是同一目标函数。“所有组件同时升级后效果变好”只能证明系统组合有效；要归因于 RL，需要固定任务集、base、harness、工具版本和测试预算，做有无 RL、compaction、PRM、异步策略等消融，记录成功率与单位成功成本。[R08,R09,R12；表格及实验设计为原创归纳]

### 深入追问

#### 1. 能不能直接照搬 AlphaZero 的自博弈？

答：前提差异很大。棋类具有可克隆状态、明确规则与胜负；真实工具可能慢、随机且有外部副作用。可以在隔离、可复位的代码环境做搜索与课程学习，但不能把 judge 自评当作可靠胜负，也不能把重复尝试的 pass@k 与单次能力混报。

#### 2. 只训练成功轨迹会怎样？

答：模型可能学不到错误恢复，也容易集中到简单任务。保留失败状态用于分析、重采样、偏好构造或 RL，不等于把失败动作当成 SFT 正例；训练池、开发集、封存评测集按任务家族隔离。

#### 3. 异步吞吐提高是否就表示训练更高效？

答：要同时看每 GPU 小时的独立评测提升、有效训练 token 比例、队列等待和长任务覆盖。大量陈旧轨迹被 mask 掉，或快速简单任务淹没困难任务，都可能使吞吐上升但有效学习下降。SAO 机制见 glm_sao。[R01]

## agent_eval_observability

**面试官常问：**

1. LLM 输出 rubric 与 Agent 评测 rubric 的对象有何不同？
2. task success、tool success、pass@k、pass^k 应怎样定义？
3. 怎样设计可复现、可审计、不会奖励“看起来完成”的 grader？
4. 模型、harness、检索和 reward 同时更新后，如何判断谁带来了提升？

### 原理与关键公式

答：文本 rubric 评价回答是否正确、清晰、有依据；Agent 还要评价**环境最终状态、行为约束、工具过程与成本**。相同最终话术可能对应完全不同的真实操作结果。先定义 task、初始快照、允许动作、预算、终态判定，再定义一次 trial。代码修复看隔离测试和实际 diff；数据库任务看目标行及约束；不是检查模型是否说了“已完成”。[R11,R12]

建议 rubric 分开保存以下维度，避免一个加权总分遮住严重错误：

| 维度 | 可检查证据 | 典型反例 |
|---|---|---|
| 结果 | 测试、数据库状态、文件产物 | 回答成功，实际没保存 |
| 约束 | 授权范围、不可修改项、预算 | 完成目标但误改其他记录 |
| 证据 | 引用可访问且支持结论 | 引用存在，内容不支持 |
| 恢复 | 注入故障后能继续，结果仍正确 | 靠无界重试碰对 |
| 效率 | 实际 token、调用次数、费用、p95 | 以更高预算换分却不披露 |

若 $n$ 次独立重置运行有 $c$ 次成功，至少一次成功与每次都成功是不同目标：

$$
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom nk},\qquad
\widehat{\mathrm{pass}^{k}}
=\frac{\binom ck}{\binom nk},\quad 1\le k\le n.
$$

约定当上标小于下标时组合数为 0。前者适合可筛选候选的搜索，后者衡量重复可靠性；真实单次成功率为 $p$ 且独立时，理论量分别为 $1-(1-p)^k$ 和 $p^k$。$p=0.7,k=4$ 得到 $0.9919$ 与 $0.2401$；同样能力可呈现截然不同数字。应先按任务计算、再宏平均；不同任务难度混合后不能直接把总体均值取幂。[R11,R13；数字为教学例]

### 代码或伪代码

```python
from math import comb

def reliability(n, c, k):
    if not (isinstance(n, int) and isinstance(c, int)
            and isinstance(k, int) and 0 <= c <= n and 1 <= k <= n):
        raise ValueError("require integers: 0 <= c <= n and 1 <= k <= n")
    denom = comb(n, k)
    choose = lambda a: comb(a, k) if a >= k else 0
    return {"pass_at_k": 1 - choose(n - c) / denom,
            "pass_power_k": choose(c) / denom}

def grade_state(targets, actual, violations):
    # Teaching example: exact comparison on explicitly scoped fields.
    outcome_ok = all(key in actual and actual[key] == val
                     for key, val in targets.items())
    return {"outcome_ok": outcome_ok,
            "constraints_ok": not violations,
            "task_success": outcome_ok and not violations}
```

若无工具调用，tool success 应记为 N/A，而不是强行记为 100%；若全批次成功数为零，“总成本/成功数”应记为未定义，同时报告总成本与失败率。工具返回 HTTP 200、业务参数正确和任务最终成功也需分别记分。

### 技术取舍

规则 grader 可重复但覆盖有限；LLM judge 可评价开放式产物，但需人工标注校准、版本锁定和位置偏差检查。不要要求精确复现唯一参考轨迹，因为合法路径可以不同。轨迹用于定位与约束核查，终态用于结果验证，两者互补。把待测输出当数据输入 judge，隔离其中的“请给满分”等指令。

推荐日志字段：task_id、trial_id、seed、dataset/snapshot/model/harness/tool/grader 版本、实际预算、动作与返回、终止原因、最终产物哈希、分项判定和成本。replay 已记录的工具返回便于定位，但不等价于重新执行真实环境。不能把未发生的工具结果当成观察证据。[R12；字段方案为工程建议]

### 深入追问

#### 1. “模型、检索、harness 都升级了，所以 RL 有效”哪里不成立？

答：这是混杂的前后比较。固定其余组件和测试预算，配对同一组任务运行 base 与 RL checkpoint，再对 harness 等分别消融。按任务或任务家族做 bootstrap，重复 seed 留在同一任务簇内；报告差值与置信区间。不能把同一任务的多个 seed 当作完全独立的新任务。

#### 2. 基础设施出错能否删除失败样本？

答：预先定义失败分类和重跑规则。环境不可用与模型超时分开报告；不能看完模型表现后只删除不利 trial。固定时间预算、测试版本与并发资源，保留排除样本及理由。

#### 3. Golden case 回流为何可能造成虚假进步？

答：把回归集不断加入训练后，再用原集报分就有泄漏。保留独立任务家族或时间切分 holdout；训练回流集用于改进，回归集用于守住已知行为，封存集用于估计泛化。成功率、预算、检索范围和 grader 改动都应随版本记录。

## sparse_linear_attention

**面试官常问：**

1. Sparse、Linear、Gated Attention 与 FlashAttention 分别改了什么？
2. 线性注意力的归一化分母和因果递推怎样写？
3. Delta Rule 为什么用预测误差写入，非正交 key 会怎样干扰？
4. 与 AttnRes、CSA2 能否组合，不能混淆哪些复杂度口径？

### 原理与关键公式

答：稀疏注意力减少**序列位置的连接**；核化线性注意力改变聚合形式；门控控制信息通过或遗忘；FlashAttention 改 exact attention 的执行和 IO。普通稠密 attention 后乘一个 gate，通常仍是二次序列复杂度；**门控本身不保证线性**。

对因果核化线性注意力，$\phi(k)\in\mathbb R^m$，$v\in\mathbb R^{d_v}$：

$$
S_t=\sum_{j\le t}\phi(k_j)v_j^\top,\quad
z_t=\sum_{j\le t}\phi(k_j),\qquad
o_t=\frac{\phi(q_t)^\top S_t}
{\phi(q_t)^\top z_t+\epsilon}.
$$

非负特征映射有助于分母稳定。这是核化 attention 的精确递推，不普遍等价于 softmax attention。状态为 $O(md_v+m)$，序列扫描为 $O(Tmd_v)$；“线性”指固定特征维度后关于 $T$ 线性，不能丢掉通道维度。[R15]

Delta Rule 是关联记忆的误差校正写入。以下采用 $S\in\mathbb R^{d_k\times d_v}$、单位范数 key：

$$
\bar S_t=\alpha_t S_{t-1},\quad
e_t=v_t-\bar S_t^\top k_t,\quad
S_t=\bar S_t+\beta_t k_te_t^\top,\qquad
o_t=S_t^\top q_t.
$$

这是标量遗忘 gate 的教学形式；$\alpha$ 控制旧记忆，$\beta$ 控制当前纠错强度。若 $\alpha=1,\beta=1,\|k\|=1$，写入后 $S_t^\top k=v_t$；若 key 不单位化，这一结论不成立。KDA 等更细粒度 gate 不能不加说明地等同于此标量公式。[R14]

### 代码或伪代码

```python
import numpy as np

def delta_update(state, key, value, alpha=1.0, beta=1.0):
    # state:[dk,dv], key:[dk] (unit norm), value:[dv]
    decayed = alpha * state
    error = value - key @ decayed
    return decayed + beta * np.outer(key, error)

def causal_linear(q, k, v, eps=1e-12):
    # q,k are nonnegative feature vectors [T,m]; v:[T,dv]
    state = np.zeros((k.shape[1], v.shape[1]))
    normalizer = np.zeros(k.shape[1])
    outputs = []
    for qi, ki, vi in zip(q, k, v):
        state += np.outer(ki, vi)
        normalizer += ki
        outputs.append((qi @ state) / (qi @ normalizer + eps))
    return np.asarray(outputs)
```

### 技术取舍

原创小例：$k_1=(1,0)$，$k_2=(0.9,\sqrt{0.19})$，两者单位化但内积为 0.9。先写 $v_1=1$，再按 $\alpha=\beta=1$ 写 $v_2=0$，第二次误差为 $-0.9$，对第一个 key 的读取改变 $-0.81$。误差校正能覆写当前 key，却仍会干扰相似 key，固定大小记忆不是无限容量字典。

| 机制 | 主要成本 | 主要风险 |
|---|---|---|
| 全量 softmax | $O(T^2d)$；decode 每步扫历史 | KV 读取与长序列成本 |
| 稀疏 softmax，每 query 选 k 项 | 注意力约 $O(Tkd)$，另加选择成本 | 漏掉重要连接，indexer 可能仍扫全历史 |
| 核化/递推状态 | 固定维度下 $O(Tmd_v)$ | 状态容量、干扰与数值稳定 |
| 混合层 | 取决于各类层比例及缓存 | 保留精确检索但不会自动消除所有 KV |

### 深入追问

#### 1. sparse 的 k 固定，为什么系统仍未必是严格线性？

答：Top-k 前的索引打分可能扫描所有历史；必须把 indexer、压缩和局部窗口一起计入。CSA2 的首次全局扫描与后续候选池重索引见 deepseek_v41_ced_csa2。[R07]

#### 2. Attention Residual 是否也在压缩历史 token？

答：不是。AttnRes 处理同一 token 的深度来源，序列 attention 处理不同 token 的关系。两种轴可以组合，详见 attention_residual。

#### 3. 固定状态内存是否意味着百万 token 的任意精确回忆？

答：不能。状态大小不随长度增加的优势，同时意味着信息瓶颈。需要测相似 key 干扰、长距离精确检索和多次覆写；单看吞吐不够。

## glm_sao

**面试官常问：**

1. SAO 的准确全称是什么，和 GLM-5.2/5.3 是什么关系？
2. 为什么同 prompt 多轨迹组采样会加剧长程异步训练的等待？
3. DIS 与 PPO clipping 有何不同，为什么要保存 rollout logprob？
4. 单轨迹如何估计优势，为什么 critic 不会因此消失？

### 原理与关键公式

答：此处 SAO 指 **Single-Rollout Asynchronous Optimization（单轨迹异步优化）**，不是 Self-Adaptive Optimization。SAO 论文明确给出在 GLM-5.2 agentic RL 中的部署；GLM-5.3 官方卡说明它沿用 GLM-5.2 base、通过后训练提升能力。两份证据不能推出“SAO 首次由 5.3 提出”或“5.3 所有提升均由 SAO 导致”，也不应套用到架构不同的 GLM-5.3-Flash。[R01,R02]

每 prompt 采样一条轨迹，完成后即可入训练队列，避免为同组慢轨迹等待；这不意味着 batch size=1，也不意味着只训练单步。critic 提供单轨迹所需基线；论文研究了更频繁的 critic 更新、冻结 critic attention 及动作间 GAE。冻结的是价值模型的 attention，不能解读成 actor attention 不训练。[R01]

行为策略可能随采样更新。对每个动作 token 保存采样器当时记录的概率，直接计算：

$$
\rho_t=\exp\!\left(\log\pi_\theta(a_t\mid h_t)
-\log\pi_{\mathrm{rollout},t}(a_t\mid h_t)\right),\qquad
w_t=\rho_t\,\mathbf1[1-\epsilon_l<\rho_t<1+\epsilon_h].
$$

DIS 在上下界之外直接排除 token，而 PPO 的 min/clip 取决于优势方向。二者不能写成同一机制。下式是用于解释 policy-gradient 权重的 **stop-gradient 教学 surrogate**：

$$
\mathcal L_{\mathrm{actor}}
=-\frac{\sum_{t\in\mathcal A}
\operatorname{sg}(w_t\widehat A_t)\log\pi_\theta(a_t\mid h_t)}
{|\mathcal A|}.
$$

这里显式停止权重和优势的梯度；否则对 $\rho\log\pi$ 两条路径都求导会改变估计器。论文展示 DIS 目标与校准函数，正式复现还需核对实现的 detach、序列归一化、正则项及采样概率口径。下面只演示 mask 权重，**不是完整 SAO 实现**。[R01,R08]

### 代码或伪代码

```python
import numpy as np

def dis_weights(logp_current, logp_behavior, action_mask,
                eps_low=0.2, eps_high=0.2):
    # Bounds are illustrative, not reported GLM hyperparameters.
    if not (0 < eps_low < 1 and eps_high > 0):
        raise ValueError("invalid DIS bounds")
    delta = np.asarray(logp_current) - np.asarray(logp_behavior)
    action_mask = np.asarray(action_mask, bool)
    keep = (action_mask & np.isfinite(delta)
            & (delta > np.log1p(-eps_low))
            & (delta < np.log1p(eps_high)))
    weights = np.zeros_like(delta, dtype=float)
    weights[keep] = np.exp(delta[keep])  # never exponentiate outliers
    return weights

def dis_surrogate(logp_current, frozen_weights, frozen_advantages, action_mask):
    # Caller freezes weights/advantages before this loss evaluation.
    valid = np.asarray(action_mask, bool)
    if not valid.any():
        raise ValueError("empty action-token batch")
    lp = np.asarray(logp_current)
    w = np.asarray(frozen_weights)
    a = np.asarray(frozen_advantages)
    used = valid & (w != 0)
    return -np.sum(w[used] * a[used] * lp[used]) / valid.sum()
```

### 技术取舍

单轨迹提高完成即消费的灵活性，却需支付 critic 成本；DIS 减少异常更新，却引入截断偏差并丢失样本。它是实际稳定化策略，不能声称严格恢复完整 on-policy 状态分布。组方法省 critic，但更依赖组内方差及同步等待。选择应按环境时延、奖励分布和单位算力质量提升衡量。

工程建议：同时监控 queue age、policy-version lag、token 保留率、任务长度分层、value error、独立评测与环境失败率。采样器/训练器的 temperature、截断分布和数值实现若不一致，logprob 差异不全是“模型更新”；应记录实际概率定义，不要把新模型重算的概率当行为概率。

### 深入追问

#### 1. GRPO 设 group size=1 不就是 SAO 吗？

答：不是。单元素组的中心化奖励为 0，原组优势退化。SAO 依赖 learned value baseline 及相关训练设计，而不是只改一个 group size。

#### 2. 所有 token 都被 mask 后怎么办？

答：actor 任务梯度为零，不应通过除以极小有效数量伪造放大。记录保留率、检查策略滞后和概率口径；代码分母固定为动作 token 数，全部被排除得到 0。critic 是否更新需按配置单独决定。

#### 3. 消费最快完成的轨迹是否有偏？

答：可能过度代表短、容易或工具快的任务。单轨迹减少组等待，不会自动保证任务采样分布。需要分层队列、受控抽样及按任务难度/长度验证，这是系统设计问题。

## attention_residual

**面试官常问：**

1. 普通残差为何可看作固定权重的深度累积？
2. AttnRes 的 attention 在序列轴还是层轴，Q/K/V 是什么？
3. Full 与 Block AttnRes 的计算、显存和通信开销如何变化？
4. 它与 self-attention、门控残差、mHC 是什么关系？

### 原理与关键公式

答：把每个 attention 或 FFN 子层都作为一层，设输入 embedding 为 $v_0$，子层新产生的输出为 $v_i=f_i(h_i)$。普通残差展开为：

$$
h_\ell=v_0+\sum_{i=1}^{\ell-1}v_i.
$$

这里各历史输出系数为 1，表示深度方向的固定累加，**不是概率平均**。Full AttnRes 则让同一 token 对历史深度输出做归一化选择：

$$
\alpha_{\ell i}
=\frac{\exp(w_\ell^\top\operatorname{RMSNorm}(v_i))}
{\sum_{j<\ell}\exp(w_\ell^\top\operatorname{RMSNorm}(v_j))},
\qquad
h_\ell=\sum_{i<\ell}\alpha_{\ell i}v_i.
$$

query $w_\ell$ 是每个目标子层的可学习向量；key 来自历史输出的归一化表示，value 保留原输出。query 本身不依赖 token，但 key 依赖，因此权重可以逐 token 不同。softmax 归一化轴是**深度来源**，没有在这一步混合不同 token；序列 self-attention 仍在子层内部工作。[R06]

Block AttnRes 把若干子层输出相加为 block 表示 $b_n$；目标子层可读 embedding、已完成 blocks 及当前 block 的非空 partial sum。它保留块内普通求和，跨块采用 attention。必须缓存子层的**新输出或其块和**，若把所有已经累加过的 hidden state 再当独立增量相加，会重复计算信息。[R06]

### 代码或伪代码

下面独立实现无可学习 RMSNorm scale 的教学聚合器。输入 source 可以是 Full 的输出列表，也可以是 Block 的块和；这是聚合 primitive，不是完整 Transformer。

```python
import numpy as np

def depth_mix(sources, query, eps=1e-6):
    # sources:[N,B,T,D], query:[D]; never softmax across T
    values = np.asarray(sources)
    if values.ndim != 4 or values.shape[0] == 0:
        raise ValueError("sources must be nonempty [N,B,T,D]")
    keys = values / np.sqrt(np.mean(values ** 2, axis=-1, keepdims=True) + eps)
    logits = np.einsum("nbtd,d->nbt", keys, query)
    weights = np.exp(logits - logits.max(axis=0, keepdims=True))
    weights /= weights.sum(axis=0, keepdims=True)
    output = np.einsum("nbt,nbtd->btd", weights, values)
    return output, weights
```

### 技术取舍

按单 token、总子层数 $L$、hidden $d$ 估算 Full 的聚合计算 $O(L^2d)$、保留来源状态 $O(Ld)$；推广到 batch 和序列需乘 $BT$。若 $N$ 个 blocks，来源状态降为 $O(Nd)$，聚合计算粗略为 $O(LNd)$。这是残差模块的额外成本，不能写成整个 Transformer 复杂度。

depth attention 允许选择历史层，却改变了原有单位直通路径；不能自动继承普通残差的所有梯度性质。Block 降低保存与跨 pipeline 传输开销，但丢失块内单层选择精度。实际要看重计算策略、并行切分与算子融合，论文测得的开销不能直接移植到任意系统。[R06；复杂度展开为教学分析]

### 深入追问

#### 1. 所有 alpha 相等时，是不是恢复普通 residual？

答：等权 softmax 是平均，普通残差是和，幅度不同。初始化 $w=0$ 得到等权平均，不能说数值完全恢复标准残差；需另有缩放或专门的参数化。

#### 2. AttnRes 是否替代 self-attention？

答：它改跨深度的聚合方式；self-attention 处理序列内部关系，两者可共存。Block 版本中普通求和仍用于块内，不能笼统说“整个模型取消残差”。

#### 3. 它与一个 sigmoid gate 或 mHC 等价吗？

答：不等价。单 gate 通常只控制当前分支/累计流，AttnRes 可对多个历史来源分别分配权重。mHC 属于另一种多流连接及混合约束方案，不能把它当 AttnRes 别名。比较需固定参数、训练预算与系统成本。

## deepseek_v41_ced_csa2

**面试官常问：**

1. Causal Encoder–Decoder 为什么能减轻长输入 prefill，和双向 encoder 有何不同？
2. CSA2 的 Full、Reindex、Reuse 到底共享什么？
3. 分层 indexer 哪一步仍扫描全历史，为什么不能说全部变 O(1)？
4. SWA bounded replay 是数学精确复用还是质量可控的近似？

### 原理与关键公式

答：这里 encoder 仍然是**因果**的。CED 把下半层作为 causal encoder，decoder 的 global KV 从 encoder 最终 hidden 通过投影取得；decoder 的局部 SWA KV 仍依赖各层自身状态。用报告的记号，$H^{L/2}$ 为 encoder 输出，decoder Full 层的全局条目及压缩权重来自：

$$
C_\ell=H^{L/2}W_\ell^{KV},\qquad
Z_\ell=H^{L/2}W_\ell^{Z},\qquad \ell>L/2.
$$

长 prompt 主体因此主要经过 encoder，decoder 用有限窗口 replay 准备局部状态；生成新 token 仍要经过模型的生成路径。以“每 token 每层工作量”作**简化成本模型**，输入长度 $T\gg W$ 时：

$$
C_{\mathrm{prefill}}\sim T\frac L2+W\frac L2
+\text{global projections/indexing}.
$$

相较常规 $TL$ 可明显减少工作量；这不是完整 FLOPs 公式，不能忽略 attention、MoE、视觉及 cache miss 成本后宣称精确减半。[R07 §2.2]

CSA2 把 global KV 共享与 sparse index 共享分开；模式按层静态分配，而不是每个 token 随意选择：[R07 §2.3]

| 模式 | main KV / indexer K | indexer Q / Top-k | 当前层 main Q、SWA KV 与 attention 输出 |
|---|---|---|---|
| Full | 产生新表示 | 新建 Q，重新选索引 | 重新计算 |
| Reindex | 复用对应旧表示 | 新建 Q，重新选索引 | 重新计算 |
| Reuse | 复用对应旧表示 | 复用该 KV 对应的最近索引 | 重新计算 |

Reuse **没有复用整个 attention 输出**。保存的索引必须与共享 KV 的版本、位置编号一致；更换 KV 源以后不可继续使用不匹配的旧索引。

### 代码或伪代码

以下原创伪代码展示状态依赖，省略 tensor/kernel 细节；candidate_pool 仅指 decoder 的分层索引域，encoder 使用全部因果可见位置：

```pseudocode
for layer in layers:
    main_q, local_kv = project_current_layer(hidden)
    if layer.mode == Full:
        source = encoder_final if layer.in_decoder else hidden
        main_kv, index_k = compress_and_project(source)
        kv_version = new_version()
        scores = score_all_causally_visible(layer.index_q(hidden), index_k)
        index = topk(scores, kv_version)
        candidate_pool = block_candidates(scores) if layer.in_decoder else ALL_VISIBLE
    elif layer.mode == Reindex:
        assert main_kv exists
        index = reselect(layer.index_q(hidden), index_k, candidate_pool, kv_version)
    elif layer.mode == Reuse:
        assert index.kv_version == kv_version
    hidden = attend_and_transform(main_q, main_kv[index], local_kv)
```

decoder 分层索引先由一个 Full 层扫描全局位置，按块分数建立比最终 Top-k 更大的候选池；后续 Reindex 在池内重新打分。设全局条目数 $M$，候选数 $C$，后续有 $R$ 次重索引，索引工作量可从约 $(R+1)M$ 改为：

$$
O(M+RC).
$$

这是固定打分维度和忽略选择算法常数后的**单 query 索引成本**。首层 $M$ 仍在，prefill 有多个 query；不能改写为整个网络、整个序列都是常数成本。过小候选池会漏掉深层需要的条目。[R07 §2.3.2]

### 技术取舍

SWA 跨层依赖扩展，精确重建可能需回放更长历史。bounded replay 只回放最近窗口，接受近似状态；即便报告实验质量损失很小，也不是对所有输入都等价。需要评估首次 decode、cache 命中位置、长间隔会话与局部检索。报告中的持久缓存节省还涉及移除 SSD 中 SWA 缓存、共享和精度，不能只归因于 CED。[R07 §3.2]

与 MLA 的区别：MLA 主要压每 token 的表示维度；CSA2 还包含序列压缩、跨层共享及索引复用。缓存大小可粗略写为“保留层组 × 保留位置数 × 每条目维度 × dtype bytes”，再加本地窗口和元数据。这是资源核算分解，不能把各优化百分比不加条件地相乘。[R07,R16]

### 深入追问

#### 1. decoder 全局 KV 都来自 encoder，decoder 的深度还有用吗？

答：每层 query、局部 SWA、FFN/MoE 和 hidden 转换仍在更新；共享 KV 不等于共享 query 或输出。尤其 Reindex 允许不同深度重新选择全局位置。

#### 2. 候选池在不同 query 间能直接复用吗？

答：不能由“跨层共享”推出“跨 query 共享”。报告描述的是对应 query 的跨索引层候选限制；应保留因果 mask，并核对压缩条目覆盖范围，防止使用包含未来 token 的未完成块。

#### 3. 如何证明速度提升不是 cache 命中率差异造成的？

答：控制输入输出长度、batch、精度、硬件与命中位置，分别报告冷启动 prefill、命中 prefill、decode 和端到端时延。对共享、索引和 replay 逐项消融，并同时报告质量。这是工程验证建议，不是本次已运行的模型实验。
