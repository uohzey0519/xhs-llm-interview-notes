# Full Run Report 2026-05-24

## Scope

- Target page: a logged-in Xiaohongshu 收藏 -> 笔记 page configured locally through `XHS_FAVORITES_URL` or `XHS_PROFILE_ID`.
- Displayed page count: `笔记・353`.
- Accepted accessible inventory: 315 unique notes, confirmed by user during the run.
- Extraction method: Chrome AppleScript DOM extraction after inventory scrolling.

## Results

| Metric | Count |
|---|---:|
| Accessible notes opened and extracted | 315 |
| LLM/interview-related notes | 204 |
| Normalized question clusters | 36 |
| Detailed answer sections with interviewer questions and `$$` formulas/code | 36 |
| `text-ok` notes | 255 |
| `title-only` notes | 60 |
| Related `text-ok` notes | 198 |
| Related `title-only` notes | 6 |

## Top High-Frequency Questions

| Rank | Question | Frequency |
|---:|---|---:|
| 1 | 大模型算法/Agent 岗公司面经中最常被深挖哪些方向？ | 115 |
| 2 | SFT、RLHF、DPO、PPO、GRPO、DAPO、GSPO 等后训练方法如何区分？ | 72 |
| 3 | DeepSeek R1/V3/V4、Qwen、Kimi 等技术报告常问哪些架构创新？ | 52 |
| 4 | GRPO 的核心思想、loss、优势估计和 PPO 区别是什么？ | 51 |
| 5 | Transformer 的基本架构和 Self-Attention 计算流程是什么？ | 35 |
| 6 | Agent 的上下文、短期记忆、长期记忆和上下文压缩怎么做？ | 30 |
| 7 | MoE 的原理、Dense 对比、路由和负载均衡难点是什么？ | 27 |
| 8 | 为什么常说 SFT 更像正向 KL，RL/偏好优化更像反向 KL？ | 24 |
| 9 | 大模型训练、RAG 和上线效果如何评估？PPL 有什么意义？ | 23 |
| 10 | LoRA、QLoRA 和全量微调有什么区别？LoRA rank 怎么选？ | 23 |

## Partial Related Sources

These related sources were retained but only title-level text was available in the extracted page DOM:

| Source ID | Title |
|---|---|
| `xhs_6a0aa47b` | 从 KL 的方向看 SFT 与 RL：模型到底是在“ |
| `xhs_69d67834` | 小红书 ai 应用开发一面 |
| `xhs_69d9d3c8` | 腾讯 ai 应用开发 一面 |
| `xhs_68159779` | 做RL？先搞清楚Reward Model怎么搞 |
| `xhs_682ab42c` | 为什么你简历上的大模型SFT经历很水？ |
| `xhs_67ad8f07` | 美团大模型算法面经 |

## Generated Files

- `notes/llm-interview-master.md`
- `data/processed/detailed_answers.md`
- `data/processed/question_bank.md`
- `data/processed/sources.csv`
- `data/processed/run_summary_2026-05-24.json`

## Privacy Check

Raw extraction files are ignored under `data/raw/`. Tracked outputs use source IDs and public note URLs instead of committing personal profile URLs, local absolute paths, full Xiaohongshu note dumps, or temporary tokenized links.
