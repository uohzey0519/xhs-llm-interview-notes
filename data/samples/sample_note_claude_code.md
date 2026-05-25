# 样本笔记：Claude Code 检索代码方式

- Source title: 面试被问到claude code的检索是咋做的？
- Author: 算法狗
- URL: https://www.xiaohongshu.com/explore/69eb90ca00000000360003c5
- Extraction status: text-ok
- Checked: 2026-05-24

## Extracted Question

面试中如果被问到：Claude Code 检索代码用的是什么方式？它是不是典型 RAG？

## Source-Grounded Summary

这条笔记的核心判断是：Claude Code 早期并不主要依赖传统 RAG，而是更接近使用 grep 类命令进行代码检索。原因是代码库经常变化，传统向量索引需要同步更新和重新 embedding；而代码检索更看重函数名、类名、精确字符串和上下文匹配，grep 类工具在这些场景下更直接。

## Interview-Ready Answer

Claude Code 的代码检索不能简单等同于传统 RAG。传统 RAG 通常会先把文档切块、embedding、建向量索引，再用语义相似度召回；这对静态知识库很有用。但代码仓库变化频繁，且代码问题经常依赖精确符号、文件路径、函数名、类名、调用关系和局部上下文。对这类任务，grep/ripgrep、文件树遍历、语言服务、AST 或符号索引往往比纯语义向量召回更可靠。

一个好的回答可以分层说：

1. 第一层是关键词/符号级检索，例如 grep、ripgrep、文件名搜索。
2. 第二层是结构化代码理解，例如 AST、LSP、调用图、定义跳转。
3. 第三层才可能叠加语义检索，用来召回注释、文档或不精确描述相关的代码片段。

所以面试里可以回答：Claude Code 这类代码 Agent 更像是结合工具调用的代码库探索，而不是单一 RAG。RAG 可以是其中一环，但不应把它理解成“向量库检索 + 生成”的固定范式。

## Tags

- RAG
- Agent
- Claude Code
- 代码检索
- 面试
