# final essay — 论文目录

本文件夹包含数学建模竞赛（HDMCM 2026 — B题）的 LaTeX 论文完整源码，基于国赛中文模板构建。

## 目录结构

```
final essay/
├── main.tex                          # 论文主文件（入口）
├── reference.bib                     # BibLaTeX 参考文献库
├── mcode.sty                         # MATLAB 代码高亮样式文件
│
├── config/                           # 模板配置
│   ├── packages.tex                  # 宏包加载 + 数学命令 + 定理环境 + 标题格式
│   ├── format.tex                    # 页面格式：页眉页脚、标题页、页边距
│   ├── bibSetting.tex               # 参考文献配置（biblatex + gb7714-2015）
│   └── bibPrint.tex                 # 参考文献输出控制
│
├── content/                          # 论文章节（通过 \input 引入）
│   ├── problem-statement.tex         # 问题重述
│   ├── problem-analysis.tex          # 问题分析（含流程分析）
│   ├── assumptions-and-symbols.tex   # 模型假设 + 符号约定（三线表）
│   ├── model-construction.tex        # 模型建立（三大模型）
│   ├── problem-solving.tex           # 模型应用与问题求解
│   ├── model-evaluation.tex          # 模型评价（优点/创新 + 反思/改进）
│   ├── appendix.tex                  # 附录：AI工具使用详情
│   ├── appendix-code.tex             # 附录：Python 代码清单（lstinputlisting）
│   └── template-guide.tex            # 模板使用说明
│
└── figure/                           # 论文插图（8张，由 ../visualize.py 生成）
    ├── 图1_问题一_成本构成饼图.png
    ├── 图2_问题一_各城市运输成本柱状图.png
    ├── 图3_问题二_优化前后成本对比.png
    ├── 图4_问题三_熵权法权重饼图.png
    ├── 图5_问题三_仓库综合得分排名.png
    ├── 图6_问题三_三层优化瀑布图.png
    ├── 图7_固定成本构成饼图.png
    └── 图8_可变成本构成饼图.png
```

## 论文结构

| 章节 | 内容 |
|------|------|
| 摘要 | 分问题概述方法、模型和关键结果 |
| 问题重述 | 重新组织题目背景和要求 |
| 问题分析 | 逐一分析三问题 + 流程分析 |
| 模型假设与符号 | 10条假设 + 符号三线表 |
| 模型建立 | 成本核算 / CW拼车优化 / 熵权法评估 |
| 模型应用与求解 | 三个问题的数值求解、表格、图表和敏感性分析 |
| 模型评价 | 6项优点/创新 + 6项反思/改进 |
| 参考文献 | 12篇，手动 thebibliography + 方括号引用 |
| 附录 | AI工具使用详情 + Python 代码清单 |

## 模板规范

- **文档类**：`ctexart` 12pt，fandol 字库
- **编译器**：`xelatex`
- **纸张**：A4，四边 2.5cm 边距
- **字体**：标题三号黑体、一级标题四号黑体、正文小四号宋体
- **表格**：booktabs 三线表，caption 在上方
- **公式**：`equation` / `align` / `equation`+`aligned` 环境
- **交叉引用**：`\cref{}` 自动补上对象名称（图/表/式），禁止手工编号
- **参考文献**：`thebibliography` 环境手动列出，正文用 `\cite{}`，方括号标示 [1]

## 编译命令

```bash
cd "final essay"
xelatex main.tex && xelatex main.tex

# 清理辅助文件
rm -f main.{aux,log,out,toc,synctex.gz,fls,fdb_latexmk}
```

编译两次即可更新所有交叉引用编号。
