# 基于情感词典与机器学习的中文社交媒体情绪测度构建及其有效性检验

> NLP 测度构建方法课程 · 预演毕业论文模拟稿（含全部可复现代码与结果表）

## 一、项目简介

本研究以公开真实语料 **weibo_senti_100k**（10 万条带人工情感标签的中文微博）为基础，
完整演示"**数据获取 → 文本预处理 → 词典构建 → 测度构造 → 有效性检验 → 稳健性检验**"
六步 NLP 测度构建流水线，构造两类文本情绪测度：

| 测度 | 方法 | 取值范围 |
|---|---|---|
| **BF**（词典法净情绪指数） | 数据驱动卡方词典 + 加权词频差 | (-1, 1) |
| **ML**（机器学习情绪概率） | TF-IDF + L2 Logistic 回归 | [0, 1] |

并用 OLS 回归与样本外分类指标（准确率 / F1 / AUC）系统检验两类测度的有效性，
辅以词典规模敏感性、随机子样本、极性翻转一致性三项稳健性检验。

## 二、仓库结构

```
nlp-measure-thesis/
├── thesis/
│   ├── 论文全文.md            # 论文 Markdown 版全文
│   └── 论文全文.docx          # 论文 Word 版（含封面、目录）
├── code/                      # 可复现代码（运行顺序 01 → 06）
│   ├── 01_data_collection.py       # 数据获取与样本构建
│   ├── 02_text_preprocessing.py    # 文本清洗与中文分词
│   ├── 03_lexicon_construction.py  # 卡方法情感词典构建
│   ├── 04_measure_construction.py  # BF / ML 两类测度构造
│   ├── 05_model_validation.py      # 回归 + 分类有效性检验
│   ├── 06_robustness.py            # 三项稳健性检验
│   └── chi2_utils.py               # 公共卡方模块
├── data/                      # 原始与中间数据（>25MB 的原始语料不入库，见"数据来源说明"）
├── output/                    # 调度结果表（CSV）
│   ├── sentiment_lexicon.csv       # 情感词典（词、极性、卡方值、权重）
│   ├── regression_results.csv      # 三个 OLS 回归模型系数表
│   ├── classification_results.csv  # 分类有效性指标对比
│   └── robustness_results.csv      # 稳健性检验结果
└── 数据来源说明.md            # 每个数据/结果文件的来源与处理步骤
```

## 三、复现步骤

```bash
# 0. 环境：Python 3.10+，安装依赖
pip install pandas scikit-learn jieba

# 1. 依次运行（在仓库根目录执行）
python code/01_data_collection.py     # 自动下载原始语料并抽样 20000 条
python code/02_text_preprocessing.py  # 清洗 + jieba 分词 + 去停用词
python code/03_lexicon_construction.py # 构建正/负各 300 词的情感词典
python code/04_measure_construction.py # 构造 BF 与 ML 两类测度
python code/05_model_validation.py     # 回归与分类有效性检验
python code/06_robustness.py           # 稳健性检验

# 全部随机过程固定 seed=42，一次运行约 5–10 分钟，可完整复现论文全部数字。
```

## 四、核心结果速览

详见 `output/` 下四张结果表；论文正文第三节有完整解读。

## 五、参考引用

若使用本仓库代码或词典，请引用原始数据集与停用词表来源：

- weibo_senti_100k 数据集（10 万条标注微博）
- 哈工大停用词表（Hit Stopwords）
