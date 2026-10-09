# -*- coding: utf-8 -*-
"""
06_robustness.py —— 第六步：稳健性检验
========================================
三项稳健性检验（对应论文"分析"部分）：

  R1 词典规模敏感性：词典 TOP_K ∈ {100, 300, 500}，重新构建词典法测度
     并检验分类 AUC，观察测度对词典规模的依赖程度；
  R2 随机子样本稳健性：从测试集随机抽取 50% 样本 20 次，检验 ML 测度
     分类准确率的波动（均值 ± 标准差）；
  R3 极性翻转一致性：将测试集中被词典法判为正/负的文档互换标签后，
     词典法测度应随之翻转符号，检验测度方向的一致性（相关系数 ≈ -1）。

输出：
    output/robustness_results.csv （R1/R2/R3 结果汇总）
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

from chi2_utils import chi2_scores  # 复用第三步的卡方实现

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
RANDOM_SEED = 42


def build_lexicon(train: pd.DataFrame, top_k: int) -> tuple:
    """在训练集上重新构建词典（规模 = top_k），返回 (正向dict, 负向dict)。"""
    vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=3, max_df=0.5)
    X = vec.fit_transform(train["seg"].astype(str))
    y = train["label"].values
    vocab = np.array(vec.get_feature_names_out())
    chi2, p_pos = chi2_scores(X, y)
    pos, neg = {}, {}
    for polarity, mask, d in (("pos", p_pos > 0.5, pos), ("neg", p_pos <= 0.5, neg)):
        idx = np.where(mask)[0]
        order = idx[np.argsort(-chi2[idx])][:top_k]
        mx = chi2[order].max() if len(order) else 1.0
        for i in order:
            d[vocab[i]] = float(chi2[i] / mx)
    return pos, neg


def bf_values(seg_texts, pos, neg) -> np.ndarray:
    vals = []
    for text in seg_texts.astype(str):
        p = sum(pos.get(t, 0.0) for t in text.split())
        n = sum(neg.get(t, 0.0) for t in text.split())
        vals.append((p - n) / (p + n + 1.0))
    return np.array(vals)


def main() -> None:
    train = pd.read_csv(DATA / "train_seg.csv", encoding="utf-8-sig")
    test = pd.read_csv(DATA / "measure_test.csv", encoding="utf-8-sig")
    seg_test = pd.read_csv(DATA / "test_seg.csv", encoding="utf-8-sig")
    y = test["label"].values
    rows = []

    # ---------- R1 词典规模敏感性 ----------
    for k in (100, 300, 500):
        pos, neg = build_lexicon(train, k)
        bf = bf_values(seg_test["seg"], pos, neg)
        auc = roc_auc_score(y, bf)
        rows.append({"test": f"R1_词典规模{k}", "metric": "AUC",
                     "value": auc, "note": f"词典正负各{k}词"})
        print(f"[R1] TOP_K={k}: 测试集 AUC = {auc:.4f}")

    # ---------- R2 随机子样本稳健性 ----------
    rng = np.random.default_rng(RANDOM_SEED)
    accs = []
    for b in range(20):
        idx = rng.choice(len(y), size=len(y) // 2, replace=False)
        pred = (test["ML"].values[idx] > 0.5).astype(int)
        accs.append((pred == y[idx]).mean())
    rows.append({"test": "R2_随机子样本(×20)", "metric": "准确率均值",
                 "value": float(np.mean(accs)),
                 "note": f"标准差 {np.std(accs):.4f}"})
    print(f"[R2] 子样本准确率 = {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # ---------- R3 极性翻转一致性 ----------
    pos, neg = build_lexicon(train, 300)
    bf = bf_values(seg_test["seg"], pos, neg)
    flipped = 1 - y                                  # 人为翻转标签
    # 测度与"真实标签"的相关应显著为正，与"翻转标签"的相关应显著为负
    corr_true = np.corrcoef(bf, y)[0, 1]
    corr_flip = np.corrcoef(bf, flipped)[0, 1]
    rows.append({"test": "R3_极性一致性", "metric": "与真实标签相关",
                 "value": corr_true, "note": f"与翻转标签相关 {corr_flip:.4f}"})
    print(f"[R3] 与真实标签相关 = {corr_true:.4f}，"
          f"与翻转标签相关 = {corr_flip:.4f}")

    pd.DataFrame(rows).to_csv(OUT / "robustness_results.csv",
                              index=False, encoding="utf-8-sig")
    print("[OK] 稳健性结果已写入 output/robustness_results.csv")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
