# -*- coding: utf-8 -*-
"""
chi2_utils.py —— 公共模块：卡方（Chi-square）词-极性关联度量
=============================================================
被 03_lexicon_construction.py 与 06_robustness.py 共同复用，
保证两处词典构建逻辑完全一致。

    对每个候选词 w 构造 2x2 列联表（是否出现 w × 情感正负），
    卡方统计量越大，说明"该词出现"与"情感极性"关联越强，
    越适合作为情感词典词条。
"""
import numpy as np


def chi2_scores(X, y) -> tuple:
    """计算词频矩阵中每个词的卡方统计量与条件极性概率。

    参数：
        X: scipy 稀疏词频矩阵 (n_docs, n_words)
        y: 0/1 情感标签数组

    返回：
        chi2:      形状 (n_words,) 的卡方统计量
        p_pos_w:   P(正类 | 词出现)，用于判定词极性
    """
    y = np.asarray(y).astype(bool)
    Xb = (X > 0)                                  # 词是否出现（二值化）
    A = np.asarray(Xb[y].sum(axis=0)).ravel()     # 出现且为正
    B = np.asarray(Xb[~y].sum(axis=0)).ravel()    # 出现且为负
    C = y.sum() - A                               # 不出现且为正
    D = (~y).sum() - B                            # 不出现且为负
    A, B, C, D = (np.asarray(t, dtype=float) for t in (A, B, C, D))
    n = A + B + C + D
    chi2 = n * (A * D - B * C) ** 2 / (
        (A + B) * (C + D) * (A + C) * (B + D) + 1e-12)
    p_pos_w = A / (A + B + 1e-12)
    return chi2, p_pos_w
