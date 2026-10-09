# -*- coding: utf-8 -*-
"""
04_measure_construction.py —— 第四步：情绪测度构造（核心变量生成）
===================================================================
构造两种文本情绪测度（对应论文"方法"部分）：

  测度一 词典法净情绪指数 BF_i（Bag-of-words Frequency）：
      pos_i = Σ_{w∈正词典∩文档i} tf(w)·weight(w)
      neg_i = Σ_{w∈负词典∩文档i} tf(w)·weight(w)
      BF_i  = (pos_i - neg_i) / (pos_i + neg_i + 1)   ∈ (-1, 1)
      取值越接近 1 情绪越正面，越接近 -1 越负面；分母加 1 避免
      未命中任何词典词时出现 0/0。

  测度二 机器学习法情绪概率 ML_i：
      TF-IDF + L2 正则 Logistic 回归（在训练集上拟合），
      输出每条文本属于"正面"的后验概率 ML_i ∈ [0, 1]。

  两者均在【测试集】上计算，供第五步样本外有效性检验。

输出：
    data/measure_dataset.csv  （label, BF, ML，及中间变量 pos/neg）
"""
import io
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RANDOM_SEED = 42
LOGREG_C = 1.0        # 正则化强度的倒数（稳健性检验中可替换）


def build_lexicon_dicts() -> tuple:
    lex = pd.read_csv(ROOT / "output" / "sentiment_lexicon.csv",
                      encoding="utf-8-sig")
    pos = dict(zip(lex[lex.polarity == "pos"]["word"],
                   lex[lex.polarity == "pos"]["weight"]))
    neg = dict(zip(lex[lex.polarity == "neg"]["word"],
                   lex[lex.polarity == "neg"]["weight"]))
    return pos, neg


def bf_measure(seg_texts: pd.Series, pos: dict, neg: dict) -> pd.DataFrame:
    """逐条计算词典法净情绪指数 BF。"""
    out = []
    for text in seg_texts.astype(str):
        pos_s = neg_s = 0.0
        for tok in text.split():
            if tok in pos:
                pos_s += pos[tok]
            elif tok in neg:
                neg_s += neg[tok]
        bf = (pos_s - neg_s) / (pos_s + neg_s + 1.0)
        out.append((pos_s, neg_s, bf))
    return pd.DataFrame(out, columns=["pos_score", "neg_score", "BF"])


def ml_model(train: pd.DataFrame):
    """拟合 TF-IDF + Logistic 回归模型，返回 (tfidf, clf)。"""
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",
                            ngram_range=(1, 2), min_df=3,
                            max_features=50000, sublinear_tf=True)
    Xtr = tfidf.fit_transform(train["seg"].astype(str))
    clf = LogisticRegression(C=LOGREG_C, max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(Xtr, train["label"].values)
    print(f"[OK] ML 模型训练完成：TF-IDF 特征 {Xtr.shape[1]} 个")
    return tfidf, clf


def main() -> None:
    train = pd.read_csv(DATA / "train_seg.csv", encoding="utf-8-sig")
    test = pd.read_csv(DATA / "test_seg.csv", encoding="utf-8-sig")

    pos, neg = build_lexicon_dicts()
    tfidf, clf = ml_model(train)

    # 两类测度在训练集与测试集上均计算（训练集供回归检验，测试集供样本外检验）
    for name, df in (("train", train), ("test", test)):
        bf = bf_measure(df["seg"], pos, neg)
        ml = clf.predict_proba(tfidf.transform(df["seg"].astype(str)))[:, 1]
        part = pd.concat([df[["label"]].reset_index(drop=True),
                          bf.reset_index(drop=True)], axis=1)
        part["ML"] = ml
        cols = ["label", "pos_score", "neg_score", "BF", "ML"]
        part[cols].to_csv(DATA / f"measure_{name}.csv",
                          index=False, encoding="utf-8-sig")
        print(f"[OK] {name} 测度已写入 data/measure_{name}.csv"
              f"（BF 均值 {part['BF'].mean():.4f}）")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
