# -*- coding: utf-8 -*-
"""
03_lexicon_construction.py —— 第三步：数据驱动情感词典构建
===========================================================
方法说明（对应论文"方法"部分）：
    本研究不直接采用现成通用词典，而是用监督语料"数据驱动"地构建领域
    情感词典，核心指标为 **卡方统计量（Chi-square）**：

        对每个词 w 计算 2x2 列联表（是否出现 w × 情感正负），
        chi2(w) 衡量词出现与情感类别之间的统计关联强度。

    词极性由该词在正类文档中的出现占比判定：
        polarity = P(正类 | w 出现)，> 0.5 记为正向词，否则为负向词。
    词权重取归一化卡方值 chi2(w)（区分能力越强的词权重越高）。

    词典规模：正负各取卡方值最高的 TOP_K=300 个词（稳健性检验中
    会对比 100/300/500 的设定）。

输出：
    output/sentiment_lexicon.csv  （word, polarity, chi2, weight）
    data/lexicon_stats.csv        （词典规模与重合度统计）
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from chi2_utils import chi2_scores  # 复用公共卡方实现（与 06 稳健性检验一致）

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

TOP_K = 300          # 正/负词典各取前 K 个词
RANDOM_SEED = 42


def main() -> None:
    train = pd.read_csv(DATA / "train_seg.csv", encoding="utf-8-sig")

    # 词频矩阵（min_df=3 过滤低频噪声词，max_df=0.5 过滤无处不在的词）
    vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=3, max_df=0.5)
    X = vec.fit_transform(train["seg"].astype(str))
    y = train["label"].values
    vocab = np.array(vec.get_feature_names_out())
    print(f"[OK] 训练集词表规模：{len(vocab)}")

    chi2, p_pos = chi2_scores(X, y)

    # 按 P(正|w) 判定极性，并在各自极性内部按卡方值排序取 TOP_K
    rows = []
    for polarity, mask in (("pos", p_pos > 0.5), ("neg", p_pos <= 0.5)):
        idx = np.where(mask)[0]
        order = idx[np.argsort(-chi2[idx])][:TOP_K]
        for i in order:
            rows.append({"word": vocab[i], "polarity": polarity,
                         "chi2": float(chi2[i]),
                         "weight": float(chi2[i] / chi2[idx].max())})
    lexicon = pd.DataFrame(rows).sort_values("chi2", ascending=False)
    lexicon.to_csv(OUT / "sentiment_lexicon.csv", index=False, encoding="utf-8-sig")

    stats = pd.DataFrame([{
        "pos_words": int((lexicon.polarity == "pos").sum()),
        "neg_words": int((lexicon.polarity == "neg").sum()),
        "total_words": len(lexicon),
        "max_chi2": float(lexicon.chi2.max()),
        "median_chi2": float(lexicon.chi2.median()),
    }])
    stats.to_csv(DATA / "lexicon_stats.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] 词典构建完成：正向 {stats.pos_words[0]} 词 / "
          f"负向 {stats.neg_words[0]} 词 -> output/sentiment_lexicon.csv")
    print("[示例-正向词]", lexicon[lexicon.polarity == 'pos']['word'].head(15).tolist())
    print("[示例-负向词]", lexicon[lexicon.polarity == 'neg']['word'].head(15).tolist())


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
