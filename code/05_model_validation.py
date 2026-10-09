# -*- coding: utf-8 -*-
"""
05_model_validation.py —— 第五步：测度有效性检验（回归 + 分类）
================================================================
检验设计（对应论文"分析"部分）：

  A. 回归分析（测度的"经济含义"检验）：
      以人工标注情感标签 y_i 为被解释变量，分别对两种测度做 OLS 回归：
          y_i = α + β·BF_i + ε_i        （模型 1）
          y_i = α + γ·ML_i + ε_i        （模型 2）
          y_i = α + β·BF_i + γ·ML_i + ε_i（模型 3，竞争性检验）
      输出系数、标准误、t 值、p 值、R²。

  B. 分类有效性（测度的"判别能力"检验）：
      词典法：BF > 0 判为正面；机器学习法：ML > 0.5 判为正面；
      计算准确率、精确率、召回率、F1 与 AUC，对比基准（多数类）。

输出：
    output/regression_results.csv   （三个回归模型的系数表）
    output/classification_results.csv（各方法分类指标对比）
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"


def ols(y: np.ndarray, X: np.ndarray, names: list) -> list:
    """手工 OLS：b = (X'X)^-1 X'y，含 Newey-West 未校正的经典标准误。"""
    n, k = X.shape
    XTX_inv = np.linalg.pinv(X.T @ X)
    b = XTX_inv @ X.T @ y
    resid = y - X @ b
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(XTX_inv) * s2)
    t_stat = b / se
    # 正态近似 p 值（大样本下与 t 分布几乎无差异）
    from math import erf, sqrt
    p = np.array([2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))) for t in t_stat])
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return [{"variable": names[i], "coef": b[i], "std_err": se[i],
             "t_value": t_stat[i], "p_value": p[i]} for i in range(k)] + \
           [{"variable": "R2", "coef": r2, "std_err": np.nan,
             "t_value": np.nan, "p_value": np.nan}]


def main() -> None:
    tr = pd.read_csv(DATA / "measure_train.csv", encoding="utf-8-sig")
    te = pd.read_csv(DATA / "measure_test.csv", encoding="utf-8-sig")

    # ---------- A. 回归分析（训练集，避免用测试集"调"模型后再报告） ----------
    y_tr = tr["label"].values.astype(float)
    ones_tr = np.ones(len(tr))
    specs = {
        "模型1_BF":   (np.column_stack([ones_tr, tr["BF"]]),   ["常数项", "BF"]),
        "模型2_ML":   (np.column_stack([ones_tr, tr["ML"]]),   ["常数项", "ML"]),
        "模型3_竞争": (np.column_stack([ones_tr, tr["BF"], tr["ML"]]),
                       ["常数项", "BF", "ML"]),
    }
    reg_rows = []
    for name, (X, names) in specs.items():
        for row in ols(y_tr, X, names):
            reg_rows.append({"model": name, **row})
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(OUT / "regression_results.csv", index=False, encoding="utf-8-sig")
    print("[OK] 回归结果已写入 output/regression_results.csv")
    print(reg.to_string(index=False,
                        float_format=lambda v: f"{v:.4f}"))

    # ---------- B. 分类有效性（测试集，样本外） ----------
    y = te["label"].values
    pred_bf = (te["BF"] > 0).astype(int)
    pred_ml = (te["ML"] > 0.5).astype(int)
    majority = int(pd.Series(y).mode()[0])           # 基准：多数类预测

    def metrics(name, pred, score=None):
        return {"method": name, "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "auc": roc_auc_score(y, score) if score is not None else np.nan}

    rows = [
        metrics("基准_多数类", np.full(len(y), majority)),
        metrics("词典法_BF", pred_bf, te["BF"]),
        metrics("机器学习_ML", pred_ml, te["ML"]),
    ]
    clf = pd.DataFrame(rows)
    clf.to_csv(OUT / "classification_results.csv", index=False,
               encoding="utf-8-sig")
    print("\n[OK] 分类结果已写入 output/classification_results.csv")
    print(clf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
