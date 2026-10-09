# -*- coding: utf-8 -*-
"""
01_data_collection.py —— 第一步：数据获取与样本构建
=====================================================
研究设计说明：
    本研究使用公开的真实语料 weibo_senti_100k（10 万条带人工情感标签的中文
    微博短文本，label=1 为正面、label=0 为负面），作为社交媒体文本情绪测度
    构建与检验的训练与验证语料。

    数据获取方式（可复现）：
      1. 手动下载：
         https://github.com/WAng91An/sentiment_classification_pytorch
         （仓库 data/ 目录下含 weibo_senti_100k.csv 与哈工大停用词表）
      2. 或运行本脚本，脚本会自动尝试从上述仓库下载并解压提取。

处理步骤（本脚本完成）：
    (1) 读取原始 CSV（UTF-8 编码，含 BOM 需处理）；
    (2) 去除空值与重复文本，保证样本独立；
    (3) 分层随机抽样 20000 条（保持正负类比例约 1:1），控制计算成本；
    (4) 按 70%/30% 划分训练集（用于词典构建与模型训练）与测试集
        （用于样本外有效性检验）；
    (5) 输出 data/sample_data.csv、data/train.csv、data/test.csv。
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

# ---- 路径设定：仓库根目录 = 本脚本所在目录的上一级 ----
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA / "weibo_senti_100k.csv"
ZIP_URL = "https://codeload.github.com/WAng91An/sentiment_classification_pytorch/zip/refs/heads/master"

SAMPLE_N = 20000      # 抽样规模（兼顾代表性与计算成本）
TEST_RATIO = 0.30     # 测试集比例
RANDOM_SEED = 42      # 固定随机种子，保证结果可复现


def ensure_raw_data() -> Path:
    """若本地无原始数据，则从 GitHub 下载仓库 zip 并提取所需文件。"""
    if RAW_CSV.exists():
        print(f"[OK] 已存在原始数据：{RAW_CSV}")
        return RAW_CSV

    import urllib.request
    print("[..] 本地无原始数据，正在从 GitHub 下载 ...")
    zip_path = DATA / "_raw_repo.zip"
    urllib.request.urlretrieve(ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        # 在 zip 中定位 weibo_senti_100k.csv 与停用词表
        for name in zf.namelist():
            if name.endswith("data/weibo_senti_100k.csv"):
                target = DATA / "weibo_senti_100k.csv"
                target.write_bytes(zf.read(name))
            elif name.endswith("data/hit_stopword"):
                (DATA / "stopwords_hit.txt").write_bytes(zf.read(name))
    zip_path.unlink()
    print(f"[OK] 下载并提取完成：{RAW_CSV}")
    return RAW_CSV


def main() -> None:
    raw = ensure_raw_data()

    # (1) 读取：utf-8-sig 自动去除 BOM（原始文件首字符含 \ufeff）
    df = pd.read_csv(raw, encoding="utf-8-sig")
    print(f"[OK] 原始语料规模：{len(df)} 条；列名：{list(df.columns)}")

    # (2) 清洗：去空值、去重复文本（重复文本会泄漏到测试集，造成测度高估）
    df = df.dropna(subset=["label", "review"])
    df = df.drop_duplicates(subset=["review"])
    df["label"] = df["label"].astype(int)
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"].str.len() >= 4]          # 剔除过短文本（信息量不足）
    print(f"[OK] 清洗后规模：{len(df)} 条；正类占比 {df['label'].mean():.3f}")

    # (3) 分层抽样：保持正负类比例，固定种子保证可复现
    frac = SAMPLE_N / len(df)
    parts = [g.sample(n=max(1, round(frac * len(g))),
                      random_state=RANDOM_SEED)
             for _, g in df.groupby("label")]
    sample = pd.concat(parts, axis=0)
    sample = sample.sample(frac=1.0, random_state=RANDOM_SEED)  # 打乱顺序
    sample = sample.reset_index(drop=True)
    sample.to_csv(DATA / "sample_data.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] 抽样样本：{len(sample)} 条，已写入 data/sample_data.csv")

    # (4) 训练/测试集划分（分层，保持两类比例一致）
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        sample, test_size=TEST_RATIO, stratify=sample["label"],
        random_state=RANDOM_SEED)
    train.to_csv(DATA / "train.csv", index=False, encoding="utf-8-sig")
    test.to_csv(DATA / "test.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] 训练集 {len(train)} 条 / 测试集 {len(test)} 条，"
          f"已写入 data/train.csv 与 data/test.csv")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
