# -*- coding: utf-8 -*-
"""
02_text_preprocessing.py —— 第二步：文本预处理与中文分词
=========================================================
处理步骤（本脚本完成）：
    (1) 文本清洗：
        - 去除转发前缀（"//@用户名:" 及其之前内容，转发链不含原创情绪）；
        - 去除 @用户名、URL、话题标签符 #；
        - 统一全角/半角、去除多余空白；
        - 保留表情文本 [xx]（如 [爱你][怒]），其本身携带情绪信息，
          统一映射为特殊标记 EMOJI_POS / EMOJI_NEG。
    (2) 中文分词：jieba 精确模式；
    (3) 去停用词：使用哈工大停用词表（data/stopwords_hit.txt）；
    (4) 输出分词结果：data/train_seg.csv、data/test_seg.csv
        （新增列 seg，为空格分隔的词序列）。
"""
import io
import re
import sys
from pathlib import Path

import jieba
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 高频正/负表情白名单（微博表情体系中最具情绪区分度的一批，人工整理）
EMOJI_POS = {"爱你", "赞", "哈哈", "开心", "笑cry", "心", "鼓掌", "花心",
             "可爱", "偷笑", "太开心", "good", "给力", "憧憬", "喜悦"}
EMOJI_NEG = {"怒", "怒骂", "伤心", "泪", "悲伤", "衰", "汗", "失望", "可怜",
             "委屈", "大哭", "抓狂", "黑人问号", "吐", "费解", "嘘"}


def load_stopwords() -> set:
    """读取哈工大停用词表；文件缺失时返回空集合（不影响流程贯通）。"""
    path = DATA / "stopwords_hit.txt"
    if not path.exists():
        return set()
    words = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        w = line.strip()
        if w:
            words.add(w)
    return words


def clean_text(text: str) -> str:
    """(1) 文本清洗：剔除转发链、@、URL 等，规范表情标记。"""
    text = str(text)
    text = re.sub(r"^.*?//@[\w\-]+:", "", text)            # 去转发前缀链
    text = re.sub(r"//@[\w\-]+:?", " ", text)              # 去残余转发标记
    text = re.sub(r"@\S+", " ", text)                      # 去 @用户名
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)     # 去 URL
    text = re.sub(r"\[([\u4e00-\u9fa5a-zA-Z]+)\]",
                  lambda m: (" EMOJI_POS " if m.group(1) in EMOJI_POS
                             else " EMOJI_NEG " if m.group(1) in EMOJI_NEG
                             else " "), text)              # 表情 → 情绪标记
    text = re.sub(r"[#【】《》「」]", " ", text)            # 去话题符号
    text = re.sub(r"\s+", " ", text).strip()               # 规范空白
    return text


STOPWORDS = load_stopwords()


def segment(text: str) -> str:
    """(2)(3) 分词 + 去停用词，返回空格分隔的词序列。"""
    tokens = jieba.lcut(text)
    keep = [t for t in tokens
            if t not in STOPWORDS and len(t) > 0 and t.strip() != ""]
    return " ".join(keep)


def process_file(in_name: str, out_name: str) -> None:
    df = pd.read_csv(DATA / in_name, encoding="utf-8-sig")
    df["clean"] = df["review"].map(clean_text)      # 清洗列（供词典法直接使用）
    df["seg"] = df["clean"].map(segment)            # 分词列（供模型与词典使用）
    df = df[df["seg"].str.len() > 0]                # 剔除分词后为空的样本
    df.to_csv(DATA / out_name, index=False, encoding="utf-8-sig")
    print(f"[OK] {in_name} -> {out_name}：{len(df)} 条")


def main() -> None:
    jieba.setLogLevel(60)                           # 关闭 jieba 日志
    process_file("train.csv", "train_seg.csv")
    process_file("test.csv", "test_seg.csv")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
