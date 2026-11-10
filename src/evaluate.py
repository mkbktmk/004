"""离线评测：4 组检索配置对比。

用法：python -m src.evaluate [--limit N] [--only A,C]
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.citation import format_warning, verify
from src.config import RESULTS_DIR, TESTSET_PATH, TOPK_FINAL, configure_logging
from src.generator import answer
from src.retriever import get_retriever

logger = logging.getLogger(__name__)

CN_DIGITS = "零一二三四五六七八九"
GOLD_RE = re.compile(r"([^;；,，\s]+?)第(\d{1,3})条")
# 方案名 -> (检索模式, 是否启用引用校验)
CONFIGS: Dict[str, Tuple[str, bool]] = {
    "A": ("vector", False),
    "B": ("bm25", False),
    "C": ("hybrid", False),
    "D": ("hybrid", True),
}


def num_to_cn(number: int) -> str:
    """把阿拉伯数字转成中文数字（条号须保留中文原文）。"""
    if number < 10:
        return CN_DIGITS[number]
    if number < 20:
        return "十" + (CN_DIGITS[number % 10] if number % 10 else "")
    if number < 100:
        return CN_DIGITS[number // 10] + "十" + (CN_DIGITS[number % 10] if number % 10 else "")
    hundreds, rest = divmod(number, 100)
    head = CN_DIGITS[hundreds] + "百"
    if rest == 0:
        return head
    if rest < 10:
        return head + "零" + CN_DIGITS[rest]
    return head + num_to_cn(rest)


def parse_gold(raw: str) -> List[Tuple[str, str]]:
    """解析 gold_articles，如"劳动合同法第19条;劳动法第44条"。"""
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [(law.strip(), f"第{num_to_cn(int(num))}条") for law, num in GOLD_RE.findall(raw)]


def match_law(item_law: str, gold_law: str) -> bool:
    """法规简称宽松匹配（允许一方是另一方的子串）。"""
    return gold_law in item_law or item_law in gold_law


def hit_at_k(items: Sequence[Dict], gold: List[Tuple[str, str]], k: int) -> int:
    """前 k 条是否命中任一 gold 条号。"""
    for item in items[:k]:
        if any(item["article_no"] == article and match_law(item["law"], law) for law, article in gold):
            return 1
    return 0


def recall_at_k(items: Sequence[Dict], gold: List[Tuple[str, str]], k: int) -> float:
    """前 k 条对 gold 条号的覆盖率。"""
    if not gold:
        return 0.0
    found = {
        (law, article)
        for law, article in gold
        if any(item["article_no"] == article and match_law(item["law"], law) for item in items[:k])
    }
    return len(found) / len(gold)


def load_testset(limit: Optional[int] = None) -> pd.DataFrame:
    """读取评测集并跳过 gold_articles 为空的行。"""
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(f"评测集不存在：{TESTSET_PATH}")
    frame = pd.read_csv(TESTSET_PATH, dtype=str).fillna("")
    frame = frame[(frame["gold_articles"].str.strip() != "") | (frame["answerable"].str.strip() == "0")]
    frame = frame.reset_index(drop=True)
    if limit:
        frame = frame.head(limit)
    logger.info("载入评测集 %d 题（已跳过空 gold_articles 行）", len(frame))
    return frame


def run_one(row: pd.Series, mode: str, use_check: bool, topk: int) -> Dict:
    """跑单题：检索 + 生成 + 引用校验。"""
    gold = parse_gold(row["gold_articles"])
    start = time.time()
    items = get_retriever().retrieve(row["question"], topk_final=topk, mode=mode)
    result = answer(row["question"], items)
    check = verify(result["answer"], items)
    elapsed = time.time() - start
    answerable = str(row["answerable"]).strip() == "1"
    return {
        "qid": row["qid"],
        "question": row["question"],
        "type": row["type"],
        "answerable": int(answerable),
        "hit@3": hit_at_k(items, gold, 3),
        "recall@5": round(recall_at_k(items, gold, 5), 4),
        "citation_accuracy": check["coverage"],
        "refusal_correct": (1 if "无法确定" in result["answer"] else 0) if not answerable else "",
        "invalid_citations": "、".join(check["invalid"]),
        "warning": format_warning(check) if use_check else "",
        "answer": result["answer"] or f"[生成失败] {result.get('error', '')}",
        "retrieved": " | ".join(f"{i['law']}{i['article_no']}({i['score']})" for i in items),
        "elapsed": round(elapsed, 2),
    }


def run_config(key: str, frame: pd.DataFrame, topk: int) -> Tuple[pd.DataFrame, Dict]:
    """跑完一组配置并落盘明细，返回 (明细表, 指标)。"""
    mode, use_check = CONFIGS[key]
    name = f"{key}_{mode}" + ("_verify" if use_check else "")
    rows = []
    for _, row in frame.iterrows():
        try:
            rows.append(run_one(row, mode, use_check, topk))
        except Exception as exc:  # noqa: BLE001 - 单题失败不影响整组
            logger.error("qid=%s 评测失败：%s", row["qid"], exc)
    detail = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(RESULTS_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    answerable = detail[detail["answerable"] == 1]
    refused = detail[detail["answerable"] == 0]
    metrics = {
        "config": name,
        "n": len(detail),
        "hit@3": round(answerable["hit@3"].mean(), 4) if len(answerable) else float("nan"),
        "recall@5": round(answerable["recall@5"].mean(), 4) if len(answerable) else float("nan"),
        "citation_accuracy": round(answerable["citation_accuracy"].mean(), 4)
        if len(answerable)
        else float("nan"),
        "refusal_correct": round(refused["refusal_correct"].astype(float).mean(), 4)
        if len(refused)
        else float("nan"),
    }
    logger.info("方案 %s 完成：%s", name, metrics)
    return detail, metrics


def _fmt(value: float) -> str:
    """数字格式化，缺失值显示为 "-"。"""
    return "-" if value != value else f"{value:.3f}"


def print_table(metrics: List[Dict]) -> None:
    """打印对齐的汇总表。"""
    header = f"{'方案':<20}{'题数':>6}{'hit@3':>9}{'recall@5':>10}{'引文准确':>10}{'拒答正确':>10}"
    print(header)
    print("-" * len(header))
    for item in metrics:
        print(
            f"{item['config']:<20}{item['n']:>6}{_fmt(item['hit@3']):>9}"
            f"{_fmt(item['recall@5']):>10}{_fmt(item['citation_accuracy']):>10}"
            f"{_fmt(item['refusal_correct']):>10}"
        )


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="劳动法 RAG 离线评测")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题，便于小样本调试")
    parser.add_argument("--only", type=str, default="", help="只跑指定方案，如 A,C")
    parser.add_argument("--topk", type=int, default=TOPK_FINAL, help="送入模型的条文数")
    args = parser.parse_args()
    configure_logging()
    selected = [key.strip().upper() for key in args.only.split(",") if key.strip()] or list(CONFIGS)
    unknown = [key for key in selected if key not in CONFIGS]
    if unknown:
        raise SystemExit(f"未知方案：{unknown}，可选 {list(CONFIGS)}")
    frame = load_testset(args.limit)
    metrics: List[Dict] = []
    for key in selected:
        _, item = run_config(key, frame, args.topk)
        metrics.append(item)
    summary = pd.DataFrame(metrics)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    print_table(metrics)
    print(f"\n明细与汇总已写入：{RESULTS_DIR}")


if __name__ == "__main__":
    main()
