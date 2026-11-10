"""多轮场景测试：验证追问改写、主题回溯、换主题不拼接、元问题分流。

单轮测试集（testset.csv / testset_extra.csv）测不到这些逻辑，因为它们只调
retrieve_auto(question)，而多轮的关键在于"用改写后的查询去检索"。

用法：py -3.11 eval/run_multi.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cite_expand import retrieve_auto  # noqa: E402
from src.evaluate import hit_at_k, match_law, parse_gold  # noqa: E402
from web.routing import build_search_query, is_meta_question  # noqa: E402

CASES = Path(__file__).resolve().parent / "testset_multi.csv"


def coverage(items: list, gold: list) -> float:
    """gold 条文中进入上下文的比例（含引用补全追加的条文）。"""
    if not gold:
        return 0.0
    hit = sum(
        1
        for law, article in gold
        if any(i["article_no"] == article and match_law(i["law"], law) for i in items)
    )
    return hit / len(gold)


def main() -> None:
    """逐题跑多轮场景并汇总。"""
    rows = list(csv.DictReader(open(CASES, encoding="utf-8-sig")))
    passed = 0
    for row in rows:
        history = [
            {"role": "user", "content": part.strip()}
            for part in row["history"].split("||")
            if part.strip()
        ]
        question = row["question"].strip()
        expect = row["expect_mode"].strip()
        meta = is_meta_question(question)
        mode = "meta" if meta else "law"
        search_query = "" if meta else build_search_query(question, history)
        joined = bool(search_query) and search_query != question
        items = [] if meta else retrieve_auto(search_query, 5)

        gold = parse_gold(row["gold_articles"])
        hit = hit_at_k(items, gold, 3) if (gold and items) else None
        cov = coverage(items, gold) if gold else None
        ok = mode == expect and (hit is None or hit or (cov or 0) == 1.0)
        passed += 1 if ok else 0

        print(f"\n[{row['qid']}] {row['intent']}")
        print(f"  问题      : {question}")
        print(f"  模式      : {mode}（期望 {expect}）{'✓' if mode == expect else '✗'}")
        if not meta:
            print(f"  是否拼接  : {'是' if joined else '否'}")
            print(f"  检索词    : {search_query[:52]}")
            print("  命中      : " + (" | ".join(
                f"{i['law']}{i['article_no']}" + ("[补]" if i.get("via") else "")
                for i in items[:4])))
        if gold:
            print(f"  hit@3     : {'✓' if hit else '✗'} | 上下文覆盖 {cov:.2f}"
                  f"（期望 {'、'.join(a for _, a in gold)}）")

    print(f"\n通过 {passed}/{len(rows)}")
    if passed < len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
