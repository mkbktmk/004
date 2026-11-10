"""跑测试集并输出指标与逐题明细。

区分两个口径：hit@3/recall@5 只看前 5 条，而引用补全会把条文追加到第 6 条之后，
所以另算「上下文 gold 覆盖率」——看送进模型的全部条文是否覆盖标注答案。

用法：
    py -3.11 eval/run_extra.py                      # 跑 eval/testset_extra.csv
    py -3.11 eval/run_extra.py --testset eval/testset.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cite_expand import retrieve_auto  # noqa: E402
from src.evaluate import hit_at_k, match_law, parse_gold, recall_at_k  # noqa: E402

RESULT_DIR = Path(__file__).resolve().parent / "results"


def coverage(items: list, gold: list) -> float:
    """gold 条文中进入上下文的比例（不限前 5 条）。"""
    if not gold:
        return 0.0
    hit = sum(
        1
        for law, article in gold
        if any(i["article_no"] == article and match_law(i["law"], law) for i in items)
    )
    return hit / len(gold)


def main() -> None:
    """读取测试集，逐题检索并汇总指标。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=str(Path(__file__).parent / "testset_extra.csv"))
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.testset, encoding="utf-8-sig")))
    hits, recalls, covers, detail = [], [], [], []
    for row in rows:
        if str(row.get("answerable", "")).strip() != "1":
            continue
        question = row["question"].strip()
        gold = parse_gold(row["gold_articles"])
        items = retrieve_auto(question, 5)
        hit, rec, cov = hit_at_k(items, gold, 3), recall_at_k(items, gold, 5), coverage(items, gold)
        hits.append(hit)
        recalls.append(rec)
        covers.append(cov)
        detail.append(
            [
                row["qid"],
                row["type"],
                question,
                len(items),
                hit,
                round(rec, 3),
                round(cov, 3),
                " | ".join(f"{i['law']}{i['article_no']}" for i in items),
            ]
        )
        print(f"{row['qid']:>3} {'✓' if hit else '✗'} rec={rec:.2f} cov={cov:.2f}  {question[:28]}")

    if not hits:
        print("没有可答题（answerable=1）")
        return
    print(
        f"\n{Path(args.testset).name}：hit@3={sum(hits) / len(hits):.3f}  "
        f"recall@5={sum(recalls) / len(recalls):.3f}  "
        f"上下文覆盖率={sum(covers) / len(covers):.3f}  题数={len(hits)}"
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / (Path(args.testset).stem + "_detail.csv")
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "type", "question", "上下文条数", "hit@3", "recall@5", "gold覆盖率", "命中条文"])
        writer.writerows(detail)
    print("明细已写入", out)


if __name__ == "__main__":
    main()
