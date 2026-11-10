"""离线测试集评测接口：让管理后台直接看到检索质量。

跑的是 eval/ 下的三套用例（原集、补充集、多轮），指标口径与命令行脚本一致：
hit@3 看排序，上下文覆盖率看"该给的条文有没有给"，多轮看路由是否正确。
结果缓存 2 分钟，避免管理员反复刷新时重复跑。
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Depends

from src.cite_expand import retrieve_auto
from src.evaluate import hit_at_k, match_law, parse_gold, recall_at_k
from web.auth import require_admin
from web.routing import build_search_query, is_meta_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
CHUNKS = Path(__file__).resolve().parent.parent / "data" / "chunks.jsonl"
CACHE_TTL = 120
_cache: Dict = {"at": 0.0, "version": 0.0, "data": None}


def _index_version() -> float:
    """知识库切块文件的修改时间：语料一变，缓存的指标就作废。

    只按时间过期不够——刚上传完法规时，看板还是旧数字，会让人误以为
    "补了条文却没效果"。
    """
    return CHUNKS.stat().st_mtime if CHUNKS.exists() else 0.0


def coverage(items: List[Dict], gold: List) -> float:
    """gold 条文中进入上下文的比例（含引用补全追加的条文）。"""
    if not gold:
        return 0.0
    found = sum(
        1
        for law, article in gold
        if any(i["article_no"] == article and match_law(i["law"], law) for i in items)
    )
    return found / len(gold)


def _wilson(hits: int, total: int, z: float = 1.96) -> List[float]:
    """Wilson 置信区间：样本少时比正态近似更稳，用来提示指标的可信范围。"""
    p = hits / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return [max(0.0, center - margin), min(1.0, center + margin)]


def run_single(filename: str, title: str) -> Dict:
    """跑单轮测试集，返回三项指标（只统计可答题）。"""
    path = EVAL_DIR / filename
    if not path.exists():
        return {"title": title, "total": 0, "hit3": 0.0, "recall5": 0.0, "coverage": 0.0}
    hits, recalls, covers = [], [], []
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        if str(row.get("answerable", "")).strip() != "1":
            continue
        gold = parse_gold(row["gold_articles"])
        items = retrieve_auto(row["question"].strip(), 5)
        hits.append(hit_at_k(items, gold, 3))
        recalls.append(recall_at_k(items, gold, 5))
        covers.append(coverage(items, gold))
    total = len(hits) or 1
    return {
        "title": title,
        "total": len(hits),
        "hit3": round(sum(hits) / total, 3),
        "recall5": round(sum(recalls) / total, 3),
        "coverage": round(sum(covers) / total, 3),
    }


def run_multi(filename: str = "testset_multi.csv") -> Dict:
    """跑多轮用例：判定标准是"路由模式正确，且 gold 进前 3 或全部进上下文"。"""
    path = EVAL_DIR / filename
    if not path.exists():
        return {"total": 0, "passed": 0, "cases": []}
    passed, cases = 0, []
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        history = [
            {"role": "user", "content": part.strip()}
            for part in row["history"].split("||")
            if part.strip()
        ]
        question = row["question"].strip()
        meta = is_meta_question(question)
        items = [] if meta else retrieve_auto(build_search_query(question, history), 5)
        gold = parse_gold(row["gold_articles"])
        hit = hit_at_k(items, gold, 3) if (gold and items) else None
        cov = coverage(items, gold) if gold else None
        ok = (meta == (row["expect_mode"].strip() == "meta")) and (
            hit is None or hit or (cov or 0) == 1.0
        )
        passed += 1 if ok else 0
        cases.append({"qid": row["qid"], "intent": row["intent"], "ok": bool(ok)})
    return {"total": len(cases), "passed": passed, "cases": cases}


@router.get("/eval")
def eval_report(refresh: bool = False, user: Dict = Depends(require_admin)) -> Dict:
    """返回三套测试集的指标；refresh=true 强制重跑。"""
    now = time.time()
    version = _index_version()
    cached = _cache["data"] is not None
    fresh = now - _cache["at"] < CACHE_TTL and _cache["version"] == version
    if not refresh and cached and fresh:
        return _cache["data"]

    start = time.time()
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "origin": run_single("testset.csv", "原集（单轮）"),
        "extra": run_single("testset_extra.csv", "补充集（单轮）"),
        "v2": run_single("testset_v2.csv", "扩充集（单轮）"),
        "multi": run_multi(),
        "elapsed": 0.0,
    }
    data["elapsed"] = round(time.time() - start, 1)
    # 合计口径：三套单轮合并统计，附 95% 置信区间（Wilson），样本少于 30 提一嘴
    totals = [s for s in (data["origin"], data["extra"], data["v2"]) if s["total"]]
    n = sum(s["total"] for s in totals)
    if n:
        hit = sum(s["hit3"] * s["total"] for s in totals) / n
        lo, hi = _wilson(round(hit * n), n)
        data["overall"] = {
            "total": n,
            "hit3": round(hit, 3),
            "recall5": round(sum(s["recall5"] * s["total"] for s in totals) / n, 3),
            "coverage": round(sum(s["coverage"] * s["total"] for s in totals) / n, 3),
            "ci": [round(lo, 2), round(hi, 2)],
        }
    _cache.update({"at": now, "version": version, "data": data})
    logger.info("离线评测完成：耗时 %.1fs", data["elapsed"])
    return data
