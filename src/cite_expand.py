"""自动检索：混合召回 + 条文引用链补全。

用户不需要理解向量/BM25 的区别——固定走混合检索，再由本模块把命中条文里
"依照本法第X条"引用的条文一并补进上下文。法律场景下这种引用关系比任何
相似度都可靠：第八十七条只说"按第四十七条标准的二倍"，漏掉第四十七条就
算不出金额，而两路检索都未必能把它排进前五。

用法：retrieve_auto("公司违法解除，赔偿金怎么算？", 5)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from src.config import CHUNKS_PATH
from src.retriever import get_retriever

logger = logging.getLogger(__name__)

CN_NUM = "零一二三四五六七八九十百"
REF_RE = re.compile(rf"(?:本法|本条例)第([{CN_NUM}]+)条")
MAX_EXTRA = 2

_CHUNKS: Dict[str, Dict] | None = None


def _load_chunks() -> Dict[str, Dict]:
    """按 id 懒加载条文库（只读一次）。"""
    global _CHUNKS
    if _CHUNKS is None:
        _CHUNKS = {}
        if CHUNKS_PATH.exists():
            with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        chunk = json.loads(line)
                        _CHUNKS.setdefault(chunk["id"], chunk)
    return _CHUNKS


def parse_refs(text: str) -> List[str]:
    """提取条文中"本法/本条例第X条"的引用条号（保留中文数字原文）。

    Args:
        text: 条文正文。

    Returns:
        条号列表，如 ["第四十七条"]，按出现顺序去重。
    """
    refs: List[str] = []
    for match in REF_RE.finditer(text):
        article = f"第{match.group(1)}条"
        if article not in refs:
            refs.append(article)
    return refs


def expand_refs(items: List[Dict], limit: int = MAX_EXTRA) -> List[Dict]:
    """把命中条文引用的条文补进结果（同法规、去重、按出现顺序）。

    Args:
        items: 检索结果（需含 law / article_no / text）。
        limit: 最多补几条，避免上下文膨胀。

    Returns:
        新增条文列表，带 via 字段标注来源，score 为 None。
    """
    chunks = _load_chunks()
    seen = {(item["law"], item["article_no"]) for item in items}
    extra: List[Dict] = []
    for item in items:
        if len(extra) >= limit:
            break
        for article in parse_refs(item["text"]):
            if len(extra) >= limit:
                break
            key = (item["law"], article)
            chunk = chunks.get(f"{item['law']}-{article}")
            if key in seen or chunk is None:
                continue
            seen.add(key)
            extra.append(
                {
                    "id": chunk["id"],
                    "law": chunk["law"],
                    "article_no": chunk["article_no"],
                    "chapter": chunk.get("chapter", ""),
                    "text": chunk["text"],
                    "score": None,
                    "via": f"《{item['article_no']}》引用补全",
                }
            )
    if extra:
        logger.info(
            "引用补全 %d 条：%s", len(extra), "、".join(e["article_no"] for e in extra)
        )
    return extra


def _reverse_index() -> Dict[str, List[str]]:
    """构建反向引用索引：被引用条号 -> 引用了它的条文 id 列表。

    正向补全解决"命中的条文引用了谁"，反向解决"谁引用了命中的条文"。
    后者是 q23 的关键：第四十五条引用第四十二条，但从第四十二条出发正向找不到它。
    """
    index: Dict[str, List[str]] = {}
    for chunk in _load_chunks().values():
        for article in parse_refs(chunk["text"]):
            index.setdefault(article, []).append(chunk["id"])
    return index


def expand_reverse(items: List[Dict], limit: int = 1) -> List[Dict]:
    """补入引用了命中条文的条文（仅同法规、去重、最多 limit 条）。

    Args:
        items: 检索结果。
        limit: 最多补几条。

    Returns:
        新增条文列表，via 标注为"某条引用了它"。
    """
    chunks = _load_chunks()
    reverse = _reverse_index()
    seen = {(item["law"], item["article_no"]) for item in items}
    extra: List[Dict] = []
    for item in items:
        for chunk_id in reverse.get(item["article_no"], []):
            chunk = chunks.get(chunk_id)
            if chunk is None or chunk["law"] != item["law"]:
                continue
            key = (chunk["law"], chunk["article_no"])
            if key in seen:
                continue
            seen.add(key)
            extra.append(
                {
                    "id": chunk["id"],
                    "law": chunk["law"],
                    "article_no": chunk["article_no"],
                    "chapter": chunk.get("chapter", ""),
                    "text": chunk["text"],
                    "score": None,
                    "via": f"引用《{item['article_no']}》的条文",
                }
            )
            if len(extra) >= limit:
                return extra
    if extra:
        logger.info(
            "反向补全 %d 条：%s", len(extra), "、".join(e["article_no"] for e in extra)
        )
    return extra


def list_laws() -> List[str]:
    """返回知识库当前包含的法规名（按条文块数降序），用于能力介绍。

    Returns:
        法规简称列表，如 ["劳动合同法", "劳动法", ...]。
    """
    counter: Dict[str, int] = {}
    for chunk in _load_chunks().values():
        counter[chunk["law"]] = counter.get(chunk["law"], 0) + 1
    return [law for law, _ in sorted(counter.items(), key=lambda item: -item[1])]


def retrieve_auto(query: str, topk_final: int = 5) -> List[Dict]:
    """默认检索入口：混合召回 + 引用补全，用户无需选择模式。

    Args:
        query: 用户问题。
        topk_final: 混合检索返回条数。

    Returns:
        检索结果，前段为混合检索命中，末尾为引用补全条文。
    """
    items = get_retriever().retrieve(query, topk_final=topk_final, mode="hybrid")
    # 正向与反向各留一个额度：正向补"它引用了谁"，反向补"谁引用了它"
    extra = expand_refs(items, limit=1)
    extra += expand_reverse(items, limit=MAX_EXTRA - len(extra))
    return items + extra
