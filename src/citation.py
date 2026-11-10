"""引用校验：检查答案中的条号是否可溯源到检索结果。"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零]+条")


def extract_articles(text: str) -> Set[str]:
    """提取文本中出现的全部"第X条"。

    Args:
        text: 待检查文本（通常是模型答案）。

    Returns:
        条号集合，如 {"第四十七条", "第八十七条"}。
    """
    if not text:
        return set()
    return set(ARTICLE_RE.findall(text))


def verify(answer: str, retrieved: List[Dict]) -> Dict:
    """校验答案引用是否全部来自检索结果。

    Args:
        answer: 模型答案。
        retrieved: 检索结果列表。

    Returns:
        {cited, invalid, coverage}；coverage = 有效引用数 / 总引用数，无引用时为 0.0。
    """
    cited = extract_articles(answer)
    allowed = {item["article_no"] for item in retrieved}
    invalid = cited - allowed
    coverage = (len(cited) - len(invalid)) / len(cited) if cited else 0.0
    result = {
        "cited": sorted(cited),
        "invalid": sorted(invalid),
        "coverage": round(coverage, 4),
    }
    logger.info(
        "引用校验：引用 %d 个条号，无效 %d 个，coverage=%.2f",
        len(cited),
        len(invalid),
        coverage,
    )
    return result


def format_warning(result: Dict) -> str:
    """生成引用异常的中文提示，无异常时返回空字符串。

    Args:
        result: verify() 的返回值。

    Returns:
        警告文案或空字符串。
    """
    invalid = result.get("invalid") or []
    if not invalid:
        return ""
    articles = "、".join(invalid)
    return (
        f"以下条号未能追溯到本次检索结果，可能存在模型自行补充：{articles}。"
        "请以【查看检索到的法条】中的原文为准。"
    )
