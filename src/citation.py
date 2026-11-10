"""引用校验：检查答案中的条号是否可溯源到检索结果。"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零]+条")
# 只取「」里的原文摘录，不绑定前面的条号：句子常出现"第四十五条：…遇第四十二条
# 规定情形的…「摘录」"这种结构，绑定条号会把摘录错归给中间那个条号，误判正确答案。
QUOTE_RE = re.compile(r"「([^」]{2,40})」")


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


def check_quotes(answer: str, retrieved: List[Dict]) -> Dict:
    """校验答案里的原文摘录是否真的出自本次检索到的条文。

    只校验条号是不够的：答案可能写着"依第四十五条，合同应续延至情形消失"，
    而第四十五条根本不在本次上下文里——条号没标错，表述却是模型自己补的。

    这里只判断"摘录是否出现在任一检索条文中"，不做条号与摘录的精细配对：
    中文句子常把多个条号挤在一句话里，强行配对会误伤正确答案。代价是
    "把 A 条的话安到 B 条上"抓不到，但能抓住"整段是编的"，这更常见也更危险。

    Args:
        answer: 模型答案。
        retrieved: 检索结果。

    Returns:
        {total, matched, bad, coverage}；无摘录时 total 为 0、coverage 记为 1.0
        （不惩罚"没按格式写"，避免格式波动被误判成幻觉）。
    """
    corpus = "".join(str(item.get("text", "")) for item in retrieved).replace(" ", "")
    total = matched = 0
    bad: List[str] = []
    for quote in QUOTE_RE.findall(answer or ""):
        total += 1
        if quote.replace(" ", "") in corpus:
            matched += 1
        else:
            bad.append(quote)
    return {
        "total": total,
        "matched": matched,
        "bad": bad,
        "coverage": round(matched / total, 4) if total else 1.0,
    }


def verify(answer: str, retrieved: List[Dict]) -> Dict:
    """校验答案引用是否全部来自检索结果，并核对原文摘录。

    Args:
        answer: 模型答案。
        retrieved: 检索结果列表。

    Returns:
        {cited, invalid, coverage, quotes}；coverage = 有效条号 / 总条号。
    """
    cited = extract_articles(answer)
    allowed = {item["article_no"] for item in retrieved}
    invalid = cited - allowed
    coverage = (len(cited) - len(invalid)) / len(cited) if cited else 0.0
    quotes = check_quotes(answer, retrieved)
    result = {
        "cited": sorted(cited),
        "invalid": sorted(invalid),
        "coverage": round(coverage, 4),
        "quotes": quotes,
    }
    logger.info(
        "引用校验：条号 %d 个（无效 %d），摘录 %d 条（不符 %d）",
        len(cited),
        len(invalid),
        quotes["total"],
        len(quotes["bad"]),
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
    bad = (result.get("quotes") or {}).get("bad") or []
    parts = []
    if invalid:
        parts.append(f"以下条号未能追溯到本次检索结果：{'、'.join(invalid)}")
    if bad:
        shown = "、".join(f"「{item[:14]}」" for item in bad[:3])
        parts.append(f"以下原文摘录在检索到的条文中找不到：{shown}")
    if not parts:
        return ""
    return "；".join(parts) + "。请以【查看检索到的法条】中的原文为准。"
