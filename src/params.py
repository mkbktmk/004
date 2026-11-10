"""计算参数表：法条给出了标准但没给折算口径，这类数字单独维护。

典型场景：劳动法第四十四条只说"按工资的 150% 支付"，而"月工资怎么折算成
小时工资"写在劳社部发〔2008〕3 号通知里（月计薪天数 21.75）。这份通知不在
知识库中，导致加班费永远算不出具体金额。

参数与法条分开存，是因为它们性质不同：法条要检索、参数按关键词直接注入，
且每条都要能追溯到出处，避免模型自己编数字。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

PARAM_PATH = Path(__file__).resolve().parent.parent / "data" / "params.json"
_CACHE: List[Dict] | None = None


def load_params() -> List[Dict]:
    """读取参数表（进程内缓存，文件改动需重启生效）。

    Returns:
        参数条目列表；文件缺失或损坏时返回空列表。
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not PARAM_PATH.exists():
        logger.warning("参数表不存在：%s", PARAM_PATH)
        _CACHE = []
        return _CACHE
    try:
        _CACHE = json.loads(PARAM_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("参数表读取失败：%s", exc)
        _CACHE = []
    return _CACHE


def match_params(text: str) -> List[Dict]:
    """按关键词挑出与问题相关的参数。

    Args:
        text: 用于匹配的文本（当前问题 + 历史里的用户提问）。

    Returns:
        命中的参数条目，最多 3 条，避免把无关参数塞进 Prompt。
    """
    hits = [item for item in load_params() if any(k in text for k in item.get("keywords", []))]
    return hits[:3]


def format_params(items: List[Dict]) -> str:
    """把参数渲染成 Prompt 片段，附出处便于核查。

    Args:
        items: match_params 的返回值。

    Returns:
        待插入的文本块；无命中时返回空字符串。
    """
    if not items:
        return ""
    lines = ["【计算参数】（配套规定给出的折算口径，可直接用于计算，引用时注明出处）"]
    for item in items:
        lines.append(f"- {item['name']}：{item['value']}")
        if item.get("formula"):
            lines.append(f"  公式：{item['formula']}")
        lines.append(f"  出处：{item['source']}")
    return "\n".join(lines) + "\n\n"
