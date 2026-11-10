"""查询扩展：同义词映射 + 关键词扩展，只改检索查询，不改法条原文。

法条原文用"延长工作时间""工资报酬"这类法定表述，用户口语却说"加班""加班费"。
BM25 是字面匹配，字面不通就召不回；本模块在检索侧把口语词映射为法条用语，
并剔除纯阿拉伯数字（"6000""10" 这类个案数值在法条中多写作中文数字，只会带来噪声）。

用法：build_query_tokens("月薪6000元，工作日加班10小时，加班费多少？")
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

# 口语/提问用词 -> 法条用语（值必须是法条原文里真实出现的词，否则扩展无效）
SYNONYMS: Dict[str, Sequence[str]] = {
    "加班费": ("加班", "延长工作时间", "工资报酬"),
    "加班": ("延长工作时间",),
    "加点": ("延长工作时间",),
    "法定节假日": ("法定休假日",),
    "法定假日": ("法定休假日",),
    "节假日": ("法定休假日",),
    "休息日加班": ("休息日", "补休"),
    "经济补偿金": ("经济补偿",),
    "经济补偿": ("经济补偿金",),
    "赔偿金": ("经济补偿", "二倍"),
    "二倍工资": ("二倍的工资",),
    "双倍工资": ("二倍的工资",),
    "辞退": ("解除劳动合同",),
    "开除": ("解除劳动合同",),
    "裁员": ("裁减人员",),
    "年假": ("年休假",),
    "社保": ("社会保险",),
    "交社保": ("未依法缴纳", "社会保险费", "缴纳"),
    "不交社保": ("未依法缴纳", "社会保险费", "缴纳"),
    "没交社保": ("未依法缴纳", "社会保险费", "缴纳"),
    "不缴社保": ("未依法缴纳", "社会保险费", "缴纳"),
    "怀孕": ("孕期", "产期", "哺乳期"),
    "产假": ("产期",),
    "生育": ("产期",),
    # 用户说"合同到期/不续签"，法条写"劳动合同期满…应当续延"。
    # 劳动法全本入库后，同主题的第二十九条（三期不得解除）会抢位，故加强"期满/续延"权重。
    "合同到期": ("劳动合同期满", "续延", "期满", "续延"),
    "合同期满": ("续延", "期满"),
    "提前通知": ("提前三十日",),
    "竞业限制": ("竞业限制", "二年"),
    "工伤": ("工伤", "事故伤害"),
    "仲裁时效": ("仲裁", "时效"),
}

# 扩展词重复注入次数：BM25 无逐词权重，重复 token 即加权，2 次是实测较稳的折中
EXPAND_REPEAT: int = 2

# 模式类扩展：口语里问"多少小时"其实在问工时时长，而法条写的是"工时制度""标准工时"。
# 用正则而非固定词，是因为问法太多（多少小时 / 几小时 / 工作多久）。
PATTERN_RULES: Sequence[Tuple[re.Pattern, Sequence[str]]] = (
    (re.compile(r"多少(?:个)?小时|几(?:个)?小时|工作.{0,6}小时"), ("工时", "标准工时")),
    (re.compile(r"工作时长|工作时间.{0,4}上限"), ("工时制度",)),
    # "干10小时合法吗"这类合规性提问，法条用的是"延长工作时间"
    (re.compile(r"(?=.*小时)(?=.*(?:合法|违法|违规|超时))"), ("工时", "延长工作时间")),
    # 问条款有效性，实际要查的是"无效"情形。只扩"劳动合同无效"，
    # 单独扩"无效"会过度加权第二十七条（部分无效）与第八十六条（确认无效）。
    (re.compile(r"有效吗|有效么|算数吗|是否有效|有没有效"), ("劳动合同无效",)),
)

_NUMERIC_RE = re.compile(r"^[\d.]+$")
_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)

QueryTokens = Tuple[List[str], List[str]]


def expand_terms(query: str) -> List[str]:
    """从查询里识别口语词并给出对应法条用语。

    Args:
        query: 用户原始问题。

    Returns:
        去重后的扩展词列表；查询里的原词不重复返回。
    """
    added: List[str] = []
    for cue, targets in SYNONYMS.items():
        if cue not in query:
            continue
        for target in targets:
            if target not in query and target not in added:
                added.append(target)
    for pattern, targets in PATTERN_RULES:
        if not pattern.search(query):
            continue
        for target in targets:
            if target not in query and target not in added:
                added.append(target)
    return added


def is_noise(token: str) -> bool:
    """判断 token 是否为检索噪声（纯数字或纯标点/空白）。

    Args:
        token: 分词结果中的单个 token。

    Returns:
        需要丢弃时返回 True。
    """
    return bool(_NUMERIC_RE.match(token) or _PUNCT_RE.match(token))


def build_query_tokens(query: str, repeat: int = EXPAND_REPEAT) -> QueryTokens:
    """把查询切成 BM25 可用的 token 序列（含同义词扩展）。

    Args:
        query: 用户原始问题。
        repeat: 扩展词重复次数，用于给扩展词加权。

    Returns:
        (原始 token 列表, 扩展 token 列表)，调用方按需拼接打分。
    """
    import jieba

    from src.config import DOMAIN_WORDS

    jieba.setLogLevel(logging.WARNING)
    for word in DOMAIN_WORDS:
        jieba.add_word(word)
    terms = expand_terms(query)
    # 注意：扩展词不能再 add_word。索引侧只加过 DOMAIN_WORDS，若查询侧把
    # "延长工作时间" 合成一个 token，就会与索引里的 "延长/工作时间" 对不上而漏召。
    base = [token for token in jieba.cut(query) if not is_noise(token)]
    extra: List[str] = []
    for term in terms:
        extra.extend([token for token in jieba.cut(term) if not is_noise(token)] * repeat)
    if terms:
        logger.info("查询扩展：%s -> %s", query, "、".join(terms))
    return base, extra
