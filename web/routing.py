"""问题路由：决定一句话走法条检索还是对话模式，以及检索用什么查询词。

两类判定都只做字面匹配，规则可读、行为可预测，不做语义模型：
- 指代式追问（"那如果我月薪9000呢"）：拼上一轮问题再检索，否则会丢主题；
- 对话元问题（"我刚才问了什么"）：跳过检索，直接依据历史回答。
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# 指代/省略信号：出现这些说明当前句单独检索会丢主题
ANAPHORA = ("那", "这", "它", "呢", "如果", "改成", "换成", "还有", "同样")
# 询问对话本身的信号：不该检索法条
META_PATTERNS = (
    "我刚才问",
    "我刚刚问",
    "上一个问题",
    "上一条",
    "之前问",
    "前面问",
    "问了什么",
    "重复一下",
    "再说一遍",
    "总结一下",
)
# 询问系统能力/用法：既不检索，也不调模型（避免编造能力）
SYSTEM_PATTERNS = (
    "你是谁",
    "你叫什么",
    "怎么用",
    "如何使用",
    "有什么功能",
    "使用说明",
    "支持哪些法规",
    "能帮我做什么",
)
# 能力类提问的动词短语：需配合"你/您/系统"这类主语才判定，避免误伤
# "你能帮我看下这份合同吗"这种真实咨询
CAPABILITY_VERBS = ("做些什么", "做什么", "干什么", "会什么", "能干啥", "有啥功能")
SELF_WORDS = ("你", "您", "系统", "本助手")


def _has_topic_noun(question: str) -> bool:
    """判断问题里是否含可独立检索的主题名词。

    "你就告诉我违反没有"这类句子只有代词和动词，单独拿去检索会命中无关条文
    （"违反"匹配到"劳动者违反本法规定"），必须拼上上一轮问题。

    Args:
        question: 用户问题。

    Returns:
        含两个汉字以上的名词时返回 True。
    """
    import jieba.posseg as pseg

    return any(flag.startswith("n") and len(word) >= 2 for word, flag in pseg.cut(question))


def build_search_query(question: str, history: List[Dict]) -> str:
    """为检索构造查询：追问式提问时拼上上一轮问题。

    三层判定，任一命中即拼接：含指代词、无主题名词（如"你就告诉我违反没有"）。
    拼接的第二条规则里当前问题重复一次，避免历史主题压过当前提问。

    Args:
        question: 当前问题。
        history: 之前的消息（不含当前问题）。

    Returns:
        用于检索的查询文本。
    """
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    if not previous:
        return question
    if any(word in question for word in ANAPHORA):
        logger.info("追问改写：%s + %s", previous[:20], question[:20])
        return f"{previous} {question}"
    if not _has_topic_noun(question):
        logger.info("无主题名词，拼接上轮问题：%s + %s", previous[:20], question[:20])
        return f"{previous} {question} {question}"
    return question


def is_meta_question(question: str) -> bool:
    """判断是否在询问本次对话本身（如"我刚才问了什么"）。

    Args:
        question: 用户问题。

    Returns:
        是这类问题时返回 True。
    """
    return any(word in question for word in META_PATTERNS)


def is_system_question(question: str) -> bool:
    """判断是否在询问系统能力或用法。

    这类问题若走法条检索，必然命中无关条文并触发"依据不足"拒答。
    判定要求"能力动词 + 第一人称主语"同时出现，否则"你能帮我算加班费吗"
    这种真实咨询会被误判成问能力。

    Args:
        question: 用户问题。

    Returns:
        是能力类问题时返回 True。
    """
    if any(word in question for word in SYSTEM_PATTERNS):
        return True
    asks_self = any(word in question for word in SELF_WORDS)
    return asks_self and any(verb in question for verb in CAPABILITY_VERBS)


def help_text(laws: List[str]) -> str:
    """生成能力介绍文案（模板化，不由模型生成，避免夸大能力）。

    Args:
        laws: 知识库当前包含的法规名。

    Returns:
        固定结构的中文说明。
    """
    library = "、".join(laws) if laws else "（尚未导入法规）"
    return (
        "我是劳动法问答助手，能做这些事：\n\n"
        "· 依据本地法规库回答劳动法问题，并标注具体条号\n"
        "· 支持连续追问（例如「那如果我月薪9000呢」）\n"
        "· 依据不足时直接说「无法确定」，不编造结论\n\n"
        f"当前知识库：{library}\n\n"
        "可以这样问：\n"
        "· 试用期最长能约定多久？\n"
        "· 月薪6000元，工作日加班10小时，加班费多少？\n"
        "· 公司不签书面合同，能要二倍工资吗？\n\n"
        "我不能做：出具正式法律意见、代理个案、判断地方司法口径差异。"
    )
