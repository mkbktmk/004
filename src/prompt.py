"""Prompt 构造：把检索结果拼成带编号的参考法条上下文。"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一名严谨的劳动法问答助手，只依据用户提供的参考法条作答，"
    "不引用参考法条之外的任何规定，不编造条号，不提供法律意见以外的推测。"
)

# 简称 -> 全称，用于按《法规全名》第X条 的格式标注来源
LAW_FULL_NAME: Dict[str, str] = {
    "劳动合同法": "中华人民共和国劳动合同法",
    "劳动法": "中华人民共和国劳动法",
    "劳动争议调解仲裁法": "中华人民共和国劳动争议调解仲裁法",
    "社会保险法": "中华人民共和国社会保险法",
    "职工带薪年休假条例": "职工带薪年休假条例",
    "工伤保险条例": "工伤保险条例",
}

USER_TEMPLATE = """你是一名严谨的劳动法助手，服务对象是普通劳动者。
请严格依据【参考法条】回答问题，遵守以下规则：
1. 只能使用【参考法条】中的内容，不得使用你记忆中的其他法律规定。
2. 每条结论后标注来源，格式为（《法规全名》第X条）；同一条法条只标注一次，后文再提不重复加括号。
3. 按可答程度分三档回答，不要一律拒答：
   （a）法条足以回答：直接给结论；
   （b）法条给出了标准或公式、但缺少计算所需参数（如月计薪天数、当地职工月平均工资）：先说能确定什么，再写公式与已知条件，最后指明缺哪个参数——不得自行假设参数数值。注意区分：一年约 52 周、每月约 4.33 周、每日 8 小时属于常识换算，可以直接用；只有法律口径的参数（月计薪天数、当地职工月平均工资、最低工资标准）才算缺失；
   （c）法条与问题无关：第一句说"根据现有法条无法确定"，并说明缺少哪方面规定，再引导用户补充情形。
4. 涉及金额或天数计算时，按"公式 → 代入 → 结果"三步写；参数缺失时也要写出公式，把未知量标出来，不要跳过计算过程。若问题问的是"多少小时/多少天/多少倍"这类量化标准，即使没给具体数值，也要直接给出基准数字（如每日 8 小时、每月工时约 173 小时），再说明超限判断方法。
5. 不要复述与问题无关的条文，不要输出法条原文全文；【依据】只列条号与一句话要点，不要重复【结论】里已说过的话；与结论无关的参考法条不要列进【依据】，特殊用工形式（非全日制、劳务派遣、以完成一定工作任务为期限）的条款仅在用户明确涉及该情形时引用。
6. 语言简洁，面向非法律专业读者，必要时用一句话解释术语。
输出结构（按需选用：简单问题只写【结论】和【依据】即可）：
【结论】一到三句话给出直接答案
【依据】引用的法条及一句话要点，不复述结论
【计算过程】仅计算题需要；缺参数时给出公式与待定项
【提示】仅在确有易误解之处时写，最多两条
{history_block}【参考法条】
{context}
【问题】
{question}"""

HISTORY_BLOCK = """【对话历史】（仅用于理解"那我呢""这种情况"等指代，不得作为法条依据）
{history}

"""

META_SYSTEM = (
    "你是一名劳动法问答助手。当前用户问的是本次对话本身"
    "（例如刚才问了什么、要求重复或总结），请仅依据【对话历史】如实回答。"
)

META_TEMPLATE = """用户正在询问本次对话本身，请依据下面的【对话历史】回答。

规则：
1. 只回答对话相关的内容（刚才问了什么、之前讨论过什么、复述或总结之前的回答）。
2. 不要引用法条，不要输出"根据现有法条无法确定"。
3. 历史里确实没有相关内容时，直接说明"本次对话里还没有相关提问"。

【对话历史】
{history}

【用户当前问题】
{question}"""


def format_history(history: List[Dict] | None, limit: int = 4, hint: bool = True) -> str:
    """把最近若干轮对话压成短文本，供 Prompt 使用。

    Args:
        history: 消息列表，元素含 role 与 content；为空表示首轮。
        limit: 最多保留的消息条数（一问一答算两条）。
        hint: 是否附加"不得作为法条依据"的说明（对话元问题不需要）。

    Returns:
        历史文本；无历史时返回空字符串。
    """
    if not history:
        return ""
    recent = history[-limit:]
    lines = []
    for message in recent:
        who = "用户" if message.get("role") == "user" else "助手"
        text = " ".join(str(message.get("content", "")).split())
        lines.append(f"{who}：{text[:200]}")
    body = "\n".join(lines)
    return HISTORY_BLOCK.format(history=body) if hint else body


def full_law_name(law: str) -> str:
    """把法规简称补全为全称，未收录的原样返回。"""
    return LAW_FULL_NAME.get(law, law)


def build_context(retrieved: List[Dict]) -> str:
    """拼接参考法条上下文，每条前加序号与《法规名》条号。"""
    blocks: List[str] = []
    for index, item in enumerate(retrieved, start=1):
        header = f"[{index}]《{full_law_name(item['law'])}》{item['article_no']}"
        blocks.append(f"{header}\n{item['text'].strip()}")
    return "\n\n".join(blocks) if blocks else "（本次未检索到相关法条）"


def build_prompt(
    question: str,
    retrieved: List[Dict],
    history: List[Dict] | None = None,
    meta: bool = False,
) -> Tuple[str, str]:
    """构造 system 与 user 消息。

    Args:
        question: 用户问题。
        retrieved: 检索结果列表。
        history: 历史消息（含 role/content），为空表示首轮对话。
        meta: 是否为"询问对话本身"的元问题；为真时改用对话模式，不引用法条。

    Returns:
        (system_prompt, user_prompt)。
    """
    if meta:
        user = META_TEMPLATE.format(
            history=format_history(history, limit=6, hint=False) or "（本次对话暂无历史）",
            question=question.strip(),
        )
        logger.info("对话模式 Prompt 构造完成：历史 %d 条，长度 %d 字", len(history or []), len(user))
        return META_SYSTEM, user
    context = build_context(retrieved)
    user = USER_TEMPLATE.format(
        history_block=format_history(history), context=context, question=question.strip()
    )
    logger.info(
        "Prompt 构造完成：参考法条 %d 条，历史 %d 条，user 长度 %d 字",
        len(retrieved),
        len(history or []),
        len(user),
    )
    return SYSTEM_PROMPT, user
