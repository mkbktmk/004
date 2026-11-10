"""Streamlit 界面：劳动法智能问答助手。

运行：streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cite_expand import retrieve_auto  # noqa: E402
from src.citation import format_warning, verify  # noqa: E402
from src.config import TOPK_FINAL  # noqa: E402
from src.generator import answer  # noqa: E402
from src.prompt import build_prompt  # noqa: E402
from src.retriever import get_retriever  # noqa: E402

MODE_LABELS = {
    "智能检索（推荐）": "auto",
    "向量检索（调试）": "vector",
    "BM25 关键词检索（调试）": "bm25",
    "混合检索 RRF（调试）": "hybrid",
}


def render_sources(items: list) -> None:
    """渲染检索证据列表。"""
    for index, item in enumerate(items, start=1):
        chapter = f"（{item['chapter']}）" if item.get("chapter") else ""
        tag = item.get("via") or f"得分 {item['score']}"
        st.markdown(f"**{index}. {item['law']} {item['article_no']}**{chapter}　`{tag}`")
        text = item["text"]
        st.write(text if len(text) <= 300 else text[:300] + "……")
        st.divider()


def run_query(question: str, mode: str, topk: int) -> dict:
    """执行一次完整问答链路并汇总耗时。"""
    retriever = get_retriever()
    start = time.time()
    if mode == "auto":
        items = retrieve_auto(question, topk)
    else:
        items = retriever.retrieve(question, topk_final=topk, mode=mode)
    retrieve_elapsed = time.time() - start
    started = time.time()
    result = answer(question, items)
    check = verify(result["answer"], items)
    system, user = build_prompt(question, items)
    mode_text = next((k for k, v in MODE_LABELS.items() if v == mode), mode)
    return {
        "question": question,
        "mode": mode_text,
        "topk": topk,
        "items": items,
        "result": result,
        "check": check,
        "retrieve_elapsed": retrieve_elapsed,
        "total_elapsed": time.time() - started + retrieve_elapsed,
        "prompt": {"system": system, "user": user},
    }


def main() -> None:
    """页面主流程。"""
    st.set_page_config(page_title="劳动法智能问答助手", page_icon="⚖️", layout="wide")
    st.title("劳动法智能问答助手")
    st.write("输入劳动法问题，系统从本地法规库检索条文，仅依据检索结果作答并标注出处。")

    with st.sidebar:
        st.header("检索设置")
        mode_label = st.selectbox("检索模式", list(MODE_LABELS.keys()), index=0)
        topk = st.slider("送入模型的条文数 Top-K", min_value=1, max_value=10, value=TOPK_FINAL)
        show_evidence = st.checkbox("显示检索证据", value=True)
        st.caption("默认无需调整：系统混合召回后自动补齐条文引用链。")
        st.caption("嵌入模型：BAAI/bge-small-zh-v1.5")

    question = st.text_area("请输入你的问题", height=100, placeholder="例如：月薪8000，工作3年7个月，经济补偿多少？")
    submitted = st.button("提交", type="primary")

    if submitted:
        if not question.strip():
            st.warning("请先输入问题。")
        else:
            with st.spinner("正在检索法条并生成答案……"):
                try:
                    outcome = run_query(question.strip(), MODE_LABELS[mode_label], topk)
                    st.session_state["outcome"] = outcome
                except Exception as exc:  # noqa: BLE001 - 界面上只提示，不崩
                    st.session_state["outcome"] = {"error": str(exc), "question": question}

    outcome = st.session_state.get("outcome")
    if not outcome:
        return
    if outcome.get("error"):
        st.error(f"处理失败：{outcome['error']}")
        return

    items = outcome["items"]
    result = outcome["result"]

    st.subheader("回答")
    if result.get("error"):
        st.error(result["error"])
    if result.get("answer"):
        st.markdown(result["answer"])
    else:
        st.info("模型未返回内容，请检查 .env 中的大模型配置。")

    warning = format_warning(outcome["check"])
    if warning:
        st.warning(warning)
    elif outcome["check"]["cited"]:
        st.success(f"引用校验通过：{len(outcome['check']['cited'])} 个条号均可在检索结果中溯源。")
    else:
        st.warning("答案中未检测到条号引用，请人工核对依据。")

    if show_evidence:
        with st.expander("查看检索到的法条", expanded=False):
            if items:
                render_sources(items)
            else:
                st.write("未检索到相关条文。")
        with st.expander("查看送入模型的 Prompt", expanded=False):
            st.markdown("**System**")
            st.code(outcome["prompt"]["system"], language="text")
            st.markdown("**User**")
            st.code(outcome["prompt"]["user"], language="text")

    st.caption(
        f"检索模式：{outcome['mode']}　检索耗时：{outcome['retrieve_elapsed']:.2f}s　"
        f"总耗时：{outcome['total_elapsed']:.2f}s　"
        f"模型：{result.get('model') or '-'}　tokens：{result.get('usage', {}).get('total_tokens', '-')}"
    )


if __name__ == "__main__":
    main()
