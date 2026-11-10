"""答案生成：调用 OpenAI 兼容接口，仅基于检索结果作答。"""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterator, List

from src.config import (
    MAX_ANSWER_TOKENS,
    REQUEST_TIMEOUT,
    TEMPERATURE,
    configure_logging,
    get_llm_config,
)
from src.prompt import build_prompt

logger = logging.getLogger(__name__)

_CLIENT = None
_MODEL_NAME = ""


def get_client():
    """初始化并复用 OpenAI 客户端单例。"""
    global _CLIENT, _MODEL_NAME
    if _CLIENT is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
            raise RuntimeError("未安装 openai SDK，请执行 pip install -r requirements.txt") from exc
        cfg = get_llm_config()
        _CLIENT = OpenAI(
            api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=REQUEST_TIMEOUT
        )
        _MODEL_NAME = cfg["model"]
        logger.info("大模型客户端初始化完成：model=%s base_url=%s", cfg["model"], cfg["base_url"])
    return _CLIENT


def _call_once(client, model: str, system: str, user: str) -> Dict:
    """单次调用 chat.completions.create 并整理返回值。"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_ANSWER_TOKENS,
    )
    content = (response.choices[0].message.content or "").strip()
    usage = {}
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return {"content": content, "usage": usage}


def answer(
    question: str,
    retrieved: List[Dict],
    history: List[Dict] | None = None,
    meta: bool = False,
) -> Dict:
    """基于检索结果生成答案，失败重试 1 次，仍失败返回结构化错误。

    Args:
        question: 用户问题。
        retrieved: 检索结果列表。
        history: 历史对话消息（含 role/content），用于多轮指代理解。
        meta: 是否为"询问对话本身"的元问题；为真时不要求引用法条。

    Returns:
        {answer, model, usage, elapsed}；出错时额外含 error 字段。
    """
    start = time.time()
    system, user = build_prompt(question, retrieved, history, meta)
    model = ""
    last_error = ""
    try:
        client = get_client()
        model = _MODEL_NAME
    except Exception as exc:  # noqa: BLE001 - 配置缺失不回抛界面
        logger.error("大模型配置错误：%s", exc)
        return {
            "answer": "",
            "model": "",
            "usage": {},
            "elapsed": 0.0,
            "error": str(exc),
        }
    for attempt in range(2):
        try:
            result = _call_once(client, model, system, user)
            elapsed = time.time() - start
            logger.info(
                "生成完成：tokens=%s 耗时=%.2fs model=%s",
                result["usage"].get("total_tokens", "-"),
                elapsed,
                model,
            )
            return {
                "answer": result["content"],
                "model": model,
                "usage": result["usage"],
                "elapsed": round(elapsed, 3),
            }
        except Exception as exc:  # noqa: BLE001 - 网络/额度/超时等统一降级
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("第 %d 次调用失败：%s", attempt + 1, last_error)
            if attempt == 0:
                time.sleep(2)
    elapsed = time.time() - start
    logger.error("生成最终失败：%s", last_error)
    return {
        "answer": "",
        "model": model,
        "usage": {},
        "elapsed": round(elapsed, 3),
        "error": f"调用大模型失败（已重试 1 次）：{last_error}",
    }


def stream_answer(
    question: str,
    retrieved: List[Dict],
    history: List[Dict] | None = None,
    meta: bool = False,
) -> Iterator[str]:
    """流式生成答案，逐块产出增量文本。

    与 answer() 的区别：不做"失败重试到成功"的兜底——已经吐出去的字收不回来，
    重复输出会让前端拼出两段答案。只在一个分片都没产生时重试 1 次。

    Args:
        question: 用户问题。
        retrieved: 检索结果列表。
        history: 历史消息，用于多轮指代。
        meta: 是否为对话元问题。

    Yields:
        模型输出的增量文本片段。

    Raises:
        RuntimeError: 配置缺失或重试后仍失败。
    """
    system, user = build_prompt(question, retrieved, history, meta)
    client = get_client()
    last_error = ""
    for attempt in range(2):
        produced = False
        try:
            stream = client.chat.completions.create(
                model=_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_ANSWER_TOKENS,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    produced = True
                    yield delta
            logger.info("流式生成完成：%d 字", len(user))
            return
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("流式第 %d 次失败：%s", attempt + 1, last_error)
            if produced:
                raise
            if attempt == 0:
                time.sleep(2)
    raise RuntimeError(f"调用大模型失败（已重试 1 次）：{last_error}")


configure_logging()
