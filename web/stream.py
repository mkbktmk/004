"""流式问答接口（SSE）：先推检索到的条文，再逐字推答案。

事件顺序：meta（session_id + 条文 + 模式）→ delta（增量文本，多次）→ done（耗时与校验）。
非流式的 /api/chat 保留，评测与调试仍走它，不受流式影响。

复用 web.chat 里的 prepare_turn / save_turn / _compact，避免两套落库逻辑走偏。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Iterator, List

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.citation import format_warning, verify
from src.cite_expand import list_laws, retrieve_auto
from src.config import LLM_MODEL
from src.generator import stream_answer
from web.auth import require_user
from web.chat import _compact, cite_flag, prepare_turn, save_turn
from web.routing import build_search_query, help_text, is_meta_question, is_system_question
from web.schemas import ChatIn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

SLICE = 8


def _event(name: str, data: Dict) -> str:
    """把一条事件编码成 SSE 帧。"""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _slice(text: str, size: int = SLICE) -> Iterator[str]:
    """把定长文本切成小块，让内置文案也有逐字出现的节奏。"""
    for start in range(0, len(text), size):
        yield text[start : start + size]


@router.post("/chat/stream")
def chat_stream(payload: ChatIn, user: Dict = Depends(require_user)) -> StreamingResponse:
    """流式提问：SSE 推送检索结果与逐字答案。"""
    question = payload.question.strip()
    sid, history = prepare_turn(payload, user)
    skip = is_system_question(question)
    is_meta = (not skip) and is_meta_question(question)
    items: List[Dict] = []
    if not skip and not is_meta:
        items = retrieve_auto(build_search_query(question, history), payload.topk)
    mode = "builtin" if skip else ("meta" if is_meta else "law")

    def gen() -> Iterator[str]:
        start = time.time()
        yield _event(
            "meta",
            {"session_id": sid, "sources": _compact(items), "mode": mode},
        )
        parts: List[str] = []
        if skip:
            for piece in _slice(help_text(list_laws())):
                parts.append(piece)
                yield _event("delta", {"text": piece})
        else:
            try:
                for delta in stream_answer(question, items, history, meta=is_meta):
                    parts.append(delta)
                    yield _event("delta", {"text": delta})
            except Exception as exc:  # noqa: BLE001 - 已输出的内容无法撤回，只能提示
                logger.error("流式生成失败：%s", exc)
                yield _event("error", {"detail": str(exc)})
        answer_text = "".join(parts)
        check = verify(answer_text, items)
        elapsed = round(time.time() - start, 2)
        save_turn(sid, question, answer_text, items, check, elapsed, skip)
        yield _event(
            "done",
            {
                "session_id": sid,
                "elapsed": elapsed,
                "warning": format_warning(check),
                "citation_ok": cite_flag(check, skip),
                "model": "builtin" if skip else LLM_MODEL,
            },
        )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
