"""会话与问答接口：多轮记忆、历史会话、消息持久化。

会话只对创建者本人可见；管理员不越权查看他人对话。
多轮指代靠最近 4 条消息进 Prompt 的【对话历史】段实现。

用法：uvicorn web.main:app 后访问 /api/chat。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException

from src.citation import format_warning, verify
from src.cite_expand import list_laws, retrieve_auto
from src.generator import answer as generate
from web import db
from web.auth import require_user
from web.routing import build_search_query, help_text, is_meta_question, is_system_question
from web.schemas import ChatIn, SessionIn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

HISTORY_LIMIT = 6
SOURCE_KEYS = ("law", "article_no", "chapter", "score", "via")


def _own_session(conn, sid: str, user_id: int):
    """取本人会话，不存在或非本人抛 404。"""
    row = conn.execute(
        "SELECT id, title FROM chat_sessions WHERE id = ? AND user_id = ?", (sid, user_id)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return row


def _compact(items: List[Dict]) -> List[Dict]:
    """裁掉条文正文，只留展示与溯源需要的字段（正文另查）。"""
    return [{key: item.get(key) for key in SOURCE_KEYS} | {"text": item["text"][:160]} for item in items]


@router.get("/sessions")
def list_sessions(user: Dict = Depends(require_user)) -> Dict:
    """列出当前用户的历史会话（按最近更新排序）。"""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT s.id, s.title, s.updated_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS msg_count "
            "FROM chat_sessions s WHERE s.user_id = ? ORDER BY s.updated_at DESC LIMIT 50",
            (user["id"],),
        ).fetchall()
    return {"sessions": [dict(row) for row in rows]}


@router.post("/sessions")
def create_session(payload: SessionIn, user: Dict = Depends(require_user)) -> Dict:
    """新建空会话。"""
    sid = uuid.uuid4().hex[:12]
    title = (payload.title or "新对话").strip()[:30]
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO chat_sessions(id, user_id, title, created_at, updated_at) "
            "VALUES(?,?,?,?,?)",
            (sid, user["id"], title, db.now(), db.now()),
        )
    return {"id": sid, "title": title}


@router.get("/sessions/{sid}")
def get_session(sid: str, user: Dict = Depends(require_user)) -> Dict:
    """读取会话及其全部消息。"""
    with db.connect() as conn:
        row = _own_session(conn, sid, user["id"])
        messages = conn.execute(
            "SELECT id, role, content, sources, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
    return {
        "id": row["id"],
        "title": row["title"],
        "messages": [
            {**dict(msg), "sources": json.loads(msg["sources"] or "[]")} for msg in messages
        ],
    }


@router.delete("/sessions/{sid}")
def delete_session(sid: str, user: Dict = Depends(require_user)) -> Dict:
    """删除本人会话（消息级联删除）。"""
    with db.connect() as conn:
        _own_session(conn, sid, user["id"])
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (sid,))
    return {"ok": True}


def prepare_turn(payload: ChatIn, user: Dict) -> Tuple[str, List[Dict]]:
    """定位或新建会话，取出最近若干条消息作为历史（流式与非流式共用）。"""
    question = payload.question.strip()
    with db.connect() as conn:
        if payload.session_id:
            _own_session(conn, payload.session_id, user["id"])
            sid = payload.session_id
        else:
            sid = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO chat_sessions(id, user_id, title, created_at, updated_at) "
                "VALUES(?,?,?,?,?)",
                (sid, user["id"], question[:30], db.now(), db.now()),
            )
        history = [
            dict(row)
            for row in conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (sid, HISTORY_LIMIT),
            ).fetchall()
        ][::-1]
    return sid, history


def cite_flag(check: Dict, skip: bool = False) -> int:
    """引用可溯源标记：1 全部有效 / 0 存在无效条号 / -1 不适用。"""
    if skip or not check["cited"]:
        return -1
    return 0 if check["invalid"] else 1


def save_turn(sid: str, question: str, answer: str, items: List[Dict], check: Dict,
              elapsed: float, skip: bool = False) -> None:
    """落库一轮问答：用户问题 + 助手回答 + 会话更新时间。"""
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content, citation_ok, created_at) "
            "VALUES(?,?,?,?,?)",
            (sid, "user", question, -1, db.now()),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, sources, citation_ok, elapsed, "
            "source_count, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                sid,
                "assistant",
                answer,
                json.dumps(_compact(items), ensure_ascii=False),
                cite_flag(check, skip),
                elapsed,
                len(items),
                db.now(),
            ),
        )
        conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (db.now(), sid))


@router.post("/chat")
def chat(payload: ChatIn, user: Dict = Depends(require_user)) -> Dict:
    """提问（非流式）：检索 + 生成 + 落库，一次性返回。"""
    question = payload.question.strip()
    sid, history = prepare_turn(payload, user)
    start = time.time()
    items: List[Dict] = []
    model_name = "builtin"
    skip = is_system_question(question)
    if skip:
        logger.info("能力咨询，直接应答：%s", question[:20])
        answer_text = help_text(list_laws())
    else:
        if is_meta_question(question):
            logger.info("对话元问题，跳过检索：%s", question[:20])
            result = generate(question, items, history, meta=True)
        else:
            items = retrieve_auto(build_search_query(question, history), payload.topk)
            result = generate(question, items, history)
        answer_text = result["answer"] or f"[生成失败] {result.get('error', '')}"
        model_name = result.get("model", "")
    check = verify(answer_text, items)
    elapsed = round(time.time() - start, 2)
    save_turn(sid, question, answer_text, items, check, elapsed, skip)

    logger.info("问答完成 user=%s session=%s 耗时 %.2fs", user["username"], sid, elapsed)
    return {
        "session_id": sid,
        "answer": answer_text,
        "sources": _compact(items),
        "warning": format_warning(check),
        "elapsed": elapsed,
        "model": model_name,
    }
