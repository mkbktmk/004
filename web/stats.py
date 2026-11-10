"""平台使用统计：用户数、提问量、拒答率、引用可溯源率、趋势与热门问题。

指标口径说明（避免把"没标准答案"硬算成正确率）：
- 拒答率：答案首段含"无法确定"的占比，法律场景拒答是安全阀，不是缺陷；
- 引用可溯源率：有引用的回答中，条号全部能在检索结果里找到的比例（citation_ok）；
- 正确率无法线上计算，以离线评测 hit@3 作为质量基线，不混入这里。

时间统一按北京时间（UTC+8）分组展示，库里存的是 UTC。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List

from fastapi import APIRouter, Depends

from web import db
from web.auth import require_admin
from web.routing import is_meta_question, is_system_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

BJ = "+8 hours"


def _scalar(conn, sql: str, args=()) -> float:
    """取单值查询结果，空值返回 0。"""
    row = conn.execute(sql, args).fetchone()
    value = row[0] if row else 0
    return value or 0


@router.get("/stats")
def platform_stats(days: int = 14, user: Dict = Depends(require_admin)) -> Dict:
    """返回概览指标、按天趋势与热门问题。"""
    days = max(1, min(days, 90))
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    today = date.today().isoformat()

    with db.connect() as conn:
        users_total = _scalar(conn, "SELECT COUNT(*) FROM users")
        users_off = _scalar(conn, "SELECT COUNT(*) FROM users WHERE status = 'disabled'")
        sessions = _scalar(conn, "SELECT COUNT(*) FROM chat_sessions")
        questions = _scalar(conn, "SELECT COUNT(*) FROM messages WHERE role = 'user'")
        answers = _scalar(conn, "SELECT COUNT(*) FROM messages WHERE role = 'assistant'")
        asked_users = _scalar(
            conn,
            "SELECT COUNT(DISTINCT s.user_id) FROM messages m "
            "JOIN chat_sessions s ON s.id = m.session_id WHERE m.role = 'user'",
        )
        refusals = _scalar(
            conn,
            "SELECT COUNT(*) FROM messages WHERE role = 'assistant' "
            "AND content LIKE '%无法确定%'",
        )
        cited_total = _scalar(conn, "SELECT COUNT(*) FROM messages WHERE citation_ok >= 0")
        cited_ok = _scalar(conn, "SELECT COUNT(*) FROM messages WHERE citation_ok = 1")
        avg_elapsed = _scalar(
            conn, "SELECT AVG(elapsed) FROM messages WHERE role = 'assistant' AND elapsed > 0"
        )
        avg_sources = _scalar(
            conn, "SELECT AVG(source_count) FROM messages WHERE role = 'assistant' "
            "AND source_count > 0",
        )
        today_questions = _scalar(
            conn,
            f"SELECT COUNT(*) FROM messages WHERE role = 'user' "
            f"AND date(datetime(created_at, '{BJ}')) = ?",
            (today,),
        )
        today_users = _scalar(
            conn,
            f"SELECT COUNT(DISTINCT s.user_id) FROM messages m "
            f"JOIN chat_sessions s ON s.id = m.session_id "
            f"WHERE m.role = 'user' AND date(datetime(m.created_at, '{BJ}')) = ?",
            (today,),
        )
        rows = conn.execute(
            f"SELECT date(datetime(m.created_at, '{BJ}')) AS d, "
            f"COUNT(*) AS q, COUNT(DISTINCT s.user_id) AS u "
            f"FROM messages m JOIN chat_sessions s ON s.id = m.session_id "
            f"WHERE m.role = 'user' AND date(datetime(m.created_at, '{BJ}')) >= ? "
            f"GROUP BY d ORDER BY d",
            (since,),
        ).fetchall()
        top = conn.execute(
            "SELECT content AS question, COUNT(*) AS times FROM messages "
            "WHERE role = 'user' GROUP BY content ORDER BY times DESC, MAX(id) DESC LIMIT 8"
        ).fetchall()
        gap = conn.execute(
            "SELECT u.content AS question, COUNT(*) AS times FROM messages u "
            "JOIN messages a ON a.session_id = u.session_id AND a.id = u.id + 1 "
            "AND a.role = 'assistant' "
            "WHERE u.role = 'user' AND a.content LIKE '%无法确定%' "
            "GROUP BY u.content ORDER BY times DESC LIMIT 12"
        ).fetchall()

    by_day = {row["d"]: row for row in rows}
    trend: List[Dict] = []
    for offset in range(days):
        day = (date.today() - timedelta(days=days - 1 - offset)).isoformat()
        hit = by_day.get(day)
        trend.append(
            {
                "date": day,
                "questions": hit["q"] if hit else 0,
                "users": hit["u"] if hit else 0,
            }
        )

    # 能力咨询/对话元问题不是知识库缺口，排除掉（历史数据里可能残留旧回答）
    gaps = [
        dict(row)
        for row in gap
        if not is_system_question(row["question"]) and not is_meta_question(row["question"])
    ][:5]

    return {
        "overview": {
            "users_total": users_total,
            "users_disabled": users_off,
            "users_asked": asked_users,
            "sessions": sessions,
            "questions": questions,
            "answers": answers,
            "today_questions": today_questions,
            "today_users": today_users,
            "refusal_rate": round(refusals / answers, 4) if answers else 0.0,
            "citation_rate": round(cited_ok / cited_total, 4) if cited_total else 0.0,
            "avg_elapsed": round(avg_elapsed, 2),
            "avg_sources": round(avg_sources, 2),
        },
        "trend": trend,
        "top_questions": [dict(row) for row in top],
        "gap_questions": gaps,
    }
