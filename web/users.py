"""管理员用户管理：列表、启用/停用、改角色、重置密码、删除。

三条硬规则防止把自己锁在门外：
1. 不能停用或降级自己；
2. 不能删除自己；
3. 不能把最后一个可用的管理员降级或删除。
停用会同时清空该用户的登录令牌，立刻生效。
"""

from __future__ import annotations

import logging
import secrets
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from web import auth, db
from web.auth import require_admin
from web.schemas import UserPatch

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

USER_SQL = """
SELECT u.id, u.username, u.role, u.status, u.created_at,
       (SELECT COUNT(*) FROM chat_sessions s WHERE s.user_id = u.id) AS sessions,
       (SELECT COUNT(*) FROM messages m JOIN chat_sessions s ON s.id = m.session_id
        WHERE s.user_id = u.id AND m.role = 'user') AS questions,
       (SELECT MAX(m.created_at) FROM messages m JOIN chat_sessions s ON s.id = m.session_id
        WHERE s.user_id = u.id) AS last_active
FROM users u ORDER BY u.id
"""


def _active_admins(conn) -> int:
    """统计当前可用的管理员数量。"""
    row = conn.execute(
        "SELECT COUNT(*) FROM users WHERE role = 'admin' AND status = 'active'"
    ).fetchone()
    return int(row[0]) if row else 0


def _get_user(conn, uid: int):
    """按 id 取用户，不存在抛 404。"""
    row = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return row


@router.get("/users")
def list_users(admin: Dict = Depends(require_admin)) -> Dict:
    """列出全部用户及其使用情况。"""
    with db.connect() as conn:
        rows = conn.execute(USER_SQL).fetchall()
    return {"users": [dict(row) for row in rows], "self_id": admin["id"]}


@router.patch("/users/{uid}")
def update_user(uid: int, payload: UserPatch, admin: Dict = Depends(require_admin)) -> Dict:
    """启用/停用账号或调整角色。"""
    if uid == admin["id"]:
        raise HTTPException(status_code=400, detail="不能修改自己的角色或状态")
    with db.connect() as conn:
        target = _get_user(conn, uid)
        if payload.status and payload.status not in ("active", "disabled"):
            raise HTTPException(status_code=400, detail="状态只能是 active 或 disabled")
        if payload.role and payload.role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="角色只能是 user 或 admin")
        losing_admin = target["role"] == "admin" and (
            payload.role == "user" or payload.status == "disabled"
        )
        if losing_admin and _active_admins(conn) <= 1:
            raise HTTPException(status_code=400, detail="至少要保留一个可用的管理员")
        if payload.status:
            conn.execute("UPDATE users SET status = ? WHERE id = ?", (payload.status, uid))
            if payload.status == "disabled":
                conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (uid,))
        if payload.role:
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (payload.role, uid))
        updated = _get_user(conn, uid)
    logger.info("管理员 %s 修改用户 %s：%s", admin["username"], updated["username"], payload)
    return {"id": updated["id"], "role": updated["role"], "status": updated["status"]}


@router.post("/users/{uid}/reset-password")
def reset_password(uid: int, admin: Dict = Depends(require_admin)) -> Dict:
    """把用户密码重置为随机串，返回明文供管理员转达。"""
    new_password = secrets.token_urlsafe(9)
    secret = auth.hash_password(new_password)
    with db.connect() as conn:
        target = _get_user(conn, uid)
        conn.execute(
            "UPDATE users SET pwd_hash = ?, salt = ? WHERE id = ?",
            (secret["pwd_hash"], secret["salt"], uid),
        )
        conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (uid,))
    logger.info("管理员 %s 重置了 %s 的密码", admin["username"], target["username"])
    return {"username": target["username"], "password": new_password}


@router.delete("/users/{uid}")
def delete_user(uid: int, admin: Dict = Depends(require_admin)) -> Dict:
    """删除用户及其全部会话（消息级联删除）。"""
    if uid == admin["id"]:
        raise HTTPException(status_code=400, detail="不能删除自己")
    with db.connect() as conn:
        target = _get_user(conn, uid)
        if target["role"] == "admin" and _active_admins(conn) <= 1:
            raise HTTPException(status_code=400, detail="至少要保留一个可用的管理员")
        conn.execute("DELETE FROM users WHERE id = ?", (uid,))
    logger.info("管理员 %s 删除了用户 %s", admin["username"], target["username"])
    return {"ok": True, "username": target["username"]}
