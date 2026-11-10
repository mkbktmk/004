"""认证与鉴权：密码哈希、登录令牌、角色依赖。

密码用标准库 pbkdf2_hmac（20 万轮），不引入 passlib/bcrypt；令牌是随机串存库，
不用 JWT —— 令牌可即时吊销，且少一个依赖。

用法：路由签名里写 user: dict = Depends(require_user) 即完成鉴权。
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from typing import Dict, Optional

from fastapi import Depends, HTTPException, Request

from web import db

logger = logging.getLogger(__name__)

COOKIE_NAME = "sid"
ROLE_ADMIN = "admin"
ROLE_USER = "user"


def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    """生成密码哈希。

    Args:
        password: 明文密码。
        salt: 复用已有盐；为空时随机生成。

    Returns:
        含 pwd_hash / salt 的字典。
    """
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return {"pwd_hash": digest.hex(), "salt": salt}


def create_user(username: str, password: str, role: str = ROLE_USER) -> int:
    """创建用户，用户名重复时抛 409。

    Args:
        username: 登录名。
        password: 明文密码。
        role: user 或 admin。

    Returns:
        新用户 id。
    """
    secret = hash_password(password)
    with db.connect() as conn:
        exists = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="用户名已存在")
        cursor = conn.execute(
            "INSERT INTO users(username, pwd_hash, salt, role, created_at) VALUES(?,?,?,?,?)",
            (username, secret["pwd_hash"], secret["salt"], role, db.now()),
        )
        return int(cursor.lastrowid)


def verify_login(username: str, password: str) -> Optional[Dict]:
    """校验账号密码。

    Args:
        username: 登录名。
        password: 明文密码。

    Returns:
        通过时返回用户字典，否则 None。
    """
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return None
    if row["status"] != "active":
        raise HTTPException(status_code=403, detail="该账号已被停用，请联系管理员")
    expected = hash_password(password, row["salt"])["pwd_hash"]
    if not secrets.compare_digest(expected, row["pwd_hash"]):
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


def issue_token(user_id: int) -> str:
    """签发登录令牌并落库。"""
    token = secrets.token_urlsafe(32)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO auth_tokens(token, user_id, created_at, expires_at) VALUES(?,?,?,?)",
            (token, user_id, db.now(), db.future()),
        )
    return token


def revoke_token(token: str) -> None:
    """吊销令牌（退出登录）。"""
    with db.connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))


def current_user(request: Request) -> Dict:
    """从 Cookie 解析当前用户，未登录抛 401。"""
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    with db.connect() as conn:
        row = conn.execute(
            "SELECT u.id, u.username, u.role FROM auth_tokens t "
            "JOIN users u ON u.id = t.user_id "
            "WHERE t.token = ? AND t.expires_at > ? AND u.status = 'active'",
            (token, db.now()),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


def require_user(user: Dict = Depends(current_user)) -> Dict:
    """登录即可访问的依赖（语义化别名，供路由签名使用）。"""
    return user


def require_admin(user: Dict = Depends(current_user)) -> Dict:
    """管理员专用依赖：非 admin 抛 403。"""
    if user["role"] != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="仅管理员可执行该操作")
    return user


def ensure_admin() -> None:
    """确保存在管理员账号；未配置密码时生成随机密码并打印一次。"""
    username = os.getenv("ADMIN_USERNAME", "admin").strip() or "admin"
    password = os.getenv("ADMIN_PASSWORD", "").strip()
    with db.connect() as conn:
        exists = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    if exists:
        return
    if not password:
        password = secrets.token_urlsafe(9)
        logger.warning(
            "未在 .env 配置 ADMIN_PASSWORD，已为 %s 生成随机密码：%s（仅本次显示，请写入 .env）",
            username,
            password,
        )
    create_user(username, password, role=ROLE_ADMIN)
    logger.info("管理员账号已创建：%s", username)
