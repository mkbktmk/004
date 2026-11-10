"""SQLite 存储层：用户、登录令牌、会话、消息、文档元数据。

所有表统一用 IF NOT EXISTS 建，重复启动幂等。每次操作独立连接（WAL 模式），
避免 FastAPI 多线程复用同一连接导致的跨线程错误。

用法：init_db() 后通过 connect() 上下文管理器读写。
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "app.sqlite3"
TOKEN_DAYS = 7

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    pwd_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '',
    citation_ok INTEGER NOT NULL DEFAULT -1,
    elapsed REAL NOT NULL DEFAULT 0,
    source_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    law TEXT NOT NULL,
    article_count INTEGER NOT NULL DEFAULT 0,
    uploaded_by INTEGER REFERENCES users(id),
    status TEXT NOT NULL DEFAULT 'done',
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """为已存在的旧库补列（幂等）。"""
    doc_cols = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
    if "note" not in doc_cols:
        conn.execute("ALTER TABLE documents ADD COLUMN note TEXT NOT NULL DEFAULT ''")
    user_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    if "status" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
    msg_cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    if "citation_ok" not in msg_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN citation_ok INTEGER NOT NULL DEFAULT -1")
        conn.execute("ALTER TABLE messages ADD COLUMN elapsed REAL NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE messages ADD COLUMN source_count INTEGER NOT NULL DEFAULT 0")


def now() -> str:
    """返回 UTC 时间字符串（秒级，统一排序用）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def future(days: int = TOKEN_DAYS) -> str:
    """返回若干天后的 UTC 时间字符串。"""
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """打开数据库连接，自动提交/回滚并关闭。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """建表并清理过期令牌（幂等）。"""
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.execute("DELETE FROM auth_tokens WHERE expires_at < ?", (now(),))
    logger.info("数据库就绪：%s", DB_PATH)
