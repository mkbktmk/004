"""FastAPI 应用装配：认证、会话、静态页面。

启动：uvicorn web.main:app --reload --port 8000
页面：/ 登录、/chat 咨询界面；管理员接口见 web/docs.py。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import configure_logging
from web import auth, db
from web.chat import router as chat_router
from web.docs import router as docs_router
from web.schemas import LoginIn, RegisterIn
from web.quality import router as quality_router
from web.stats import router as stats_router
from web.stream import router as stream_router
from web.users import router as users_router

configure_logging()
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
COOKIE_MAX_AGE = 7 * 24 * 3600


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建表并确保管理员账号存在。"""
    db.init_db()
    auth.ensure_admin()
    logger.info("Web 服务就绪")
    yield


app = FastAPI(title="劳动法智能问答助手", version="1.0.0", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(stream_router)
app.include_router(docs_router)
app.include_router(stats_router)
app.include_router(quality_router)
app.include_router(users_router)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/api/auth/login")
def login(payload: LoginIn, response: Response) -> dict:
    """登录成功后在 HttpOnly Cookie 写入令牌。"""
    user = auth.verify_login(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = auth.issue_token(user["id"])
    response.set_cookie(
        auth.COOKIE_NAME, token, httponly=True, samesite="lax", max_age=COOKIE_MAX_AGE
    )
    return user


@app.post("/api/auth/register")
def register(payload: RegisterIn) -> dict:
    """注册普通用户（管理员只能由初始化脚本或接口创建）。"""
    user_id = auth.create_user(payload.username, payload.password)
    return {"ok": True, "id": user_id}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict:
    """退出登录：吊销令牌并清 Cookie。"""
    token = request.cookies.get(auth.COOKIE_NAME, "")
    if token:
        auth.revoke_token(token)
    response.delete_cookie(auth.COOKIE_NAME)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(auth.require_user)) -> dict:
    """返回当前登录用户。"""
    return user


@app.get("/", include_in_schema=False)
def page_login() -> FileResponse:
    """登录页。"""
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/chat", include_in_schema=False)
def page_chat() -> FileResponse:
    """咨询页。"""
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/admin", include_in_schema=False)
def page_admin() -> FileResponse:
    """知识库管理页（前端会校验角色，非管理员跳回咨询页）。"""
    return FileResponse(STATIC_DIR / "admin.html")
