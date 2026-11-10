"""请求体模型（pydantic）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoginIn(BaseModel):
    """登录请求：只校验非空，不校验密码强度（老账号可能较短）。"""

    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=1, max_length=64)


class RegisterIn(BaseModel):
    """注册请求：用户名与密码都做长度约束。"""

    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=6, max_length=64)


class SessionIn(BaseModel):
    """新建会话请求。"""

    title: str = ""


class ChatIn(BaseModel):
    """提问请求；session_id 为空表示新建会话。"""

    question: str = Field(min_length=1, max_length=500)
    session_id: str = ""
    topk: int = Field(default=5, ge=1, le=10)


class UserPatch(BaseModel):
    """管理员调整用户：字段留空表示该项不变。"""

    status: str = ""
    role: str = ""
