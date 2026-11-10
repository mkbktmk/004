"""全局配置：路径、检索参数、模型参数、密钥读取。

所有可调参数集中在此文件，其余模块只允许 `from src.config import X`。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# ---------------------------------------------------------------- 路径配置
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RAW_CLEAN_DIR: Path = DATA_DIR / "raw_clean"
CHUNKS_PATH: Path = DATA_DIR / "chunks.jsonl"
INDEX_DIR: Path = BASE_DIR / "index"
CHROMA_DIR: Path = INDEX_DIR / "chroma"
BM25_PATH: Path = INDEX_DIR / "bm25.pkl"
EVAL_DIR: Path = BASE_DIR / "eval"
TESTSET_PATH: Path = EVAL_DIR / "testset.csv"
RESULTS_DIR: Path = EVAL_DIR / "results"

# ---------------------------------------------------------------- 检索参数
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
CHUNK_MAX_LEN: int = int(os.getenv("CHUNK_MAX_LEN", "400"))
COLLECTION_NAME: str = "labor_law"
TOPK_VEC: int = int(os.getenv("TOPK_VEC", "10"))
TOPK_BM25: int = int(os.getenv("TOPK_BM25", "10"))
TOPK_FINAL: int = int(os.getenv("TOPK_FINAL", "5"))
RRF_K: int = int(os.getenv("RRF_K", "60"))

# ---------------------------------------------------------------- 生成参数
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "60"))
MAX_ANSWER_TOKENS: int = int(os.getenv("MAX_ANSWER_TOKENS", "800"))

# ---------------------------------------------------------------- 大模型配置
load_dotenv(BASE_DIR / ".env")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "").strip()
LLM_MODEL: str = os.getenv("LLM_MODEL", "").strip()

# 关键词检索领域词表：jieba 必须先把这些词加入词典，否则会被切碎
DOMAIN_WORDS = [
    "经济补偿",
    "赔偿金",
    "代通知金",
    "竞业限制",
    "医疗期",
    "年休假",
    "违法解除",
    "二倍工资",
    "工伤保险",
    "劳动仲裁",
    "试用期",
    "加班费",
]


def get_llm_config() -> Dict[str, str]:
    """读取并校验大模型配置。

    Returns:
        含 api_key / base_url / model 的字典。

    Raises:
        RuntimeError: 任一环境变量缺失时抛出中文提示。
    """
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not OPENAI_BASE_URL:
        missing.append("OPENAI_BASE_URL")
    if not LLM_MODEL:
        missing.append("LLM_MODEL")
    if missing:
        raise RuntimeError(
            "缺少大模型配置："
            + "、".join(missing)
            + "。请复制 .env.example 为 .env 并填写后重试。"
        )
    return {"api_key": OPENAI_API_KEY, "base_url": OPENAI_BASE_URL, "model": LLM_MODEL}


def ensure_dirs() -> None:
    """创建项目运行所需目录（幂等）。"""
    for path in (DATA_DIR, RAW_DIR, RAW_CLEAN_DIR, INDEX_DIR, CHROMA_DIR, EVAL_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> None:
    """配置全局日志格式（幂等），并压低第三方库噪声。"""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        root.setLevel(level)
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "jieba", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


configure_logging()
ensure_dirs()
