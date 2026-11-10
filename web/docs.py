"""管理员接口：知识文档上传、库内法规概览、删除。

上传是长任务（切块 + 编码 + BM25 全量重建），走 BackgroundTasks，
前端轮询 /documents 看 status 从 indexing 变 done/failed。

用法：仅 admin 可访问，普通用户返回 403。
"""

from __future__ import annotations

import io
import logging
from collections import Counter
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from src.chunker import short_law_name
from src.clean import clean_text
from src.index_ops import add_law_document, read_all_chunks, remove_law_document
from web import db
from web.auth import require_admin

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

ALLOWED = {".txt", ".md", ".pdf"}
MAX_BYTES = 10 * 1024 * 1024


def extract_text(filename: str, raw: bytes) -> str:
    """抽取纯文本：pdf 走 pypdf，其余按 utf-8 解码，统一清洗。

    Args:
        filename: 原始文件名，用来判断后缀。
        raw: 文件字节内容。

    Returns:
        清洗后的文本；pdf 无文本层时可能为空串。
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        text = raw.decode("utf-8", errors="ignore")
    return clean_text(text)


def _run_index(doc_id: int, law: str, text: str) -> None:
    """后台任务：切块入库，回写 done / failed 状态。"""
    try:
        result = add_law_document(law, text)
        with db.connect() as conn:
            conn.execute(
                "UPDATE documents SET status = 'done', article_count = ?, note = '' WHERE id = ?",
                (result["chunks"], doc_id),
            )
    except Exception as exc:  # noqa: BLE001 - 失败要落库给管理员看
        logger.error("法规入库失败 %s：%s", law, exc)
        with db.connect() as conn:
            conn.execute(
                "UPDATE documents SET status = 'failed', note = ? WHERE id = ?",
                (str(exc)[:200], doc_id),
            )


@router.get("/documents")
def list_documents(user: Dict = Depends(require_admin)) -> Dict:
    """返回上传记录 + 当前库内法规概览。"""
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY id DESC").fetchall()
    counter = Counter(chunk["law"] for chunk in read_all_chunks())
    library = [{"law": law, "chunks": count} for law, count in counter.most_common()]
    return {
        "documents": [dict(row) for row in rows],
        "library": library,
        "total_chunks": sum(counter.values()),
    }


@router.post("/documents")
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    user: Dict = Depends(require_admin),
) -> Dict:
    """上传法规文件（txt/md/pdf），异步建索引。"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail="只支持 .txt / .md / .pdf")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="文件超过 10MB")
    text = extract_text(file.filename or "", raw)
    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空或无法解析（扫描版 PDF 需先 OCR）")
    law = short_law_name(Path(file.filename or "未命名").stem)
    with db.connect() as conn:
        cursor = conn.execute(
            "INSERT INTO documents(filename, law, article_count, uploaded_by, status, note, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (file.filename, law, 0, user["id"], "indexing", "", db.now()),
        )
        doc_id = int(cursor.lastrowid)
    background.add_task(_run_index, doc_id, law, text)
    logger.info("收到上传：%s -> %s（doc_id=%d）", file.filename, law, doc_id)
    return {"id": doc_id, "law": law, "status": "indexing"}


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: int, user: Dict = Depends(require_admin)) -> Dict:
    """删除上传记录，并把该法规从知识库中移除。"""
    with db.connect() as conn:
        row = conn.execute("SELECT law FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    removed = remove_law_document(row["law"])
    with db.connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    return {"ok": True, "law": row["law"], "removed_chunks": removed}
