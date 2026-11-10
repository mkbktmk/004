"""知识库增量维护：上传新法规时切片入库、重建 BM25、刷新运行时缓存。

增量写入只影响 chroma（按 law 覆盖），但 BM25 的 IDF 是全局统计量，
新增条文后所有词权重都会变，必须全量重建——所以一次上传天然是长任务。

用法：add_law_document("工资支付暂行规定", text)
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Dict, List

from src.build_index import build_bm25, encode_texts
from src.chunker import split_law
from src.config import CHROMA_DIR, CHUNKS_PATH, COLLECTION_NAME

logger = logging.getLogger(__name__)

# 建索引任务串行化：并发上传会同时写 chunks.jsonl 与 chroma
_LOCK = threading.Lock()


def read_all_chunks() -> List[Dict]:
    """读取 chunks.jsonl 全部切块。"""
    if not CHUNKS_PATH.exists():
        return []
    chunks: List[Dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def _write_all_chunks(chunks: List[Dict]) -> None:
    """全量重写 chunks.jsonl。"""
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _upsert_vectors(chunks: List[Dict]) -> None:
    """按 law 覆盖写入向量库（先删同法规旧块，再写新块）。"""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    law = chunks[0]["law"]
    collection.delete(where={"law": law})
    vectors, _ = encode_texts([chunk["text"] for chunk in chunks])
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=vectors,
        metadatas=[
            {
                "law": chunk["law"],
                "article_no": chunk["article_no"],
                "chapter": chunk.get("chapter", ""),
            }
            for chunk in chunks
        ],
    )
    logger.info("向量写入：%s %d 块", law, len(chunks))


def _delete_vectors(law: str) -> None:
    """删除某部法规的全部向量。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.delete(where={"law": law})


def reload_runtime() -> None:
    """清空检索器与条文缓存，让下次查询读到新索引。"""
    from src import cite_expand
    from src.retriever import Retriever

    Retriever._instance = None  # 单例置空，下次 get_retriever() 重新加载
    cite_expand._CHUNKS = None
    logger.info("检索器缓存已刷新")


def add_law_document(law: str, text: str) -> Dict:
    """把一部法规全量替换进知识库（切块 → 向量 → BM25）。

    Args:
        law: 法规简称，作为切块 id 前缀与覆盖依据。
        text: 清洗后的法规全文，需保持"一行一条"格式。

    Returns:
        {"law", "chunks"}；无法切出条文时抛 ValueError。
    """
    chunks = split_law(law, text)
    if not chunks:
        raise ValueError("未识别到任何“第X条”，请确认文件是法规条文格式")
    with _LOCK:
        remaining = [chunk for chunk in read_all_chunks() if chunk["law"] != law]
        merged = remaining + chunks
        _write_all_chunks(merged)
        _upsert_vectors(chunks)
        build_bm25(merged)
    reload_runtime()
    logger.info("法规入库完成：%s %d 块，库内共 %d 块", law, len(chunks), len(merged))
    return {"law": law, "chunks": len(chunks)}


def remove_law_document(law: str) -> int:
    """从知识库删除某部法规，返回删除的块数。"""
    with _LOCK:
        current = read_all_chunks()
        remaining = [chunk for chunk in current if chunk["law"] != law]
        removed = len(current) - len(remaining)
        _write_all_chunks(remaining)
        _delete_vectors(law)
        if remaining:
            build_bm25(remaining)
        else:
            logger.warning("知识库已空，BM25 保持旧索引以免除零错误")
    reload_runtime()
    logger.info("法规移除完成：%s 共 %d 块", law, removed)
    return removed
