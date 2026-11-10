"""构建索引：向量库（chroma）与 BM25。

用法：python -m src.build_index
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from typing import Dict, List, Sequence, Tuple

import jieba
from rank_bm25 import BM25Okapi

from src.config import (
    BM25_PATH,
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    DOMAIN_WORDS,
    EMBED_MODEL,
    RAW_DIR,
    configure_logging,
)

logger = logging.getLogger(__name__)

SAMPLE_TEXT = """第一章 总则
第一条 【示例数据】用人单位与劳动者建立劳动关系，应当订立书面劳动合同。
第二条 【示例数据】劳动合同期限一年以上不满三年的，试用期不得超过二个月。
第三章 工作时间和休息休假
第三条 【示例数据】用人单位安排劳动者延长工作时间的，支付不低于工资百分之一百五十的工资报酬。"""


def load_chunks() -> List[Dict]:
    """读取 data/chunks.jsonl。

    Returns:
        切块列表，文件不存在时返回空列表。
    """
    if not CHUNKS_PATH.exists():
        logger.warning("未找到 %s", CHUNKS_PATH)
        return []
    chunks: List[Dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("载入切块 %d 条", len(chunks))
    return chunks


def bootstrap_sample() -> List[Dict]:
    """法规库为空时生成 3 条【示例数据】条文，保证流程可跑通。

    Returns:
        示例切块列表，已写入 chunks.jsonl。
    """
    from src.chunker import split_law  # 局部导入，避免模块循环引用

    logger.warning("data/raw 为空：自动生成 3 条【示例数据】条文用于跑通流程")
    chunks = split_law("示例劳动法", SAMPLE_TEXT)[:3]
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return chunks


def encode_texts(texts: Sequence[str]) -> Tuple[List[List[float]], float]:
    """用 sentence-transformers 编码文本。

    Args:
        texts: 文本列表。

    Returns:
        (向量列表, 编码耗时秒)。

    Raises:
        RuntimeError: 模型加载或编码失败时抛出中文提示。
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL)
    except Exception as exc:  # noqa: BLE001 - 需覆盖下载、路径、依赖等各类异常
        raise RuntimeError(
            f"嵌入模型 {EMBED_MODEL} 加载失败：请检查网络或改用本地模型路径"
            "（设置环境变量 EMBED_MODEL 指向本地目录）。"
        ) from exc
    start = time.time()
    try:
        vectors = model.encode(
            list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"文本编码失败：{exc}") from exc
    elapsed = time.time() - start
    logger.info("编码完成：%d 条，耗时 %.2fs", len(texts), elapsed)
    return [vector.tolist() for vector in vectors], elapsed


def build_vector_index(chunks: List[Dict], vectors: List[List[float]]) -> None:
    """写入 chroma 持久化向量库（幂等重建）。

    Args:
        chunks: 切块列表。
        vectors: 与 chunks 一一对应的向量。
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("已删除旧 collection：%s", COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - 首次构建时 collection 不存在
        logger.info("collection 不存在，直接创建")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    for start in range(0, len(chunks), 256):
        batch = chunks[start : start + 256]
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=vectors[start : start + len(batch)],
            metadatas=[
                {
                    "law": chunk["law"],
                    "article_no": chunk["article_no"],
                    "chapter": chunk.get("chapter", ""),
                }
                for chunk in batch
            ],
        )
    logger.info("向量写入完成：%d 条 -> %s", len(chunks), CHROMA_DIR)


def build_bm25(chunks: List[Dict]) -> Tuple[List[str], BM25Okapi]:
    """构建 BM25 索引并落盘。

    Args:
        chunks: 切块列表。

    Returns:
        (chunk_id 列表, BM25 对象)。
    """
    for word in DOMAIN_WORDS:
        jieba.add_word(word)
    jieba.setLogLevel(logging.WARNING)
    tokenized = [list(jieba.cut(chunk["text"])) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as handle:
        pickle.dump(([chunk["id"] for chunk in chunks], bm25), handle)
    logger.info("BM25 构建完成：%d 篇 -> %s", len(tokenized), BM25_PATH)
    return [chunk["id"] for chunk in chunks], bm25


def main() -> None:
    """命令行入口：编码 + 建向量库 + 建 BM25。"""
    configure_logging()
    chunks = load_chunks()
    if not chunks:
        if not any(RAW_DIR.glob("*.txt")):
            chunks = bootstrap_sample()
        else:
            from src.chunker import build_chunks

            chunks = build_chunks()
    if not chunks:
        print("没有可用条文，请先准备 data/raw/*.txt 并执行 python -m src.clean && python -m src.chunker")
        return
    vectors, elapsed = encode_texts([chunk["text"] for chunk in chunks])
    build_vector_index(chunks, vectors)
    build_bm25(chunks)
    logger.info("索引构建完成")
    print(f"块数：{len(chunks)}")
    print(f"编码耗时：{elapsed:.2f}s")
    print(f"向量库：{CHROMA_DIR}")
    print(f"BM25：{BM25_PATH}")


if __name__ == "__main__":
    main()
