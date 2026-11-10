"""检索器：向量检索 + BM25 + RRF 融合。

用法：Retriever().retrieve("试用期最长多久")
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from typing import Dict, List, Sequence, Tuple

from src.config import (
    BM25_PATH, CHROMA_DIR, CHUNKS_PATH, COLLECTION_NAME, DOMAIN_WORDS, EMBED_MODEL,
    RRF_K, TOPK_BM25, TOPK_FINAL, TOPK_VEC,
)

logger = logging.getLogger(__name__)


class Retriever:
    """单例懒加载检索器：模型、chroma collection、BM25 首次使用时初始化。"""

    _instance: "Retriever | None" = None

    def __init__(self) -> None:
        """初始化空壳，不做任何 IO。"""
        self._model = None
        self._collection = None
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._chunks: Dict[str, Dict] | None = None

    @classmethod
    def instance(cls) -> "Retriever":
        """获取全局单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_chunks(self) -> Dict[str, Dict]:
        """把 chunks.jsonl 全量载入内存字典，供按 id 反查原文。"""
        if self._chunks is None:
            self._chunks = {}
            if CHUNKS_PATH.exists():
                with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            chunk = json.loads(line)
                            self._chunks[chunk["id"]] = chunk
            logger.info("内存条文库载入：%d 条", len(self._chunks))
        return self._chunks

    def _ensure_model(self):
        """加载嵌入模型，失败时抛出中文提示。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                start = time.time()
                self._model = SentenceTransformer(EMBED_MODEL)
                logger.info("嵌入模型加载完成：%s（%.2fs）", EMBED_MODEL, time.time() - start)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"嵌入模型 {EMBED_MODEL} 加载失败：请检查网络或改用本地模型路径"
                ) from exc
        return self._model

    def _ensure_collection(self):
        """连接 chroma 持久化 collection。"""
        if self._collection is None:
            import chromadb

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = client.get_collection(COLLECTION_NAME)
            logger.info("chroma collection 已连接：%s", COLLECTION_NAME)
        return self._collection

    def _ensure_bm25(self) -> List[str]:
        """读取 BM25 pickle；读取失败则从 chunks.jsonl 自动重建并覆盖落盘。"""
        if self._bm25 is None:
            try:
                with BM25_PATH.open("rb") as handle:
                    self._bm25_ids, self._bm25 = pickle.load(handle)
                logger.info("BM25 载入成功：%d 篇", len(self._bm25_ids))
            except Exception as exc:  # noqa: BLE001
                logger.warning("BM25 载入失败（%s），从 chunks.jsonl 重建", exc)
                from src.build_index import build_bm25

                self._bm25_ids, self._bm25 = build_bm25(list(self._load_chunks().values()))
        return self._bm25_ids

    def vector_search(self, query: str, k: int = TOPK_VEC) -> List[Tuple[str, float]]:
        """向量检索，返回 [(chunk_id, 余弦相似度)]，按分数降序。"""
        start = time.time()
        model = self._ensure_model()
        collection = self._ensure_collection()
        total = collection.count()
        if total == 0:
            logger.warning("向量库为空，请先执行 python -m src.build_index")
            return []
        embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
        result = collection.query(
            query_embeddings=[embedding], n_results=min(k, total), include=["distances"]
        )
        pairs = [
            (chunk_id, float(1.0 - distance))
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0])
        ]
        logger.info("向量检索命中 %d 条，耗时 %.3fs", len(pairs), time.time() - start)
        return pairs

    def bm25_search(
        self, query: str, k: int = TOPK_BM25, expand: bool = True
    ) -> List[Tuple[str, float]]:
        """BM25 关键词检索，返回 [(chunk_id, bm25 分数)]，按分数降序。

        expand=True 时先做同义词/关键词扩展（只扩展查询，不改法条原文）。
        """
        import jieba

        from src.query_expand import build_query_tokens

        jieba.setLogLevel(logging.WARNING)
        start = time.time()
        ids = self._ensure_bm25()
        if expand:
            base_tokens, extra_tokens = build_query_tokens(query)
            tokens = base_tokens + extra_tokens
        else:
            for word in DOMAIN_WORDS:
                jieba.add_word(word)
            tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        pairs = [(ids[i], float(scores[i])) for i in ranked]
        logger.info(
            "BM25 检索命中 %d 条（expand=%s），耗时 %.3fs", len(pairs), expand, time.time() - start
        )
        return pairs

    @staticmethod
    def rrf_fuse(rank_lists: Sequence[Sequence[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
        """RRF 融合多路排名：score = Σ 1/(k + rank + 1)，rank 从 0 开始。"""
        scores: Dict[str, float] = {}
        for rank_list in rank_lists:
            for rank, chunk_id in enumerate(rank_list):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def retrieve(
        self,
        query: str,
        topk_final: int = TOPK_FINAL,
        mode: str = "hybrid",
        expand: bool = True,
    ) -> List[Dict]:
        """统一检索入口；mode 取 vector / bm25 / hybrid，expand 只影响关键词侧。"""
        if mode not in ("vector", "bm25", "hybrid"):
            raise ValueError(f"不支持的检索模式：{mode}，可选 vector / bm25 / hybrid")
        start = time.time()
        chunks = self._load_chunks()
        if mode == "vector":
            ranked = self.vector_search(query, TOPK_VEC)
        elif mode == "bm25":
            ranked = self.bm25_search(query, TOPK_BM25, expand=expand)
        else:
            vector_ids = [cid for cid, _ in self.vector_search(query, TOPK_VEC)]
            bm25_ids = [cid for cid, _ in self.bm25_search(query, TOPK_BM25, expand=expand)]
            ranked = self.rrf_fuse([vector_ids, bm25_ids], RRF_K)
        results: List[Dict] = []
        for chunk_id, score in ranked[:topk_final]:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                logger.warning("检索命中未知 chunk_id：%s", chunk_id)
                continue
            results.append(
                {
                    "id": chunk["id"],
                    "law": chunk["law"],
                    "article_no": chunk["article_no"],
                    "chapter": chunk.get("chapter", ""),
                    "text": chunk["text"],
                    "score": round(float(score), 6),
                }
            )
        logger.info(
            "检索完成 mode=%s 返回 %d 条，总耗时 %.3fs", mode, len(results), time.time() - start
        )
        return results


def get_retriever() -> Retriever:
    """获取全局检索器单例（唯一的模块级入口）。"""
    return Retriever.instance()
