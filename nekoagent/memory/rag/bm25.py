"""BM25 关键词检索：jieba 中文分词 + rank-bm25，支持增量更新与落盘回放。

Fallback：若 jieba 未安装，使用按字符切分（对中文近似可用，召回较粗）。
BM25 索引以 JSON 落盘，重启时重建。
"""

from __future__ import annotations

import json
import os
from typing import Any

from nekoagent.config.loader import Config, get_config
from nekoagent.observability.exceptions import RAGError
from nekoagent.observability.logging import get_logger

_log = get_logger("memory.rag.bm25")


def _tokenize(text: str) -> list[str]:
    """优先 jieba 切词，缺失时退化为按字符切分。"""

    if not text:
        return []
    try:
        import jieba  # type: ignore
        return [t for t in jieba.lcut(text) if t and t.strip()]
    except ImportError:
        return [ch for ch in text if ch.strip()]


def _score_to_rank(docs: list[str], scores: list[float] | list[int]) -> list[tuple[int, float]]:
    """rank-bm25 返回分数，转为 (doc_index, normalized_score)。"""

    pairs = list(enumerate(float(s) for s in scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


class BM25Retriever:
    """单实例 BM25 索引，doc_id 由调用方提供；增量 add + persist 回放。"""

    def __init__(self, persist_path: str | None = None) -> None:
        self.persist_path = persist_path
        self._doc_ids: list[str] = []
        self._corpus: list[list[str]] = []   # 已分词的 tokens 列表
        self._bm25: Any | None = None
        if persist_path and os.path.exists(persist_path):
            self._load()

    def _rebuild(self) -> None:
        if not self._corpus:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RAGError("未安装 rank-bm25", cause=exc) from exc
        self._bm25 = BM25Okapi(self._corpus)

    def add(self, doc_id: str, text: str) -> None:
        # 同 doc_id 视为相同文档：若已存在则替换内容
        if doc_id in self._doc_ids:
            idx = self._doc_ids.index(doc_id)
            self._corpus[idx] = _tokenize(text)
        else:
            self._doc_ids.append(doc_id)
            self._corpus.append(_tokenize(text))
        self._rebuild()
        self._persist()

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._doc_ids:
            return
        idx = self._doc_ids.index(doc_id)
        self._doc_ids.pop(idx)
        self._corpus.pop(idx)
        self._rebuild()
        self._persist()

    def clear(self) -> None:
        self._doc_ids = []
        self._corpus = []
        self._bm25 = None
        self._persist()

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """返回 [(doc_id, score)]，按分数降序。"""

        if self._bm25 is None or not self._doc_ids:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(float(s) for s in scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._doc_ids[i], score) for i, score in ranked if score > 0]

    def all_ids(self) -> list[str]:
        return list(self._doc_ids)

    def _persist(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump({"doc_ids": self._doc_ids, "corpus": self._corpus}, f, ensure_ascii=False)
        except OSError as exc:
            _log.warning("BM25 索引落盘失败 %s: %s", self.persist_path, exc)

    def _load(self) -> None:
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._doc_ids = list(data.get("doc_ids", []))
            self._corpus = list(data.get("corpus", []))
            self._rebuild()
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("BM25 索引加载失败 %s: %s", self.persist_path, exc)
            self._doc_ids = []
            self._corpus = []


_bm25_singleton: BM25Retriever | None = None


def get_bm25(cfg: Config | None = None) -> BM25Retriever:
    global _bm25_singleton
    if _bm25_singleton is None:
        cfg = cfg or get_config()
        _bm25_singleton = BM25Retriever(persist_path=cfg.rag.bm25_index_path)
    return _bm25_singleton


def clear_cache() -> None:
    global _bm25_singleton
    _bm25_singleton = None