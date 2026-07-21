"""Chroma 向量存储：用户记忆向量化 + 持久化本地文件夹。

- 嵌入模型由 `llm.embeding.get_embedder()` 提供（HuggingFaceEmbeddings）。
- 集合名取自 `config.memory.user_memory_namespace`。
- 落盘路径取自 `config.rag.chroma_persist_dir`。
- 增量 add / 按 doc_id 删除 / 重建。
"""

from __future__ import annotations

from typing import Any

from nekoagent.config.loader import Config, get_config
from nekoagent.llm.embedding import get_embedder
from nekoagent.observability.exceptions import LLMConfigError, RAGError
from nekoagent.observability.logging import get_logger

_log = get_logger("memory.rag.chroma")


class ChromaMemoryStore:
    """Thin adapter over LangChain Chroma store. doc_id 唯一标识一条记忆片段。"""

    def __init__(self, persist_dir: str, namespace: str) -> None:
        self.persist_dir = persist_dir
        self.namespace = namespace
        self._store: Any | None = None

    def _ensure_store(self) -> Any:
        if self._store is not None:
            return self._store
        try:
            from langchain_chroma import Chroma  # type: ignore
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma  # type: ignore
            except ImportError as exc:
                raise RAGError(
                    "未安装 langchain-chroma 或 langchain-community；无法初始化 Chroma",
                    cause=exc,
                ) from exc
        try:
            embedder = get_embedder()
        except LLMConfigError as exc:
            # 嵌入模型路径缺失/加载失败 → 转为 RAG 友好降级
            raise RAGError(
                "无法加载嵌入模型：" + exc.friendly_message,
                cause=exc,
            ) from exc
        try:
            import os
            os.makedirs(self.persist_dir, exist_ok=True)
            self._store = Chroma(
                collection_name=self.namespace,
                embedding_function=embedder,
                persist_directory=self.persist_dir,
            )
        except Exception as exc:
            raise RAGError(f"Chroma 初始化失败: {exc}", cause=exc) from exc
        return self._store

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        store = self._ensure_store()
        # 同 doc_id 强制唯一：先尝试删除旧条目
        self.remove(doc_id)
        try:
            store.add_texts(
                texts=[text],
                metadatas=[{"doc_id": doc_id, **(metadata or {})}],
                ids=[doc_id],
            )
        except Exception as exc:
            raise RAGError(f"Chroma.add 失败: {exc}", cause=exc) from exc

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """返回 [(doc_id, similarity_score)]，按相似度降序。"""

        store = self._ensure_store()
        try:
            results = store.similarity_search_with_score(query, k=top_k)
        except Exception as exc:
            raise RAGError(f"Chroma.search 失败: {exc}", cause=exc) from exc
        out: list[tuple[str, float]] = []
        for doc, score in results:
            metadata = getattr(doc, "metadata", {}) or {}
            doc_id = metadata.get("doc_id") or getattr(doc, "id", None) or metadata.get("id")
            if doc_id is None:
                continue
            # langchain_chroma 返回的 score 是距离（越小越相似），归一化为相似度
            sim = 1.0 / (1.0 + float(score))
            out.append((str(doc_id), sim))
        return out

    def remove(self, doc_id: str) -> None:
        if self._store is None:
            return
        try:
            self._store.delete(ids=[doc_id])
        except Exception:
            # doc_id 不存在时 Chroma 会抛错，忽略
            pass

    def clear(self) -> None:
        try:
            store = self._ensure_store()
            from chromadb import PersistentClient  # type: ignore
            client = PersistentClient(path=self.persist_dir)
            try:
                client.delete_collection(self.namespace)
            except Exception:
                pass
            self._store = None
        except ImportError:
            _log.warning("chromadb 未安装，无法清空集合 %s", self.namespace)
        except Exception as exc:
            _log.warning("Chroma.clear 失败：%s", exc)


_chroma_singleton: ChromaMemoryStore | None = None


def get_chroma(cfg: Config | None = None) -> ChromaMemoryStore:
    global _chroma_singleton
    if _chroma_singleton is None:
        cfg = cfg or get_config()
        _chroma_singleton = ChromaMemoryStore(
            persist_dir=cfg.rag.chroma_persist_dir,
            namespace=cfg.memory.user_memory_namespace,
        )
    return _chroma_singleton


def clear_cache() -> None:
    global _chroma_singleton
    _chroma_singleton = None