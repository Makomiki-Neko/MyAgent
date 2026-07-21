"""RAG 流水线编排 + 注入决策 + 重建索引。

`rag_search(query, cfg)` 返回 [(doc_id, text, score)] 已排序、阈值过滤。
`inject_or_skip(messages, query, cfg)` 在 messages 列表前部注入 system 记忆片段（仅当达到门槛）。
`rebuild_index(cfg)` 切换嵌入模型后重建索引。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from nekoagent.config.loader import Config, get_config
from nekoagent.llm import embedding as embedding_module
from nekoagent.llm import reranker as reranker_module
from nekoagent.memory.rag.bm25 import get_bm25
from nekoagent.memory.rag.chroma_store import get_chroma
from nekoagent.memory.rag.rrf import reciprocal_rank_fusion
from nekoagent.observability.exceptions import RAGError
from nekoagent.observability.logging import get_logger

_log = get_logger("memory.rag.pipeline")


def _metadata_text_by_id(doc_id: str, cfg: Config | None = None) -> str:
    """从 Chroma 集合中按 id 反查文档全文。

    用于把 Reranker 排序后的 doc_id 映射回原文以注入对话上下文。
    """

    store = get_chroma(cfg)
    try:
        col = store._ensure_store()
        # langchain Chroma 提供 _collection.get(ids=...)
        col_collection = getattr(col, "_collection", None)
        if col_collection is None:
            return ""
        result = col_collection.get(ids=[doc_id], include=["documents", "metadatas"])
        docs = result.get("documents") if result else None
        if docs:
            return docs[0] or ""
    except Exception as exc:  # 内部元数据回查失败，不阻塞主流程
        _log.warning("Chroma 反查 doc_id=%s 失败: %s", doc_id, exc)
    return ""


def rag_search(query: str, cfg: Config | None = None) -> list[tuple[str, str, float]]:
    """返回 [(doc_id, text, reranker_score)] 按 Reranker 分数降序，仅含达到相似度门槛且在 Top-K 内。"""

    cfg = cfg or get_config()
    rag_cfg = cfg.rag
    if not query or not query.strip():
        return []

    bm25_results: list[tuple[str, float]] = []
    chroma_results: list[tuple[str, float]] = []
    try:
        bm25 = get_bm25(cfg)
        bm25_results = bm25.search(query, top_k=rag_cfg.top_k_before_rerank)
    except RAGError as exc:
        _log.warning("BM25 检索不可用，本轮仅有语义路径：%s", exc.friendly_message)
    try:
        chroma = get_chroma(cfg)
        chroma_results = chroma.search(query, top_k=rag_cfg.top_k_before_rerank)
    except RAGError as exc:
        _log.warning("Chroma 检索不可用，本轮仅有关键词路径：%s", exc.friendly_message)

    if not bm25_results and not chroma_results:
        return []

    fused = reciprocal_rank_fusion(bm25_results, chroma_results, k=rag_cfg.rrf_k)
    if not fused:
        return []

    candidate_ids = [doc_id for doc_id, _ in fused[: rag_cfg.top_k_before_rerank]]
    candidate_texts = [(cid, _metadata_text_by_id(cid, cfg)) for cid in candidate_ids]
    candidate_texts = [(cid, t) for cid, t in candidate_texts if t]
    if not candidate_texts:
        # Chroma 不可用时无法反查 doc_id → 文本。优雅跳过（避免注入空文本）
        return []

    try:
        ranked_pairs = reranker_module.rerank(
            query=query,
            documents=[t for _, t in candidate_texts],
            top_k=rag_cfg.top_k_after_rerank,
            cfg=cfg,
        )
    except RAGError as exc:
        _log.warning("Reranker 不可用，回退 RRF 顺序（分数按 rank 归一化）：%s", exc.friendly_message)
        # 用 RRF 排名本身作为分数：第一名=1.0，依次衰减，使低排序自然过不去阈值
        ranked_pairs = [(i, 1.0 / (1.0 + i)) for i in range(min(rag_cfg.top_k_after_rerank, len(candidate_texts)))]

    out: list[tuple[str, str, float]] = []
    for idx, score in ranked_pairs:
        if idx >= len(candidate_texts):
            continue
        doc_id, text = candidate_texts[idx]
        if score < rag_cfg.similarity_threshold:
            continue
        out.append((doc_id, text, float(score)))
    return out


def inject_or_skip(messages: list[BaseMessage], query: str, cfg: Config | None = None) -> list[BaseMessage]:
    """达到阈值时在 messages 前部注入记忆 system 消息；否则原样返回。"""

    cfg = cfg or get_config()
    if not cfg.rag.inject_when_above_threshold:
        return messages
    results = rag_search(query, cfg)
    if not results:
        return messages
    preview = "\n\n".join(f"- {text}" for _, text, score in results)
    injection_text = (
        "[长期记忆参考 - 以下为已检索到的历史信息，请谨慎使用]\n"
        f"{preview}\n"
        "[长期记忆参考结束]"
    )
    return [SystemMessage(content=injection_text), *messages]


def rebuild_index(cfg: Config | None = None) -> None:
    """切换嵌入/Reranker 模型或主 Agent 记忆整理后调用：清空并按 main_store 重建。"""

    cfg = cfg or get_config()
    embedding_module.clear_cache()
    reranker_module.clear_cache()
    from nekoagent.memory.rag.bm25 import clear_cache as bm25_clear
    from nekoagent.memory.rag.chroma_store import clear_cache as chroma_clear
    bm25_clear()
    chroma_clear()

    bm25 = get_bm25(cfg)
    chroma = get_chroma(cfg)
    bm25.clear()
    chroma.clear()

    # 重新拉取主 Agent 所有未归档总结消息作为重建源
    try:
        from nekoagent.memory.stores import get_user_memory_store
        store = get_user_memory_store()
        rows = store.query_recent(limit=10000)
    except Exception as exc:
        raise RAGError(f"索引重建读取主 Agent 记忆失败: {exc}", cause=exc) from exc

    for row in rows:
        if row.get("archived"):
            continue
        doc_id = f"main_msg_{row['id']}"
        text = row.get("content", "")
        if not text:
            continue
        bm25.add(doc_id, text)
        try:
            chroma.add(doc_id, text, metadata={"session_id": row.get("session_id"), "created_at": str(row.get("created_at"))})
        except RAGError as exc:
            _log.warning("Chroma 写入 skipped（doc_id=%s）：%s", doc_id, exc.friendly_message)
            break

    _log.info("RAG 索引重建完成，bm25 docs=%s", len(bm25.all_ids()))


def clear_caches() -> None:
    """强制释放单例；热更新或单测用。"""

    from nekoagent.memory.rag.bm25 import clear_cache as bm25_clear
    from nekoagent.memory.rag.chroma_store import clear_cache as chroma_clear
    bm25_clear()
    chroma_clear()