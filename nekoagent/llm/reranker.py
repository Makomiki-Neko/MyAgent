"""HuggingFace Reranker（CrossEncoder）懒加载。

- 从 config.auxiliary_models.reranker_model_path 加载本地 HF 重排模型。
- 单例缓存，热更新需 `clear_cache()`。
- 失败时抛 LLMConfigError。
- 对外暴露统一接口 `rerank(query, documents, top_k)` 返回 (doc, score) 列表。
"""

from __future__ import annotations

from typing import Any

from nekoagent.config.loader import Config, get_config
from nekoagent.observability.exceptions import LLMConfigError
from nekoagent.observability.logging import get_logger

_log = get_logger("llm.reranker")

_reranker: Any = None


def get_reranker(cfg: Config | None = None):
    """返回缓存的 sentence-transformers CrossEncoder 实例。"""

    global _reranker
    if _reranker is not None:
        return _reranker
    cfg = cfg or get_config()
    aux = cfg.auxiliary_models
    if not aux.reranker_model_path:
        raise LLMConfigError("config.auxiliary_models.reranker_model_path 未配置")

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise LLMConfigError(
            "未安装 sentence-transformers，无法加载 Reranker",
            cause=exc,
        ) from exc

    try:
        _reranker = CrossEncoder(aux.reranker_model_path, device=aux.reranker_device)
    except Exception as exc:
        raise LLMConfigError(
            f"无法加载 Reranker 模型 {aux.reranker_model_path}：{exc}",
            cause=exc,
        ) from exc

    _log.info("Reranker 已加载：%s (device=%s)", aux.reranker_model_path, aux.reranker_device)
    return _reranker


def rerank(query: str, documents: list[str], top_k: int | None = None, cfg: Config | None = None) -> list[tuple[int, float]]:
    """对 documents 按 query 重排。返回 [(doc_index, score)] 列表，分数归一化到 [0,1]。

    - 使用 sigmoid 将 logits 归一化以匹配 rag.similarity_threshold 门槛。
    - top_k=None 时返回全量排序；否则截断。
    """

    reranker = get_reranker(cfg)
    if not documents:
        return []

    pairs = [(query, doc) for doc in documents]
    try:
        scores = reranker.predict(pairs, batch_size=cfg.auxiliary_models.rerank_batch_size if cfg else 16)
    except TypeError:
        scores = reranker.predict(pairs)
    import numpy as np
    arr = np.asarray(scores, dtype="float32").reshape(-1)
    # sigmoid 归一化
    arr = 1.0 / (1.0 + np.exp(-arr))

    ranked = sorted(enumerate(arr.tolist()), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def clear_cache() -> None:
    global _reranker
    _reranker = None