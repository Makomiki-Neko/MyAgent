"""HuggingFace 嵌入模型懒加载。

- 从 config.auxiliary_models.embedding_model_path 加载本地 HF 嵌入模型。
- 单例缓存，热更新需 `clear_cache()`。
- 失败时抛 LLMConfigError（友好文案 + dev log）。
"""

from __future__ import annotations

from typing import Any

from nekoagent.config.loader import Config, get_config
from nekoagent.observability.exceptions import LLMConfigError
from nekoagent.observability.logging import get_logger

_log = get_logger("llm.embedding")

_embedder: Any = None


def get_embedder(cfg: Config | None = None):
    """返回缓存的 HuggingFace 嵌入模型封装（LangChain HuggingFaceEmbeddings 兼容）。"""

    global _embedder
    if _embedder is not None:
        return _embedder
    cfg = cfg or get_config()
    aux = cfg.auxiliary_models
    if not aux.embedding_model_path:
        raise LLMConfigError("config.auxiliary_models.embedding_model_path 未配置")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        except ImportError as exc:
            raise LLMConfigError(
                "未安装 langchain-huggingface / langchain-community 嵌入组件，无法加载 HuggingFace 嵌入模型",
                cause=exc,
            ) from exc

    try:
        _embedder = HuggingFaceEmbeddings(
            model_name=aux.embedding_model_path,
            model_kwargs={"device": aux.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise LLMConfigError(
            f"无法加载嵌入模型 {aux.embedding_model_path}：{exc}",
            cause=exc,
        ) from exc

    _log.info("嵌入模型已加载：%s (device=%s)", aux.embedding_model_path, aux.embedding_device)
    return _embedder


def clear_cache() -> None:
    global _embedder
    _embedder = None