"""RAG 库自动扫描 + Agent 工具注册。

启动时扫描 vector/ 目录下所有带 .ragindex 标记的子目录，
为每个 RAG 库生成一个工具函数，按配置检索 Chroma/BM25，
并将工具注册到主 Agent 和执行子 Agent 的可用工具列表中。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nekoagent.config.loader import Config, get_config
from nekoagent.observability.logging import get_logger

_log = get_logger("rag_tools")

_VECTOR_DIR = Path("vector")
_registered_libs: dict[str, dict] = {}


def scan_rag_libraries() -> list[dict]:
    """扫描 vector/ 目录，返回所有 RAG 库的元信息列表。"""
    libs = []
    if not _VECTOR_DIR.exists():
        return libs
    for sub in _VECTOR_DIR.iterdir():
        if not sub.is_dir():
            continue
        marker = sub / ".ragindex"
        if not marker.exists():
            continue
        meta_file = sub / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta["_path"] = str(sub.resolve())
            libs.append(meta)
        except Exception as exc:
            _log.warning("RAG 库元数据读取失败 %s: %s", sub, exc)
    return libs


def _search_library(lib_name: str, query: str, top_k: int = 5) -> str:
    """在指定 RAG 库中检索 query，返回格式化文本。"""
    try:
        from nekoagent.memory.rag.bm25 import get_bm25
        from nekoagent.memory.rag.chroma_store import get_chroma
        from nekoagent.memory.rag.rrf import reciprocal_rank_fusion
    except Exception as exc:
        return f"[RAG] 检索模块加载失败：{exc}"

    cfg = get_config()
    lib_dir = _VECTOR_DIR / lib_name
    meta_file = lib_dir / "meta.json"
    if not meta_file.exists():
        return f"[RAG] 库 {lib_name} 不存在"

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return f"[RAG] 库 {lib_name} 元数据读取失败"

    index_type = meta.get("index_type", "hybrid")
    use_reranker = meta.get("use_reranker", True)
    top_k = meta.get("top_k", top_k)

    bm25_results = []
    chroma_results = []

    if index_type in ("bm25", "hybrid"):
        try:
            bm25 = get_bm25(cfg)
            bm25_results = bm25.search(query, k=top_k * 2)
        except Exception as exc:
            _log.debug("BM25 检索失败：%s", exc)

    if index_type in ("embedding", "hybrid"):
        try:
            chroma = get_chroma(cfg)
            # 用 collection name 区分不同库
            chroma_results = chroma.similarity_search_with_score(query, k=top_k * 2)
        except Exception as exc:
            _log.debug("Chroma 检索失败：%s", exc)

    if index_type == "hybrid":
        fused = reciprocal_rank_fusion(bm25_results, chroma_results)
        results = fused[:top_k]
    elif index_type == "bm25":
        results = bm25_results[:top_k]
    else:
        results = chroma_results[:top_k]

    if not results:
        return f"[RAG] 在库 {lib_name} 中未找到相关内容"

    lines = [f"在 RAG 库「{lib_name}」中找到以下相关内容："]
    for i, (doc_id, text, score) in enumerate(results, 1):
        lines.append(f"\n{i}. (相关度 {score:.3f})\n{text[:300]}")

    return "\n".join(lines)


def _build_tool_description(lib_meta: dict) -> str:
    """为 RAG 库构建工具描述。"""
    name = lib_meta.get("name", "unknown")
    desc = lib_meta.get("description", "")
    return f"检索 RAG 知识库「{name}」：{desc}。当用户询问与「{name}」相关内容时，调用此工具获取信息。输入为搜索关键词。"


def register_rag_tools() -> None:
    """扫描并注册所有 RAG 库工具到 Agent。"""
    libs = scan_rag_libraries()
    if not libs:
        _log.info("未发现 RAG 库，跳过工具注册")
        return

    for lib in libs:
        name = lib.get("name", "")
        if not name:
            continue
        _registered_libs[name] = lib
        _log.info("已注册 RAG 工具：%s - %s", name, lib.get("description", ""))

    # 注入 _search_library 作为 Agent 可用函数（通过 bind_tools / tool 机制）
    # 各 Agent 可在 system prompt 中引用这些库
    _log.info("共注册 %d 个 RAG 工具", len(libs))


def get_search_tool(lib_name: str) -> Any:
    """获取指定 RAG 库的搜索函数。"""
    if lib_name not in _registered_libs:
        return None
    meta = _registered_libs[lib_name]

    def _search(query: str) -> str:
        return _search_library(lib_name, query, meta.get("top_k", 5))

    _search.__name__ = f"search_{lib_name}"
    _search.__doc__ = _build_tool_description(meta)
    return _search


def get_all_search_tools() -> list[tuple[str, Any, str]]:
    """返回所有已注册 RAG 工具 (name, func, description)。"""
    result = []
    for name, meta in _registered_libs.items():
        func = get_search_tool(name)
        if func:
            result.append((name, func, _build_tool_description(meta)))
    return result


def delete_library(lib_name: str) -> bool:
    """删除指定 RAG 库。"""
    lib_dir = _VECTOR_DIR / lib_name
    if not lib_dir.exists():
        return False
    import shutil
    shutil.rmtree(lib_dir)
    _registered_libs.pop(lib_name, None)
    _log.info("RAG 库已删除：%s", lib_name)
    return True


def update_library_meta(lib_name: str, description: str) -> bool:
    """更新 RAG 库的描述。"""
    lib_dir = _VECTOR_DIR / lib_name
    meta_file = lib_dir / "meta.json"
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta["description"] = description
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if lib_name in _registered_libs:
            _registered_libs[lib_name]["description"] = description
        return True
    except Exception:
        return False
