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

try:
    from langchain_core.tools import tool as _langchain_tool
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    _langchain_tool = None

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
    """在指定 RAG 库中检索 query，使用该库独立的 Chroma + BM25 索引。"""
    _log.info("收到RAG检索请求, Lib:%s , Q:%s.", lib_name, query[:80])
    lib_dir = _VECTOR_DIR / lib_name
    meta_file = lib_dir / "meta.json"
    if not meta_file.exists():
        return f"[RAG] 库 {lib_name} 不存在"
    try:
        import json as _json
        meta = _json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return f"[RAG] 库 {lib_name} 元数据读取失败"

    index_type = meta.get("index_type", "hybrid")
    top_k = meta.get("top_k", top_k)

    from nekoagent.memory.rag.rrf import reciprocal_rank_fusion as _rrf
    from langchain_core.documents import Document

    unified: list[tuple[str, str, float]] = []  # (doc_id, text, score)

    # ---- BM25 检索（按库独立索引）----
    if index_type in ("bm25", "hybrid"):
        bm25_path = lib_dir / "bm25_index.json"
        if bm25_path.exists():
            try:
                from nekoagent.memory.rag.bm25 import BM25Retriever
                bm25 = BM25Retriever(persist_path=str(bm25_path))
                for doc_id, score in bm25.search(query, top_k * 2):
                    unified.append((doc_id, doc_id, score))
            except Exception as exc:
                _log.debug("BM25 检索失败：%s", exc)

    # ---- Chroma 向量检索（按库独立目录加载）----
    if index_type in ("embedding", "hybrid"):
        chroma_dir = lib_dir / "chroma_db"
        if chroma_dir.exists() and any(chroma_dir.iterdir()):
            try:
                from nekoagent.llm.embedding import get_embedder as _ge
                from langchain_chroma import Chroma
                embedder = _ge()
                vector_db = Chroma(
                    persist_directory=str(chroma_dir),
                    collection_name="langchain",
                    embedding_function=embedder,
                )
                retriever = vector_db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": top_k * 2, "fetch_k": top_k * 3, "lambda_mult": 0.7},
                )
                docs: list[Document] = retriever.invoke(query)
                for doc in docs:
                    doc_id = doc.metadata.get("doc_id", doc.id if hasattr(doc, 'id') else str(hash(doc.page_content)))
                    unified.append((str(doc_id), doc.page_content, 1.0))
                _log.info("Chroma 检索成功: docs=%d", len(docs))
            except Exception as exc:
                _log.debug("Chroma 检索失败：%s", exc)

    if not unified:
        return f"[RAG] 在库 {lib_name} 中未找到相关内容"

    # 按分数降序排列
    unified.sort(key=lambda x: x[2], reverse=True)
    results = unified[:top_k]



    lines = [f"在 RAG 库「{lib_name}」中找到以下相关内容："]
    for i, (doc_id, text, score) in enumerate(results, 1):
        lines.append(f"\n{i}. (相关度 {score:.3f})\n{text[:500]}")
    _log.info(f"RAG 检索完成: lib={lib_name} results={len(results)} result={results}")
    return "\n".join(lines)


def _build_tool_description(lib_meta: dict) -> str:
    """为 RAG 库构建工具描述。"""
    name = lib_meta.get("name", "unknown")
    desc = lib_meta.get("description", "")
    return f"检索 RAG 知识库「{name}」：{desc}。当用户询问与「{desc}」有关时，调用此工具获取信息。输入为搜索关键词。"


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


def get_rag_tools_for_agent(agent_name: str = "main_agent") -> list:
    """返回 LangChain tool 对象列表。"""
    if not _HAS_LANGCHAIN:
        return []
    tools = []
    for name, meta in _registered_libs.items():
        desc = meta.get("description", f"检索 RAG 知识库 {name}")
        top_k = meta.get("top_k", 5)

        @_langchain_tool(description=f"检索 RAG 知识库「{name}」：{desc}。输入为搜索关键词。")
        def _search(query: str, _lib_name=name, _tk=top_k) -> str:
            """在知识库中搜索与 query 相关的内容"""
            return _search_library(_lib_name, query, _tk)

        _search.name = f"search_{name}"
        tools.append(_search)
    return tools


def bind_rag_tools_to_llm(llm, agent_name: str = "main_agent"):
    tools = get_rag_tools_for_agent(agent_name)
    if not tools:
        return llm
    try:
        return llm.bind_tools(tools)
    except Exception:
        return llm


def execute_rag_tool(func_name: str, raw_args) -> str:
    """执行 RAG 搜索工具调用，返回结果文本。供 Agent ReAct 循环使用。"""
    query = raw_args.get('query', '') if isinstance(raw_args, dict) else str(raw_args) if raw_args else ''
    _log.info("RAG检索: tool=%s query=%s", func_name, query[:100])
    for tname in _registered_libs:
        if func_name == f"search_{tname}":
            result = _search_library(tname, query, _registered_libs[tname].get("top_k", 5))
            _log.info("RAG检索完成: lib=%s result_len=%d", tname, len(result))
            return result
    _log.warning("未找到 RAG 工具: %s", func_name)
    return f"[RAG] 未找到工具 {func_name}"


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
