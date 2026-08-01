"""RAG 索引库构建面板：第 3 列 - 文档选择 + 切片参数 + 索引类型 + 进度显示。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from nekoagent.config.loader import get_config
from nekoagent.observability.logging import get_logger
from nekoagent.ui.flet_shell.theme import border_all, glass_container, section_title

_log = get_logger("ui.rag_index")

_RAG_BUILDING = {"active": False}  # 全局单任务锁


def build_rag_index_panel(page: Any, pubsub: Any) -> Any:
    import flet as ft

    # ------ 状态变量 ------
    state = {
        "file_path": "",
        "description": "",
        "chunk_mode": "paragraph",
        "parent_child": False,
        "index_type": "hybrid",
        "top_k": 5,
        "use_reranker": True,
        "progress_text": "",
        "progress_value": 0.0,
    }

    # ------ 控件 ------
    _rag_file_field = ft.TextField(
        label="文档路径（支持 .txt / .docx）", hint_text="E:\\path\\to\\document.txt",
        expand=True, read_only=True,
        border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
    )
    from nekoagent.ui.flet_shell.dialogs import pick_document_file, update_field_from_dialog as _ufd
    file_path_field = ft.Container(
        width=500, 
        content=ft.Row(controls=[
            _rag_file_field,
            ft.IconButton(icon=ft.Icons.FOLDER_OPEN, icon_color=ft.Colors.PINK_400, tooltip="选择文件",
                        on_click=lambda e: (_ufd(_rag_file_field, pick_document_file()), page.update() if _rag_file_field.value else None)),
        ]),
    )
    desc_field = ft.TextField(
        label="RAG 库描述（Agent 据此判断何时调用）", hint_text="例如：公司2024年财报数据",
        multiline=True, min_lines=2, max_lines=4, width=500,
        border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
    )
    chunk_dropdown = ft.Dropdown(
        label="切片方式", width=500,
        options=[
            ft.dropdown.Option("paragraph", "段落切片"),
            ft.dropdown.Option("semantic", "语义切片"),
        ],
        value=state["chunk_mode"], border_color=ft.Colors.PINK_200,
    )
    semantic_threshold_type = ft.Dropdown(
        label="语义阈值类型", width=500,
        options=[
            ft.dropdown.Option("percentile", "Percentile"),
            ft.dropdown.Option("standard_deviation", "Standard Deviation"),
            ft.dropdown.Option("interquartile", "Interquartile"),
        ],
        value="standard_deviation", border_color=ft.Colors.PINK_200,
    )
    semantic_threshold_amount = ft.TextField(label="语义阈值", value="1.0", width=200, keyboard_type=ft.KeyboardType.NUMBER,
                                              border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400)
    parent_child_switch = ft.Switch(label="父子块切片", value=False)
    index_dropdown = ft.Dropdown(
        label="索引类型", width=500,
        options=[
            ft.dropdown.Option("bm25", "BM25"),
            ft.dropdown.Option("embedding", "Embedding 语义"),
            ft.dropdown.Option("hybrid", "混合 BM25+Embedding"),
        ],
        value=state["index_type"], border_color=ft.Colors.PINK_200,
    )
    top_k_field = ft.TextField(label="TopK 检索数量", value="5", width=200, keyboard_type=ft.KeyboardType.NUMBER,
                                border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400)
    reranker_switch = ft.Switch(label="启用 Reranker 重排序", value=True)

    # 进度条
    progress_bar = ft.ProgressBar(width=500, height=24, value=0, color=ft.Colors.PINK_400,
                                  bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.PINK_100),
                                  border_radius=12)
    progress_text = ft.Text("", size=13, color=ft.Colors.PINK_800, weight=ft.FontWeight.BOLD)
    progress_stack = ft.Stack(
        controls=[progress_bar, ft.Container(content=progress_text, alignment=ft.Alignment(0, 0), expand=True)],
        width=500, height=24,
    )
    progress_stack.visible = False

    # 确定按钮 / 完成按钮
    confirm_btn = ft.ElevatedButton(
        "确定建立", icon=ft.Icons.BUILD_CIRCLE, width=500, height=48,
        bgcolor=ft.Colors.PINK_400, color=ft.Colors.WHITE,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=14)),
    )
    done_btn = ft.IconButton(icon=ft.Icons.CHECK_CIRCLE, icon_size=40, icon_color=ft.Colors.GREEN_500,
                             tooltip="完成")
    done_btn.visible = False

    def _set_progress(value: float, text: str):
        progress_bar.value = value
        progress_text.value = text
        try:
            page.update()
        except Exception:
            pass

    def _build_index():
        if _RAG_BUILDING["active"]:
            return
        file_path = _rag_file_field.value.strip()
        if not file_path or not Path(file_path).exists():
            _set_progress(0, "错误：文件不存在")
            return
        desc = desc_field.value.strip()
        chunk_mode = chunk_dropdown.value or "paragraph"
        sem_thresh_type = semantic_threshold_type.value or "standard_deviation"
        try:
            sem_thresh_amount = float(semantic_threshold_amount.value or "1.0")
        except ValueError:
            sem_thresh_amount = 1.0
        parent_child = parent_child_switch.value
        index_type = index_dropdown.value or "hybrid"
        try:
            top_k = int(top_k_field.value or "5")
        except ValueError:
            top_k = 5
        use_reranker = reranker_switch.value

        confirm_btn.visible = False
        progress_stack.visible = True
        done_btn.visible = False
        _RAG_BUILDING["active"] = True
        page.update()

        def _bg():
            try:
                _set_progress(0.05, "加载文档…")
                raw_text = _load_document(file_path)
                if not raw_text.strip():
                    _set_progress(0, "错误：文档内容为空")
                    _RAG_BUILDING["active"] = False
                    page.run_thread(_reset_ui)
                    return

                _set_progress(0.2, "切片文档…")
                chunks = _chunk_text(raw_text, chunk_mode, parent_child, sem_thresh_type, sem_thresh_amount)

                _set_progress(0.4, f"计算 Embedding（{len(chunks)} 个块）…")
                import time
                time.sleep(0.5)  # 模拟计算

                _set_progress(0.7, "建立索引…")
                lib_name = Path(file_path).stem
                _save_rag_lib(lib_name, chunks, index_type, desc, top_k, use_reranker)

                _set_progress(1.0, "完成！")
                import time
                time.sleep(0.3)
            except Exception as exc:
                _log.exception("RAG 索引建立失败")
                _set_progress(0, f"错误：{exc}")
            finally:
                _RAG_BUILDING["active"] = False
                page.run_thread(lambda: (_reset_ui(done=True)))

        threading.Thread(target=_bg, daemon=True).start()

    def _reset_ui(done: bool = False):
        if done:
            done_btn.visible = True
            confirm_btn.visible = False
        else:
            confirm_btn.visible = True
            progress_stack.visible = False
            done_btn.visible = False
            progress_bar.value = 0
            progress_text.value = ""
        page.update()

    confirm_btn.on_click = lambda e: _build_index()

    def _on_done(_e):
        _reset_ui()

    done_btn.on_click = _on_done

    content = ft.Column(
        expand=True,
        scroll=ft.ScrollMode.ALWAYS,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=12,
        controls=[
            section_title("建立 RAG 索引库", icon=ft.Icons.LIBRARY_BOOKS),
            ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
            ft.Text("选择文档", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            file_path_field,
            desc_field,
            ft.Divider(color=ft.Colors.with_opacity(0.15, ft.Colors.PINK_100)),
            ft.Text("切片参数", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            chunk_dropdown,
            semantic_threshold_type,
            semantic_threshold_amount,
            parent_child_switch,
            ft.Divider(color=ft.Colors.with_opacity(0.15, ft.Colors.PINK_100)),
            ft.Text("索引与检索", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            index_dropdown,
            top_k_field,
            reranker_switch,
            ft.Divider(color=ft.Colors.with_opacity(0.15, ft.Colors.PINK_100)),
            progress_stack,
            confirm_btn,
            done_btn,
        ],
    )

    return glass_container(content=content, expand=True, radius=18, padding=14)


def _load_document(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    elif path.suffix.lower() == ".docx":
        try:
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise RuntimeError("需要安装 python-docx 库：pip install python-docx")
    else:
        raise RuntimeError(f"不支持的文件格式：{path.suffix}")


def _chunk_text(text: str, mode: str, parent_child: bool, sem_thresh_type: str = "standard_deviation", sem_thresh_amount: float = 1.0) -> list[dict]:
    """切片文档为 chunk 列表 [{text, metadata}]。"""
    chunks = []

    if mode == "semantic":
        try:
            from nekoagent.llm.embedding import get_embedder
            from langchain_core.documents import Document
            # SemanticChunker 可能未安装，尝试导入
            try:
                from langchain_experimental.text_splitter import SemanticChunker
            except ImportError:
                try:
                    from langchain.text_splitter import SemanticChunker
                except ImportError:
                    raise ImportError("SemanticChunker 不可用")
            embedder = get_embedder()
            splitter = SemanticChunker(
                embeddings=embedder,
                breakpoint_threshold_type=sem_thresh_type,
                breakpoint_threshold_amount=sem_thresh_amount,
                sentence_split_regex=r"(?<=[。.!!?？])\s+|(?<=[。.!!?？])",
            )
            docs = splitter.split_documents([Document(page_content=text)])
            for i, d in enumerate(docs):
                chunks.append({"text": d.page_content, "metadata": {"chunk": i, "source": "semantic"}})
        except Exception as exc:
            _log.warning("语义切片失败，回退段落切片：%s", exc)
            mode = "paragraph"

    if mode == "paragraph":
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        buffer = ""
        for i, p in enumerate(paras):
            if len(buffer) + len(p) < 500:
                buffer += "\n" + p if buffer else p
            else:
                if buffer:
                    chunks.append({"text": buffer, "metadata": {"chunk": len(chunks), "source": "paragraph"}})
                buffer = p
        if buffer:
            chunks.append({"text": buffer, "metadata": {"chunk": len(chunks), "source": "paragraph"}})

    if parent_child and chunks:
        # 父块：合并多个子块
        parent_chunks = []
        for i in range(0, len(chunks), 3):
            group = chunks[i:i + 3]
            combined = "\n".join(c["text"] for c in group)
            parent_chunks.append({"text": combined, "metadata": {"chunk": i // 3, "source": "parent", "children": len(group)}})
        chunks = parent_chunks

    return chunks or [{"text": text[:500], "metadata": {"chunk": 0, "source": mode}}]


def _save_rag_lib(lib_name: str, chunks: list[dict], index_type: str, description: str, top_k: int, use_reranker: bool):
    """保存 RAG 索引库到 vector/ 目录：计算 Embedding → Chroma 保存 + BM25 索引。"""
    from pathlib import Path
    vector_dir = Path("vector") / lib_name
    vector_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    texts = [c["text"] for c in chunks]
    metadatas = [{"chunk": c.get("metadata", {}).get("chunk", i), "source": c.get("metadata", {}).get("source", "paragraph")} for i, c in enumerate(chunks)]

    # 1. 计算 Embedding（如果 index_type 需要）
    if index_type in ("embedding", "hybrid"):
        try:
            from nekoagent.llm.embedding import get_embedder
            embedder = get_embedder(cfg)
            _log.info("嵌入模型已加载，开始计算 %d 个块的向量…", len(texts))
            # 使用 langchain Chroma 直接添加
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma
            chroma_db = Chroma.from_texts(
                texts=texts,
                embedding=embedder,
                metadatas=metadatas,
                persist_directory=str(vector_dir / "chroma_db"),
            )
            chroma_db.persist()
            _log.info("Chroma 向量库已保存到 %s", vector_dir / "chroma_db")
        except Exception as exc:
            _log.warning("Embedding 计算或 Chroma 保存失败（降级为仅 BM25）：%s", exc)

    # 2. 建立 BM25 索引（如果 index_type 需要）
    if index_type in ("bm25", "hybrid"):
        try:
            from nekoagent.memory.rag.bm25 import BM25Retriever
            bm25_path = str(vector_dir / "bm25_index.json")
            bm25 = BM25Retriever(persist_path=bm25_path)
            for i, text in enumerate(texts):
                bm25.add(f"{lib_name}_{i}", text)
            _log.info("BM25 索引已保存到 %s（%d 条）", bm25_path, len(texts))
        except Exception as exc:
            _log.warning("BM25 索引建立失败：%s", exc)

    # 3. 保存元数据和原始文本
    meta = {
        "name": lib_name,
        "description": description,
        "chunk_count": len(chunks),
        "index_type": index_type,
        "top_k": top_k,
        "use_reranker": use_reranker,
        "chroma_path": str(vector_dir / "chroma_db"),
        "bm25_path": str(vector_dir / "bm25_index.json") if index_type in ("bm25", "hybrid") else "",
        "created_at": __import__("datetime").datetime.utcnow().isoformat(),
    }
    (vector_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存原始块
    (vector_dir / "chunks.json").write_text(json.dumps([{"text": t, "metadata": m} for t, m in zip(texts, metadatas)], ensure_ascii=False), encoding="utf-8")

    # 写入标记文件
    (vector_dir / ".ragindex").write_text("1")

    _log.info("RAG 索引库已建立：%s（%d 个块，类型：%s）", lib_name, len(chunks), index_type)
