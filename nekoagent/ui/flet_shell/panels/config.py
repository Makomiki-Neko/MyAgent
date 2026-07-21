"""配置面板：可视化查看 / 修改 / 保存 / 重置 部分核心 YAML 项（MVP 范围：人设 + 阈值 + 模型路径）。"""

from __future__ import annotations

from typing import Any

from nekoagent.config.loader import get_config
from nekoagent.config.writeback import reset_persona_cache, save_config, save_persona
from nekoagent.llm.factory import clear_cache as llm_clear_cache
from nekoagent.observability.exceptions import format_friendly_error
from nekoagent.observability.logging import get_logger

_log = get_logger("ui.config")


def _show_snack(page: Any, message: str) -> None:
    try:
        import flet as ft
        snack = ft.SnackBar(ft.Text(message))
        page.overlay.append(snack)
        snack.open = True
        page.update()
    except Exception:
        try:
            page.update()
        except Exception:
            pass


def build_config_panel(page: Any, pubsub: Any, in_col2: bool = False) -> Any:
    import flet as ft
    from nekoagent.ui.flet_shell.theme import glass_container, section_title

    cfg = get_config()
    persona = cfg.get_persona_cached("main_agent")

    # 当配置作为第 2 列时使用更紧凑的字段宽度
    field_width = 220 if in_col2 else 800

    persona_field = ft.TextField(
        label="主 Agent 人设 system_prompt",
        value=persona.system_prompt,
        multiline=True,
        min_lines=4,
        max_lines=8,
        width=field_width if in_col2 else 800,
        border_color=ft.Colors.PINK_200,
        focused_border_color=ft.Colors.PINK_400,
    )
    threshold_field = ft.TextField(
        label="总结裁剪 Token 阈值",
        value=str(cfg.memory.summarize_token_threshold),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=(field_width - 10) if in_col2 else 200,
        border_color=ft.Colors.PINK_200,
        focused_border_color=ft.Colors.PINK_400,
    )
    rag_threshold_field = ft.TextField(
        label="RAG 相似度门槛",
        value=str(cfg.rag.similarity_threshold),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=(field_width - 10) if in_col2 else 200,
        border_color=ft.Colors.PINK_200,
        focused_border_color=ft.Colors.PINK_400,
    )
    embedding_path_field = ft.TextField(
        label="嵌入模型路径 (HuggingFace 本地)",
        value=str(cfg.auxiliary_models.embedding_model_path),
        width=field_width,
        border_color=ft.Colors.PINK_200,
        focused_border_color=ft.Colors.PINK_400,
    )

    def _save(_e: Any) -> None:
        try:
            new_threshold = int(threshold_field.value)
            new_rag_threshold = float(rag_threshold_field.value)
            if new_threshold <= 0:
                raise ValueError("总结 Token 阈值必须为正整数")
            if not (0.0 <= new_rag_threshold <= 1.0):
                raise ValueError("RAG 门槛需在 [0,1] 之间")

            # 1. 写回主 Agent persona yaml
            persona_dict = persona.model_dump()
            persona_dict["system_prompt"] = persona_field.value
            save_persona(cfg.agents["main_agent"].system_prompt_file, persona_dict)
            reset_persona_cache()

            # 2. 写回主 config.yaml
            import copy
            new_raw = copy.deepcopy(cfg.raw)
            new_raw.setdefault("memory", {})["summarize_token_threshold"] = new_threshold
            new_raw.setdefault("rag", {})["similarity_threshold"] = new_rag_threshold
            new_raw.setdefault("auxiliary_models", {})["embedding_model_path"] = embedding_path_field.value
            save_config(new_raw, cfg.source_path)
            reset_persona_cache()
            llm_clear_cache()
            _show_snack(page, "配置保存成功，已重新加载生效。")
            pubsub.notify("config_changed", {"saved_at": __import__("datetime").datetime.utcnow().isoformat()})
        except Exception as exc:
            _show_snack(page, format_friendly_error(exc))

    def _rebuild_index(_e: Any) -> None:
        try:
            from nekoagent.memory.rag.pipeline import rebuild_index
            rebuild_index()
            _show_snack(page, "RAG 索引已重建（如切换了嵌入模型，已完成向量重算）")
        except Exception as exc:
            _show_snack(page, format_friendly_error(exc))

    inner = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        expand=True,
        controls=[
            section_title("系统配置", icon=ft.Icons.SETTINGS),
            ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),

            ft.Text("🎭 主 Agent 人设", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            persona_field,

            ft.Text("⏱ 阈值", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            ft.Row([threshold_field, rag_threshold_field]) if not in_col2 else ft.Column([threshold_field, rag_threshold_field]),

            ft.Text("🤗 嵌入模型", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            embedding_path_field,

            ft.Row([
                ft.ElevatedButton(
                    "保存配置",
                    icon=ft.Icons.SAVE,
                    on_click=_save,
                    bgcolor=ft.Colors.PINK_400,
                    color=ft.Colors.WHITE,
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
                ),
                ft.OutlinedButton(
                    "重建 RAG 索引",
                    icon=ft.Icons.REFRESH,
                    on_click=_rebuild_index,
                    icon_color=ft.Colors.PINK_700,
                ),
            ]),

            ft.Container(
                content=ft.Text(
                    "提示：配置修改会写入 YAML 文件并以新值生效，重启后仍然保留。",
                    italic=True, size=10, color=ft.Colors.GREY_700,
                ),
                padding=ft.Padding(left=8, top=4, right=8, bottom=4),
                bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.WHITE),
                border_radius=8,
            ),
        ],
    )
    # 在 col2 中作为 holder 内可滚动玻璃卡片；当独立显示时也保持玻璃感
    return glass_container(
        content=inner,
        expand=True,
        radius=18,
        padding=12,
    )