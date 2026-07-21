"""Flet 桌面应用 main app - 4 列毛玻璃粉白渐变布局。

布局：
  Col1: 菜单（72px）  Col2: 会话列表（270px）  Col3: 对话/设置(expand=3)  Col4: 状态监视(310px)

点击 Col1 设置按钮 → Col3 在对话与设置之间切换。
"""

from __future__ import annotations

from typing import Any

from nekoagent.config.loader import get_config
from nekoagent.observability.exceptions import NekoAgentError, format_friendly_error
from nekoagent.observability.logging import get_logger

_log = get_logger("ui.flet_app")


def run_app() -> None:
    try:
        import flet as ft
    except ImportError as exc:
        raise NekoAgentError("未安装 Flet，请先 `pip install flet`", cause=exc) from exc

    from nekoagent.observability.logging import configure_logging
    cfg = get_config()
    configure_logging(cfg.logging_cfg.model_dump())

    try:
        from nekoagent.memory.mysql_checkpointer import ensure_tables
        ensure_tables(cfg)
    except Exception as exc:
        _log.warning("数据库表初始化失败（下游可能继续 graceful skip）：%s", exc)

    from nekoagent.queue import get_scheduler
    get_scheduler()
    from nekoagent.skills import get_skill_registry
    get_skill_registry(cfg)
    from nekoagent.rag_tools import register_rag_tools
    register_rag_tools()
    try:
        from nekoagent.queue import get_scheduler as _gs
        _gs().restore_pending_tasks()
    except Exception as exc:
        _log.warning("恢复任务状态失败（忽略）：%s", exc)

    def main(page: ft.Page) -> None:
        from nekoagent.ui.flet_shell.panels.chat import build_chat_panel
        from nekoagent.ui.flet_shell.panels.menu import build_menu_panel
        from nekoagent.ui.flet_shell.panels.sessions import build_sessions_panel
        from nekoagent.ui.flet_shell.panels.settings import build_settings_panel
        from nekoagent.ui.flet_shell.panels.status import build_status_panel
        from nekoagent.ui.flet_shell.pubsub import UIPubSub
        from nekoagent.ui.flet_shell.theme import page_gradient_background

        page.title = "NekoAgent"
        page.scroll = None
        page.padding = 0
        page.window_min_width = 1280
        page.window_min_height = 720
        # 默认窗口大小 = 屏幕 2/3，居中
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            root.destroy()
        except Exception:
            sw, sh = 1920, 1080
        win_w = int(sw * 2 / 3)
        win_h = int(sh * 2 / 3)
        page.window_width = win_w
        page.window_height = win_h
        page.window_left = max(0, (sw - win_w) // 2)
        page.window_top = max(0, (sh - win_h) // 2)
        page.update()
        page.theme = ft.Theme(
            color_scheme_seed=ft.Colors.PINK,
            color_scheme=ft.ColorScheme(primary=ft.Colors.PINK_400, on_primary=ft.Colors.WHITE, secondary=ft.Colors.PURPLE_300),
        )
        page.bgcolor = ft.Colors.with_opacity(0.92, ft.Colors.WHITE)

        pubsub = UIPubSub(page)

        # ---- 任务主管事件 → UI ----
        try:
            from nekoagent.queue import get_scheduler as _get_sched
            _sched = _get_sched()
            _sched.on_new_plan(lambda _task: page.run_thread(
                lambda: pubsub.notify("plan_ready", {
                    "thread_id": _task.thread_id,
                    "plan_text": getattr(_task, "plan_presented_to_user_text", ""),
                })
            ))
            _sched.on_decision_made(lambda _task: page.run_thread(
                lambda: pubsub.notify("decision_made", {
                    "thread_id": _task.thread_id,
                    "task_id": _task.task_id,
                    "status": _task.status.value,
                    "summary": _task.task_summary,
                })
            ))
            _sched.on_complete(lambda _task: page.run_thread(
                lambda: pubsub.notify("task_complete", {
                    "thread_id": _task.thread_id,
                    "result_text": getattr(_task, "final_result_presented", getattr(_task, "final_result", "")),
                })
            ))
            _sched.on_subtask_change(lambda _task: page.run_thread(
                lambda: pubsub.notify("subtask_change", {
                    "thread_id": _task.thread_id,
                    "subtask": getattr(_task, "current_subtask", ""),
                })
            ))
            _sched.on_info_request(lambda _task: page.run_thread(
                lambda: pubsub.notify("info_request", {
                    "thread_id": _task.thread_id,
                    "question": getattr(_task, "info_request", ""),
                })
            ))
        except Exception:
            pass

        # ---- 当前 Session ----
        active_session_id = {"value": ""}
        from nekoagent.session import list_sessions, create_session
        try:
            sessions = list_sessions()
            if sessions:
                active_session_id["value"] = sessions[0].session_id
            else:
                active_session_id["value"] = create_session("新会话").session_id
        except Exception:
            pass

        # ---- Col 3: 对话 / 设置 切换 ----
        col3_holder = ft.Container(expand=True)
        _cached_chat = {"panel": None}
        _history_loader = {"fn": None}

        def _show_chat():
            if _cached_chat["panel"] is None:
                _cached_chat["panel"] = build_chat_panel(
                    page, pubsub, lambda: active_session_id["value"],
                    history_loader=_history_loader,
                )
            col3_holder.content = _cached_chat["panel"]
            page.update()
            sid = active_session_id["value"]
            if sid and _history_loader["fn"]:
                _history_loader["fn"](sid)

        def _show_settings():
            col3_holder.content = build_settings_panel(page, pubsub)
            page.update()

        _show_chat()

        # ---- 第 2 列：sessions（不变） ----
        def _on_session_switch(new_session):
            active_session_id["value"] = new_session.session_id
            try:
                pubsub.notify("session_switch", {"session_id": new_session.session_id})
            except Exception:
                pass

        refresh_sig: dict = {}
        # 注意：chat_log_ref 通过 pubsub 里的 ref 拿到已创建的 chat 控件
        col2_content = build_sessions_panel(
            page=page, on_switch_session=_on_session_switch,
            refresh_signal=refresh_sig, pubsub=pubsub,
            history_loader=_history_loader,
        )

        # ---- toggle 逻辑 ----
        _col3_mode = "chat"  # chat | settings | rag

        def _show_rag():
            from nekoagent.ui.flet_shell.panels.rag_index import build_rag_index_panel
            col3_holder.content = build_rag_index_panel(page, pubsub)
            page.update()

        def _toggle_settings(force_sessions: bool | str | None = None):
            nonlocal _col3_mode
            if force_sessions is True:
                _show_chat()
                _col3_mode = "chat"
            elif force_sessions is False:
                _show_settings()
                _col3_mode = "settings"
            elif force_sessions == "rag":
                _show_rag()
                _col3_mode = "rag"
            else:
                if _col3_mode == "chat":
                    _show_settings()
                    _col3_mode = "settings"
                elif _col3_mode == "rag":
                    _show_chat()
                    _col3_mode = "chat"
                else:
                    _show_chat()
                    _col3_mode = "chat"

        # ---- Col 1: 菜单 ----
        menu_col = build_menu_panel(_toggle_settings, page, pubsub=pubsub)

        # ---- Col 4: 状态 ----
        status_col = build_status_panel(page, pubsub, lambda: active_session_id["value"])

        # ---- 整体布局 ----
        body = ft.Row(
            expand=True,
            alignment=ft.MainAxisAlignment.START,
            spacing=12,
            controls=[
                ft.Container(content=menu_col, width=110, expand=True),
                ft.Container(content=col2_content, width=270, expand=True),
                ft.Container(content=col3_holder, expand=3),
                ft.Container(content=status_col, width=310, expand=True),
            ],
        )

        page.add(
            ft.Stack(
                expand=True,
                controls=[
                    page_gradient_background(),
                    ft.Container(content=body, padding=12, expand=True),
                ],
            )
        )

        pubsub.notify("init", {"message": "NekoAgent 已就绪 ✨"})

    try:
        ft.app(target=main)
    except Exception as exc:
        raise NekoAgentError("UI 启动失败：" + str(exc), cause=exc) from exc


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