"""会话列表面板：第 2 列 - 从 DB 加载历史会话 + 双击重命名 + 点击加载消息。"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from nekoagent.observability.exceptions import format_friendly_error
from nekoagent.session import create_session, delete_session, list_sessions, rename_session, switch_session
from nekoagent.ui.flet_shell.theme import border_all, glass_container, section_title


def build_sessions_panel(page: Any, on_switch_session: Any, refresh_signal: Any, pubsub: Any, history_loader: dict | None = None) -> Any:
    import flet as ft

    cards_holder = ft.Column(controls=[], scroll=ft.ScrollMode.ALWAYS, expand=True, spacing=8)
    _last_click_time: dict[str, float] = {}
    _double_click_threshold = 0.35

    def _render():
        cards_holder.controls.clear()
        try:
            sessions = list_sessions()
        except Exception as exc:
            cards_holder.controls.append(ft.Text(format_friendly_error(exc), color=ft.Colors.RED, size=12))
            page.update()
            return

        for s in sessions:
            title_text = ft.Text(s.title, size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_900, max_lines=1)

            card = ft.Container(
                padding=10, border_radius=14,
                bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.WHITE),
                border=border_all(1, ft.Colors.with_opacity(0.25, ft.Colors.PINK_100)),
                ink=True,
                content=ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Column(expand=True, controls=[
                            title_text,
                            ft.Text(s.created_at.strftime("%Y-%m-%d %H:%M"), size=10, color=ft.Colors.GREY_600),
                        ]),
                        ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE, icon_color=ft.Colors.PINK_400, icon_size=18,
                            tooltip="删除会话",
                            style=ft.ButtonStyle(bgcolor=ft.Colors.TRANSPARENT),
                        ),
                    ],
                ),
            )

            # Fix 2: 通过闭包正确捕获 card
            def _make_handler(_card, _sid, _title):
                def _handler(e):
                    now = time.time()
                    last = _last_click_time.get(_sid, 0)
                    _last_click_time[_sid] = now
                    if now - last < _double_click_threshold:
                        _replace_with_edit(_card, _sid, _title.value)
                    else:

                        def _delayed_click():
                            if time.time() - _last_click_time.get(_sid, 0) >= _double_click_threshold + 0.04:
                                page.run_thread(lambda: _on_pick(_sid))

                        threading.Timer(_double_click_threshold + 0.05, _delayed_click).start()
                return _handler

            card.on_click = _make_handler(card, s.session_id, title_text)

            # Delete button must be set inside the closure
            delete_btn = card.content.controls[1]
            delete_btn.on_click = lambda e, sid=s.session_id: _on_delete(sid)
            cards_holder.controls.append(card)

        if not cards_holder.controls:
            cards_holder.controls.append(
                ft.Text("暂无会话，点击下方按钮新建一个吧~", size=12, color=ft.Colors.GREY_700, italic=True))
        page.update()

    def _replace_with_edit(card: ft.Container, session_id: str, old_title: str):
        tf = ft.TextField(value=old_title, dense=True, width=140, autofocus=True,
                          border_color=ft.Colors.PINK_300, focused_border_color=ft.Colors.PINK_400)

        def _commit(_e=None):
            new_title = tf.value or old_title
            try:
                rename_session(session_id, new_title)
            except Exception as exc:
                _show_snack(page, format_friendly_error(exc))
            _render()

        tf.on_submit = _commit
        tf.on_blur = _commit
        card.content = ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[tf, ft.IconButton(icon=ft.Icons.CHECK, icon_color=ft.Colors.PINK_400, icon_size=18, on_click=_commit)],
        )
        page.update()

    def _on_pick(session_id: str):
        try:
            new_active = switch_session(session_id)
            on_switch_session(new_active)
            if history_loader and history_loader.get("fn"):
                history_loader["fn"](session_id)
        except Exception as exc:
            _show_snack(page, format_friendly_error(exc))

    def _on_delete(session_id: str):
        try:
            delete_session(session_id)
            _render()
        except Exception as exc:
            _show_snack(page, format_friendly_error(exc))

    def _on_create(_e=None):
        import datetime as _dt
        try:
            create_session(f"会话 {_dt.datetime.now().strftime('%H:%M')}")
            _render()
        except Exception as exc:
            _show_snack(page, format_friendly_error(exc))

    # Fix 3: 新建按钮居中
    create_btn = ft.ElevatedButton(
        "新建会话", icon=ft.Icons.ADD, on_click=_on_create,
        bgcolor=ft.Colors.PINK_300, color=ft.Colors.WHITE,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
    )

    root = glass_container(
        content=ft.Column(
            expand=True,
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Column(expand=True, controls=[
                    section_title("会话", icon=ft.Icons.FORUM),
                    ft.Divider(color=ft.Colors.with_opacity(0.25, ft.Colors.PINK_200), height=1),
                    cards_holder,
                ]),
                ft.Container(
                    alignment=ft.alignment.Alignment.CENTER,
                    content=create_btn,
                    padding=ft.Padding(left=0, top=8, right=0, bottom=4),
                ),
            ],
        ),
        expand=True, radius=18, padding=12,
    )

    refresh_signal["render"] = _render
    _render()
    return root


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