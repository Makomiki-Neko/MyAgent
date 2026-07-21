from __future__ import annotations

from pathlib import Path
from typing import Any

from nekoagent.ui.flet_shell.theme import glass_container
from nekoagent.observability.logging import get_logger

_log = get_logger("ui.flet_menu")


def build_menu_panel(toggle_settings_callback, page: Any, pubsub: Any = None) -> Any:
    import flet as ft

    avatar = ft.Container(
        width=56, height=56, border_radius=28,
        bgcolor=ft.Colors.PINK_300,
        content=ft.Icon(ft.Icons.PETS, color=ft.Colors.WHITE, size=28),
        shadow=ft.BoxShadow(color=ft.Colors.with_opacity(0.3, ft.Colors.PINK_200), offset=ft.Offset(0, 2), blur_radius=8),
    )

    def _menu_btn(icon, label, on_click):
        return ft.Container(
            width=92, height=36,
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.WHITE),
            border_radius=8,
            on_click=on_click,
            content=ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=6,
                controls=[
                    ft.Icon(icon, color=ft.Colors.PINK_700, size=18),
                    ft.Text(label, size=13, color=ft.Colors.PINK_800, weight=ft.FontWeight.BOLD),
                ],
            ),
        )

    def _on_settings(_e):
        toggle_settings_callback()

    def _build_profile_image():
        import flet as ft
        img_path = ""
        try:
            from nekoagent.config.loader import get_config
            img_path = get_config().get_persona_cached("main_agent").avatar_path or ""
        except Exception:
            pass
        if not img_path or not Path(img_path).exists():
            return ft.Container(expand=True)

        return ft.Container(
            expand=True,
            padding=ft.Padding(4,4,4,4),
            content=ft.Image(
                src=str(Path(img_path).absolute()),
                fit="contain",
                repeat="noRepeat",
            )
        )

    # 横幅图片：从项目根目录加载 banner.png / banner.jpg
    _banner_path = None
    for _ext in ["png", "jpg", "jpeg", "webp"]:
        _p = Path.cwd() / f"banner.{_ext}"
        if _p.exists():
            _banner_path = str(_p.absolute())
            break
    if _banner_path is None:
        _log.warning("Not Find Banner Image")

    if _banner_path:
        _banner_content = ft.Stack(
            expand=True,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            controls=[
                # 底层容器：填满 Stack，内部 Image 使用 fitHeight
                ft.Container(
                    expand=True,
                    content=ft.Image(
                        src=_banner_path,
                        fit="fitHeight",          # 字符串形式，兼容 0.86
                        repeat="noRepeat",
                    ),
                ),
            ],
        )
    else:
        _banner_content = None
    btn_row = ft.Row(
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=4,
        wrap=True,
        controls=[
            _menu_btn(ft.Icons.CHAT, "对话", lambda e: toggle_settings_callback(True)),
            _menu_btn(ft.Icons.LIBRARY_BOOKS, "RAG", lambda e: toggle_settings_callback("rag")),
            _menu_btn(ft.Icons.SETTINGS, "设置", _on_settings),
        ]
    )

    # 横幅外层容器
    _profile_img_container = ft.Container(
        content=_banner_content,
        expand=True,
        alignment=ft.Alignment.CENTER,
        bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.PINK_200) if not _banner_path else None,
        padding=ft.Padding(0, 0, 0, 0),
    )

    return ft.Container(
        width=110,
        padding=ft.Padding(left=8, top=10, right=8, bottom=10),
        content=glass_container(
            content=ft.Column(
                alignment=ft.MainAxisAlignment.START,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    avatar,
                    ft.Container(height=4),
                    ft.Text("NekoAgent", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
                    ft.Divider(color=ft.Colors.with_opacity(0.3, ft.Colors.PINK_200), height=1),

                    _profile_img_container,

                    ft.Divider(color=ft.Colors.with_opacity(0.3, ft.Colors.PINK_200), height=1),
                    ft.Container(height=6),
                    btn_row,
                ],
                expand=True,
            ),
            radius=22, padding=10,
            expand=True,
        ),
    )