"""毛玻璃毛色玻璃容器辅助：统一呈现半透明粉白渐变的玻璃面板。"""

from __future__ import annotations

from typing import Any


def border_all(width: int, color: Any) -> Any:
    """Flet 0.86 兼容的 four-side border helper。"""
    import flet as ft
    side = ft.BorderSide(width, color)
    return ft.Border(left=side, top=side, right=side, bottom=side)


def glass_container(content: Any, *, width: float | int | None = None, expand: bool = False, padding: int = 16, radius: int = 18) -> Any:
    """粉色毛玻璃容器：半透明白色填充 + 圆角 + 软阴影 + 粉色边框。"""

    import flet as ft

    return ft.Container(
        content=content,
        width=width,
        expand=expand,
        padding=padding,
        border_radius=radius,
        bgcolor=ft.Colors.with_opacity(0.55, ft.Colors.WHITE),
        border=border_all(1, ft.Colors.with_opacity(0.4, ft.Colors.PINK_100)),
        shadow=ft.BoxShadow(
            color=ft.Colors.with_opacity(0.18, ft.Colors.PINK_200),
            offset=ft.Offset(0, 4),
            blur_radius=18,
            spread_radius=0,
        ),
    )


def excel_button(text: str, icon: Any | None = None, on_click: Any | None = None, *, primary: bool = True, width: float | int | None = None) -> Any:
    """粉色系按钮。"""

    import flet as ft

    if primary:
        bgcolor = ft.Colors.PINK_300
        fgcolor = ft.Colors.WHITE
    else:
        bgcolor = ft.Colors.with_opacity(0.5, ft.Colors.WHITE)
        fgcolor = ft.Colors.PINK_700
    return ft.ElevatedButton(
        text=text,
        icon=icon,
        on_click=on_click,
        bgcolor=bgcolor,
        color=fgcolor,
        width=width,
        style=ft.ButtonStyle(
            elevation=2,
            padding=ft.Padding(left=14, top=8, right=14, bottom=8),
            shape=ft.RoundedRectangleBorder(radius=12),
        ),
    )


def tag_chip(text: str, color: Any, *, online: bool = True) -> Any:
    """带圆点 + 文字的状态 chip。"""

    import flet as ft

    dot = ft.Container(
        width=8,
        height=8,
        border_radius=4,
        bgcolor=color if online else ft.Colors.GREY_400,
    )
    return ft.Container(
        content=ft.Row([dot, ft.Text(text, size=12, color=ft.Colors.GREY_800)]),
        padding=ft.Padding(left=10, top=4, right=14, bottom=4),
        bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.WHITE),
        border_radius=8,
    )


def section_title(text: str, icon: Any | None = None) -> Any:
    import flet as ft
    return ft.Row(
        controls=[
            ft.Icon(icon, color=ft.Colors.PINK_400, size=18) if icon else ft.Container(width=0),
            ft.Text(text, size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_900),
        ],
        spacing=8,
    )


def page_gradient_background() -> Any:
    """粉白对角渐变背景容器——半透明毛玻璃感（Fix 4：增强透光）。"""

    import flet as ft
    return ft.Container(
        expand=True,
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=[
                ft.Colors.with_opacity(0.85, ft.Colors.PINK_50),
                ft.Colors.with_opacity(0.95, ft.Colors.WHITE),
                ft.Colors.with_opacity(0.85, ft.Colors.PINK_100),
            ],
            stops=[0.0, 0.55, 1.0],
        ),
        opacity=0.92,
    )