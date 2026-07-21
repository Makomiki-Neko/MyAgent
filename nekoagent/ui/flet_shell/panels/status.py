"""系统状态监视：第 4 列 - Agent 在线状态(Timer ping) + Token 双管线 + 主管任务状态。"""

from __future__ import annotations

import threading
from typing import Any, Callable

from nekoagent.config.loader import get_config
from nekoagent.observability.exceptions import LLMConfigError, format_friendly_error
from nekoagent.observability.logging import get_logger
from nekoagent.observability.token_counter import get_global_totals, get_session_stats
from nekoagent.ui.flet_shell.theme import border_all, glass_container, section_title

_log = get_logger("ui.status")

_PING_INTERVAL_SEC = 30


def _ping_llm(agent_name: str, cfg: Any) -> bool:
    """尝试用 openai 客户端快速探测模型 API 是否可达。"""

    try:
        from nekoagent.config.loader import resolve_api_key
        from nekoagent.llm.profile_loader import build_llm_for_profile
        agent_cfg = cfg.agents.get(agent_name)
        if agent_cfg is None:
            return False
        profile = cfg.llm_profiles.get(agent_cfg.llm_profile)
        if profile is None:
            return False
        # 构造轻量 openai client
        import httpx
        from openai import OpenAI as _OpenAI
        api_key = resolve_api_key(agent_cfg.llm_profile, cfg)
        client = _OpenAI(api_key=api_key, base_url=profile.base_url, http_client=httpx.Client(timeout=3.0))
        client.models.list()
        return True
    except Exception as exc:
        _log.warning("Agent %s ping 失败：%s", agent_name, exc)
        return False


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def build_status_panel(page: Any, pubsub: Any, get_active_session_id: Callable[[], str]) -> Any:
    import flet as ft

    online = {"main_agent": False, "supervisor_agent": False, "executor_agent": False, "summary_agent": False}

    agent_dots: dict[str, ft.Container] = {}
    agent_chips: dict[str, ft.Container] = {}

    def build_dot_online(label: str, key: str) -> tuple[ft.Container, ft.Container]:
        dot = ft.Container(width=10, height=10, border_radius=5, bgcolor=ft.Colors.GREY_400)
        status_label = ft.Container(
            padding=ft.Padding(left=10, top=4, right=10, bottom=4),
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.GREY_200),
            border_radius=8,
            content=ft.Text("检测中…", size=10, color=ft.Colors.GREY_700),
        )
        agent_dots[key] = dot
        agent_chips[key] = status_label
        return dot, status_label

    def agent_row(label: str, key: str, hint: str) -> ft.Container:
        dot, chip = build_dot_online(label, key)
        return ft.Container(
            padding=ft.Padding(left=10, top=8, right=10, bottom=8),
            bgcolor=ft.Colors.with_opacity(0.6, ft.Colors.WHITE),
            border_radius=10,
            border=border_all(1, ft.Colors.with_opacity(0.15, ft.Colors.PINK_100)),
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.Row(controls=[dot, ft.Text(label, size=12, color=ft.Colors.GREY_800)], spacing=8),
                    chip,
                ],
            ),
        )

    agents_section = ft.Column(
        spacing=6,
        controls=[
            agent_row("主 Agent", "main_agent", "对话流"),
            agent_row("任务主管 Agent", "supervisor_agent", "调度"),
            agent_row("执行 Agent", "executor_agent", "原子工具"),
            agent_row("总结用 Agent", "summary_agent", "裁剪记忆"),
        ],
    )

    dialogue_tokens_text = ft.Text("0", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_700)
    task_tokens_text = ft.Text("0", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.PURPLE_400)
    global_tokens_text = ft.Text("0", size=14, weight=ft.FontWeight.NORMAL, color=ft.Colors.GREY_700)
    dialogue_bar = ft.ProgressBar(value=0, color=ft.Colors.PINK_300, bgcolor=ft.Colors.with_opacity(0.3, ft.Colors.GREY_200), height=6, border_radius=4)
    task_bar = ft.ProgressBar(value=0, color=ft.Colors.PURPLE_400, bgcolor=ft.Colors.with_opacity(0.3, ft.Colors.GREY_200), height=6, border_radius=4)

    supervisor_status_chip = ft.Container(
        padding=ft.Padding(left=10, top=4, right=10, bottom=4),
        bgcolor=ft.Colors.GREY_300,
        border_radius=8,
        content=ft.Text("空闲", size=11, color=ft.Colors.GREY_700),
    )
    pending_tasks_count = ft.Text("0", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_700)
    subtask_name_text = ft.Text("—", size=11, color=ft.Colors.GREY_700)
    subtask_status_chip = ft.Container(
        padding=ft.Padding(left=10, top=4, right=10, bottom=4),
        bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.GREY_200),
        border_radius=8,
        content=ft.Text("空闲", size=11, color=ft.Colors.GREY_700),
    )

    def _update_agent_status(agent_key: str, online_status: bool) -> None:
        dot = agent_dots.get(agent_key)
        chip = agent_chips.get(agent_key)
        if dot:
            dot.bgcolor = ft.Colors.GREEN_400 if online_status else ft.Colors.GREY_400
        if chip:
            chip.bgcolor = ft.Colors.with_opacity(0.7, ft.Colors.GREEN_200) if online_status else ft.Colors.with_opacity(0.5, ft.Colors.GREY_200)
            chip.content = ft.Text("在线" if online_status else "离线", size=10, color=ft.Colors.GREY_800 if online_status else ft.Colors.GREY_600)

    def _ping_all_agents() -> None:
        try:
            cfg = get_config()
            for akey, agent_label in [("main_agent", "main_agent"), ("supervisor_agent", "supervisor_agent"),
                                       ("executor_agent", "executor_agent"), ("summary_agent", None)]:
                if agent_label is None:
                    continue
                ok = _ping_llm(agent_label, cfg)
                online[akey] = ok
                _update_agent_status(akey, ok)
            # summary 用: 查 auxiliary_models.summary_llm_profile 对应的 profile
            try:
                profile_name = cfg.auxiliary_models.summary_llm_profile
                agent_cfg = cfg.agents.get("main_agent")
                # 直接用 summary 对应的 profile API
                from nekoagent.llm.profile_loader import build_llm_for_profile
                profile = cfg.llm_profiles.get(profile_name)
                if profile:
                    import httpx
                    from openai import OpenAI as _OpenAI
                    api_key_resolved = profile.api_key or "dummy"
                    client = _OpenAI(api_key=api_key_resolved, base_url=profile.base_url, http_client=httpx.Client(timeout=3.0))
                    client.models.list()
                    online["summary_agent"] = True
                else:
                    online["summary_agent"] = False
            except Exception:
                online["summary_agent"] = False
            _update_agent_status("summary_agent", online["summary_agent"])
        except Exception as exc:
            _log.warning("ping_all_agents 异常：%s", exc)
        # 无论如何，未完成的 Agent 保持 label "检测中…" → 改为离线
        for key in ["main_agent", "supervisor_agent", "executor_agent", "summary_agent"]:
            dot = agent_dots.get(key)
            if dot and dot.bgcolor == ft.Colors.GREY_400:
                _update_agent_status(key, False)
        page.update()

    def _refresh_token_and_task(_e: Any = None, _payload: Any = None) -> None:
        try:
            sid = get_active_session_id()
            stats = get_session_stats(sid) if sid else {"dialogue_tokens": 0, "task_tokens": 0, "total": 0}
            dialogue_tokens_text.value = _fmt_tokens(stats["dialogue_tokens"])
            task_tokens_text.value = _fmt_tokens(stats["task_tokens"])
            total_for_bar = max(1, stats.get("total", 1))
            dialogue_bar.value = stats["dialogue_tokens"] / total_for_bar
            task_bar.value = stats["task_tokens"] / total_for_bar
            global_stats = get_global_totals()
            global_tokens_text.value = _fmt_tokens(global_stats["total"])
        except Exception as exc:
            _log.debug("token refresh fail: %s", exc)

        try:
            from nekoagent.queue import get_scheduler
            sched = get_scheduler()
            pending_count = sum(1 for t in sched._tasks.values() if t.status.value in ("planning", "awaiting_user", "running"))
            pending_tasks_count.value = str(pending_count)
            if pending_count > 0:
                supervisor_status_chip.bgcolor = ft.Colors.PINK_300
                supervisor_status_chip.content = ft.Text("运行中", size=11, color=ft.Colors.WHITE)
            else:
                supervisor_status_chip.bgcolor = ft.Colors.with_opacity(0.5, ft.Colors.GREY_200)
                supervisor_status_chip.content = ft.Text("空闲", size=11, color=ft.Colors.GREY_700)
        except Exception as exc:
            _log.debug("supervisor status refresh fail: %s", exc)

        try:
            from nekoagent.queue import get_scheduler
            sched = get_scheduler()
            running_tasks = [t for t in sched._tasks.values() if t.status.value in ("running",)]
            if running_tasks:
                st = running_tasks[0].current_subtask
                subtask_name_text.value = st if st else "等待中"
                subtask_status_chip.bgcolor = ft.Colors.PURPLE_300
                subtask_status_chip.content = ft.Text("执行中", size=11, color=ft.Colors.WHITE)
            else:
                subtask_name_text.value = "—"
                subtask_status_chip.bgcolor = ft.Colors.with_opacity(0.5, ft.Colors.GREY_200)
                subtask_status_chip.content = ft.Text("空闲", size=11, color=ft.Colors.GREY_700)
        except Exception:
            pass
        page.update()

    def _full_refresh(_e: Any = None) -> None:
        _ping_all_agents()
        _refresh_token_and_task()

    # 启动周期性 ping
    def _ping_loop():
        import time
        time.sleep(2)
        try:
            page.run_thread(_full_refresh)
        except Exception:
            pass
        while True:
            time.sleep(_PING_INTERVAL_SEC)
            try:
                page.run_thread(_full_refresh)
            except Exception:
                pass

    _ping_thread = threading.Thread(target=_ping_loop, daemon=True, name="agent-ping")
    _ping_thread.start()

    root = glass_container(
        content=ft.Column(
            expand=True,
            scroll=ft.ScrollMode.ALWAYS,
            spacing=12,
            controls=[
                section_title("系统状态", icon=ft.Icons.MONITOR),
                ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
                ft.Text("🤖 Agent 在线状况", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
                agents_section,
                ft.Divider(color=ft.Colors.with_opacity(0.15, ft.Colors.PINK_100)),
                ft.Text("📊 Token 使用（当前会话）", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Column(controls=[ft.Text("对话消耗", size=10, color=ft.Colors.GREY_700), dialogue_tokens_text]),
                        ft.Column(controls=[ft.Text("任务消耗", size=10, color=ft.Colors.GREY_700), task_tokens_text]),
                    ],
                ),
                ft.Text("对话", size=10, color=ft.Colors.PINK_700),
                dialogue_bar,
                ft.Text("任务", size=10, color=ft.Colors.PURPLE_700),
                task_bar,
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[ft.Text("全局累计", size=10, color=ft.Colors.GREY_700), global_tokens_text],
                ),
                ft.Divider(color=ft.Colors.with_opacity(0.15, ft.Colors.PINK_100)),
                ft.Text("🎯 主管 Agent 状态", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                       controls=[ft.Text("实时状态", size=11, color=ft.Colors.GREY_700), supervisor_status_chip]),
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                       controls=[ft.Text("活跃任务数", size=11, color=ft.Colors.GREY_700), pending_tasks_count]),
                ft.Divider(color=ft.Colors.with_opacity(0.15, ft.Colors.PINK_100)),
                ft.Text("⚡ 子 Agent 状态", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                       controls=[ft.Text("当前子任务", size=11, color=ft.Colors.GREY_700), subtask_name_text]),
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                       controls=[ft.Text("执行状态", size=11, color=ft.Colors.GREY_700), subtask_status_chip]),
                ft.Divider(color=ft.Colors.with_opacity(0.15, ft.Colors.PINK_100)),
            ],
        ),
        expand=True,
        radius=18,
        padding=12,
    )

    pubsub.subscribe("status_change", lambda p: _refresh_token_and_task())
    pubsub.subscribe("task_complete", lambda p: _refresh_token_and_task())
    pubsub.subscribe("new_plan", lambda p: _refresh_token_and_task())
    pubsub.subscribe("exception", lambda p: _refresh_token_and_task())
    pubsub.subscribe("init", lambda p: _refresh_token_and_task())
    pubsub.subscribe("session_switch", lambda p: _refresh_token_and_task())
    pubsub.subscribe("subtask_change", lambda p: _refresh_token_and_task())

    return root