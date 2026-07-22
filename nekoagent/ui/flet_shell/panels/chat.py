"""对话窗口：第 3 列 - 历史消息流 + Token流式输出 + 状态提示 + Enter/ShiftEnter 键绑。"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable

from nekoagent.agents.main.agent import MainAgent, handle_user_message, get_main_agent
from nekoagent.config.loader import get_config
from nekoagent.memory.stores import get_user_memory_store
from nekoagent.queue.scheduler import handle_user_decision as sched_user_decision
from nekoagent.observability.exceptions import format_friendly_error
from nekoagent.ui.flet_shell.theme import border_all, glass_container, section_title

_chat_log_holder: list[Any] = [None]


def build_chat_panel(page: Any, pubsub: Any, get_active_session_id: Callable[[], str], history_loader: dict | None = None) -> Any:
    import flet as ft

    chat_log = ft.ListView(expand=True, spacing=10, padding=10, auto_scroll=True)
    _chat_log_holder[0] = chat_log

    def _user_bubble(text: str) -> ft.Container:
        return ft.Container(
            content=ft.Text(text, selectable=True),
            padding=ft.Padding(left=14, top=10, right=14, bottom=10),
            bgcolor=ft.Colors.with_opacity(0.85, ft.Colors.PINK_50),
            border=border_all(1, ft.Colors.with_opacity(0.35, ft.Colors.PINK_200)),
            border_radius=14,
            shadow=ft.BoxShadow(color=ft.Colors.with_opacity(0.15, ft.Colors.PINK_200), offset=ft.Offset(0, 2), blur_radius=6),
        )

    def _strip_italic(text: str) -> str:
        """去除 Markdown 斜体标记（*text* / _text_），保留 **bold**。"""
        import re
        # 将 *text* 替换为 text（但不影响 **bold**）
        text = re.sub(r'(?<!\*)\*(?!\*)([^*]+)(?<!\*)\*(?!\*)', r'\1', text)
        # 将 _text_ 替换为 text（但不影响 __ 等）
        text = re.sub(r'(?<!_)_(?!_)([^_]+)(?<!_)_(?!_)', r'\1', text)
        return text

    def _assistant_bubble(text: str) -> ft.Container:
        text = _strip_italic(text)
        return ft.Container(
            content=ft.Container(
                content=ft.Markdown(text, auto_follow_links=True, selectable=True),
                expand=1,
            ),
            padding=ft.Padding(left=14, top=10, right=14, bottom=10),
            bgcolor=ft.Colors.with_opacity(0.85, ft.Colors.WHITE),
            border=border_all(1, ft.Colors.with_opacity(0.35, ft.Colors.PINK_100)),
            border_radius=14,
            shadow=ft.BoxShadow(color=ft.Colors.with_opacity(0.15, ft.Colors.PINK_100), offset=ft.Offset(0, 2), blur_radius=6),
        )

    assistant_slot: dict = {"container": None, "text": ""}

    def _append_user(text: str):
        chat_log.controls.append(
            ft.Container(
                content=_user_bubble(text),
                expand=True,
                alignment=ft.Alignment(1, 0),
            )
        )

    def _append_assistant_slot(text: str):
        bubble = _assistant_bubble(text)
        ctrl = ft.Container(
            content=bubble,
            expand=True,
            alignment=ft.Alignment(-1, 0),
        )
        assistant_slot["container"] = ctrl
        assistant_slot["text"] = text
        chat_log.controls.append(ctrl)
        return ctrl

    def _update_assistant_slot(text: str):
        ctrl = assistant_slot["container"]
        if ctrl is not None:
            ctrl.content = _assistant_bubble(text)
            assistant_slot["text"] = text

    def _stream_token(token: str):
        try:
            _pname = get_config().get_persona_cached("main_agent").name
        except Exception:
            _pname = ""
        if assistant_slot["text"] == f"{_pname}正在思考……":
            assistant_slot["text"] = ""
        assistant_slot["text"] += token
        _update_assistant_slot(assistant_slot["text"])

    def _send(_e: Any = None) -> None:
        try:
            _pname = get_config().get_persona_cached("main_agent").name
        except Exception:
            _pname = ""
        text = input_box.value or ""
        if text.endswith("\n") and not text.endswith("\n\n"):
            text = text[:-1]
        text = text.strip()
        if not text:
            return
        _append_user(text)
        input_box.value = ""
        _append_assistant_slot(f"{_pname}正在思考……")
        try:
            page.update()
        except Exception:
            pass

        sid = get_active_session_id()

        def _background():
            try:
                agent = get_main_agent()
                if agent.is_task_request(text):
                    reply, _ = handle_user_message(sid, text)
                    page.run_thread(lambda: (_update_assistant_slot(reply or "(空回复)"), pubsub.notify("status_change", {})))
                else:
                    for token in agent.chat_stream(sid, text):
                        page.run_thread(lambda t=token: _stream_token(t))
                    page.run_thread(lambda: pubsub.notify("status_change", {}))
            except Exception as exc:
                page.run_thread(lambda: _update_assistant_slot(format_friendly_error(exc)))
            try:
                page.run_thread(lambda: page.update())
            except Exception:
                pass

        threading.Thread(target=_background, daemon=True).start()

    send_btn = ft.ElevatedButton(
        "发送", icon=ft.Icons.SEND_ROUNDED, on_click=_send,
        bgcolor=ft.Colors.PINK_400, color=ft.Colors.WHITE,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=14), elevation=2),
    )

    input_box = ft.TextField(
        hint_text="输入消息，以 /task 开头将自动转交主管 Agent 规划",
        multiline=True, min_lines=1, max_lines=5, expand=True,
        border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
        bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.WHITE), border_radius=14,
        on_submit=lambda e: _send(),
    )

    header = ft.Row(
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        controls=[
            section_title("对话", icon=ft.Icons.CHAT_BUBBLE),
            ft.Container(
                content=ft.Text("Enter 发送", size=10, color=ft.Colors.GREY_700, italic=True),
                padding=ft.Padding(left=10, top=2, right=10, bottom=2),
                bgcolor=ft.Colors.with_opacity(0.6, ft.Colors.WHITE), border_radius=10,
            ),
        ],
    )

    # ---- 历史消息加载 ----
    def _load_history(session_id: str):
        try:
            from nekoagent.memory.stores import get_user_memory_store
            store = get_user_memory_store()
            rows = store.fetch_active_messages(session_id, limit=200)
            chat_log.controls.clear()
            for row in rows:
                if row["role"] == "user":
                    chat_log.controls.append(
                        ft.Container(content=_user_bubble(row["content"]), expand=True, alignment=ft.Alignment(1, 0))
                    )
                elif row["role"] == "assistant":
                    chat_log.controls.append(
                        ft.Container(content=_assistant_bubble(row["content"]), expand=True, alignment=ft.Alignment(-1, 0))
                    )
            try:
                asyncio.create_task(chat_log.scroll_to(offset=-1, duration=100))
            except Exception:
                pass
            page.update()
        except Exception:
            pass

    if history_loader is not None:
        history_loader["fn"] = _load_history

    # ---- HITL 消息持久化辅助 ----
    def _save_msg(role, content):
        try:
            store = get_user_memory_store()
            sid = get_active_session_id()
            if sid and content:
                store.add_message(session_id=sid, role=role, content=content)
        except Exception:
            pass

    # ---- 任务规划到达（HITL 人在回路） ----
    _mod_row_ref = {"row": None, "thread_id": None}
    _btn_row_ref = {"row": None}

    def _make_decision(_sid, decision):
        action = decision.get("action", "")
        mod = decision.get("modification", "")
        if action == "execute":
            _save_msg("user", "批准执行任务计划")
            _append_user("批准执行任务计划")
        elif action == "modify":
            _save_msg("user", f"要求修改任务计划，修改意见：{mod}")
            _append_user(f"要求修改任务计划：{mod}")
        elif action == "cancel":
            _save_msg("user", "取消任务")
            _append_user("取消任务")
        # 立即隐藏按钮，阻止重复点击
        if _btn_row_ref["row"] is not None:
            try:
                chat_log.controls.remove(_btn_row_ref["row"])
            except Exception:
                pass
            _btn_row_ref["row"] = None
        if _mod_row_ref["row"] is not None:
            try:
                chat_log.controls.remove(_mod_row_ref["row"])
            except Exception:
                pass
            _mod_row_ref["row"] = None
        page.update()
        _append_assistant_slot("正在处理您的请求……")
        page.update()
        def _bg_ack():
            sid = get_active_session_id()
            try:
                agent = get_main_agent()
                ack = agent.generate_decision_ack(action, mod, session_id=sid)
            except Exception:
                ack = {"execute": "好的，收到！马上安排执行喵~", "modify": "好的，收到修改意见啦~", "cancel": "好哒，已取消任务喵~"}.get(action, "好的喵~")
            _save_msg("assistant", ack)
            def _update():
                if assistant_slot.get("container") is not None:
                    c = assistant_slot["container"]
                    if c and c.content:
                        c.content = _assistant_bubble(ack)
                        assistant_slot["text"] = ack
                        page.update()
            page.run_thread(_update)
        threading.Thread(target=_bg_ack, daemon=True).start()
        # 后台处理决策（supervisor 执行任务）
        threading.Thread(target=lambda: sched_user_decision(_sid, decision), daemon=True).start()

    def _toggle_mod_row(_sid):
        if _mod_row_ref["row"] is not None:
            try:
                chat_log.controls.remove(_mod_row_ref["row"])
            except Exception:
                pass
            _mod_row_ref["row"] = None
            page.update()
            return
        tf = ft.TextField(hint_text="输入修改意见", dense=True, expand=True, border_color=ft.Colors.PINK_300)
        def _submit_mod(_e):
            mod_text = tf.value or ""
            if mod_text.strip():
                _make_decision(_sid, {"action": "modify", "modification": mod_text})
        tf.on_submit = _submit_mod
        send_mod = ft.IconButton(icon=ft.Icons.SEND, icon_color=ft.Colors.PINK_400, on_click=_submit_mod)
        mod_row = ft.Row(controls=[tf, send_mod])
        _mod_row_ref["row"] = mod_row
        chat_log.controls.append(mod_row)
        page.update()

    def _on_plan_ready(payload):
        try:
            plan_text = payload.get("plan_text", "")
            thread_id = payload.get("thread_id", "")
            if not plan_text or not thread_id:
                return
            _save_msg("assistant", plan_text)
            _mod_row_ref["thread_id"] = thread_id
            _append_assistant_slot(plan_text)
            btn_approve = ft.ElevatedButton("同意", icon=ft.Icons.CHECK_CIRCLE, on_click=lambda e: _make_decision(thread_id, {"action": "execute"}),
                                            bgcolor=ft.Colors.GREEN_400, color=ft.Colors.WHITE, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)))
            btn_modify = ft.ElevatedButton("修正", icon=ft.Icons.EDIT, on_click=lambda e: _toggle_mod_row(thread_id),
                                           bgcolor=ft.Colors.ORANGE_400, color=ft.Colors.WHITE, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)))
            btn_cancel = ft.ElevatedButton("取消", icon=ft.Icons.CANCEL, on_click=lambda e: _make_decision(thread_id, {"action": "cancel"}),
                                           bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)))
            btn_row = ft.Row(
                alignment=ft.MainAxisAlignment.CENTER, spacing=12,
                controls=[btn_approve, btn_modify, btn_cancel],
            )
            _btn_row_ref["row"] = btn_row
            chat_log.controls.append(btn_row)
            page.update()
        except Exception:
            pass

    # ---- 用户决策后状态更新（ack 已在 _make_decision 中立即生成） ----
    def _on_decision_made(payload):
        try:
            status = payload.get("status", "")
            if status == "error":
                _append_assistant_slot("处理决策时出现了一些问题，请稍后重试~")
                page.update()
        except Exception:
            pass

    # ---- 任务完成 ----
    def _on_task_complete(payload):
        try:
            result_text = payload.get("result_text", "")
            if result_text:
                _save_msg("assistant", result_text)
                _append_assistant_slot(result_text)
                page.update()
        except Exception:
            pass

    # ---- 主管请求用户补充信息 ----
    _info_row_ref = {"row": None, "thread_id": None}
    _info_btn_row_ref = {"row": None}
    _last_info_question = { "q": "" }

    def _on_info_request(payload):
        try:
            question = payload.get("question", "")
            thread_id = payload.get("thread_id", "")
            if not question or not thread_id:
                return
            _last_info_question["q"] = question
            def _bg_info():
                sid = get_active_session_id()
                try:
                    agent = get_main_agent()
                    text = agent.generate_info_request_notification(question, session_id=sid)
                except Exception:
                    text = f"人家需要你补充一下信息呢：{question}"
                _save_msg("assistant", text)
                page.run_thread(lambda: _show_info_buttons(text, thread_id))
            threading.Thread(target=_bg_info, daemon=True).start()
        except Exception:
            pass

    def _show_info_buttons(info_text, thread_id):
        _info_row_ref["thread_id"] = thread_id
        _append_assistant_slot(info_text)
        def _make_info_decision(_sid, decision):
            info_val = decision.get("info", "")
            action_label = "provide_info" if info_val else "cancel"
            _save_msg("user", f"补充信息：{info_val}" if info_val else "取消信息请求")
            _append_user(f"补充信息：{info_val}" if info_val else "取消信息请求")
            if _info_row_ref["row"] is not None:
                try:
                    chat_log.controls.remove(_info_row_ref["row"])
                except Exception:
                    pass
                _info_row_ref["row"] = None
            if _info_btn_row_ref["row"] is not None:
                try:
                    chat_log.controls.remove(_info_btn_row_ref["row"])
                except Exception:
                    pass
                _info_btn_row_ref["row"] = None
            page.update()
            def _bg_ack():
                sid = get_active_session_id()
                try:
                    agent = get_main_agent()
                    ack = agent.generate_info_request_feedback(action_label, info_val, session_id=sid)
                except Exception:
                    ack = {"provide_info": "好的，收到你补充的信息啦，我马上告诉主管~", "cancel": "好的，已取消~"}.get(action_label, "好的喵~")
                _save_msg("assistant", ack)
                page.run_thread(lambda: (_append_assistant_slot(ack), page.update()))
            threading.Thread(target=_bg_ack, daemon=True).start()
            if decision.get("action") == "cancel":
                threading.Thread(target=lambda: sched_user_decision(_sid, {"action": "cancel"}), daemon=True).start()
            else:
                threading.Thread(target=lambda: sched_user_decision(_sid, decision), daemon=True).start()
        def _toggle_info_row(_sid):
            if _info_row_ref["row"] is not None:
                try:
                    chat_log.controls.remove(_info_row_ref["row"])
                except Exception:
                    pass
                _info_row_ref["row"] = None
                page.update()
                return
            tf = ft.TextField(hint_text="输入补充信息", dense=True, expand=True, border_color=ft.Colors.PURPLE_300)
            def _submit_info(_e):
                info_text = tf.value or ""
                if info_text.strip():
                    _make_info_decision(_sid, {"action": "provide_info", "info": info_text})
            tf.on_submit = _submit_info
            send_btn2 = ft.IconButton(icon=ft.Icons.SEND, icon_color=ft.Colors.PURPLE_400, on_click=_submit_info)
            row = ft.Row(controls=[tf, send_btn2])
            _info_row_ref["row"] = row
            chat_log.controls.append(row)
            page.update()
        def _auto_decide(_sid):
            question = _last_info_question.get("q", "")
            _save_msg("user", "让主Agent代为决定")
            _append_user("让主Agent代为决定")
            if _info_row_ref["row"] is not None:
                try:
                    chat_log.controls.remove(_info_row_ref["row"])
                except Exception:
                    pass
                _info_row_ref["row"] = None
            if _info_btn_row_ref["row"] is not None:
                try:
                    chat_log.controls.remove(_info_btn_row_ref["row"])
                except Exception:
                    pass
                _info_btn_row_ref["row"] = None
            page.update()
            def _bg_info():
                sid = get_active_session_id()
                try:
                    agent = get_main_agent()
                    answer = agent.generate_info_request_answer(question, session_id=sid)
                except Exception:
                    answer = ""
                if answer.strip():
                    _save_msg("user", f"（由主Agent代为决定）{answer[:200]}")
                    try:
                        feedback = agent.generate_info_request_feedback("auto_decide", answer, session_id=sid)
                    except Exception:
                        feedback = "人家已经帮你决定啦，这就告诉主管~"
                    _save_msg("assistant", feedback)
                    page.run_thread(lambda: (_append_assistant_slot(feedback), page.update()))
                    threading.Thread(target=lambda: sched_user_decision(_sid, {"action": "provide_info", "info": answer}), daemon=True).start()
                else:
                    try:
                        agent = get_main_agent()
                        fb = agent.generate_info_request_feedback("cancel", session_id=sid)
                    except Exception:
                        fb = "好的，已取消~"
                    _save_msg("assistant", fb)
                    page.run_thread(lambda: (_append_assistant_slot(fb), page.update()))
                    threading.Thread(target=lambda: sched_user_decision(_sid, {"action": "cancel"}), daemon=True).start()
            threading.Thread(target=_bg_info, daemon=True).start()
        # 获取角色名
        try:
            _pname = get_config().get_persona_cached("main_agent").name
        except Exception:
            _pname = "助手"
        btn_continue = ft.ElevatedButton("继续", icon=ft.Icons.ARROW_FORWARD, on_click=lambda e: _toggle_info_row(thread_id),
                                         bgcolor=ft.Colors.PURPLE_400, color=ft.Colors.WHITE, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)))
        btn_auto = ft.ElevatedButton(f"让{_pname}决定", icon=ft.Icons.PSYCHOLOGY, on_click=lambda e: _auto_decide(thread_id),
                                     bgcolor=ft.Colors.INDIGO_400, color=ft.Colors.WHITE, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)))
        btn_cancel = ft.ElevatedButton("取消", icon=ft.Icons.CANCEL, on_click=lambda e: _make_info_decision(thread_id, {"action": "cancel"}),
                                       bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)))
        btn_row = ft.Row(alignment=ft.MainAxisAlignment.CENTER, spacing=12, controls=[btn_continue, btn_auto, btn_cancel])
        _info_btn_row_ref["row"] = btn_row
        chat_log.controls.append(btn_row)
        page.update()

    pubsub.subscribe("plan_ready", lambda p: _on_plan_ready(p))
    pubsub.subscribe("decision_made", lambda p: _on_decision_made(p))
    pubsub.subscribe("task_complete", lambda p: _on_task_complete(p))
    pubsub.subscribe("info_request", lambda p: _on_info_request(p))

    return glass_container(
        content=ft.Column(
            expand=True,
            controls=[
                header,
                ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
                chat_log,
                ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
                ft.Row(controls=[input_box, send_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ],
        ),
        expand=True, radius=18, padding=12,
    )