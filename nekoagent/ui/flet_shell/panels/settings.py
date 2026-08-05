"""设置面板：在 col 3 显示，顶部 tabs 切分子页面。

子页：
- 角色设置：人物名称 + 人物设定 Prompt（写回 personas/main.yaml + config.yaml 人物名）
- 主 Agent / 任务主管 Agent / 执行子 Agent / 总结用 Agent：
    LLM URL、API Key、Model Name、Temperature 滑动条(0-2)、MaxTokens
- RAG 配置：Embedding 路径、Reranker 路径、设备选择(CPU/CUDA)
"""

from __future__ import annotations

from typing import Any

from nekoagent.config.loader import get_config
from nekoagent.config.schema import PersonaConfig
from nekoagent.config.writeback import reset_persona_cache, save_config, save_persona
from nekoagent.llm.factory import clear_cache as llm_clear_cache
from nekoagent.observability.exceptions import format_friendly_error
from nekoagent.observability.logging import get_logger
from nekoagent.ui.flet_shell.theme import border_all, glass_container, section_title

_log = get_logger("ui.settings")


def build_settings_panel(page: Any, pubsub: Any) -> Any:
    import flet as ft
    import copy
    import datetime as _dt

    cfg = get_config()
    persona = cfg.get_persona_cached("main_agent")

    # save 共享回调
    def _save(_e: Any) -> None:
        try:
            # ---- 1. 主 config.yaml ----
            new_raw = copy.deepcopy(cfg.raw)

            # 更新 LLM profiles（每个 agent 对应的 profile 字段）
            for (agent_key, profile_key, fields) in _agent_field_sources:
                if profile_key not in new_raw.get("llm_profiles", {}):
                    continue
                profile = new_raw["llm_profiles"][profile_key]
                url_val = fields["url_field"].value.strip()
                api_val = fields["api_field"].value.strip()
                model_val = fields["model_field"].value.strip()
                temp_val = fields["temp_field"].value
                tok_val = fields["tok_field"].value.strip()
                if url_val:
                    profile["base_url"] = url_val
                if api_val:
                    profile["api_key"] = api_val
                if model_val:
                    profile["model"] = model_val
                if temp_val is not None:
                    profile["temperature"] = float(temp_val)
                if tok_val.isdigit():
                    profile["max_tokens"] = int(tok_val)

            # 更新 TTS 字段
            new_raw.setdefault("tts", {})
            new_raw["tts"]["model_path"] = _tts_model_field.value.strip()
            new_raw["tts"]["device"] = _tts_device.value
            # 获取当前激活的克隆声音
            try:
                from nekoagent.tts.engine import get_engine
                engine = get_engine()
                active = engine.get_active_clone()
                new_raw["tts"]["active_clone"] = active["name"] if active else ""
            except Exception:
                new_raw["tts"]["active_clone"] = ""

            # 更新 RAG 字段
            new_raw.setdefault("auxiliary_models", {})
            new_raw["auxiliary_models"]["embedding_model_path"] = _rag_embedding_field.value.strip()
            new_raw["auxiliary_models"]["reranker_model_path"] = _rag_reranker_field.value.strip()
            new_raw["auxiliary_models"]["embedding_device"] = _rag_embed_device.value
            new_raw["auxiliary_models"]["reranker_device"] = _rag_rerank_device.value

            # 更新 MCP 服务器列表
            new_raw["mcp_servers"] = {}
            for i, entry in enumerate(mcp_addr_list):
                addr = entry["field"].value.strip()
                if not addr:
                    continue
                srv_name = f"mcp_server_{i}"
                if addr.startswith("http://") or addr.startswith("https://"):
                    new_raw["mcp_servers"][srv_name] = {"transport": "sse", "url": addr}
                else:
                    parts = addr.split()
                    new_raw["mcp_servers"][srv_name] = {
                        "transport": "stdio",
                        "command": parts[0],
                        "args": parts[1:] if len(parts) > 1 else [],
                        "env": {},
                    }
            # 同步更新各 agent 的 mcp_servers 引用
            srv_names = list(new_raw["mcp_servers"].keys())
            agents_raw = new_raw.get("agents", {})
            for agent_key in ("main_agent", "supervisor_agent", "executor_agent"):
                if agent_key in agents_raw:
                    agents_raw[agent_key]["mcp_servers"] = list(srv_names)

            # 写回主 config
            save_config(new_raw, cfg.source_path)
            reset_persona_cache()
            llm_clear_cache()
            # 重新初始化 TTS 引擎
            try:
                from nekoagent.tts.engine import init_engine
                tts_cfg = new_raw.get("tts", {})
                init_engine(
                    model_path=tts_cfg.get("model_path", ""),
                    device=tts_cfg.get("device", "cpu"),
                )
            except Exception:
                pass

            # ---- 2. persona yaml ----
            persona_dict = persona.model_dump()
            persona_dict["name"] = _persona_name_field.value.strip() or persona.name
            persona_dict["system_prompt"] = _persona_prompt_field.value.strip() or persona.system_prompt
            save_persona(cfg.agents["main_agent"].system_prompt_file, persona_dict)
            reset_persona_cache()

            # 改名同步 + 角色立绘刷新
            pubsub.notify("persona_name_changed", {"name": persona_dict["name"]})

            _show_snack(page, "配置保存成功，已重新加载生效！")
        except Exception as exc:
            _show_snack(page, format_friendly_error(exc))

    # ======== 辅助：agent 行构建 ========
    def _agent_section(label, profile_name) -> list:
        fields = []
        url = ft.TextField(
            label="LLM API",
            value=cfg.llm_profiles.get(profile_name, getattr(cfg.llm_profiles, "_", None)).base_url if cfg.llm_profiles.get(profile_name) else "",
            border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
            width=420,
        )
        api = ft.TextField(
            label="API Key",
            value=cfg.llm_profiles.get(profile_name, None).api_key if cfg.llm_profiles.get(profile_name) else "",
            password=True, can_reveal_password=True,
            border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
            width=420,
        )
        model = ft.TextField(
            label="Model Name",
            value=cfg.llm_profiles.get(profile_name, None).model if cfg.llm_profiles.get(profile_name) else "",
            border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
            width=420,
        )
        temp_val = cfg.llm_profiles.get(profile_name, None).temperature if cfg.llm_profiles.get(profile_name) else 0.7
        temp_label = ft.Text(f"Temperature: {temp_val:.1f}", size=12, color=ft.Colors.GREY_700)
        temp = ft.Slider(
            min=0, max=2, divisions=20, value=float(temp_val),
            label="{value}", active_color=ft.Colors.PINK_400,
            width=420, round=1,
            on_change=lambda e: setattr(temp_label, 'value', f"Temperature: {e.control.value:.1f}"),
        )
        token_val = cfg.llm_profiles.get(profile_name, None).max_tokens if cfg.llm_profiles.get(profile_name) else 2048
        tok = ft.TextField(
            label="Max Tokens",
            value=str(token_val),
            keyboard_type=ft.KeyboardType.NUMBER,
            border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
            width=200,
        )
        nonlocal _agent_field_sources
        _agent_field_sources.append(("", profile_name, {"url_field": url, "api_field": api, "model_field": model, "temp_field": temp, "tok_field": tok}))
        return [
            ft.Text(label, size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
            url, api, model,
            temp_label,
            temp, tok,
        ]

    _agent_field_sources = []  # 存每个 agent 表中编辑后的字段

    # ======== 子页面控件 ========
    # 角色设置
    _persona_name_field = ft.TextField(
        label="人物名称",
        value=persona.name,
        border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
        width=420,
    )
    _persona_prompt_field = ft.TextField(
        label="人物设定 System Prompt",
        value=persona.system_prompt,
        multiline=True, min_lines=6, max_lines=16,
        border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
        width=420,
    )


    persona_page = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            ft.Text("角色设置", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
            _persona_name_field,
            _persona_prompt_field,
        ],
    )

    # 各 Agent 页（profile name 必须在 config.yaml.llm_profiles 中存在）
    main_page = ft.Column(scroll=ft.ScrollMode.ALWAYS, horizontal_alignment=ft.CrossAxisAlignment.CENTER, controls=_agent_section("主 Agent 配置", "main"))
    supervisor_page = ft.Column(scroll=ft.ScrollMode.ALWAYS, horizontal_alignment=ft.CrossAxisAlignment.CENTER, controls=_agent_section("任务主管 Agent 配置", "supervisor"))
    executor_page = ft.Column(scroll=ft.ScrollMode.ALWAYS, horizontal_alignment=ft.CrossAxisAlignment.CENTER, controls=_agent_section("执行子 Agent 配置", "executor"))
    summary_page = ft.Column(scroll=ft.ScrollMode.ALWAYS, horizontal_alignment=ft.CrossAxisAlignment.CENTER, controls=_agent_section("总结用 Agent 配置", "summary_local"))

    # RAG 页
    _rag_embedding_field = ft.TextField(
        label="Embedding 模型路径", expand=True, read_only=True,
        value=cfg.auxiliary_models.embedding_model_path,
        border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
    )
    _rag_reranker_field = ft.TextField(
        label="Reranker 模型路径", expand=True, read_only=True,
        value=cfg.auxiliary_models.reranker_model_path,
        border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
    )

    from nekoagent.ui.flet_shell.dialogs import pick_model_directory, update_field_from_dialog as _ufd
    _rag_embedding_row = ft.Container(
        width=500, 
        content=ft.Row(controls=[_rag_embedding_field,
        ft.IconButton(icon=ft.Icons.FOLDER_OPEN, icon_color=ft.Colors.PINK_400, tooltip="选择文件夹",
                       on_click=lambda e: (_ufd(_rag_embedding_field, pick_model_directory()), page.update() if _rag_embedding_field.value else None))]),
    )
    _rag_reranker_row = ft.Container(
        width=500, 
        content=ft.Row(
        controls=[_rag_reranker_field,
            ft.IconButton(icon=ft.Icons.FOLDER_OPEN, icon_color=ft.Colors.PINK_400, tooltip="选择文件夹",
            on_click=lambda e: (_ufd(_rag_reranker_field, pick_model_directory()), page.update() if _rag_reranker_field.value else None))],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
    )

    _rag_embed_device = ft.Dropdown(
        label="Embedding 设备",
        options=[ft.dropdown.Option("cpu", "CPU"), ft.dropdown.Option("cuda", "CUDA")],
        value=cfg.auxiliary_models.embedding_device,
        width=250, border_color=ft.Colors.PINK_200,
    )
    _rag_rerank_device = ft.Dropdown(
        label="Reranker 设备",
        options=[ft.dropdown.Option("cpu", "CPU"), ft.dropdown.Option("cuda", "CUDA")],
        value=cfg.auxiliary_models.reranker_device,
        width=250, border_color=ft.Colors.PINK_200,
    )
    # RAG 库列表
    # ======== MCP 服务器地址列表（可编辑） ========
    mcp_addr_list: list[dict] = []  # 每项 {"field": TextField, "result": Text}
    mcp_list = ft.Column(spacing=20, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def _rebuild_mcp_list():
        mcp_list.controls.clear()
        mcp_addr_list.clear()
        # 从 config 读取已有服务器地址（支持 command 和 url 字段）
        for srv_name, srv_cfg in cfg.mcp_servers.items():
            addr = srv_cfg.url or srv_cfg.command or ""
            if addr:
                _add_mcp_entry(addr)
        if not mcp_addr_list:
            _add_mcp_entry("")
        page.update()

    def _add_mcp_entry(default_value: str = ""):
        idx = len(mcp_addr_list)
        field = ft.TextField(label=f"MCP 服务器 {idx + 1}", value=default_value, expand=True,
                              border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400)
        result_text = ft.Text("", size=10, color=ft.Colors.GREY_700)

        def _do_ping(e):
            addr = field.value.strip()
            result_text.value = "测试中…"
            page.update()
            if not addr:
                result_text.value = "❌ 地址为空"
                page.update()
                return
            try:
                import httpx
                r = httpx.get(addr, timeout=5)
                if r.status_code < 500:
                    result_text.value = f"✅ 可达"
                else:
                    result_text.value = f"❌ 服务端错误（{r.status_code}）"
            except httpx.ConnectError:
                result_text.value = "❌ 连接失败"
            except httpx.TimeoutException:
                result_text.value = "⏱ 超时"
            except Exception as exc:
                result_text.value = f"❌ {exc}"
            page.update()

        ping_btn = ft.IconButton(icon=ft.Icons.WIFI, icon_size=18, icon_color=ft.Colors.PINK_400,
                                  tooltip="测试连接", on_click=_do_ping)
        del_btn = ft.IconButton(icon=ft.Icons.DELETE, icon_size=18, icon_color=ft.Colors.RED_400,
                                 tooltip="删除", on_click=lambda e: _remove_mcp_entry(idx))
        row = ft.Row(controls=[field, ping_btn, del_btn, result_text], alignment=ft.MainAxisAlignment.CENTER,
                      vertical_alignment=ft.CrossAxisAlignment.CENTER)
        mcp_addr_list.append({"field": field, "result": result_text})
        mcp_list.controls.append(row)
        page.update()

    def _remove_mcp_entry(idx: int):
        if 0 <= idx < len(mcp_addr_list):
            mcp_addr_list.pop(idx)
        _rebuild_mcp_list()

    add_btn = ft.ElevatedButton("+ 添加服务器", icon=ft.Icons.ADD, on_click=lambda e: _add_mcp_entry(""),
                                 bgcolor=ft.Colors.PINK_300, color=ft.Colors.WHITE,
                                 style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)))

    _rebuild_mcp_list()

    mcp_list_container = ft.Container(
        width=500,  # 固定宽度
        content=mcp_list,
    )

    mcp_page = ft.Column(scroll=ft.ScrollMode.ALWAYS, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10, controls=[
        ft.Text("MCP 服务器配置", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
        ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
        ft.Text("添加 MCP 服务器地址（支持 http/https 或 stdio 命令）", size=11, color=ft.Colors.GREY_700),
        mcp_list_container,
        add_btn,
    ])

    rag_lib_container = ft.Column(spacing=8)

    def _refresh_lib_list():
        rag_lib_container.controls.clear()
        try:
            from nekoagent.rag_tools import scan_rag_libraries, delete_library, update_library_meta
            libs = scan_rag_libraries()
        except Exception:
            libs = []
        if not libs:
            rag_lib_container.controls.append(
                ft.Container(
                    content=ft.Text("暂无已建立的 RAG 库", size=12, italic=True, color=ft.Colors.GREY_600),
                    padding=10, border_radius=8,
                    bgcolor=ft.Colors.with_opacity(0.4, ft.Colors.WHITE),
                )
            )
        else:
            for lib in libs:
                name = lib.get("name", "未知")
                desc = lib.get("description", "无描述")
                idx_type = lib.get("index_type", "hybrid")
                chunk_count = lib.get("chunk_count", 0)
                desc_edit = ft.TextField(value=desc, dense=True, expand=True, border_color=ft.Colors.PINK_200,
                                          focused_border_color=ft.Colors.PINK_400)
                desc_label = ft.Text(desc, size=11, color=ft.Colors.GREY_700, expand=True)
                _editing = {"active": False}

                def _toggle_edit(_name, _label, _field):
                    if _editing["active"]:
                        _label.visible = True
                        _field.visible = False
                        _editing["active"] = False
                    else:
                        _label.visible = False
                        _field.visible = True
                        _editing["active"] = True
                    page.update()

                def _save_desc(_name, _field, _label):
                    try:
                        from nekoagent.rag_tools import update_library_meta
                        if update_library_meta(_name, _field.value.strip()):
                            _label.value = _field.value.strip()
                    except Exception:
                        pass
                    _label.visible = True
                    _field.visible = False
                    _editing["active"] = False
                    page.update()

                def _delete_lib(_name):
                    try:
                        from nekoagent.rag_tools import delete_library
                        delete_library(_name)
                    except Exception:
                        pass
                    _refresh_lib_list()
                    page.update()

                edit_btn = ft.IconButton(icon=ft.Icons.EDIT, icon_size=16, icon_color=ft.Colors.PINK_400,
                                          tooltip="编辑描述", on_click=lambda e, n=name, l=desc_label, f=desc_edit: _toggle_edit(n, l, f))
                save_btn = ft.IconButton(icon=ft.Icons.SAVE, icon_size=16, icon_color=ft.Colors.GREEN_500,
                                          tooltip="保存", on_click=lambda e, n=name, f=desc_edit, l=desc_label: _save_desc(n, f, l))
                del_btn = ft.IconButton(icon=ft.Icons.DELETE, icon_size=16, icon_color=ft.Colors.RED_400,
                                         tooltip="删除", on_click=lambda e, n=name: _delete_lib(n))
                desc_edit.visible = False

                card = ft.Container(
                    padding=8, border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.WHITE),
                    border=border_all(1, ft.Colors.with_opacity(0.2, ft.Colors.PINK_100)),
                    content=ft.Column(controls=[
                        ft.Row(controls=[ft.Text(name, size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
                                          ft.Text(f"[{idx_type}] {chunk_count}块", size=10, color=ft.Colors.GREY_600)],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Row(controls=[desc_label, desc_edit, edit_btn, save_btn, del_btn]),
                    ]),
                )
                rag_lib_container.controls.append(card)
        page.update()

    _refresh_lib_list()

    rag_page = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            ft.Text("RAG 配置", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
            _rag_embedding_row,
            _rag_reranker_row,
            ft.Row([_rag_embed_device, _rag_rerank_device], alignment=ft.MainAxisAlignment.CENTER),
            ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
            ft.Text("已建立的 RAG 库", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            rag_lib_container,
            ft.ElevatedButton("刷新列表", icon=ft.Icons.REFRESH, on_click=lambda e: _refresh_lib_list(),
                              bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.PINK_100), color=ft.Colors.PINK_800,
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
        ],
    )

    # ======== TTS 配置页 ========
    tts_raw = cfg.raw.get("tts", {})
    _tts_model_field = ft.TextField(
        label="模型路径", expand=True, read_only=True,
        value=tts_raw.get("model_path", ""),
        border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
    )
    from nekoagent.ui.flet_shell.dialogs import pick_model_directory as _pmd, pick_audio_file as _paf
    _tts_model_row = ft.Container(
        width=500,
        content=ft.Row(controls=[_tts_model_field,
            ft.IconButton(icon=ft.Icons.FOLDER_OPEN, icon_color=ft.Colors.PINK_400, tooltip="选择模型目录",
                          on_click=lambda e: (_ufd(_tts_model_field, _pmd()), page.update() if _tts_model_field.value else None))]),
    )
    _tts_device = ft.Dropdown(
        label="设备",
        options=[ft.dropdown.Option("cpu", "CPU"), ft.dropdown.Option("cuda", "CUDA")],
        value=tts_raw.get("device", "cpu"),
        width=250, border_color=ft.Colors.PINK_200,
    )
    # 克隆声音创建
    _clone_audio_field = ft.TextField(
        label="参考音频", expand=True, read_only=True,
        value="", border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400,
    )
    _clone_audio_row = ft.Container(
        width=500,
        content=ft.Row(controls=[_clone_audio_field,
            ft.IconButton(icon=ft.Icons.AUDIO_FILE, icon_color=ft.Colors.PINK_400, tooltip="选择音频",
                          on_click=lambda e: (_ufd(_clone_audio_field, _paf()), page.update() if _clone_audio_field.value else None))]),
    )
    _clone_text_field = ft.TextField(
        label="参考文本（音频中朗读的内容）", multiline=True, min_lines=1, max_lines=4,
        value="", border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400, width=500,
    )
    _clone_name_field = ft.TextField(
        label="克隆声音名称", value="",
        border_color=ft.Colors.PINK_200, focused_border_color=ft.Colors.PINK_400, width=500,
    )
    _clone_status = ft.Text("", size=11, color=ft.Colors.GREY_700)

    def _do_clone(_e):
        audio = _clone_audio_field.value.strip()
        text = _clone_text_field.value.strip()
        name = _clone_name_field.value.strip()
        if not audio or not text or not name:
            _clone_status.value = "❌ 请填写音频、文本和名称"
            page.update()
            return
        try:
            # 使用当前表单中的模型路径和设备重新初始化引擎（无需先保存配置）
            from nekoagent.tts.engine import init_engine, get_engine
            model_path = _tts_model_field.value.strip()
            device = _tts_device.value
            engine = init_engine(model_path=model_path, device=device)
            ok = engine.clone_voice(audio, text, name)
            _clone_status.value = "✅ 克隆成功！" if ok else "❌ 克隆失败（检查模型路径和日志）"
            if ok:
                # 克隆成功后自动激活并写入配置
                import copy
                raw = copy.deepcopy(get_config().raw)
                raw.setdefault("tts", {})
                raw["tts"]["model_path"] = model_path
                raw["tts"]["device"] = device
                raw["tts"]["active_clone"] = name
                from nekoagent.config.writeback import save_config
                save_config(raw, get_config().source_path)
                # 从 yaml 重新初始化引擎
                _cfg = get_config()
                _tts = _cfg.raw.get("tts", {})
                from nekoagent.tts.engine import init_engine
                init_engine(
                    model_path=_tts.get("model_path", ""),
                    device=_tts.get("device", "cpu"),
                    active_clone=_tts.get("active_clone", None),
                )
                # 同步设置表单字段
                _tts_model_field.value = model_path
                _tts_device.value = device
                _clone_audio_field.value = ""
                _clone_text_field.value = ""
                _clone_name_field.value = ""
                _refresh_clone_list()
            page.update()
        except Exception as exc:
            _clone_status.value = f"❌ {exc}"
            page.update()

    # 克隆声音列表
    _clone_list_container = ft.Column(spacing=8)
    _active_clone_label = ft.Text("当前未激活", size=12, color=ft.Colors.GREY_600, italic=True)

    def _refresh_clone_list():
        _clone_list_container.controls.clear()
        try:
            from nekoagent.tts.engine import get_engine
            engine = get_engine()
            clones = engine.list_clones()
            active = engine.get_active_clone()
            active_name = active["name"] if active else None
        except Exception:
            clones = []
            active_name = None
        if active_name:
            _active_clone_label.value = f"已激活: {active_name}"
        else:
            _active_clone_label.value = "当前未激活"
        if not clones:
            _clone_list_container.controls.append(
                ft.Container(
                    content=ft.Text("暂无克隆声音", size=12, italic=True, color=ft.Colors.GREY_600),
                    padding=10, border_radius=8,
                    bgcolor=ft.Colors.with_opacity(0.4, ft.Colors.WHITE),
                )
            )
        else:
            for c in clones:
                cname = c.get("name", "未知")
                is_active = cname == active_name

                def _handle_activate(clone_name):
                    try:
                        _log.info("TTS 激活克隆声音: %s", clone_name)
                        import copy
                        raw = copy.deepcopy(get_config().raw)
                        raw.setdefault("tts", {})
                        raw["tts"]["active_clone"] = clone_name
                        from nekoagent.config.writeback import save_config
                        save_config(raw, get_config().source_path)
                        _cfg = get_config()
                        _tts = _cfg.raw.get("tts", {})
                        from nekoagent.tts.engine import init_engine
                        engine = init_engine(
                            model_path=_tts.get("model_path", ""),
                            device=_tts.get("device", "cpu"),
                            active_clone=_tts.get("active_clone", None),
                        )
                        _log.info("TTS 引擎已重新初始化, active_clone=%s", engine.get_active_clone())
                        _tts_model_field.value = _tts.get("model_path", "")
                        _tts_device.value = _tts.get("device", "cpu")
                        _refresh_clone_list()
                    except Exception as exc:
                        _log.error("TTS 激活失败: %s", exc)

                def _handle_delete(clone_name):
                    try:
                        import json, os
                        from nekoagent.tts.engine import _VOICE_CLONE_DIR, _CLONE_INDEX_FILE, get_engine
                        engine = get_engine()
                        engine._clones.pop(clone_name, None)
                        engine._save_clone_index()
                        clone_path = _VOICE_CLONE_DIR / f"{clone_name}.pt"
                        if clone_path.exists():
                            os.remove(str(clone_path))
                        if engine._active_clone_name == clone_name:
                            engine._active_clone_name = None
                            import copy
                            raw = copy.deepcopy(get_config().raw)
                            raw.setdefault("tts", {})
                            raw["tts"]["active_clone"] = ""
                            from nekoagent.config.writeback import save_config
                            save_config(raw, get_config().source_path)
                            _cfg = get_config()
                            _tts = _cfg.raw.get("tts", {})
                            from nekoagent.tts.engine import init_engine
                            init_engine(
                                model_path=_tts.get("model_path", ""),
                                device=_tts.get("device", "cpu"),
                                active_clone=None,
                            )
                        _refresh_clone_list()
                        page.update()
                    except Exception:
                        pass

                act_btn = ft.ElevatedButton(
                    "已激活" if is_active else "激活",
                    icon=ft.Icons.CHECK_CIRCLE if is_active else ft.Icons.PLAY_ARROW,
                    on_click=lambda e, name=cname: _handle_activate(name),
                    bgcolor=ft.Colors.RED_400 if is_active else ft.Colors.PINK_400,
                    color=ft.Colors.WHITE,
                    disabled=is_active,
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                )
                card = ft.Container(
                    padding=8, border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.85, ft.Colors.RED_50) if is_active else ft.Colors.with_opacity(0.7, ft.Colors.WHITE),
                    border=border_all(1, ft.Colors.RED_300) if is_active else border_all(1, ft.Colors.with_opacity(0.2, ft.Colors.PINK_100)),
                    content=ft.Row(controls=[
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.RED_500, size=18) if is_active else ft.Icon(ft.Icons.RADIO_BUTTON_UNCHECKED, color=ft.Colors.GREY_400, size=18),
                        ft.Text(cname, size=13, weight=ft.FontWeight.BOLD,
                                color=ft.Colors.RED_800 if is_active else ft.Colors.PINK_800,
                                expand=True),
                        act_btn,
                        ft.IconButton(icon=ft.Icons.DELETE, icon_size=18, icon_color=ft.Colors.RED_400,
                                      tooltip="删除", on_click=lambda e, name=cname: _handle_delete(name)),
                    ]),
                )
                _clone_list_container.controls.append(card)
        page.update()

    _refresh_clone_list()

    tts_page = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            ft.Text("TTS 配置", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
            ft.Text("模型配置", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            _tts_model_row,
            _tts_device,
            ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
            ft.Text("克隆声音", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            _clone_audio_row,
            _clone_text_field,
            _clone_name_field,
            ft.Row(controls=[
                ft.ElevatedButton("开始克隆", icon=ft.Icons.MIC, on_click=_do_clone,
                                  bgcolor=ft.Colors.PINK_400, color=ft.Colors.WHITE,
                                  style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
                _clone_status,
            ], alignment=ft.MainAxisAlignment.CENTER),
            ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
            ft.Text("克隆声音列表", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PINK_800),
            _active_clone_label,
            _clone_list_container,
            ft.ElevatedButton("刷新列表", icon=ft.Icons.REFRESH, on_click=lambda e: _refresh_clone_list(),
                              bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.PINK_100), color=ft.Colors.PINK_800,
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
        ],
    )

    # 子页面映射
    sub_pages = {
        "角色设置": persona_page,
        "主Agent": main_page,
        "任务主管Agent": supervisor_page,
        "执行子Agent": executor_page,
        "总结用Agent": summary_page,
        "TTS设置": tts_page,
        "RAG配置": rag_page,
        "MCP服务器": mcp_page,
    }

    # ======== TabBar + TabBarView ========
    tab_bar = ft.TabBar(
        tabs=[ft.Tab(label=t) for t in sub_pages.keys()],
        scrollable=True,
    )
    tab_bar_view = ft.TabBarView(
        expand=True,
        controls=list(sub_pages.values()),
    )
    tabs = ft.Tabs(
        length=len(sub_pages),
        expand=True,
        content=ft.Column(
            expand=True,
            controls=[tab_bar, tab_bar_view],
        ),
        selected_index=0,
        animation_duration=300,
    )

    # ======== 保存 / 关闭 ========
    save_btn = ft.ElevatedButton(
        "保存配置", icon=ft.Icons.SAVE, on_click=_save,
        bgcolor=ft.Colors.PINK_400, color=ft.Colors.WHITE,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
    )
    def _rebuild_rag(_e):
        try:
            from nekoagent.memory.rag.pipeline import rebuild_index
            rebuild_index()
            _show_snack(page, "RAG 索引已重建！")
        except Exception as exc:
            _show_snack(page, format_friendly_error(exc))

    rebuild_btn = ft.OutlinedButton(
        "重建 RAG 索引", icon=ft.Icons.REFRESH, on_click=_rebuild_rag,
        icon_color=ft.Colors.PINK_700,
    )

    wrapper = glass_container(
        content=ft.Column(
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                section_title("系统设置", icon=ft.Icons.SETTINGS),
                tabs,
                ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PINK_200)),
                ft.Row([save_btn, rebuild_btn]),
            ],
        ),
        expand=True, radius=18, padding=14,
    )

    return wrapper


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