import datetime
import json
import platform
import socket
import subprocess
import sys
import textwrap
import uuid

import psutil

from mcp.server import FastMCP

mcp = FastMCP("NekoSystemTools", host="127.0.0.1", port=8002)


def _get_gpu_info() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return "未检测到 NVIDIA GPU 或 nvidia-smi 无输出"
        parts = []
        for i, line in enumerate(lines):
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 4:
                name, mem_total, mem_used, mem_free = cols[:4]
                parts.append(
                    f"  GPU {i}: {name}\n"
                    f"    显存总量: {mem_total} MiB\n"
                    f"    已用显存: {mem_used} MiB\n"
                    f"    空闲显存: {mem_free} MiB"
                )
            else:
                parts.append(f"  GPU {i}: {line}")
        return "\n".join(parts)
    except FileNotFoundError:
        return "未安装 NVIDIA 驱动或找不到 nvidia-smi"
    except subprocess.TimeoutExpired:
        return "查询 GPU 信息超时"
    except subprocess.CalledProcessError as e:
        return f"nvidia-smi 返回错误: {e}"
    except Exception as e:
        return f"获取 GPU 信息失败: {e}"


def _get_disk_info() -> str:
    parts = []
    for part in psutil.disk_partitions():
        if part.fstype and "cdrom" not in part.opts:
            try:
                usage = psutil.disk_usage(part.mountpoint)
                parts.append(
                    f"  {part.device} ({part.mountpoint}) [{part.fstype}]\n"
                    f"    总量: {usage.total // (1024**3)} GB\n"
                    f"    已用: {usage.used // (1024**3)} GB\n"
                    f"    可用: {usage.free // (1024**3)} GB\n"
                    f"    使用率: {usage.percent}%"
                )
            except PermissionError:
                parts.append(f"  {part.device} ({part.mountpoint}) [{part.fstype}] — 无权限访问")
    return "\n".join(parts)


def _get_network_info() -> str:
    parts = []
    net = psutil.net_if_addrs()
    io = psutil.net_io_counters()
    for name, addrs in net.items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                parts.append(f"  {name}: {addr.address}")
                break
    parts.append(f"\n  总发送: {io.bytes_sent // (1024**2)} MB")
    parts.append(f"  总接收: {io.bytes_recv // (1024**2)} MB")
    return "\n".join(parts)


@mcp.tool()
def get_current_datetime() -> str:
    """获取当前系统的日期与时间"""
    now = datetime.datetime.now()
    tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    return (
        f"当前日期: {now.strftime('%Y-%m-%d')}\n"
        f"当前时间: {now.strftime('%H:%M:%S')}\n"
        f"时区: {tz}\n"
        f"ISO 格式: {now.isoformat()}"
    )


@mcp.tool()
def get_system_info() -> str:
    """获取当前系统的硬件与软件信息, 包括 PC 名称、CPU、内存、硬盘、GPU、网络等"""
    mem = psutil.virtual_memory()
    cpu_info = platform.processor() or "未知"
    cpu_count = psutil.cpu_count(logical=True)
    cpu_phys = psutil.cpu_count(logical=False)
    cpu_freq = psutil.cpu_freq()
    cpu_freq_str = f"{cpu_freq.current:.0f} MHz" if cpu_freq else "未知"
    cpu_percent = psutil.cpu_percent(interval=0.5, percpu=False)

    info = {
        "系统": {
            "PC 名称": socket.gethostname(),
            "操作系统": platform.system(),
            "系统版本": platform.version(),
            "架构": platform.machine(),
        },
        "CPU": {
            "型号": cpu_info,
            "物理核心数": cpu_phys,
            "逻辑核心数": cpu_count,
            "当前频率": cpu_freq_str,
            "使用率": f"{cpu_percent}%",
        },
        "内存": {
            "总量": f"{mem.total // (1024**3)} GB ({mem.total} bytes)",
            "已用": f"{mem.used // (1024**3)} GB ({mem.used} bytes)",
            "可用": f"{mem.available // (1024**3)} GB ({mem.available} bytes)",
            "使用率": f"{mem.percent}%",
        },
    }

    lines = ["===== 系统信息 ====="]
    for section, items in info.items():
        lines.append(f"\n--- {section} ---")
        for k, v in items.items():
            lines.append(f"  {k}: {v}")

    lines.append("\n--- 硬盘 ---")
    lines.append(_get_disk_info())

    lines.append("\n--- GPU ---")
    lines.append(_get_gpu_info())

    lines.append("\n--- 网络 ---")
    lines.append(_get_network_info())

    return "\n".join(lines)


@mcp.tool()
def run_python_script(code: str) -> str:
    """
    运行一段 Python 脚本并返回其标准输出与标准错误。

    Args:
        code: 要执行的 Python 代码
    """
    import io
    import contextlib
    import traceback

    globals_dict = {"__name__": "__mcp__"}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        compiled = compile(textwrap.dedent(code), "<mcp>", "exec")
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compiled, globals_dict)
        stdout_out = stdout_buf.getvalue()
        stderr_out = stderr_buf.getvalue()
        parts = []
        if stdout_out:
            parts.append(f"[stdout]\n{stdout_out}")
        if stderr_out:
            parts.append(f"[stderr]\n{stderr_out}")
        if not parts:
            return "脚本执行完毕，无输出。"
        return "\n".join(parts)
    except Exception:
        return f"[错误]\n{traceback.format_exc()}"


@mcp.tool()
def run_comfyui_workflow(workflow_json: str, timeout: int = 300) -> str:
    """
    向本地 ComfyUI (127.0.0.1:8188) 提交工作流，等待生成完成后返回所有输出图像。

    Args:
        workflow_json: ComfyUI 工作流 JSON 字符串（完整 UI 格式或 API 格式均可，工具会自动提取 API 格式节点）
        timeout: 最大等待时间（秒），默认 300（5 分钟）
    """
    import httpx
    import base64
    import time
    import urllib.parse

    base_url = "http://127.0.0.1:8188"
    client_id = f"mcp-{uuid.uuid4().hex[:8]}"

    # 解析传入的 workflow JSON
    try:
        workflow = json.loads(workflow_json)
    except json.JSONDecodeError as e:
        return f"[错误] 工作流 JSON 解析失败: {e}"

    # 如果顶层有 "prompt" 键，说明用户传递的是 ComfyUI API 格式的完整请求体
    prompt_data = workflow if "prompt" not in workflow else workflow["prompt"]

    # 如果包含 "last_node_id" / "number" 等 UI 元字段，尝试自动提取 API 格式
    # ComfyUI 完整 UI 格式中节点在顶层键中（数字 ID），且有额外元字段
    meta_keys = {"last_node_id", "last_link_id", "number", "version", "export_format"}
    if meta_keys.intersection(prompt_data.keys()):
        prompt_data = {k: v for k, v in prompt_data.items() if k not in meta_keys}

    payload = {"prompt": prompt_data, "client_id": client_id}

    # ── 1. 提交工作流 ──
    try:
        with httpx.Client(base_url=base_url, timeout=30) as client:
            resp = client.post("/prompt", json=payload)
            resp.raise_for_status()
            result = resp.json()
    except httpx.RequestError as e:
        return f"[错误] 无法连接 ComfyUI ({base_url}): {e}"
    except Exception as e:
        return f"[错误] 提交工作流失败: {e}"

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        return f"[错误] ComfyUI 未返回 prompt_id: {result}"

    # ── 2. 轮询等待完成 ──
    deadline = time.monotonic() + timeout
    errors = result.get("node_errors", {})
    if errors:
        err_detail = "; ".join(f"节点 {k}: {v}" for k, v in errors.items())
        return f"[错误] 工作流存在节点错误: {err_detail}"

    with httpx.Client(base_url=base_url, timeout=30) as client:
        while time.monotonic() < deadline:
            try:
                hist_resp = client.get(f"/history/{prompt_id}")
                if hist_resp.status_code != 200:
                    time.sleep(2)
                    continue
                history = hist_resp.json()
            except Exception:
                time.sleep(2)
                continue

            prompt_data_resp = history.get(prompt_id)
            if not prompt_data_resp:
                time.sleep(2)
                continue

            status = prompt_data_resp.get("status", {})
            if status.get("completed") is True or status.get("status_str") == "success":
                outputs = prompt_data_resp.get("outputs", {})
                break
            elif status.get("status_str") == "error":
                error_msg = prompt_data_resp.get("error", {}).get("message", "未知错误")
                return f"[错误] ComfyUI 执行失败: {error_msg}"

            time.sleep(2)
        else:
            return f"[错误] 等待超时（{timeout} 秒），请尝试增大 timeout 参数"

        # ── 3. 获取输出图像 ──
        image_infos = []
        for node_id, node_output in outputs.items():
            for key, value in node_output.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "filename" in item:
                            image_infos.append(item)
                elif isinstance(value, dict) and "filename" in value:
                    image_infos.append(value)

        if not image_infos:
            outputs_str = json.dumps(outputs, ensure_ascii=False, indent=2)
            return f"工作流执行完成，但未找到图像输出。完整输出:\n{outputs_str}"

        # ── 4. 下载图像并返回 base64 ──
        markdown_parts = [f"生成完成，共 {len(image_infos)} 张图像：\n"]
        for img in image_infos:
            filename = img["filename"]
            subfolder = img.get("subfolder", "")
            img_type = img.get("type", "output")
            params = urllib.parse.urlencode({
                "filename": filename,
                "subfolder": subfolder,
                "type": img_type,
            })
            try:
                img_resp = client.get(f"/view?{params}")
                img_resp.raise_for_status()
                img_b64 = base64.b64encode(img_resp.content).decode("ascii")
                markdown_parts.append(
                    f"![{filename}](data:image/png;base64,{img_b64})"
                )
            except Exception as e:
                markdown_parts.append(f"  {filename} — 下载失败: {e}")

        return "\n".join(markdown_parts)


def main():
    mcp.run(transport="sse")

if __name__ == "__main__":
    main()
