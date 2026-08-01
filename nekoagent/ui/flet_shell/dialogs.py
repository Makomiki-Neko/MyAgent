"""使用 tkinter 调用系统原生文件/文件夹选择框（绕过 Flet FilePicker 的异步桥接问题）。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


def _tk_pick_file(title: str, file_types: list[tuple[str, str]]) -> str:
    """在后台线程打开 tkinter 文件选择框，返回选中路径或空字符串。"""
    result: list[str] = [""]

    def _run():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(title=title, filetypes=file_types)
            root.destroy()
            result[0] = path or ""
        except Exception:
            result[0] = ""

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    return result[0]


def _tk_pick_directory(title: str) -> str:
    """在后台线程打开 tkinter 文件夹选择框，返回选中路径或空字符串。"""
    result: list[str] = [""]

    def _run():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title=title)
            root.destroy()
            result[0] = path or ""
        except Exception:
            result[0] = ""

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    return result[0]


def pick_document_file() -> str:
    """选择文档文件（txt/docx）。"""
    return _tk_pick_file(
        "选择文档",
        [("文档文件", "*.txt *.docx"), ("文本文件", "*.txt"), ("Word 文档", "*.docx")],
    )


def pick_image_file() -> str:
    """选择图片文件。"""
    return _tk_pick_file(
        "选择图片",
        [("图片文件", "*.png *.jpg *.jpeg *.gif *.webp"), ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")],
    )


def pick_model_directory() -> str:
    """选择模型目录。"""
    return _tk_pick_directory("选择模型目录")


def pick_audio_file() -> str:
    """选择音频文件。"""
    return _tk_pick_file(
        "选择音频文件",
        [("音频文件", "*.wav *.mp3 *.flac *.ogg"), ("WAV", "*.wav"), ("MP3", "*.mp3")],
    )


def update_field_from_dialog(field: Any, dialog_result: str) -> None:
    """更新 TextField 的值并返回是否成功。"""
    if dialog_result:
        field.value = dialog_result


def crop_and_save_image(src_path: str, target_w: int = 110, target_h: int = 120) -> str:
    """打开 tkinter 裁剪窗口，返回裁剪后保存的路径。若用户取消返回空字符串。"""
    result: list[str] = [""]

    def _run():
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
            from pathlib import Path
            import uuid

            img = Image.open(src_path)
            # 缩放显示以适应屏幕
            display_max = 600
            iw, ih = img.size
            scale = min(display_max / iw, display_max / ih, 1.0)
            display_w, display_h = int(iw * scale), int(ih * scale)
            img_display = img.resize((display_w, display_h), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img_display)

            root = tk.Toplevel()
            root.title("裁剪角色立绘")
            root.attributes("-topmost", True)
            root.resizable(False, False)

            crop_data = {"x1": 0, "y1": 0, "x2": display_w, "y2": display_h, "drawing": False, "rect_id": None}

            canvas = tk.Canvas(root, width=display_w, height=display_h, cursor="cross")
            canvas.pack()
            canvas.create_image(0, 0, anchor="nw", image=tk_img)

            def on_press(e):
                crop_data["x1"] = e.x
                crop_data["y1"] = e.y
                crop_data["drawing"] = True
                if crop_data["rect_id"]:
                    canvas.delete(crop_data["rect_id"])
                crop_data["rect_id"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)

            def on_drag(e):
                if crop_data["drawing"] and crop_data["rect_id"]:
                    canvas.coords(crop_data["rect_id"], crop_data["x1"], crop_data["y1"], e.x, e.y)

            def on_release(e):
                crop_data["drawing"] = False
                crop_data["x2"] = e.x
                crop_data["y2"] = e.y

            def do_crop():
                x1, y1 = crop_data["x1"], crop_data["y1"]
                x2, y2 = crop_data["x2"], crop_data["y2"]
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                if x2 - x1 < 10 or y2 - y1 < 10:
                    # 选区太小，使用全图
                    x1, y1, x2, y2 = 0, 0, display_w, display_h
                # 映射回原始坐标
                ox1 = int(x1 / scale)
                oy1 = int(y1 / scale)
                ox2 = int(x2 / scale)
                oy2 = int(y2 / scale)
                crop_img = img.crop((ox1, oy1, ox2, oy2))
                # 缩放到目标尺寸
                crop_img = crop_img.resize((target_w, target_h), Image.LANCZOS)
                save_dir = Path(src_path).parent.parent / "data" / "avatars" if "data" not in src_path else Path(src_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                fname = f"avatar_{uuid.uuid4().hex[:8]}.png"
                save_path = save_dir / fname
                crop_img.save(str(save_path))
                result[0] = str(save_path)
                root.destroy()

            def on_cancel():
                root.destroy()

            canvas.bind("<ButtonPress-1>", on_press)
            canvas.bind("<B1-Motion>", on_drag)
            canvas.bind("<ButtonRelease-1>", on_release)

            btn_frame = tk.Frame(root)
            btn_frame.pack(pady=6)
            tk.Button(btn_frame, text="确认裁剪", command=do_crop, bg="#4CAF50", fg="white", padx=16).pack(side="left", padx=8)
            tk.Button(btn_frame, text="取消", command=on_cancel, padx=16).pack(side="left", padx=8)

            root.protocol("WM_DELETE_WINDOW", on_cancel)
            root.grab_set()
            root.wait_window()
        except Exception:
            result[0] = ""

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    return result[0]

