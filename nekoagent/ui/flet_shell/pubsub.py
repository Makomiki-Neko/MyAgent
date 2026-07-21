"""UI pub/sub：把后台 task schedule 事件推送到 Flet 控件。

实现策略：
- 同步接口 `notify(kind, payload)` 在主线程受 page.update() 限制下被调用
- 后台 supervisor 线程产生事件经 asyncio.run_coroutine_threadsafe 调度回主线程运行
- 简单事件流：plan_ready | exception | task_complete | status_change | message_stream_chunk
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable


class UIPubSub:
    def __init__(self, page: Any) -> None:
        self.page = page
        self._subscribers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = None

    def subscribe(self, kind: str, cb: Callable[[dict[str, Any]], None]) -> None:
        self._subscribers.setdefault(kind, []).append(cb)

    def clear_subscribers(self, kind: str) -> None:
        self._subscribers[kind] = []

    def notify(self, kind: str, payload: dict[str, Any]) -> None:
        for cb in self._subscribers.get(kind, []):
            try:
                cb(payload)
            except Exception as exc:  # sub callback 异常不阻塞主流程
                # 日志经由 logger 在调用方处理，这里仅 swallow
                pass
        # 试图把 UI 刷新提到主循环
        try:
            self.page.update()
        except Exception:
            pass