"""NekoAgent CLI 入口。MVP 阶段直接拉起 Flet 桌面应用。"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from nekoagent.ui.flet_shell.app import run_app
    except ImportError as exc:
        print(f"[NekoAgent] 无法导入 UI 模块（依赖未安装？）：{exc}", file=sys.stderr)
        return 2

    try:
        run_app()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # pragma: no cover - 兜底
        from nekoagent.observability.exceptions import format_friendly_error
        print(format_friendly_error(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())