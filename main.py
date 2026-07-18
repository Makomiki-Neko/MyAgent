import argparse
import logging
import sys

from app.config import get_logging_config


def setup_logging():
    cfg = get_logging_config()
    logging.basicConfig(
        level=getattr(logging, cfg["level"], logging.INFO),
        format=cfg["format"],
        handlers=[
            logging.FileHandler(cfg["file"], encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_cli():
    from app.cli.cli import CLIApp
    app = CLIApp()
    app.run()


def run_api(host: str = "0.0.0.0", port: int = 8000):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from app.api.routes import router
    from app.agents.main_agent import MainAgent
    from app.agents.supervisor_agent import SupervisorAgent
    from app.memory.memory_manager import MemoryManager
    from app.task.task_manager import TaskManager
    import app.api.routes as routes

    mm = MemoryManager()
    ma = MainAgent(mm)
    sup = SupervisorAgent(mm.supervisor_procedural)
    tm = TaskManager(sup)
    tm.set_plan_ready_callback(lambda t: print(f"[Task Plan Ready] {t.task_id}"))
    tm.set_task_completed_callback(lambda tid, s: print(f"[Task Completed] {tid}"))
    ma.set_task_callback(lambda obj, _hist: tm.submit_task(obj))
    routes.main_agent = ma
    routes.task_manager = tm

    app = FastAPI(title="多Agent用户交互系统", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/")
    async def root():
        return {"service": "多Agent用户交互系统", "version": "0.2.0", "status": "running"}

    import uvicorn
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="多Agent用户交互系统")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli", help="运行模式")
    parser.add_argument("--host", default="0.0.0.0", help="API 监听地址")
    parser.add_argument("--port", type=int, default=8000, help="API 监听端口")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Starting in {args.mode} mode")

    if args.mode == "api":
        run_api(host=args.host, port=args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
