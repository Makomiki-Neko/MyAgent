from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.main_agent import MainAgent
from app.task.task_manager import TaskManager

router = APIRouter(prefix="/api/v1")

main_agent: MainAgent = None
task_manager: TaskManager = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class TaskSubmitRequest(BaseModel):
    objective: str


class TaskModifyRequest(BaseModel):
    task_id: str
    feedback: str


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not main_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    response = main_agent.chat(req.message)
    return ChatResponse(response=response)


@router.post("/tasks")
async def submit_task(req: TaskSubmitRequest):
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
    task_id = task_manager.submit_task(req.objective)
    return {"task_id": task_id, "status": "pending_confirmation"}


@router.get("/tasks")
async def list_tasks():
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
    tasks = task_manager.get_all_tasks()
    return [
        {
            "task_id": t.task_id,
            "objective": t.objective,
            "status": t.status.value,
            "progress": t.progress,
            "current_subtask": t.current_subtask,
            "error": t.error,
        }
        for t in tasks
    ]


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task.task_id,
        "objective": task.objective,
        "status": task.status.value,
        "progress": task.progress,
        "subtasks": [
            {
                "id": st.id,
                "description": st.description,
                "status": st.status.value,
                "result": st.result,
                "error": st.error,
            }
            for st in task.subtasks
        ],
    }


@router.post("/tasks/{task_id}/confirm")
async def confirm_task(task_id: str):
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
    ok = task_manager.confirm_task(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot confirm task")
    return {"status": "confirmed"}


@router.post("/tasks/{task_id}/modify")
async def modify_task(task_id: str, req: TaskModifyRequest):
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
    ok = task_manager.modify_task(task_id, req.feedback)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot modify task")
    return {"status": "modified"}


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
    ok = task_manager.cancel_task(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot cancel task")
    return {"status": "cancelled"}


@router.get("/persona")
async def get_persona():
    from app.config import get_agent_config
    cfg = get_agent_config("main_agent")
    return {"persona": cfg.get("persona_prompt", "")}


@router.get("/memory")
async def get_memory():
    from app.memory.memory_manager import MemoryManager
    mm = MemoryManager()
    return {
        "episodic_count": len(mm.episodic._episodes) if hasattr(mm.episodic, "_episodes") else 0,
        "semantic_data": mm.semantic.all_data if hasattr(mm.semantic, "all_data") else {},
    }
