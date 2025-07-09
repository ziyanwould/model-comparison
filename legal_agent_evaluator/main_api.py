# main_api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from starlette.responses import StreamingResponse # For SSE
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import asyncio
import uuid # For generating unique evaluation IDs

from core.evaluator import run_evaluation # run_evaluation is now async

app = FastAPI(
    title="法律智能体评测API",
    description="一个用于评测法律领域智能体表现的API，支持实时日志流。",
    version="0.3.0"
)

# --- Global store for evaluation tasks and their log queues ---
# In a production scenario, you might use Redis or another shared store
# if you have multiple API worker processes. For a single process, a dict is fine.
evaluation_tasks: Dict[str, Dict[str, Any]] = {}
# Structure: { "eval_id": {"queue": asyncio.Queue, "status": "running/completed/error", "task": asyncio.Task} }


# --- Pydantic Models for Request Body ---
class AgentApiConfig(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_url: Optional[str] = None
    token: Optional[str] = None
    model_name: Optional[str] = None

class CustomConfigs(BaseModel):
    general_agent: Optional[AgentApiConfig] = None
    finetuned_agent: Optional[AgentApiConfig] = None

class EvaluateRequest(BaseModel):
    custom_configs: Optional[CustomConfigs] = None

class EvaluateResponse(BaseModel):
    message: str
    evaluation_id: str
    logs_url: str
    results_url: Optional[str] = None # URL to fetch results later if needed


async def run_evaluation_and_manage_queue(
    eval_id: str,
    custom_configs: Optional[Dict] = None
):
    """Wrapper to run evaluation and handle queue completion."""
    log_queue = evaluation_tasks[eval_id]["queue"]
    try:
        await log_queue.put("[Backend Log] 评测任务开始...")
        results = await run_evaluation(custom_configs=custom_configs, log_queue=log_queue)
        evaluation_tasks[eval_id]["status"] = "completed"
        evaluation_tasks[eval_id]["results"] = results # Store results
        await log_queue.put("[Backend Log] 评测任务成功完成。日志流即将结束。")
    except Exception as e:
        evaluation_tasks[eval_id]["status"] = "error"
        error_message = f"[Backend Log] 评测任务发生错误: {type(e).__name__} - {str(e)}. 日志流即将结束。"
        print(error_message) # Also print to server console
        if log_queue: # Check if queue still exists
            await log_queue.put(error_message)
    finally:
        if log_queue: # Signal that no more logs will be added
             await log_queue.put(None) # Sentinel value to end log streaming

@app.post("/evaluate", summary="开始智能体评测并返回评测ID", response_model=EvaluateResponse)
async def start_evaluation_endpoint(
    request_data: EvaluateRequest = EvaluateRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    eval_id = str(uuid.uuid4())
    log_queue = asyncio.Queue()

    # Store the task and queue
    # The asyncio.Task is stored so we can potentially cancel it if needed, though not implemented here.
    task = asyncio.create_task(
        run_evaluation_and_manage_queue(
            eval_id=eval_id,
            custom_configs=request_data.custom_configs.model_dump(exclude_none=True) if request_data.custom_configs else None
        )
    )
    evaluation_tasks[eval_id] = {"queue": log_queue, "status": "running", "task": task, "results": None}

    # background_tasks.add_task(task) # This is another way to run it, create_task is also fine.

    return EvaluateResponse(
        message="评测任务已启动。请使用以下ID订阅日志流。",
        evaluation_id=eval_id,
        logs_url=f"/evaluate/stream-logs/{eval_id}",
        # results_url=f"/evaluate/results/{eval_id}" # Example for a future results endpoint
    )

# This is子步骤 1.3：在 `main_api.py` 中创建SSE日志流端点
async def log_stream_generator(eval_id: str):
    """Async generator for streaming logs."""
    if eval_id not in evaluation_tasks:
        # This might happen if client connects too fast or eval_id is wrong
        yield f"data: [System] Error: Evaluation ID '{eval_id}' not found or already cleaned up.\n\n"
        return

    log_queue = evaluation_tasks[eval_id]["queue"]
    yield f"data: [System] 连接到评测ID '{eval_id}' 的日志流...\n\n"

    try:
        while True:
            log_message = await log_queue.get()
            if log_message is None: # Sentinel to indicate end of logs
                yield f"data: [System] 日志流结束。\n\n"
                # Optionally, clean up the task here if it's fully done
                # Be careful with timing if results are fetched separately
                break
            yield f"data: {log_message}\n\n"
            await asyncio.sleep(0.01) # Small sleep to allow other tasks to run if queue is spammed
    except asyncio.CancelledError:
        yield f"data: [System] 日志流被取消。\n\n"
    except Exception as e:
        yield f"data: [System] 日志流发生错误: {e}\n\n"
    finally:
        # Clean up the specific eval_id task from evaluation_tasks after streaming is done
        # or after a certain timeout if the client disconnects.
        # For simplicity, let's remove it here.
        # A more robust system might have a separate cleanup mechanism or TTL for tasks.
        if eval_id in evaluation_tasks:
            # If status is completed or error, it's safe to remove.
            # If still running and client disconnected, the background task continues.
            # For now, let's assume we clean up after client disconnects from log stream
            # or after logs end.
            # To prevent removing while run_evaluation_and_manage_queue is still writing results:
            # Only remove if task is done or errored.
            # This part needs careful thought in a prod system.
            # For now, let's assume the None sentinel from the queue means the main task is also finishing.
            # await asyncio.sleep(2) # Give a small grace period for final messages
            if evaluation_tasks[eval_id]["status"] in ["completed", "error"]:
                 print(f"Cleaning up task for eval_id: {eval_id}")
                 del evaluation_tasks[eval_id]
            else:
                # If client disconnects but task is still running, we might not want to delete immediately.
                # This is a simplification.
                print(f"Log stream for {eval_id} ended, but task status is '{evaluation_tasks[eval_id]['status']}'. Not deleting immediately.")


@app.get("/evaluate/stream-logs/{eval_id}", summary="获取实时评测日志 (SSE)")
async def stream_logs_endpoint(eval_id: str):
    if eval_id not in evaluation_tasks:
        raise HTTPException(status_code=404, detail=f"评测ID '{eval_id}' 未找到。可能已完成或不存在。")

    # Check if the task is running or has logs to stream
    # This is a simple check. More robust would be to check queue status.
    # if evaluation_tasks[eval_id]["status"] not in ["running", "completed", "error"]:
    #     raise HTTPException(status_code=404, detail=f"评测ID '{eval_id}' 的日志当前不可用。")

    return StreamingResponse(log_stream_generator(eval_id), media_type="text/event-stream")


@app.get("/evaluate/results/{eval_id}", summary="获取特定评测任务的结果")
async def get_evaluation_results(eval_id: str):
    if eval_id not in evaluation_tasks:
        raise HTTPException(status_code=404, detail=f"评测ID '{eval_id}' 未找到。")

    task_info = evaluation_tasks[eval_id]
    if task_info["status"] == "running":
        raise HTTPException(status_code=202, detail="评测仍在进行中。请稍后再试。")
    if task_info["status"] == "error":
        raise HTTPException(status_code=500, detail="评测任务以错误结束。无法获取结果。")
    if task_info["status"] == "completed" and task_info["results"] is not None:
        return task_info["results"]

    raise HTTPException(status_code=404, detail="评测结果不可用或任务状态未知。")

# To run: cd legal_agent_evaluator && uvicorn main_api:app --reload
