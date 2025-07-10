# main_api.py
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse # Ensure StreamingResponse is imported
from pydantic import BaseModel
from typing import Optional, Dict, AsyncGenerator # Added AsyncGenerator for type hint
import json # For json.dumps

from core.evaluator import run_evaluation # run_evaluation is now a generator

app = FastAPI(
    title="法律智能体评测API",
    description="一个用于评测法律领域智能体表现的API，支持流式结果输出。",
    version="0.2.2" # Version reflects streaming results
)

# --- Pydantic Models for Request Body (from phase 1 extension) ---
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

async def evaluation_event_generator(custom_configs_dict: Optional[Dict]) -> AsyncGenerator[str, None]:
    """
    Asynchronous generator wrapper for the synchronous run_evaluation generator.
    Yields each result item as a JSON string followed by a newline.
    """
    try:
        for result_item in run_evaluation(custom_configs=custom_configs_dict):
            yield json.dumps(result_item, ensure_ascii=False) + "\n"
            # Add a small sleep if needed to prevent overwhelming the client or event loop,
            # especially if run_evaluation yields very rapidly without I/O.
            # However, since run_evaluation itself has I/O (API calls), this might not be strictly necessary.
            # await asyncio.sleep(0.001) # Optional: uncomment if needed
    except Exception as e:
        # In case of an error during the generation (e.g., from run_evaluation itself)
        # it's tricky to report this mid-stream in NDJSON without breaking format.
        # The error will likely be logged server-side by run_evaluation.
        # Client will see the stream end prematurely.
        # For a more robust solution, a wrapper could yield a final error JSON object.
        print(f"Error in evaluation_event_generator: {e}") # Log error
        # yield json.dumps({"type": "error", "detail": str(e)}) + "\n" # Optional: try to send error
        raise # Or re-raise to let FastAPI handle it if it can before streaming starts

@app.post(
    "/evaluate",
    summary="执行智能体评测 (流式NDJSON结果)",
    description="运行预定义的测试用例，流式返回每个测试用例的评测结果 (NDJSON格式)。"
    # No explicit response_model here because StreamingResponse handles it.
)
async def evaluate_agents_streaming_endpoint(request_data: EvaluateRequest = EvaluateRequest()):
    """
    触发对配置好的智能体进行评测，并以NDJSON格式流式返回每个测试用例的结果。
    每个JSON对象通过换行符分隔。
    """
    custom_configs_dict = None
    if request_data and request_data.custom_configs:
        custom_configs_dict = request_data.custom_configs.model_dump(exclude_none=True)

    # run_evaluation is a synchronous generator. StreamingResponse needs an async generator.
    # So, we wrap run_evaluation's iteration in an async generator function.
    return StreamingResponse(
        evaluation_event_generator(custom_configs_dict),
        media_type="application/x-ndjson"
    )

# To run: cd legal_agent_evaluator && uvicorn main_api:app --reload
