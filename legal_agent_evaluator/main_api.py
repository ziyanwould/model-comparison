# main_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict

# 确保在 legal_agent_evaluator 目录下运行时，可以正确导入
from core.evaluator import run_evaluation

app = FastAPI(
    title="法律智能体评测API",
    description="一个用于评测法律领域智能体表现的API。",
    version="0.2.0" # Version updated for new features
)

# --- Pydantic Models for Request Body ---
class AgentApiConfig(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None # For OpenAI-like agents
    api_url: Optional[str] = None  # For other custom URL based agents
    token: Optional[str] = None    # For token-based auth

class CustomConfigs(BaseModel):
    general_agent: Optional[AgentApiConfig] = None
    finetuned_agent: Optional[AgentApiConfig] = None

class EvaluateRequest(BaseModel):
    custom_configs: Optional[CustomConfigs] = None

@app.post("/evaluate", summary="执行智能体评测", description="运行预定义的测试用例，对比不同智能体的表现。可以提供自定义的API配置。")
async def evaluate_agents_endpoint(request_data: EvaluateRequest = EvaluateRequest()):
    """
    触发对配置好的智能体进行评测。
    评测流程包括：
    1. 加载 `data/golden_test_set.jsonl` 中的测试用例。
    2. 依次运行通用大模型和法律微调模型。
       - 如果在请求中提供了 `custom_configs`，则会使用用户指定的API端点和密钥。
       - 否则，使用在 `core/agents.py` 中定义的默认配置。
    3. 对比模型输出与标准答案，判断是否成功。
    4. 返回详细的评测结果列表。

    请求体示例 (可选):
    ```json
    {
        "custom_configs": {
            "general_agent": {
                "api_key": "user_openai_key_here",
                "base_url": "user_openai_base_url_here"
            },
            "finetuned_agent": {
                "api_url": "user_custom_agent_api_url_here",
                "token": "user_custom_agent_token_here"
            }
        }
    }
    ```
    """
    custom_configs_dict = None
    if request_data and request_data.custom_configs:
        custom_configs_dict = request_data.custom_configs.model_dump(exclude_none=True) # Use .model_dump() for Pydantic v2+
        # print(f"Received custom_configs: {custom_configs_dict}") # For debugging

    try:
        results = run_evaluation(custom_configs=custom_configs_dict)
        return results
    except ValueError as ve: # Catch ValueErrors from agent initialization (e.g., missing keys)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Log the exception here if you have a logger configured
        # logger.error(f"Error during evaluation: {e}", exc_info=True)
        print(f"Error during evaluation: {e}") # Simple print for now
        raise HTTPException(status_code=500, detail=f"评测过程中发生内部错误: {str(e)}")

# To run: cd legal_agent_evaluator && uvicorn main_api:app --reload
