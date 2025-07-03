# main_api.py
from fastapi import FastAPI
from core.evaluator import run_evaluation
# 确保在 legal_agent_evaluator 目录下运行时，可以正确导入
# 如果使用 python -m legal_agent_evaluator.main_api 这样的方式运行，可能需要调整导入路径

app = FastAPI(
    title="法律智能体评测API",
    description="一个用于评测法律领域智能体表现的API。",
    version="0.1.0"
)

@app.post("/evaluate", summary="执行智能体评测", description="运行预定义的测试用例，对比不同智能体的表现。")
async def evaluate_agents_endpoint():
    """
    触发对配置好的智能体进行评测。
    评测流程包括：
    1. 加载 `data/golden_test_set.jsonl` 中的测试用例。
    2. 依次运行通用大模型和法律微调模型。
    3. 对比模型输出与标准答案，判断是否成功。
    4. 返回详细的评测结果列表。
    """
    results = run_evaluation()
    return results

# 如果希望直接运行此文件进行测试 (uvicorn main_api:app --reload)
# 需要确保 core 和 data 目录与 main_api.py 在同一查找路径下
# 通常将 legal_agent_evaluator 设为工作目录即可
# 例如: cd legal_agent_evaluator && uvicorn main_api:app --reload
# 或者，如果 legal_agent_evaluator 是一个包，并且你在其父目录:
# python -m uvicorn legal_agent_evaluator.main_api:app --reload
# (这种情况下，core.evaluator 的导入需要改为 from .core.evaluator import run_evaluation)
# 为了简单起见，并假设直接在 legal_agent_evaluator 目录下运行，保持 from core.evaluator import run_evaluation
