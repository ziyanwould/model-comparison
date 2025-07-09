# core/evaluator.py
import json
import re
import openai
import asyncio # Added for asyncio.Queue
from typing import List, Dict, Optional, Callable, Any # Added Callable, Any
from .agents import BaseAgent, GeneralLLMAgent, FineTunedAgent

# --- Constants for Similarity Scoring ---
SIMILARITY_THRESHOLD = 0.7

# --- Prompt Template for Judgment ---
JUDGE_PROMPT_TEMPLATE = """\
请根据以下问题、标准答案和模型回答，评估模型回答的质量。
请专注于模型回答是否在语义上准确地回应了问题，并且其核心信息是否与标准答案一致或等价。
请给出一个0.0到1.0之间的小数评分。1.0表示模型回答与标准答案在语义上完全一致或同等正确地回答了问题。0.0表示模型回答完全不相关、错误或未能回答问题。

问题：
{query}

标准答案：
{ground_truth}

模型回答：
{agent_response}

请在下面仅提供评分数值 (例如: 0.8):
评分 (0.0 - 1.0):"""

# --- Type alias for the logger callback ---
# A simple logger function that takes a string message
LogCallable = Callable[[str], None]

async def _log_message(log_queue: Optional[asyncio.Queue], message: str):
    """Helper to send a message to the log_queue if it exists."""
    prefix = "[Backend Log]" # Simple prefix
    # print(f"{prefix} {message}") # Also print to server console for debugging
    if log_queue:
        try:
            await log_queue.put(f"{prefix} {message}")
        except Exception as e:
            print(f"Error putting message in log_queue: {e}")


async def get_similarity_score(
    query: str,
    agent_response: str,
    ground_truth: str,
    judge_agent_instance: GeneralLLMAgent,
    log_queue: Optional[asyncio.Queue] = None
) -> float:
    if not agent_response or not agent_response.strip():
        await _log_message(log_queue, "评委：被评估智能体的回答为空，相似度得分：0.0")
        return 0.0

    gt_str = json.dumps(ground_truth, ensure_ascii=False) if isinstance(ground_truth, dict) else str(ground_truth)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        ground_truth=gt_str,
        agent_response=str(agent_response)
    )
    await _log_message(log_queue, f"评委：调用裁判模型 ({judge_agent_instance.name}, 模型: {judge_agent_instance.model_name}) 进行相似度评分...")

    try:
        judge_response_text = ""
        # Assuming judge_agent_instance.client.chat.completions.create is an async method
        # If it's synchronous, it needs to be run in a thread pool executor for async context
        # For now, let's assume it's compatible or we'll adjust if openai client is sync

        # The openai client v1.x.x uses httpx, which can be async.
        # We need to ensure the client was initialized for async if we 'await' its methods.
        # However, GeneralLLMAgent's client is initialized synchronously.
        # To make a non-blocking call in an async def, we should use asyncio.to_thread (Python 3.9+)
        # or loop.run_in_executor for older versions.

        # For simplicity in this step, if client.chat.completions.create is sync, this will block.
        # This will be addressed when making run_evaluation fully async if necessary.
        # Let's assume for now direct call is acceptable for the flow.
        completion = await asyncio.to_thread(
            judge_agent_instance.client.chat.completions.create,
            model=judge_agent_instance.model_name,
            messages=[
                {"role": "system", "content": "You are an impartial evaluator. Your task is to provide a numerical score based on the user's instructions. Output only the numerical score."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10,
            stream=False,
            timeout=45
        )

        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            judge_response_text = completion.choices[0].message.content.strip()
        else:
            await _log_message(log_queue, "评委：裁判模型未返回有效内容。")
            return 0.0

        await _log_message(log_queue, f"评委：裁判模型原始回复: '{judge_response_text}'")

        score_match = re.search(r"(\d\.\d+)", judge_response_text)
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))
            await _log_message(log_queue, f"评委：解析得到分数 (正则): {score:.2f}")
            return score
        else:
            try:
                score = float(judge_response_text)
                score = max(0.0, min(1.0, score))
                await _log_message(log_queue, f"评委：解析得到分数 (直接转换): {score:.2f}")
                return score
            except ValueError:
                await _log_message(log_queue, f"评委：无法从回复中解析分数: '{judge_response_text}'")
                return 0.0

    except openai.APIError as e:
        await _log_message(log_queue, f"评委：裁判模型API错误: {e}")
        return 0.0
    except Exception as e:
        await _log_message(log_queue, f"评委：裁判过程中发生未知错误: {e}")
        return 0.0

def load_test_set(file_path: str) -> list:
    test_cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_cases.append(json.loads(line.strip()))
    return test_cases

async def evaluate_agent(
    agent_to_evaluate: BaseAgent,
    test_cases: list,
    judge_agent: GeneralLLMAgent,
    log_queue: Optional[asyncio.Queue] = None
) -> list:
    results = []
    await _log_message(log_queue, f"开始评测智能体: {agent_to_evaluate.name}")
    for i, case in enumerate(test_cases):
        query = case['query']
        ground_truth = case['ground_truth']

        await _log_message(log_queue, f"  正在处理案例 {i+1}/{len(test_cases)}: ID {case['id']}, 查询: '{query[:30]}...'")

        # Agent's run method might be synchronous.
        # If agent_to_evaluate.run is sync, run it in a thread_pool_executor
        agent_response = await asyncio.to_thread(agent_to_evaluate.run, query)
        await _log_message(log_queue, f"    智能体 ({agent_to_evaluate.name}) 回答: '{str(agent_response)[:50]}...'")

        exact_match = False
        if isinstance(ground_truth, dict):
            try:
                if isinstance(agent_response, str) and agent_response.strip().startswith("{") and agent_response.strip().endswith("}"):
                    agent_response_json = json.loads(agent_response)
                    exact_match = agent_response_json == ground_truth
                else:
                    exact_match = False
            except json.JSONDecodeError:
                exact_match = False
        else:
            exact_match = agent_response == ground_truth

        similarity_score = 0.0
        if exact_match:
            similarity_score = 1.0
            await _log_message(log_queue, f"    案例 {case['id']}: 完全匹配。相似度得分: 1.00")
        else:
            await _log_message(log_queue, f"    案例 {case['id']}: 非完全匹配。调用裁判模型进行相似度评分...")
            similarity_score = await get_similarity_score(query, agent_response, ground_truth, judge_agent, log_queue)
            await _log_message(log_queue, f"    案例 {case['id']}: 从裁判模型获取相似度得分: {similarity_score:.2f}")

        is_success = similarity_score >= SIMILARITY_THRESHOLD
        await _log_message(log_queue, f"    案例 {case['id']}: 最终判定是否成功 ({SIMILARITY_THRESHOLD=}): {is_success}")

        results.append({
            "id": case['id'], "type": case['type'], "query": query,
            "ground_truth": ground_truth, "agent_response": agent_response,
            "exact_match": exact_match, "similarity_score": similarity_score,
            "is_success": is_success, "agent_name": agent_to_evaluate.name
        })
    await _log_message(log_queue, f"完成对智能体 {agent_to_evaluate.name} 的评测。")
    return results

async def run_evaluation(custom_configs: Optional[Dict] = None, log_queue: Optional[asyncio.Queue] = None) -> list:
    await _log_message(log_queue, "完整评测流程开始...")
    if custom_configs:
        await _log_message(log_queue, f"使用自定义配置: {json.dumps(custom_configs, indent=2)}")
    else:
        await _log_message(log_queue, "未使用自定义配置，将使用默认值。")

    test_cases = await asyncio.to_thread(load_test_set, "data/golden_test_set.jsonl")
    await _log_message(log_queue, f"已加载 {len(test_cases)} 个测试用例。")

    general_agent_config = custom_configs.get("general_agent") if custom_configs else {}
    if general_agent_config is None: general_agent_config = {}

    # This agent is used for eval and as judge. Init it once.
    # Agent initialization is synchronous.
    judge_and_eval_agent = await asyncio.to_thread(
        GeneralLLMAgent,
        name="通用大模型（OpenAI API）",
        api_key=general_agent_config.get("api_key"),
        base_url=general_agent_config.get("base_url"),
        model_name=general_agent_config.get("model_name")
    )
    await _log_message(log_queue, f"已初始化通用大模型/裁判智能体: {judge_and_eval_agent.name} (模型: {judge_and_eval_agent.model_name})")

    finetuned_agent_config = custom_configs.get("finetuned_agent") if custom_configs else {}
    if finetuned_agent_config is None: finetuned_agent_config = {}

    fine_tuned_agent = await asyncio.to_thread(
        FineTunedAgent,
        api_url=finetuned_agent_config.get("api_url"),
        token=finetuned_agent_config.get("token")
    )
    await _log_message(log_queue, f"已初始化微调后模型: {fine_tuned_agent.name}")

    all_results = []

    await _log_message(log_queue, "\n--- 开始评测: 通用大模型 ---")
    general_agent_results = await evaluate_agent(
        agent_to_evaluate=judge_and_eval_agent,
        test_cases=test_cases,
        judge_agent=judge_and_eval_agent,
        log_queue=log_queue
    )
    for res in general_agent_results:
        all_results.append({
            "任务ID": res['id'], "任务类型": res['type'], "查询语句": res['query'],
            "标准答案": json.dumps(res['ground_truth'], ensure_ascii=False) if isinstance(res['ground_truth'], dict) else res['ground_truth'],
            "模型回答": res['agent_response'], "完全匹配": res['exact_match'],
            "相似度得分": f"{res['similarity_score']:.2f}", "是否成功": res['is_success'],
            "智能体": res['agent_name']
        })

    await _log_message(log_queue, "\n--- 开始评测: 微调后模型 ---")
    finetuned_agent_results = await evaluate_agent(
        agent_to_evaluate=fine_tuned_agent,
        test_cases=test_cases,
        judge_agent=judge_and_eval_agent,
        log_queue=log_queue
    )
    for res in finetuned_agent_results:
        all_results.append({
            "任务ID": res['id'], "任务类型": res['type'], "查询语句": res['query'],
            "标准答案": json.dumps(res['ground_truth'], ensure_ascii=False) if isinstance(res['ground_truth'], dict) else res['ground_truth'],
            "模型回答": res['agent_response'], "完全匹配": res['exact_match'],
            "相似度得分": f"{res['similarity_score']:.2f}", "是否成功": res['is_success'],
            "智能体": res['agent_name']
        })

    await _log_message(log_queue, "\n完整评测流程结束。")
    return all_results
