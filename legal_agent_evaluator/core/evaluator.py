# core/evaluator.py
import json
import re
import openai
from typing import List, Dict, Optional
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

def get_similarity_score(
    query: str,
    agent_response: str,
    ground_truth: str,
    judge_agent_instance: GeneralLLMAgent
) -> float:
    """
    使用“裁判”大模型评估模型回答与标准答案的相似度。
    """
    if not agent_response or not agent_response.strip():
        print("  [Judge] Agent response is empty, similarity score: 0.0")
        return 0.0

    gt_str = json.dumps(ground_truth, ensure_ascii=False) if isinstance(ground_truth, dict) else str(ground_truth)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        ground_truth=gt_str,
        agent_response=str(agent_response)
    )
    # print(f"  [Judge] Calling judge agent ({judge_agent_instance.name}, Model: {judge_agent_instance.model_name}) for similarity score...")
    # print(f"  [Judge] Prompt: \n{prompt[:500]}...")

    try:
        judge_response_text = ""
        # Direct synchronous call using the client
        completion = judge_agent_instance.client.chat.completions.create(
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
            print("  [Judge] Judge agent did not return content.")
            return 0.0

        # print(f"  [Judge] Raw response: '{judge_response_text}'")

        score_match = re.search(r"(\d\.\d+)", judge_response_text)
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))
            # print(f"  [Judge] Parsed score (regex): {score}")
            return score
        else:
            try:
                score = float(judge_response_text)
                score = max(0.0, min(1.0, score))
                # print(f"  [Judge] Parsed score (direct float): {score}")
                return score
            except ValueError:
                print(f"  [Judge] Could not parse score from response: '{judge_response_text}'")
                return 0.0

    except openai.APIError as e:
        print(f"  [Judge] Judge Agent API Error: {e}")
        return 0.0
    except Exception as e:
        print(f"  [Judge] Unexpected error during judging: {e}")
        return 0.0

def load_test_set(file_path: str) -> list:
    test_cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_cases.append(json.loads(line.strip()))
    return test_cases

def evaluate_agent(
    agent_to_evaluate: BaseAgent,
    test_cases: list,
    judge_agent: GeneralLLMAgent
) -> list:
    results = []
    print(f"Starting evaluation for agent: {agent_to_evaluate.name}") # Server log
    for i, case in enumerate(test_cases):
        query = case['query']
        ground_truth = case['ground_truth']

        # print(f"  Processing case {i+1}/{len(test_cases)}: ID {case['id']}, Query: '{query[:30]}...'")
        agent_response = agent_to_evaluate.run(query) # Synchronous call
        # print(f"    Agent ({agent_to_evaluate.name}) Response: '{str(agent_response)[:50]}...'")

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
            # print(f"    Case {case['id']}: Exact match.")
        else:
            # print(f"    Case {case['id']}: No exact match. Calling judge agent for similarity score.")
            similarity_score = get_similarity_score(query, agent_response, ground_truth, judge_agent) # Synchronous call
            # print(f"    Case {case['id']}: Similarity score = {similarity_score:.2f}")

        is_success = similarity_score >= SIMILARITY_THRESHOLD
        # print(f"    Case {case['id']}: Success ({SIMILARITY_THRESHOLD=}): {is_success}")

        results.append({
            "id": case['id'], "type": case['type'], "query": query,
            "ground_truth": ground_truth, "agent_response": agent_response,
            "exact_match": exact_match, "similarity_score": similarity_score,
            "is_success": is_success, "agent_name": agent_to_evaluate.name
        })
    print(f"Finished evaluation for agent: {agent_to_evaluate.name}") # Server log
    return results

def run_evaluation(custom_configs: Optional[Dict] = None) -> list:
    print("Starting full evaluation run...") # Server log
    if custom_configs:
        print(f"Using custom configurations: {json.dumps(custom_configs, indent=2)}") # Server log
    else:
        print("No custom configurations provided, using defaults.") # Server log

    test_cases = load_test_set("data/golden_test_set.jsonl")
    print(f"Loaded {len(test_cases)} test cases.") # Server log

    general_agent_config = custom_configs.get("general_agent") if custom_configs else {}
    if general_agent_config is None: general_agent_config = {}

    judge_and_eval_agent = GeneralLLMAgent(
        name="通用大模型（OpenAI API）",
        api_key=general_agent_config.get("api_key"),
        base_url=general_agent_config.get("base_url"),
        model_name=general_agent_config.get("model_name")
    )
    print(f"Initialized GeneralLLMAgent/Judge: {judge_and_eval_agent.name} (Model: {judge_and_eval_agent.model_name})") # Server log

    finetuned_agent_config = custom_configs.get("finetuned_agent") if custom_configs else {}
    if finetuned_agent_config is None: finetuned_agent_config = {}

    fine_tuned_agent = FineTunedAgent(
        api_url=finetuned_agent_config.get("api_url"),
        token=finetuned_agent_config.get("token")
    )
    print(f"Initialized FineTunedAgent: {fine_tuned_agent.name}") # Server log

    all_results = []

    print("\n--- Evaluating GeneralLLMAgent ---") # Server log
    general_agent_results = evaluate_agent(
        agent_to_evaluate=judge_and_eval_agent,
        test_cases=test_cases,
        judge_agent=judge_and_eval_agent
    )
    for res in general_agent_results:
        all_results.append({
            "任务ID": res['id'], "任务类型": res['type'], "查询语句": res['query'],
            "标准答案": json.dumps(res['ground_truth'], ensure_ascii=False) if isinstance(res['ground_truth'], dict) else res['ground_truth'],
            "模型回答": res['agent_response'], "完全匹配": res['exact_match'],
            "相似度得分": f"{res['similarity_score']:.2f}", "是否成功": res['is_success'],
            "智能体": res['agent_name']
        })

    print("\n--- Evaluating FineTunedAgent ---") # Server log
    finetuned_agent_results = evaluate_agent(
        agent_to_evaluate=fine_tuned_agent,
        test_cases=test_cases,
        judge_agent=judge_and_eval_agent
    )
    for res in finetuned_agent_results:
        all_results.append({
            "任务ID": res['id'], "任务类型": res['type'], "查询语句": res['query'],
            "标准答案": json.dumps(res['ground_truth'], ensure_ascii=False) if isinstance(res['ground_truth'], dict) else res['ground_truth'],
            "模型回答": res['agent_response'], "完全匹配": res['exact_match'],
            "相似度得分": f"{res['similarity_score']:.2f}", "是否成功": res['is_success'],
            "智能体": res['agent_name']
        })

    print("\nFull evaluation run completed.") # Server log
    return all_results
