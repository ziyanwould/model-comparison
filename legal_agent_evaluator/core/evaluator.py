# core/evaluator.py
import json
import re
import openai
from typing import List, Dict, Optional, Iterator # Changed from List to Iterator for generator type hint
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
    if not agent_response or not agent_response.strip():
        print("  [Judge] Agent response is empty, similarity score: 0.0")
        return 0.0
    gt_str = json.dumps(ground_truth, ensure_ascii=False) if isinstance(ground_truth, dict) else str(ground_truth)
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, ground_truth=gt_str, agent_response=str(agent_response))
    try:
        completion = judge_agent_instance.client.chat.completions.create(
            model=judge_agent_instance.model_name,
            messages=[
                {"role": "system", "content": "You are an impartial evaluator. Your task is to provide a numerical score based on the user's instructions. Output only the numerical score."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=10, stream=False, timeout=45
        )
        judge_response_text = ""
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            judge_response_text = completion.choices[0].message.content.strip()
        else:
            print("  [Judge] Judge agent did not return content.")
            return 0.0
        score_match = re.search(r"(\d\.\d+)", judge_response_text)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        else:
            try:
                score = float(judge_response_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                print(f"  [Judge] Could not parse score from response: '{judge_response_text}'")
                return 0.0
    except openai.APIError as e:
        print(f"  [Judge] Judge Agent API Error: {e}")
        return 0.0
    except Exception as e:
        print(f"  [Judge] Unexpected error during judging: {e}")
        return 0.0

def load_test_set(file_path: str) -> list: # Remains a list loader
    test_cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_cases.append(json.loads(line.strip()))
    return test_cases

def evaluate_agent( # This becomes a generator
    agent_to_evaluate: BaseAgent,
    test_cases: list,
    judge_agent: GeneralLLMAgent
) -> Iterator[Dict]: # Changed return type hint
    print(f"Starting evaluation for agent: {agent_to_evaluate.name}")
    for i, case in enumerate(test_cases):
        query = case['query']
        ground_truth = case['ground_truth']

        print(f"  Processing case {i+1}/{len(test_cases)} for {agent_to_evaluate.name}: ID {case['id']}") # More specific log
        agent_response = agent_to_evaluate.run(query)

        exact_match = False
        if isinstance(ground_truth, dict):
            try:
                if isinstance(agent_response, str) and agent_response.strip().startswith("{") and agent_response.strip().endswith("}"):
                    agent_response_json = json.loads(agent_response)
                    exact_match = agent_response_json == ground_truth
                # No else needed, exact_match remains False
            except json.JSONDecodeError:
                exact_match = False # Explicitly set, though already False
        else:
            exact_match = agent_response == ground_truth

        similarity_score = 0.0
        if exact_match:
            similarity_score = 1.0
        else:
            similarity_score = get_similarity_score(query, agent_response, ground_truth, judge_agent)

        is_success = similarity_score >= SIMILARITY_THRESHOLD

        # Yield individual result dictionary
        yield {
            "任务ID": case['id'], "任务类型": case['type'], "查询语句": query, # Used query variable
            "标准答案": json.dumps(ground_truth, ensure_ascii=False) if isinstance(ground_truth, dict) else ground_truth,
            "模型回答": agent_response, "完全匹配": exact_match,
            "相似度得分": f"{similarity_score:.2f}", # Format score here
            "是否成功": is_success,
            "智能体": agent_to_evaluate.name
        }
    print(f"Finished evaluation for agent: {agent_to_evaluate.name}")

def run_evaluation(custom_configs: Optional[Dict] = None) -> Iterator[Dict]: # Changed return type hint
    print("Starting full evaluation run (streaming results)...")
    if custom_configs:
        print(f"Using custom configurations: {json.dumps(custom_configs, indent=2)}")
    else:
        print("No custom configurations provided, using defaults.")

    test_cases = load_test_set("data/golden_test_set.jsonl")
    print(f"Loaded {len(test_cases)} test cases.")

    general_agent_config = custom_configs.get("general_agent") if custom_configs else {}
    if general_agent_config is None: general_agent_config = {}

    judge_and_eval_agent = GeneralLLMAgent(
        name="通用大模型（OpenAI API）",
        api_key=general_agent_config.get("api_key"),
        base_url=general_agent_config.get("base_url"),
        model_name=general_agent_config.get("model_name")
    )
    print(f"Initialized GeneralLLMAgent/Judge: {judge_and_eval_agent.name} (Model: {judge_and_eval_agent.model_name})")

    finetuned_agent_config = custom_configs.get("finetuned_agent") if custom_configs else {}
    if finetuned_agent_config is None: finetuned_agent_config = {}

    fine_tuned_agent = FineTunedAgent(
        api_url=finetuned_agent_config.get("api_url"),
        token=finetuned_agent_config.get("token")
    )
    print(f"Initialized FineTunedAgent: {fine_tuned_agent.name}")

    # Yield a system message indicating start for GeneralLLMAgent
    yield {"type": "system", "agent": judge_and_eval_agent.name, "message": f"开始评测智能体: {judge_and_eval_agent.name}"}
    for result_dict in evaluate_agent(
        agent_to_evaluate=judge_and_eval_agent,
        test_cases=test_cases,
        judge_agent=judge_and_eval_agent
    ):
        yield result_dict # Yield each result as it comes
    yield {"type": "system", "agent": judge_and_eval_agent.name, "message": f"完成评测智能体: {judge_and_eval_agent.name}"}

    # Yield a system message indicating start for FineTunedAgent
    yield {"type": "system", "agent": fine_tuned_agent.name, "message": f"开始评测智能体: {fine_tuned_agent.name}"}
    for result_dict in evaluate_agent(
        agent_to_evaluate=fine_tuned_agent,
        test_cases=test_cases,
        judge_agent=judge_and_eval_agent
    ):
        yield result_dict # Yield each result as it comes
    yield {"type": "system", "agent": fine_tuned_agent.name, "message": f"完成评测智能体: {fine_tuned_agent.name}"}

    print("\nFull evaluation run (streaming) completed.")
    yield {"type": "system", "message": "所有评测已完成。"} # Final system message
