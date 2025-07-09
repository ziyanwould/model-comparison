# core/evaluator.py
import json
from typing import List, Dict, Optional
from .agents import GeneralLLMAgent, FineTunedAgent

def load_test_set(file_path: str) -> list:
    """加载黄金测试集"""
    test_cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_cases.append(json.loads(line.strip()))
    return test_cases

def evaluate_agent(agent, test_cases: list) -> list:
    """评测单个智能体"""
    results = []
    print(f"Starting evaluation for agent: {agent.name}")
    for i, case in enumerate(test_cases):
        query = case['query']
        ground_truth = case['ground_truth']

        # print(f"  Processing case {i+1}/{len(test_cases)}: ID {case['id']}, Query: '{query[:30]}...'")
        agent_response = agent.run(query)

        is_success = False
        if isinstance(ground_truth, dict):
            try:
                if isinstance(agent_response, str) and agent_response.strip().startswith("{") and agent_response.strip().endswith("}"):
                    agent_response_json = json.loads(agent_response)
                    is_success = agent_response_json == ground_truth
                else:
                    is_success = False
            except json.JSONDecodeError:
                is_success = False
        else:
            is_success = agent_response == ground_truth

        results.append({
            "id": case['id'],
            "type": case['type'],
            "query": query,
            "ground_truth": ground_truth,
            "agent_response": agent_response,
            "is_success": is_success,
            "agent_name": agent.name
        })
    print(f"Finished evaluation for agent: {agent.name}")
    return results

def run_evaluation(custom_configs: Optional[Dict] = None) -> list:
    """
    运行所有评测。
    :param custom_configs: 一个可选的字典，包含用户自定义的API配置。
                           结构示例:
                           {
                               "general_agent": {
                                   "api_key": "user_key",
                                   "base_url": "user_url",
                                   "model_name": "user_model"
                               },
                               "finetuned_agent": {
                                   "api_url": "user_api_url",
                                   "token": "user_token"
                                   // model_name is not currently used by FineTunedAgent's API
                               }
                           }
    """
    print("Starting full evaluation run...")
    if custom_configs:
        print(f"Using custom configurations: {json.dumps(custom_configs, indent=2)}")
    else:
        print("No custom configurations provided, using defaults.")

    test_cases = load_test_set("data/golden_test_set.jsonl")

    general_agent_config = custom_configs.get("general_agent") if custom_configs else {}
    if general_agent_config is None: general_agent_config = {}

    general_agent = GeneralLLMAgent(
        api_key=general_agent_config.get("api_key"),
        base_url=general_agent_config.get("base_url"),
        model_name=general_agent_config.get("model_name") # Pass model_name
    )

    finetuned_agent_config = custom_configs.get("finetuned_agent") if custom_configs else {}
    if finetuned_agent_config is None: finetuned_agent_config = {}

    # FineTunedAgent currently doesn't take model_name, so we don't pass it.
    finetuned_agent = FineTunedAgent(
        api_url=finetuned_agent_config.get("api_url"),
        token=finetuned_agent_config.get("token")
    )

    all_results = []

    general_agent_results = evaluate_agent(general_agent, test_cases)
    for res in general_agent_results:
        all_results.append({
            "任务ID": res['id'],
            "任务类型": res['type'],
            "查询语句": res['query'],
            "标准答案": json.dumps(res['ground_truth'], ensure_ascii=False) if isinstance(res['ground_truth'], dict) else res['ground_truth'],
            "模型回答": res['agent_response'],
            "是否成功": res['is_success'],
            "智能体": res['agent_name']
        })

    finetuned_agent_results = evaluate_agent(finetuned_agent, test_cases)
    for res in finetuned_agent_results:
        all_results.append({
            "任务ID": res['id'],
            "任务类型": res['type'],
            "查询语句": res['query'],
            "标准答案": json.dumps(res['ground_truth'], ensure_ascii=False) if isinstance(res['ground_truth'], dict) else res['ground_truth'],
            "模型回答": res['agent_response'],
            "是否成功": res['is_success'],
            "智能体": res['agent_name']
        })

    print("Full evaluation run completed.")
    return all_results
