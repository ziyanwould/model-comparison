# core/agents.py
import json
import time
import random
import requests # For FineTunedAgent
import openai # For GeneralLLMAgent
import os

# --- Constants for API access ---
# For GeneralLLMAgent (OpenAI via Proxy)
OPENAI_API_KEY = "sk-STEgVqMtrgBghNJKqrzDsmKC7veRoRJAJfqAwPc4XwoGM0JC"
OPENAI_BASE_URL = "https://expose.drawaspark.com/v1" # Assuming /v1 is needed

# For FineTunedAgent ("好的智能体" API)
GOOD_AGENT_API_URL = "https://fastai.gxzgt.com:3000/api/v1/chat/completions"
GOOD_AGENT_TOKEN = "fastgpt-fEtaxAzubhd8ZRQBftG7njT7Q9RxLQ2lFwLsnaoEWDA6rEM7hnMhBn5Ia4nH"


class BaseAgent:
    """智能体基类"""
    def __init__(self, name):
        self.name = name

    def run(self, query: str) -> str:
        raise NotImplementedError

class GeneralLLMAgent(BaseAgent):
    """模拟通用大模型智能体 (调用OpenAI API)"""
    def __init__(self, name="通用大模型（OpenAI API）"):
        super().__init__(name)
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

    def run(self, query: str) -> str:
        try:
            start_time = time.time()
            stream = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # Or any other model available via your proxy
                messages=[
                    {"role": "system", "content": "You are a general helpful assistant."},
                    {"role": "user", "content": query}
                ],
                stream=True,
                timeout=60 # 60 seconds timeout
            )

            response_content = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    response_content.append(chunk.choices[0].delta.content)
                if chunk.choices and chunk.choices[0].finish_reason == "stop":
                    break

            final_response = "".join(response_content)
            end_time = time.time()
            print(f"[GeneralLLMAgent] Query: '{query[:50]}...' Response: '{final_response[:50]}...' Time: {end_time - start_time:.2f}s")
            return final_response if final_response else "抱歉，我无法回答这个问题（OpenAI API未返回有效内容）。"

        except openai.APIConnectionError as e:
            print(f"[GeneralLLMAgent] OpenAI API Connection Error: {e}")
            return f"抱歉，连接OpenAI服务失败：{e}"
        except openai.APITimeoutError as e:
            print(f"[GeneralLLMAgent] OpenAI API Timeout Error: {e}")
            return f"抱歉，请求OpenAI服务超时：{e}"
        except openai.APIStatusError as e:
            print(f"[GeneralLLMAgent] OpenAI API Status Error: {e.status_code} - {e.response}")
            return f"抱歉，OpenAI API返回错误状态 {e.status_code}：{e.message}"
        except openai.APIError as e: # Catch-all for other OpenAI errors
            print(f"[GeneralLLMAgent] OpenAI API Error: {e}")
            return f"抱歉，调用OpenAI服务时发生错误：{e}"
        except Exception as e:
            print(f"[GeneralLLMAgent] Unexpected error: {e}")
            return f"抱歉，处理请求时发生未知错误：{e}"

class FineTunedAgent(BaseAgent):
    """模拟经过法律数据微调的智能体 (调用“好的智能体”API)"""
    def __init__(self, name="微调后模型（特定API）"):
        super().__init__(name)
        self.api_url = GOOD_AGENT_API_URL
        self.headers = {
            "Authorization": f"Bearer {GOOD_AGENT_TOKEN}",
            "Content-Type": "application/json"
        }

    def run(self, query: str) -> str:
        payload = {
            "chatId": f"chat_{random.randint(1000, 9999)}_{time.time_ns()}", # Unique chatId
            "stream": True,
            "detail": False,
            "variables": { # Example variables, adjust if needed
                "uid": "jules_evaluator",
                "name": "Jules"
            },
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        try:
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=60 # 60 seconds timeout
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            response_content = []
            # Read the stream line by line
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.strip() == "[DONE]":
                        break
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[len("data: "):]
                        try:
                            chunk = json.loads(json_str)
                            if chunk.get("choices") and isinstance(chunk["choices"], list) and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    response_content.append(content)
                                if chunk["choices"][0].get("finish_reason") == "stop":
                                    break
                        except json.JSONDecodeError as e:
                            print(f"[FineTunedAgent] JSONDecodeError for line: '{json_str}', error: {e}")
                            continue # Skip malformed JSON lines

            final_response = "".join(response_content)
            end_time = time.time()
            print(f"[FineTunedAgent] Query: '{query[:50]}...' Response: '{final_response[:50]}...' Time: {end_time - start_time:.2f}s")
            return final_response if final_response else "抱歉，我无法回答这个问题（特定API未返回有效内容）。"

        except requests.exceptions.Timeout as e:
            print(f"[FineTunedAgent] API Timeout Error: {e}")
            return f"抱歉，请求特定API服务超时：{e}"
        except requests.exceptions.ConnectionError as e:
            print(f"[FineTunedAgent] API Connection Error: {e}")
            return f"抱歉，连接特定API服务失败：{e}"
        except requests.exceptions.HTTPError as e:
            print(f"[FineTunedAgent] API HTTP Error: {e.response.status_code} - {e.response.text}")
            return f"抱歉，特定API服务返回错误状态 {e.response.status_code}：{e.response.reason}"
        except requests.exceptions.RequestException as e: # Catch-all for other requests errors
            print(f"[FineTunedAgent] API Request Error: {e}")
            return f"抱歉，调用特定API服务时发生错误：{e}"
        except Exception as e:
            print(f"[FineTunedAgent] Unexpected error: {e}")
            return f"抱歉，处理请求时发生未知错误：{e}"

# Example of how the old agents were (for reference, will be overwritten)
# class OldGeneralLLMAgent(BaseAgent):
#     """模拟通用大模型智能体 (未针对法律领域优化)"""
#     def run(self, query: str) -> str:
#         time.sleep(random.uniform(0.5, 1.5))
#         if "诉讼时效" in query:
#             return "诉讼时效就是打官司的时间限制，过期了就不能告了。"
#         # ... other hardcoded responses
#         return "抱歉，我无法回答这个问题。"

# class OldFineTunedAgent(BaseAgent):
#     """模拟经过法律数据微调的智能体"""
#     def run(self, query: str) -> str:
#         time.sleep(random.uniform(0.2, 0.8))
#         if "诉讼时效" in query:
#             return "诉讼时效是指权利人在法定期间内不行使权利，该期间届满后，权利不受法律保护的制度。"
#         # ... other hardcoded responses
#         return "抱歉，我无法回答这个问题。"
