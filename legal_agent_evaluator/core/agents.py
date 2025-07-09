# core/agents.py
import json
import time
import random
import requests # For FineTunedAgent
import openai # For GeneralLLMAgent
import os

# --- Constants for API access (Defaults) ---
# For GeneralLLMAgent (OpenAI via Proxy)
DEFAULT_OPENAI_API_KEY = "sk-STEgVqMtrgBghNJKqrzDsmKC7veRoRJAJfqAwPc4XwoGM0JC"
DEFAULT_OPENAI_BASE_URL = "https://expose.drawaspark.com/v1"

# For FineTunedAgent ("好的智能体" API)
DEFAULT_GOOD_AGENT_API_URL = "https://fastai.gxzgt.com:3000/api/v1/chat/completions"
DEFAULT_GOOD_AGENT_TOKEN = "fastgpt-fEtaxAzubhd8ZRQBftG7njT7Q9RxLQ2lFwLsnaoEWDA6rEM7hnMhBn5Ia4nH"


class BaseAgent:
    """智能体基类"""
    def __init__(self, name):
        self.name = name

    def run(self, query: str) -> str:
        raise NotImplementedError

class GeneralLLMAgent(BaseAgent):
    """模拟通用大模型智能体 (调用OpenAI API)"""
    def __init__(self, name="通用大模型（OpenAI API）", api_key: str = None, base_url: str = None):
        super().__init__(name)
        self_api_key = api_key if api_key else DEFAULT_OPENAI_API_KEY
        self_base_url = base_url if base_url else DEFAULT_OPENAI_BASE_URL

        if not self_api_key:
            raise ValueError("OpenAI API key must be provided either as a parameter or as a default.")

        self.client = openai.OpenAI(
            api_key=self_api_key,
            base_url=self_base_url,
        )
        print(f"[GeneralLLMAgent] Initialized with Base URL: {self_base_url}, Key: {'Provided' if api_key else 'Default'}")

    def run(self, query: str) -> str:
        try:
            start_time = time.time()
            # Ensure client is using the potentially updated base_url if it was re-init
            # For openai client, base_url is set at init, so it's fine.
            stream = self.client.chat.completions.create(
<<<<<<< HEAD
                model="gpt-4o-mini", # Or any other model available via your proxy
=======
                model="gpt-3.5-turbo",
>>>>>>> 4299d0e50b4a5b51d9e9a519f1e2f018f6e3dfd3
                messages=[
                    {"role": "system", "content": "You are a general helpful assistant."},
                    {"role": "user", "content": query}
                ],
                stream=True,
                timeout=60
            )

            response_content = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    response_content.append(chunk.choices[0].delta.content)
                if chunk.choices and chunk.choices[0].finish_reason == "stop":
                    break

            final_response = "".join(response_content)
            end_time = time.time()
            # print(f"[GeneralLLMAgent] Query: '{query[:50]}...' Response: '{final_response[:50]}...' Time: {end_time - start_time:.2f}s")
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
        except openai.APIError as e:
            print(f"[GeneralLLMAgent] OpenAI API Error: {e}")
            return f"抱歉，调用OpenAI服务时发生错误：{e}"
        except Exception as e:
            print(f"[GeneralLLMAgent] Unexpected error: {e}")
            return f"抱歉，处理请求时发生未知错误：{e}"

class FineTunedAgent(BaseAgent):
    """模拟经过法律数据微调的智能体 (调用“好的智能体”API)"""
    def __init__(self, name="微调后模型（特定API）", api_url: str = None, token: str = None):
        super().__init__(name)
        self.api_url = api_url if api_url else DEFAULT_GOOD_AGENT_API_URL
        self_token = token if token else DEFAULT_GOOD_AGENT_TOKEN

        if not self_token:
            raise ValueError("FineTunedAgent token must be provided either as a parameter or as a default.")
        if not self.api_url:
            raise ValueError("FineTunedAgent API URL must be provided either as a parameter or as a default.")

        self.headers = {
            "Authorization": f"Bearer {self_token}",
            "Content-Type": "application/json"
        }
        print(f"[FineTunedAgent] Initialized with API URL: {self.api_url}, Token: {'Provided' if token else 'Default'}")


    def run(self, query: str) -> str:
        payload = {
            "chatId": f"chat_{random.randint(1000, 9999)}_{time.time_ns()}",
            "stream": True,
            "detail": False,
            "variables": {
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
                self.api_url, # Uses the potentially updated self.api_url
                headers=self.headers, # Uses the potentially updated self.headers (if token changed)
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            response_content = []
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
                            # print(f"[FineTunedAgent] JSONDecodeError for line: '{json_str}', error: {e}")
                            continue

            final_response = "".join(response_content)
            end_time = time.time()
            # print(f"[FineTunedAgent] Query: '{query[:50]}...' Response: '{final_response[:50]}...' Time: {end_time - start_time:.2f}s")
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
        except requests.exceptions.RequestException as e:
            print(f"[FineTunedAgent] API Request Error: {e}")
            return f"抱歉，调用特定API服务时发生错误：{e}"
        except Exception as e:
            print(f"[FineTunedAgent] Unexpected error: {e}")
            return f"抱歉，处理请求时发生未知错误：{e}"
