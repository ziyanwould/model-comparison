# core/agents.py
import json
import time
import random

class BaseAgent:
    """智能体基类"""
    def __init__(self, name):
        self.name = name

    def run(self, query: str) -> str:
        raise NotImplementedError

class GeneralLLMAgent(BaseAgent):
    """模拟通用大模型智能体 (未针对法律领域优化)"""
    def run(self, query: str) -> str:
        time.sleep(random.uniform(0.5, 1.5))
        if "诉讼时效" in query:
            return "诉讼时效就是打官司的时间限制，过期了就不能告了。"
        if "合同编号" in query:
            return "合同号是HT20250702，甲方是未来科技有限公司，乙方是智慧法律集团。"
        if "试用期" in query:
            # 通用模型可能无法很好地遵循格式指令
            return "劳动合同法规定，合同期限一到三年，试用期不能超过两个月。"
        if "拖欠房租" in query:
            return "你可以和租客商量一下，或者找个律师问问。"
        return "抱歉，我无法回答这个问题。"

class FineTunedAgent(BaseAgent):
    """模拟经过法律数据微调的智能体"""
    def run(self, query: str) -> str:
        time.sleep(random.uniform(0.2, 0.8))
        if "诉讼时效" in query:
            return "诉讼时效是指权利人在法定期间内不行使权利，该期间届满后，权利不受法律保护的制度。"
        if "合同编号" in query:
            # 经过训练，可以更好地按结构化格式提取
            return json.dumps({"合同编号": "HT20250702", "甲方": "未来科技有限公司", "乙方": "智慧法律集团"}, ensure_ascii=False)
        if "试用期" in query:
            # 经过训练，可以很好地输出JSON并引用法条
            return json.dumps({"法律依据": "《劳动合同法》第十九条", "最长试用期": "二个月"}, ensure_ascii=False)
        if "拖欠房租" in query:
            # 提供更专业、步骤化的建议
            return "房东应首先进行书面催告，要求租客在合理期限内支付租金。如果租客逾期仍不支付，房东可以依法解除租赁合同并要求其承担违约责任。"
        return "抱歉，我无法回答这个问题。"
