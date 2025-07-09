<!--
 * @Author: liujiarong 448736378@qq.com
 * @Date: 2024-07-12 10:00:00
 * @LastEditors: liujiarong 448736378@qq.com
 * @LastEditTime: 2024-07-12 10:00:00
 * @FilePath: /legal_agent_evaluator/README.md
 * @Description: 法律智能体评测仪表盘项目，用于对比不同大型语言模型在法律领域的表现。
-->

# ⚖️ 法律智能体评测仪表盘

本项目旨在提供一个可配置的仪表盘，用于评测和对比不同大型语言模型（LLMs）在特定法律任务上的表现。用户可以通过界面直观地看到通用大模型与经过法律数据微调/优化的模型之间的差异。

## ✨ 主要功能

*   **后端API**: 使用 FastAPI 构建，负责处理评测逻辑和与大模型API的交互。
*   **前端仪表盘**: 使用 Streamlit 构建，提供用户友好的界面来触发评测、输入自定义API配置并展示评测结果。
*   **可配置智能体**: 支持配置不同的大模型API端点和密钥，以便灵活测试。
*   **模拟评测数据**:包含一套预定义的法律领域测试用例 (`data/golden_test_set.jsonl`)。
*   **结果可视化**: 以图表和表格形式清晰展示各智能体的任务成功率和详细结果。

## 🚀 环境设置与安装

为了保持项目纯净和依赖清晰，建议将整个项目放在一个独立的Python虚拟环境中。

**1. 克隆或下载项目 (如果您尚未这样做)**

如果您是从版本控制系统获取项目，请先克隆。如果已下载代码，请解压到您的工作区。

**2. 创建项目文件夹并进入**

假设您的项目文件夹名为 `legal_agent_evaluator`。

```bash
# cd path/to/your/workspace
# (如果需要，先创建 legal_agent_evaluator 文件夹)
cd legal_agent_evaluator
```

**3. 创建Python虚拟环境**

在项目文件夹 (`legal_agent_evaluator`) 内，执行以下命令创建一个名为 `venv` 的虚拟环境。

```bash
python -m venv venv
```
这会创建一个 `venv` 文件夹，其中包含了Python解释器的一个独立副本和包管理工具。

**4. 激活虚拟环境**

激活环境后，您安装的所有包都将只存在于这个环境中，不会影响系统的全局Python环境。

*   **在 Windows 上**:
    ```bash
    .\venv\Scripts\activate
    ```
*   **在 macOS / Linux 上**:
    ```bash
    source venv/bin/activate
    ```
激活后，您会看到命令行提示符前面出现了 `(venv)` 字样。

**5. 安装依赖**

在已激活的虚拟环境中，使用 `pip` 安装所有项目依赖：

```bash
pip install -r requirements.txt
```

## 🛠️ 运行与体验

请确保您处于已激活的 `(venv)` 虚拟环境中，并且当前目录是 `legal_agent_evaluator`。

您需要打开 **两个终端** 窗口/标签页。

**1. 启动FastAPI后端服务 (在第一个终端中)**

```bash
# 确保你在 legal_agent_evaluator 文件夹内并且 (venv) 已激活
uvicorn main_api:app --reload
```
服务启动后，您通常会看到类似 `Uvicorn running on http://127.0.0.1:8000` 的消息。

**2. 启动Streamlit前端应用 (在第二个终端中)**

```bash
# 确保你在 legal_agent_evaluator 文件夹内并且 (venv) 已激活
streamlit run dashboard.py
```
Streamlit 应用通常会自动在您的默认浏览器中打开，地址类似于 `http://localhost:8501`。如果没有自动打开，请手动复制终端中显示的URL到浏览器访问。

**3. 使用仪表盘**

*   在浏览器中打开Streamlit应用。
*   **可选配置**: 在左侧边栏，您可以勾选 "启用自定义API配置" 并填入您希望测试的模型的API端点和密钥/Token。如果留空或不勾选，将使用后端代码中定义的默认API配置。
*   点击 "🚀 开始评测" 按钮。
*   等待评测完成，结果将显示在仪表盘上。

## 📝 文件结构简介

```
legal_agent_evaluator/
├── core/                   # 核心逻辑
│   ├── agents.py           # 定义和实现不同类型的智能体 (例如，通用LLM, 微调LLM)
│   └── evaluator.py        # 实现评测流程和指标计算
├── data/                   # 测试数据
│   └── golden_test_set.jsonl # 包含查询和标准答案的黄金测试集
├── .gitignore              # Git忽略文件配置
├── dashboard.py            # Streamlit 前端应用代码
├── main_api.py             # FastAPI 后端API服务代码
├── README.md               # 本文档
└── requirements.txt        # Python项目依赖列表
```

## 后续开发建议 (来自用户)
用户提出以下优秀建议，可作为后续迭代方向：
1.  **前端实时显示后端日志**: 在评测进行时，将后端的详细日志（例如每个测试用例的调用情况、智能体的原始输出等）实时滚动显示在前端页面，方便追踪和调试。这可能需要使用WebSocket或Server-Sent Events (SSE)。
2.  **基于相似度的酌情给分**: 对于模型给出的答案与参考答案有一定相似性但非完全匹配的情况，可以调用一个“裁判”大模型来判断其相似度或正确性，并酌情给分（例如0到1之间的浮点数），而不是简单地判定为0分（失败）。这将使评分更加 nuanced 和公平。

---
*如有任何问题或建议，欢迎提出。*
