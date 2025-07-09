<!--
 * @Author: liujiarong 448736378@qq.com
 * @Date: 2025-07-09 17:11:52
 * @LastEditors: liujiarong 448736378@qq.com
 * @LastEditTime: 2025-07-09 17:55:45
 * @FilePath: /model-comparison/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
激活虚拟环境
cd legal_agent_evaluator
激活环境后，您安装的所有包都将只存在于这个环境中，不会影响系统的全局Python环境。

在 Windows 上:

Bash

.\venv\Scripts\activate
在 macOS / Linux 上:

Bash

source venv/bin/activate
激活后，您会看到命令行提示符前面出现了 (venv) 字样。

pip install -r requirements.txt
运行与体验
请确保您处于已激活的 (venv) 虚拟环境中。

1. 在第一个终端中，启动FastAPI后端服务：

Bash

# 确保你在 legal_agent_evaluator 文件夹内并且 (venv) 已激活
uvicorn main_api:app --reload
2. 在第二个终端中，启动Streamlit前端应用：

Bash

# 确保你在 legal_agent_evaluator 文件夹内并且 (venv) 已激活
streamlit run dashboard.py