# dashboard.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json

# This threshold is defined in core/evaluator.py.
# For display purposes here, we define it again. Ideally, this could come from a config or API.
SIMILARITY_THRESHOLD = 0.7

st.set_page_config(
    page_title="法律智能体评测仪表盘",
    page_icon="⚖️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/evaluate" # Ensure this is correct

st.title("⚖️ 法律智能体准确率评测仪表盘")
st.caption("对比『通用大模型』与『法律领域微调模型』的表现 (成功标准基于相似度评分)")

st.sidebar.header("⚙️ 自定义API配置 (可选)")
use_custom_config = st.sidebar.checkbox("启用自定义API配置", key="use_custom_api_config")

# Initialize session state for API configs and new SSE/evaluation flow
default_session_state = {
    'gen_api_key': "", 'gen_base_url': "", 'gen_model_name': "",
    'ft_api_url': "", 'ft_token': "",
    'evaluation_id': None, 'results_url': None, 'full_logs_url': None,
    'logs': [], 'sse_active': False, 'evaluation_running': False,
    'evaluation_results_ready': False, 'evaluation_results': None,
    'sse_component_value': None, 'fetching_results': False
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

if not st.session_state.evaluation_running and not st.session_state.sse_active and not st.session_state.fetching_results:
    st.session_state.logs = []

with st.sidebar.expander("通用大模型 (OpenAI API)", expanded=False):
    st.session_state.gen_api_key = st.text_input(
        "OpenAI API Key", value=st.session_state.gen_api_key, type="password", help="留空则使用后端默认配置"
    )
    st.session_state.gen_base_url = st.text_input(
        "OpenAI Base URL", value=st.session_state.gen_base_url, help="例如: https://api.openai.com/v1。留空则使用后端默认配置。"
    )
    st.session_state.gen_model_name = st.text_input(
        "模型名称 (Model ID)", value=st.session_state.gen_model_name, help="例如: gpt-4o-mini。留空则使用后端默认模型。"
    )

with st.sidebar.expander("微调后模型 (特定API)", expanded=False):
    st.session_state.ft_api_url = st.text_input(
        "特定API URL", value=st.session_state.ft_api_url, help="留空则使用后端默认配置"
    )
    st.session_state.ft_token = st.text_input(
        "特定API Token", value=st.session_state.ft_token, type="password", help="留空则使用后端默认配置"
    )

# --- Main Interaction: SSE Component Value Handling ---
if st.session_state.sse_component_value:
    value_from_js = st.session_state.sse_component_value
    if isinstance(value_from_js, dict):
        log_type = value_from_js.get("type")
        log_data = value_from_js.get("data", "")
        if log_type == "log":
            st.session_state.logs.append(log_data)
        elif log_type == "status":
            if log_data == "EVAL_COMPLETE":
                st.session_state.sse_active = False
                # evaluation_running will be set to False when results are fetched or fetch fails
                st.session_state.evaluation_results_ready = True
                st.session_state.logs.append("[System] 评测流程已在后端完成，准备获取结果。")
            elif log_data == "SSE_CLOSED" or log_data == "SSE_ERROR":
                st.session_state.sse_active = False
                if not st.session_state.evaluation_results_ready:
                     st.session_state.logs.append(f"[System] 日志流连接意外关闭 ({log_data})。如果评测未完成，可能需要手动检查或重新开始。")
                     st.session_state.evaluation_running = False # Stop eval if SSE errored early
    else:
        st.session_state.logs.append(str(value_from_js))
    st.session_state.sse_component_value = None # Reset after processing
    st.experimental_rerun() # Rerun to update log display immediately


# --- Main Interaction: Start Evaluation Button ---
if st.button("🚀 开始评测", type="primary", disabled=st.session_state.evaluation_running):
    st.session_state.logs = ["[System] 初始化评测..."]
    st.session_state.evaluation_id = None
    st.session_state.results_url = None
    st.session_state.full_logs_url = None
    st.session_state.evaluation_results = None
    st.session_state.evaluation_results_ready = False
    st.session_state.evaluation_running = True
    st.session_state.sse_active = False
    st.session_state.fetching_results = False

    payload = {}
    if use_custom_config:
        custom_configs_dict = {}
        ga_config = {k: v for k, v in {"api_key": st.session_state.gen_api_key, "base_url": st.session_state.gen_base_url, "model_name": st.session_state.gen_model_name}.items() if v}
        if ga_config: custom_configs_dict["general_agent"] = ga_config
        ft_config = {k: v for k, v in {"api_url": st.session_state.ft_api_url, "token": st.session_state.ft_token}.items() if v}
        if ft_config: custom_configs_dict["finetuned_agent"] = ft_config
        if custom_configs_dict:
            payload["custom_configs"] = custom_configs_dict
            st.sidebar.info("评测将使用自定义API配置。")
        else:
            st.sidebar.warning("已勾选“启用自定义API配置”，但未填写任何有效的自定义API信息。将使用后端默认配置。")
            st.session_state.logs.append("[System] WARN: 已勾选自定义配置但未提供有效信息，将使用默认配置。")
    else:
        st.sidebar.info("评测将使用后端默认API配置。")
        st.session_state.logs.append("[System] INFO: 使用后端默认API配置。")

    try:
        st.session_state.logs.append(f"[System] 发送评测请求到 {API_URL}...")
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        eval_response_data = response.json()
        st.session_state.evaluation_id = eval_response_data.get("evaluation_id")
        st.session_state.results_url = eval_response_data.get("results_url")
        logs_url_path = eval_response_data.get("logs_url")

        if logs_url_path:
             base_api_url_for_sse = API_URL.rsplit('/', 1)[0]
             st.session_state.full_logs_url = f"{base_api_url_for_sse}{logs_url_path}"
        else:
            st.session_state.full_logs_url = None

        if st.session_state.evaluation_id and st.session_state.full_logs_url:
            st.session_state.logs.append(f"[System] 评测任务已启动 (ID: {st.session_state.evaluation_id})。")
            st.session_state.logs.append(f"[System] 准备连接到日志流: {st.session_state.full_logs_url}")
            st.session_state.sse_active = True
        else:
            st.session_state.logs.append("[System] ERROR: 后端未能返回有效的评测ID或日志URL。")
            st.error("后端未能返回有效的评测ID或日志URL。无法启动日志流。")
            st.session_state.evaluation_running = False
    except Exception as e:
        error_msg = f"启动评测时发生错误: {e}"
        st.error(error_msg)
        st.session_state.logs.append(f"[System] ERROR: {error_msg}")
        st.session_state.evaluation_running = False
    st.experimental_rerun()

# --- SSE Log Receiver Component (Rendered if sse_active) ---
if st.session_state.get('sse_active') and st.session_state.get('full_logs_url'):
    sse_html_template = """
    <!DOCTYPE html>
    <html><head><meta charset="UTF-8"><title>SSE Log Receiver</title>
    <script>
        function sendValueToPython(value) {
            if (window.Streamlit && typeof Streamlit.setComponentValue === 'function') {
                Streamlit.setComponentValue(value);
            } else { console.warn("Streamlit.setComponentValue not available."); }
        }
        function initSSE(sseUrl) {
            console.log("SSE JS: Connecting to", sseUrl);
            sendValueToPython({ type: "log", data: "[SSE_JS] Connecting to " + sseUrl });
            const eventSource = new EventSource(sseUrl);
            eventSource.onopen = function() {
                console.log("SSE JS: Connection opened.");
                sendValueToPython({ type: "log", data: "[SSE_JS] Log stream connected." });
            };
            eventSource.onmessage = function(event) {
                sendValueToPython({ type: "log", data: event.data });
                if (event.data.includes("[System] 日志流结束。") || event.data.includes("评测任务成功完成") || event.data.includes("评测任务发生错误")) {
                     console.log("SSE JS: Detected evaluation end/error in logs.");
                     sendValueToPython({ type: "status", data: "EVAL_COMPLETE" });
                     eventSource.close();
                }
            };
            eventSource.onerror = function(err) {
                console.error("SSE JS: Error:", err);
                sendValueToPython({ type: "status", data: "SSE_ERROR" });
                eventSource.close();
            };
        }
        const sseUrlFromPython = "%(sse_url)s";
        if (sseUrlFromPython && sseUrlFromPython !== "%%(sse_url)s" && (sseUrlFromPython.startsWith("http://") || sseUrlFromPython.startsWith("https://"))) {
             initSSE(sseUrlFromPython);
        } else {
            console.error("SSE JS: URL not provided or invalid:", sseUrlFromPython);
            sendValueToPython({ type: "log", data: "[SSE_JS_ERR] URL invalid: " + sseUrlFromPython });
        }
    </script>
    </head><body><!-- Invisible --></body></html>
    """
    html_to_render = sse_html_template % {"sse_url": st.session_state.full_logs_url}
    component_key = f"sse_receiver_{st.session_state.evaluation_id}"
    # sse_component_value is updated by this call when JS sends data
    captured_value = st.components.v1.html(html_to_render, height=0, key=component_key)
    if captured_value: # If JS sent a value in this run
        st.session_state.sse_component_value = captured_value
        st.experimental_rerun() # Rerun to process the new sse_component_value

# --- Log Display Area ---
if st.session_state.get('evaluation_id'):
    st.subheader("评测日志")
    log_content = "\n".join(st.session_state.get('logs', []))
    st.code(log_content, language='text', line_numbers=False)

# --- Results Fetching Logic ---
if st.session_state.get('evaluation_results_ready') and \
   st.session_state.get('results_url') and \
   not st.session_state.get('evaluation_results') and \
   not st.session_state.get('fetching_results'):

    st.session_state.fetching_results = True
    new_log_message = f"[System] 正在从 {st.session_state.results_url} 获取最终评测结果..."
    st.session_state.logs.append(new_log_message)
    st.experimental_rerun()

if st.session_state.get('fetching_results') and not st.session_state.get('evaluation_results'):
    try:
        base_api_url_for_results = API_URL.rsplit('/', 1)[0]
        full_results_url = f"{base_api_url_for_results}{st.session_state.results_url}"

        results_response = requests.get(full_results_url, timeout=60)
        results_response.raise_for_status()
        st.session_state.evaluation_results = results_response.json()
        st.session_state.logs.append("[System] 成功获取评测结果。")
        st.success("评测报告已加载。")
    except Exception as e:
        error_msg = f"获取评测结果时出错: {e}"
        st.error(error_msg)
        st.session_state.logs.append(f"[System] ERROR: {error_msg}")
    finally:
        st.session_state.evaluation_running = False
        st.session_state.sse_active = False
        st.session_state.fetching_results = False
        st.experimental_rerun()

# --- Results Display Area ---
if st.session_state.get('evaluation_results'):
    results_data = st.session_state.get('evaluation_results')
    if not isinstance(results_data, list):
        st.error(f"从后端接收到的最终结果格式不正确。期望一个列表，但收到: {type(results_data)}")
        if isinstance(results_data, dict) and 'detail' in results_data:
             st.error(f"详细错误: {results_data['detail']}")
        results_df = pd.DataFrame()
    else:
        results_df = pd.DataFrame(results_data)

    if not results_df.empty:
        st.divider()
        st.header("📊 总体指标对比")
        st.caption(f"成功标准：相似度得分 >= {SIMILARITY_THRESHOLD:.1f} (由裁判模型评估)")
        if '智能体' in results_df.columns and '是否成功' in results_df.columns:
            try:
                results_df['是否成功'] = results_df['是否成功'].astype(bool)
                success_rate = results_df.groupby('智能体')['是否成功'].mean().reset_index()
                success_rate['成功率'] = success_rate['是否成功'] * 100
                fig = px.bar(
                    success_rate, x='智能体', y='成功率',
                    title='各智能体任务成功率对比',
                    text=success_rate['成功率'].apply(lambda x: f'{x:.2f}%'),
                    color='智能体',
                    color_discrete_map={'通用大模型（OpenAI API）': '#ff9900', '微调后模型（特定API）': '#00529B'}
                )
                fig.update_layout(yaxis_title='成功率 (%)', xaxis_title=None, yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"生成图表时出错: {e}")
                st.dataframe(results_df.head())
        else:
            st.warning("评测结果数据不完整或格式不正确，无法生成总体指标图表。")
            st.dataframe(results_df.head())
        st.divider()
        st.header("🔍 详细结果与Bad Case分析")
        display_columns = ["任务ID", "任务类型", "查询语句", "标准答案", "模型回答", "完全匹配", "相似度得分", "是否成功", "智能体"]
        columns_to_show = [col for col in display_columns if col in results_df.columns]
        tab1, tab2 = st.tabs(["所有结果", "失败案例 (Bad Cases)"])
        with tab1:
            st.subheader("所有任务评测明细")
            st.dataframe(results_df[columns_to_show] if not results_df.empty else pd.DataFrame())
        with tab2:
            st.subheader("失败案例分析 (基于相似度评分)")
            if '是否成功' in results_df.columns:
                bad_cases_df = results_df[results_df['是否成功'] == False]
                if bad_cases_df.empty: st.success("太棒了！没有发现失败案例。")
                else: st.dataframe(bad_cases_df[columns_to_show])
            else: st.warning("评测结果数据中缺少 '是否成功' 列。")
elif not st.session_state.get('evaluation_running') and not st.session_state.get('evaluation_id'):
    st.info("点击“开始评测”以查看结果。")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**注意**: '是否成功'的判断基于相似度评分。后端当前使用的相似度阈值为 **{SIMILARITY_THRESHOLD:.1f}**。")
