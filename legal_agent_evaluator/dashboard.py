# dashboard.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json

SIMILARITY_THRESHOLD = 0.7

st.set_page_config(
    page_title="法律智能体评测仪表盘",
    page_icon="⚖️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/evaluate"

st.title("⚖️ 法律智能体准确率评测仪表盘")
st.caption("对比『通用大模型』与『法律领域微调模型』的表现 (成功标准基于相似度评分)")

st.sidebar.header("⚙️ 自定义API配置 (可选)")
use_custom_config = st.sidebar.checkbox("启用自定义API配置", key="use_custom_api_config")

# Initialize session state
default_session_state = {
    'gen_api_key': "", 'gen_base_url': "", 'gen_model_name': "",
    'ft_api_url': "", 'ft_token': "",
    'streamed_results': [], # To store results as they stream in
    'evaluation_complete': False, # Flag to indicate streaming is done
    'active_evaluation_run': False # Flag to indicate an evaluation is ongoing
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

with st.sidebar.expander("通用大模型 (OpenAI API)", expanded=False):
    st.session_state.gen_api_key = st.text_input("OpenAI API Key", value=st.session_state.gen_api_key, type="password", help="留空则使用后端默认配置")
    st.session_state.gen_base_url = st.text_input("OpenAI Base URL", value=st.session_state.gen_base_url, help="例如: https://api.openai.com/v1。留空则使用后端默认配置。")
    st.session_state.gen_model_name = st.text_input("模型名称 (Model ID)", value=st.session_state.gen_model_name, help="例如: gpt-4o-mini。留空则使用后端默认模型。")

with st.sidebar.expander("微调后模型 (特定API)", expanded=False):
    st.session_state.ft_api_url = st.text_input("特定API URL", value=st.session_state.ft_api_url, help="留空则使用后端默认配置")
    st.session_state.ft_token = st.text_input("特定API Token", value=st.session_state.ft_token, type="password", help="留空则使用后端默认配置")

# --- Streaming Results Display Area ---
# This area will show results as they stream in.
# We use st.empty() to replace its content dynamically.
streaming_results_placeholder = st.empty()

# --- Main Interaction: Start Evaluation Button ---
if st.button("🚀 开始评测", type="primary", disabled=st.session_state.active_evaluation_run):
    st.session_state.streamed_results = [] # Clear previous streamed results
    st.session_state.evaluation_complete = False
    st.session_state.active_evaluation_run = True # Disable button during run

    payload = {}
    if use_custom_config:
        custom_configs_dict = {}
        ga_config = {k: v for k, v in {"api_key": st.session_state.gen_api_key, "base_url": st.session_state.gen_base_url, "model_name": st.session_state.gen_model_name}.items() if v}
        if ga_config: custom_configs_dict["general_agent"] = ga_config
        ft_config = {k: v for k, v in {"api_url": st.session_state.ft_api_url, "token": st.session_state.ft_token}.items() if v}
        if ft_config: custom_configs_dict["finetuned_agent"] = ft_config
        if custom_configs_dict:
            payload["custom_configs"] = custom_configs_dict
            st.sidebar.info("评测将使用自定义API配置.")
        else:
            st.sidebar.warning("已勾选“启用自定义API配置”，但未填写任何有效的自定义API信息。将使用后端默认配置。")
    else:
        st.sidebar.info("评测将使用后端默认API配置。")

    # Use a spinner for the entire duration of the streaming request
    with st.spinner("评测进行中，结果将逐步显示... 请勿刷新页面。"):
        current_results_display = [] # Temp list for display within this run
        try:
            response = requests.post(API_URL, json=payload, stream=True, timeout=1800) # Long timeout for potentially many items
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        result_item = json.loads(decoded_line)

                        # Check for system messages from the stream
                        if result_item.get("type") == "system":
                            # Display system message in a distinct way or log it
                            # For now, add to a temporary display list or print to console
                            current_results_display.append(f"SYSTEM: {result_item.get('message', '')} (Agent: {result_item.get('agent', 'N/A')})")
                            if result_item.get("message") == "所有评测已完成。":
                                st.session_state.evaluation_complete = True
                        else: # It's a result item
                            st.session_state.streamed_results.append(result_item)
                            # Prepare a subset of info for real-time display
                            display_item = (
                                f"智能体: {result_item.get('智能体', 'N/A')}, "
                                f"任务ID: {result_item.get('任务ID', 'N/A')}, "
                                f"查询: {result_item.get('查询语句', '')[:30]}..., "
                                f"成功: {result_item.get('是否成功', False)}, "
                                f"得分: {result_item.get('相似度得分', '0.00')}"
                            )
                            current_results_display.append(display_item)

                        # Update the placeholder with the latest set of messages/results
                        # Displaying as a simple list of strings for now
                        streaming_results_placeholder.text_area(
                            "评测进度 (逐条结果):",
                            "\n".join(current_results_display),
                            height=300,
                            key=f"stream_display_{len(current_results_display)}" # Key to force update
                        )
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line: {decoded_line}")
                        current_results_display.append(f"错误: 无法解析的日志行: {decoded_line[:100]}")
                        streaming_results_placeholder.text_area("评测进度:", "\n".join(current_results_display), height=300)
                    except Exception as e_inner: # Catch other errors during line processing
                        print(f"Error processing streamed line: {e_inner}")
                        current_results_display.append(f"错误: 处理流数据时出错 - {str(e_inner)[:100]}")
                        streaming_results_placeholder.text_area("评测进度:", "\n".join(current_results_display), height=300)

            if not st.session_state.evaluation_complete: # If stream ended without explicit completion message
                st.session_state.evaluation_complete = True # Assume completion if stream ends
                current_results_display.append("SYSTEM: 后端数据流已结束。")
                streaming_results_placeholder.text_area("评测进度:", "\n".join(current_results_display), height=300)

            st.success("评测数据流接收完毕！正在生成最终报告...")

        except requests.exceptions.RequestException as e:
            st.error(f"请求后端服务时发生错误: {e}")
            st.session_state.streamed_results = [] # Clear on error
        except Exception as e_outer: # Catch other unexpected errors
            st.error(f"评测过程中发生未知错误: {e_outer}")
            st.session_state.streamed_results = []
        finally:
            st.session_state.active_evaluation_run = False # Re-enable button
            # Force a final rerun to ensure the 'evaluation_complete' state is processed for report generation
            st.experimental_rerun()


# --- Final Results Display Area (triggered after streaming is complete) ---
if st.session_state.evaluation_complete and st.session_state.streamed_results:
    results_df = pd.DataFrame(st.session_state.streamed_results)

    # Filter out any system messages if they were accidentally added to streamed_results
    # (though current logic tries to put them in current_results_display for live view only)
    if "type" in results_df.columns: # Assuming system messages have a 'type' field
        results_df = results_df[results_df["type"] != "system"]

    if not results_df.empty:
        streaming_results_placeholder.empty() # Clear the streaming display area
        st.header("📊 最终评测报告")
        st.caption(f"成功标准：相似度得分 >= {SIMILARITY_THRESHOLD:.1f} (由裁判模型评估)")

        if '智能体' in results_df.columns and '是否成功' in results_df.columns:
            try:
                # Ensure '是否成功' is boolean for calculations
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
            st.dataframe(results_df[columns_to_show] if columns_to_show else results_df)
        with tab2:
            st.subheader("失败案例分析 (基于相似度评分)")
            if '是否成功' in results_df.columns:
                bad_cases_df = results_df[results_df['是否成功'] == False]
                if bad_cases_df.empty: st.success("太棒了！没有发现失败案例。")
                else: st.dataframe(bad_cases_df[columns_to_show] if columns_to_show else bad_cases_df)
            else: st.warning("评测结果数据中缺少 '是否成功' 列。")
    else:
        st.info("未生成有效的评测数据用于最终报告。")

elif not st.session_state.active_evaluation_run: # If no eval running and no results yet
     st.info("点击“开始评测”以查看结果。")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**注意**: '是否成功'的判断基于相似度评分。后端当前使用的相似度阈值为 **{SIMILARITY_THRESHOLD:.1f}**。")
