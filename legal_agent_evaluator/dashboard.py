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
    'streamed_results_log': [], # Stores raw log strings for display as they stream in
    'final_eval_results': [], # Stores parsed result items for final report
    'evaluation_complete': False,
    'active_evaluation_run': False
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
streaming_results_placeholder = st.empty()

# --- Main Interaction: Start Evaluation Button ---
if st.button("🚀 开始评测", type="primary", disabled=st.session_state.active_evaluation_run):
    st.session_state.streamed_results_log = []
    st.session_state.final_eval_results = []
    st.session_state.evaluation_complete = False
    st.session_state.active_evaluation_run = True
    streaming_results_placeholder.text_area("评测进度:", "初始化评测...", height=300, key="log_display_initial")


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
            st.session_state.streamed_results_log.append("INFO: 评测将使用自定义API配置.")
        else:
            st.sidebar.warning("已勾选“启用自定义API配置”，但未填写任何有效的自定义API信息。将使用后端默认配置。")
            st.session_state.streamed_results_log.append("WARN: 已勾选自定义配置但未提供有效信息，将使用默认配置。")
    else:
        st.sidebar.info("评测将使用后端默认API配置。")
        st.session_state.streamed_results_log.append("INFO: 使用后端默认API配置。")

    streaming_results_placeholder.text_area("评测进度:", "\n".join(st.session_state.streamed_results_log), height=300, key="log_display_setup")

    with st.spinner("评测进行中，结果将逐步显示... 请勿刷新页面。"):
        try:
            response = requests.post(API_URL, json=payload, stream=True, timeout=1800)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        item = json.loads(decoded_line)

                        if item.get("type") == "system":
                            log_entry = f"SYSTEM: {item.get('message', '')} (Agent: {item.get('agent', 'N/A')})"
                            st.session_state.streamed_results_log.append(log_entry)
                            if item.get("message") == "所有评测已完成。":
                                st.session_state.evaluation_complete = True
                        else: # It's a result item
                            st.session_state.final_eval_results.append(item) # Store full result item
                            # Prepare a simpler log entry for real-time display
                            log_entry = (
                                f"结果: 智能体: {item.get('智能体', 'N/A')}, "
                                f"任务ID: {item.get('任务ID', 'N/A')}, "
                                f"查询: {item.get('查询语句', '')[:20]}..., " # Shorter query
                                f"成功: {item.get('是否成功', False)}, "
                                f"得分: {item.get('相似度得分', '0.00')}"
                            )
                            st.session_state.streamed_results_log.append(log_entry)

                        # Update the placeholder with the latest log messages
                        # The key change is important if we want st.empty to replace content reliably without rerun
                        streaming_results_placeholder.text_area(
                            "评测进度:",
                            "\n".join(st.session_state.streamed_results_log),
                            height=300,
                            key=f"log_update_{len(st.session_state.streamed_results_log)}"
                        )
                    except json.JSONDecodeError:
                        err_log = f"错误: 无法解析的日志行: {decoded_line[:100]}"
                        st.session_state.streamed_results_log.append(err_log)
                        streaming_results_placeholder.text_area("评测进度:", "\n".join(st.session_state.streamed_results_log), height=300)
                    except Exception as e_inner:
                        err_log = f"错误: 处理流数据时出错 - {str(e_inner)[:100]}"
                        st.session_state.streamed_results_log.append(err_log)
                        streaming_results_placeholder.text_area("评测进度:", "\n".join(st.session_state.streamed_results_log), height=300)

            if not st.session_state.evaluation_complete:
                st.session_state.streamed_results_log.append("SYSTEM: 后端数据流已结束。")
                st.session_state.evaluation_complete = True

            streaming_results_placeholder.text_area("评测进度:", "\n".join(st.session_state.streamed_results_log), height=300) # Final update to log display
            st.success("评测数据流接收完毕！正在处理最终报告...") # This will show up

        except requests.exceptions.RequestException as e:
            st.error(f"请求后端服务时发生错误: {e}")
            st.session_state.streamed_results_log.append(f"ERROR: 请求后端服务时发生错误: {e}")
            st.session_state.evaluation_complete = True # Mark as complete to allow report gen (even if empty)
        except Exception as e_outer:
            st.error(f"评测过程中发生未知错误: {e_outer}")
            st.session_state.streamed_results_log.append(f"ERROR: 评测过程中发生未知错误: {e_outer}")
            st.session_state.evaluation_complete = True
        finally:
            st.session_state.active_evaluation_run = False
            # No st.experimental_rerun() here. UI will update on next natural Streamlit cycle or interaction.

# --- Final Results Display Area (triggered after streaming is complete) ---
# This block will execute when Streamlit reruns the script AND evaluation_complete is True
if st.session_state.evaluation_complete and st.session_state.final_eval_results:
    results_df = pd.DataFrame(st.session_state.final_eval_results)

    if not results_df.empty:
        # It's better to clear or hide the streaming placeholder once final report is ready.
        # However, st.empty() needs to be called in the same script run path.
        # For now, let's just write the final report below it.
        # streaming_results_placeholder.empty() # This might not work as expected without rerun

        st.header("📊 最终评测报告")
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
            st.dataframe(results_df[columns_to_show] if columns_to_show else results_df)
        with tab2:
            st.subheader("失败案例分析 (基于相似度评分)")
            if '是否成功' in results_df.columns:
                bad_cases_df = results_df[results_df['是否成功'] == False]
                if bad_cases_df.empty: st.success("太棒了！没有发现失败案例。")
                else: st.dataframe(bad_cases_df[columns_to_show] if columns_to_show else bad_cases_df)
            else: st.warning("评测结果数据中缺少 '是否成功' 列。")
    else: # if results_df is empty but evaluation was marked complete
        st.info("评测完成，但未生成有效的评测数据用于最终报告。请检查流式日志获取更多信息。")

elif not st.session_state.active_evaluation_run:
     st.info("点击“开始评测”以查看结果。")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**注意**: '是否成功'的判断基于相似度评分。后端当前使用的相似度阈值为 **{SIMILARITY_THRESHOLD:.1f}**。")
