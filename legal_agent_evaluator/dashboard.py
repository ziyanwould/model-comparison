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

API_URL = "http://127.0.0.1:8000/evaluate"

st.title("⚖️ 法律智能体准确率评测仪表盘")
st.caption("对比『通用大模型』与『法律领域微调模型』的表现 (成功标准基于相似度评分)")

st.sidebar.header("⚙️ 自定义API配置 (可选)")
use_custom_config = st.sidebar.checkbox("启用自定义API配置", key="use_custom_api_config")

# Initialize session state for API configs
default_session_state = {
    'gen_api_key': "", 'gen_base_url': "", 'gen_model_name': "",
    'ft_api_url': "", 'ft_token': "",
    'results': None
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

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

if st.button("🚀 开始评测", type="primary"):
    payload = {}
    if use_custom_config:
        custom_configs = {}
        ga_config = {
            k: v for k, v in {
                "api_key": st.session_state.gen_api_key,
                "base_url": st.session_state.gen_base_url,
                "model_name": st.session_state.gen_model_name
            }.items() if v
        }
        if ga_config: custom_configs["general_agent"] = ga_config

        ft_config = {
            k: v for k, v in {
                "api_url": st.session_state.ft_api_url,
                "token": st.session_state.ft_token
            }.items() if v
        }
        if ft_config: custom_configs["finetuned_agent"] = ft_config

        if custom_configs:
            payload["custom_configs"] = custom_configs
            st.sidebar.info("评测将使用自定义API配置。")
            st.sidebar.json(payload)
        else:
            st.sidebar.warning("已勾选“启用自定义API配置”，但未填写任何有效的自定义API信息。将使用后端默认配置。")
    else:
        st.sidebar.info("评测将使用后端默认API配置。")

    with st.spinner("评测进行中 (可能需要几分钟，取决于模型和测试用例数量)..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=600) # Increased timeout further
            response.raise_for_status()
            st.session_state.results = response.json()
            st.success("评测完成！")
        except requests.exceptions.Timeout:
            st.error(f"连接后端服务 ({API_URL}) 超时 (600秒)。")
            st.session_state.results = None
        except requests.exceptions.ConnectionError:
            st.error(f"无法连接到后端服务 ({API_URL})。")
            st.session_state.results = None
        except requests.exceptions.HTTPError as e:
            error_detail = f"后端服务 ({API_URL}) 返回错误: {e.response.status_code} {e.response.reason}"
            try:
                error_json = e.response.json()
                error_detail += f"\n详细信息: {error_json.get('detail', e.response.text)}"
            except ValueError:
                error_detail += f"\n详细信息: {e.response.text}"
            st.error(error_detail)
            st.session_state.results = None
        except requests.exceptions.RequestException as e:
            st.error(f"请求后端服务 ({API_URL}) 时发生未知错误: {e}")
            st.session_state.results = None
        except json.JSONDecodeError:
            st.error("后端返回的响应不是有效的JSON格式。")
            st.session_state.results = None

if st.session_state.results:
    if not isinstance(st.session_state.results, list):
        st.error(f"从后端接收到的结果格式不正确。期望一个列表，但收到: {type(st.session_state.results)}")
        if isinstance(st.session_state.results, dict) and 'detail' in st.session_state.results:
             st.error(f"详细错误: {st.session_state.results['detail']}")
        results_df = pd.DataFrame()
    else:
        results_df = pd.DataFrame(st.session_state.results)

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
                # st.info("分析...") # Info message might need rephrasing based on similarity scores
            except Exception as e:
                st.error(f"生成图表时出错: {e}")
                st.dataframe(results_df.head())
        else:
            st.warning("评测结果数据不完整或格式不正确，无法生成总体指标图表。")
            st.dataframe(results_df.head())

        st.divider()
        st.header("🔍 详细结果与Bad Case分析")

        # Define columns to display, including new ones
        display_columns = [
            "任务ID", "任务类型", "查询语句", "标准答案", "模型回答",
            "完全匹配", "相似度得分", "是否成功", "智能体"
        ]
        # Filter out columns that might not exist if something went wrong
        columns_to_show = [col for col in display_columns if col in results_df.columns]

        tab1, tab2 = st.tabs(["所有结果", "失败案例 (Bad Cases)"])

        with tab1:
            st.subheader("所有任务评测明细")
            if not results_df.empty:
                st.dataframe(results_df[columns_to_show])
            else:
                st.info("没有评测结果可供展示。")

        with tab2:
            st.subheader("失败案例分析 (基于相似度评分)")
            if '是否成功' in results_df.columns:
                bad_cases_df = results_df[results_df['是否成功'] == False]
                if bad_cases_df.empty:
                    st.success("太棒了！没有发现失败案例。")
                else:
                    st.warning("以下是执行失败的案例，可用于重点分析和模型迭代。")
                    st.dataframe(bad_cases_df[columns_to_show])
            else:
                st.warning("评测结果数据中缺少 '是否成功' 列，无法筛选失败案例。")
    else:
        st.info("没有有效的评测结果可供展示。")
else:
    st.info("点击“开始评测”以查看结果。")

# Add a note about the similarity threshold being used (from backend)
# This is a bit of a hack as SIMILARITY_THRESHOLD is defined in backend.
# For a cleaner solution, backend could return this info.
# However, for now, we can assume it's known or mention it generally.
st.sidebar.markdown("---")
st.sidebar.markdown(f"**注意**: '是否成功'的判断基于相似度评分。后端当前使用的相似度阈值为 **{SIMILARITY_THRESHOLD:.1f}**。")
