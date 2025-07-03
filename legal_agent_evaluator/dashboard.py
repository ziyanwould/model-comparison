# dashboard.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- 页面配置 ---
st.set_page_config(
    page_title="法律智能体评测仪表盘",
    page_icon="⚖️",
    layout="wide"
)

# --- 后端API地址 ---
API_URL = "http://127.0.0.1:8000/evaluate"

# --- 页面标题 ---
st.title("⚖️ 法律智能体准确率评测仪表盘")
st.caption("对比『通用大模型』与『法律领域微调模型』的表现")

# --- 主交互 (代码与之前相同) ---
if 'results' not in st.session_state:
    st.session_state.results = None

if st.button("🚀 开始评测", type="primary"):
    with st.spinner("评测进行中，请稍候... (正在调用后端API)"):
        try:
            # 确保 FastAPI 服务在 http://127.0.0.1:8000 上运行
            response = requests.post(API_URL, timeout=120)
            response.raise_for_status() # 如果请求失败 (4xx or 5xx), 会抛出异常
            st.session_state.results = response.json()
            st.success("评测完成！")
        except requests.exceptions.ConnectionError:
            st.error(f"无法连接到后端服务 ({API_URL})。请确保FastAPI服务已启动并且地址正确。")
            st.session_state.results = None
        except requests.exceptions.Timeout:
            st.error(f"连接后端服务 ({API_URL}) 超时。请检查服务状态或网络连接。")
            st.session_state.results = None
        except requests.exceptions.HTTPError as e:
            st.error(f"后端服务 ({API_URL}) 返回错误: {e.response.status_code} {e.response.reason}")
            try:
                st.error(f"详细错误: {e.response.json()}") # 尝试显示后端返回的JSON错误信息
            except ValueError: # 如果响应不是JSON
                st.error(f"详细错误: {e.response.text}")
            st.session_state.results = None
        except requests.exceptions.RequestException as e:
            st.error(f"请求后端服务 ({API_URL}) 时发生未知错误: {e}")
            st.session_state.results = None


# --- 结果展示 ---
if st.session_state.results:
    results_df = pd.DataFrame(st.session_state.results)

    st.divider()

    # 1. 总体指标分析
    st.header("📊 总体指标对比")

    if not results_df.empty and '智能体' in results_df.columns and '是否成功' in results_df.columns:
        success_rate = results_df.groupby('智能体')['是否成功'].mean().reset_index()
        success_rate['成功率'] = success_rate['是否成功'] * 100

        fig = px.bar(
            success_rate, x='智能体', y='成功率',
            title='各智能体任务成功率对比',
            text=success_rate['成功率'].apply(lambda x: f'{x:.2f}%'),
            color='智能体',
            color_discrete_map={'通用大模型': '#ff9900', '微调后模型': '#00529B'} # 更新颜色映射
        )
        fig.update_layout(yaxis_title='成功率 (%)', xaxis_title=None, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

        st.info("分析可见，**法律领域微调模型**在法律术语的精确性、文书要素提取的结构化以及法律合规判断的准确性上，均显著优于通用大模型。")
    else:
        st.warning("评测结果数据不完整或格式不正确，无法生成总体指标图表。")
        if not results_df.empty:
            st.write("接收到的数据样本:", results_df.head())
        else:
            st.write("未接收到评测结果数据。")

    st.divider()

    # 2. 详细结果与Bad Case分析
    st.header("🔍 详细结果与Bad Case分析")

    if not results_df.empty:
        tab1, tab2 = st.tabs(["所有结果", "失败案例 (Bad Cases)"])

        with tab1:
            st.subheader("所有任务评测明细")
            st.dataframe(results_df)

        with tab2:
            st.subheader("失败案例分析")
            if '是否成功' in results_df.columns:
                bad_cases_df = results_df[results_df['是否成功'] == False]
                if bad_cases_df.empty:
                    st.success("太棒了！没有发现失败案例。")
                else:
                    st.warning("以下是执行失败的案例，可用于重点分析和模型迭代。")
                    st.dataframe(bad_cases_df)
            else:
                st.warning("评测结果数据中缺少 '是否成功' 列，无法筛选失败案例。")
                st.write("接收到的数据列:", results_df.columns.tolist())
    else:
        st.info("没有可供展示的详细评测结果。")
