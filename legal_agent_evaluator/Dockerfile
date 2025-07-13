# 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器
COPY . /app

# 安装依赖
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 暴露端口（FastAPI:8000，Streamlit:8501）
EXPOSE 8000 8501

# 启动脚本（可选：同时启动FastAPI和Streamlit）
CMD ["sh", "-c", "uvicorn main_api:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"]