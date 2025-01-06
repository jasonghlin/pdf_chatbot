FROM python:3.11.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

# 複製現有項目文件
COPY . /app/

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -r requirements_poetry.txt

COPY ChatPDF.py /app/

# 暴露埠
EXPOSE 8501

# 健康檢查
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 啟動應用程式
# ENTRYPOINT ["/bin/bash", "-c", "source /app/.venv/bin/activate && exec streamlit run ChatPDF.py --server.port=8501 --server.address=0.0.0.0"]
ENTRYPOINT ["streamlit", "run", "ChatPDF.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=debug"]
