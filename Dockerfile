# ============================================================
# MindPal Pro Backend — Dockerfile
# 单阶段构建：安装依赖，运行 FastAPI 服务
# ============================================================

FROM python:3.12-slim

# 创建非 root 用户
RUN groupadd -r mindpal && useradd -r -g mindpal -d /app -s /sbin/nologin mindpal

WORKDIR /app

# pip 全局设置：不缓存
ENV PIP_NO_CACHE_DIR=1

# 复制 requirements.txt 并安装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# 复制应用代码和模型
COPY backend/ backend/
COPY bert_data/ bert_data/

# 创建数据、日志目录并设置权限
RUN mkdir -p backend/data backend/logs && \
    chown -R mindpal:mindpal /app

# 环境变量默认值（可被 docker-compose environment 覆盖）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SESSION_DB_PATH=backend/data/sessions.db \
    CONTENT_DB_FILE=backend/data/content_db.json \
    LOG_DIR=backend/logs \
    BERT_MODEL_PATH=bert_data/models/v4_2_domain_only_v2/best_model \
    EMOTION_MODEL_PATH=bert_data/models/emotion_v1/best_model \
    BERT_DEVICE=cpu \
    EMOTION_DEVICE=cpu

EXPOSE 8000

USER mindpal

# 健康检查：60s 宽限期（BERT 模型延迟加载）
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["python", "backend/main.py"]
