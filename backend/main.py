# main.py
import os

# 压制 transformers/HF 非关键警告（模型已缓存到本地，无需联网检查）
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# 压制 OpenMP 运行时冲突警告（torch 与 sklearn 等库共存时常见）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import uvicorn

from api_endpoints import router
from application import create_app
from config import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = create_app(router)


# ------------------ 启动入口 ------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MindPal Pro 后端服务 v3.2 启动中...")
    print("✨ 功能：上下文感知对话系统 + 个性化推荐")
    print("🔗 API地址: http://localhost:8000")
    print("📝 接口文档: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host=config.HOST, port=config.PORT)
