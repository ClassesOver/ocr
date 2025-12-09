# 基于 Ubuntu 22.04
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖和 Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1-mesa-glx \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libwebp-dev \
        wget \
        curl \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv 到全局位置
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install -m 755 /root/.cargo/bin/uv /usr/local/bin/uv

# 创建应用目录和非 root 用户
RUN useradd -m -u 1000 -s /bin/bash ocruser \
    && mkdir -p /app \
    && chown -R ocruser:ocruser /app

# 设置工作目录
WORKDIR /app

# 使用 uv 创建 Python 3.11 虚拟环境（在 root 下创建，然后更改所有权）
RUN uv venv --python python3.11 /app/.venv && \
    chown -R ocruser:ocruser /app/.venv

# 设置虚拟环境为全局 Python 环境
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# 复制依赖文件并安装（利用缓存层）
COPY --chown=ocruser:ocruser requirements.txt .
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install -r requirements.txt && \
    uv pip install -U ultralytics && \
    chown -R ocruser:ocruser /app/.venv

# 复制应用代码
COPY --chown=ocruser:ocruser . .

# 切换到非 root 用户
USER ocruser

# 暴露端口
EXPOSE 8078

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8078/ || exit 1

# 启动应用（使用 Gunicorn）
CMD ["gunicorn", "-c", "gunicorn.conf.py", "api:app"]
