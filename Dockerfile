# 基于 Ubuntu 22.04
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（使用默认 Python 3.10）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
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

# 创建 python 符号链接
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 安装 uv 到全局位置
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install -m 755 /root/.local/bin/uv /usr/local/bin/uv

# 使用 uv 安装 Python 3.11
RUN uv python install 3.11

# 创建应用目录和非 root 用户
RUN useradd -m -u 1000 -s /bin/bash ocruser \
    && mkdir -p /app \
    && chown -R ocruser:ocruser /app

# 设置工作目录
WORKDIR /app

# 使用 uv 创建 Python 3.11 虚拟环境（在 root 下创建，然后更改所有权）
RUN uv venv --python 3.11 /app/.venv && \
    chown -R ocruser:ocruser /app/.venv


# 复制依赖文件并安装（利用缓存层）
COPY --chown=ocruser:ocruser requirements.txt .
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install -r requirements.txt
RUN uv pip install -U ultralytics
RUN chown -R ocruser:ocruser /app/.venv

# 复制应用代码
COPY --chown=ocruser:ocruser . .

# 切换到非 root 用户
USER ocruser

# 暴露端口
EXPOSE 8078

# 启动应用（使用虚拟环境中的 Gunicorn）
CMD ["/app/.venv/bin/gunicorn", "-c", "gunicorn.conf.py", "api:app"]
