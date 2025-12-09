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
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
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
        ca-certificates

# 设置 Python 3.11 为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# 升级 pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 创建应用目录和非 root 用户
RUN useradd -m -u 1000 -s /bin/bash ocruser \
    && mkdir -p /app \
    && chown -R ocruser:ocruser /app

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装（利用缓存层）
COPY --chown=ocruser:ocruser requirements.txt .
RUN pip3 install -r requirements.txt \
    && pip3 install paddleocr paddlepaddle \
    && pip3 install flask-pydantic loguru \
    && pip3 install gunicorn gevent \
    && rm -rf /root/.cache/pip

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
