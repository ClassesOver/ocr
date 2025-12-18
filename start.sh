#!/bin/bash

# 启动脚本：同时启动 Flask (Gunicorn) 和 PaddleX 服务

# 定义端口
FLASK_PORT=${FLASK_PORT:-8078}
PADDLEX_PORT=${PADDLEX_PORT:-8079}

# 定义日志目录和文件
LOG_DIR=${LOG_DIR:-/app/logs}
GUNICORN_LOG="${LOG_DIR}/gunicorn.log"
PADDLEX_LOG="${LOG_DIR}/paddlex.log"

# 创建日志目录
mkdir -p ${LOG_DIR}

# 切换到应用目录
cd /app/code

echo "正在启动 OCR 服务..."
echo "Flask 服务端口: ${FLASK_PORT}"
echo "PaddleX 服务端口: ${PADDLEX_PORT} (高性能推理模式)"
echo "日志目录: ${LOG_DIR}"

# 后台启动 Gunicorn (Flask)，输出日志到文件
/app/.venv/bin/gunicorn -c gunicorn.conf.py --bind "0.0.0.0:${FLASK_PORT}" api:app >> "${GUNICORN_LOG}" 2>&1 &
GUNICORN_PID=$!

# 等待一下确保 Gunicorn 启动
sleep 2

# 检查 Gunicorn 是否成功启动
if ! kill -0 $GUNICORN_PID 2>/dev/null; then
    echo "错误: Gunicorn 启动失败"
    exit 1
fi

# 后台启动 PaddleX 服务（通过配置文件启用高性能推理），输出日志到文件
/app/.venv/bin/paddlex --serve --pipeline PaddleOCR-VL.yaml --host 0.0.0.0 --port ${PADDLEX_PORT} >> "${PADDLEX_LOG}" 2>&1 &
PADDLEX_PID=$!

# 等待一下确保 PaddleX 服务启动
sleep 5

# 检查 PaddleX 是否成功启动
if ! kill -0 $PADDLEX_PID 2>/dev/null; then
    echo "错误: PaddleX 服务启动失败"
    kill $GUNICORN_PID 2>/dev/null
    exit 1
fi

echo "服务已启动"
echo "Gunicorn PID: ${GUNICORN_PID}, 日志文件: ${GUNICORN_LOG}"
echo "PaddleX PID: ${PADDLEX_PID}, 日志文件: ${PADDLEX_LOG}"

# 信号处理：优雅关闭
trap "echo '收到关闭信号，正在停止服务...'; kill $GUNICORN_PID $PADDLEX_PID 2>/dev/null; wait; exit 0" SIGTERM SIGINT

# 等待所有后台进程
wait

