# Gunicorn 配置文件
import multiprocessing
import os

# 服务器绑定
bind = "0.0.0.0:8078"

# 工作进程数：CPU 核心数 * 2 + 1
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))

# 工作模式：sync/gevent/eventlet
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "sync")

# 每个工作进程的线程数
threads = int(os.getenv("GUNICORN_THREADS", 2))

# 超时时间（秒）- OCR 处理较慢，设置较大值
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))

# 优雅重启超时
graceful_timeout = 30

# 最大请求数（防止内存泄漏）
max_requests = 1000
max_requests_jitter = 50

# 日志
accesslog = "-"
errorlog = "-"
loglevel = "info"

# 进程名称
proc_name = "ocr-service"

# 预加载应用
preload_app = True

# 使用内存文件系统提高性能
worker_tmp_dir = "/dev/shm"

# 钩子函数
def when_ready(server):
    print(f"Gunicorn 就绪，监听 {bind}，工作进程: {workers}")

def post_fork(server, worker):
    print(f"工作进程 {worker.pid} 已启动")

