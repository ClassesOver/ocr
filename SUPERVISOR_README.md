# Supervisor 服务管理说明

本项目使用 Supervisor 管理以下两个服务：
1. **Gunicorn (Flask)** - API 服务，端口：8078
2. **PaddleX** - 表格识别服务（table_recognition_v2），端口：8079

## 服务状态查看

进入容器后，可以使用以下命令：

```bash
# 查看所有服务状态
supervisorctl status

# 查看特定服务状态
supervisorctl status gunicorn
supervisorctl status paddlex
```

## 服务控制命令

```bash
# 启动服务
supervisorctl start gunicorn
supervisorctl start paddlex

# 停止服务
supervisorctl stop gunicorn
supervisorctl stop paddlex

# 重启服务
supervisorctl restart gunicorn
supervisorctl restart paddlex

# 重新加载配置
supervisorctl reread
supervisorctl update
```

## 查看日志

日志文件位于 `/app/logs/` 目录：
- `/app/logs/supervisord.log` - Supervisor 主日志
- `/app/logs/gunicorn.log` - Gunicorn 服务日志
- `/app/logs/paddlex.log` - PaddleX 服务日志

```bash
# 实时查看日志
tail -f /app/logs/gunicorn.log
tail -f /app/logs/paddlex.log

# 查看 supervisor 日志
tail -f /app/logs/supervisord.log
```

## 配置文件

- **Supervisor 配置**: `/etc/supervisor/conf.d/supervisord.conf`
- **PaddleX 管道配置**: `/app/code/table_recognition_v2.yaml`

## 环境变量

可以在 `docker-compose.yml` 中配置：
- `FLASK_PORT` - Flask 服务端口（默认：8078）
- `PADDLEX_PORT` - PaddleX 服务端口（默认：8079）

## 故障排查

如果服务启动失败：

1. 检查 supervisor 状态：
```bash
supervisorctl status
```

2. 查看错误日志：
```bash
cat /app/logs/gunicorn.log
cat /app/logs/paddlex.log
```

3. 手动重启服务：
```bash
supervisorctl restart all
```

4. 重新加载 supervisor 配置：
```bash
supervisorctl reread
supervisorctl update
```

