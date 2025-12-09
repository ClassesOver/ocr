#!/bin/bash

# OCR 服务快速启动脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "======================================"
echo "    OCR 服务快速部署脚本"
echo "======================================"
echo -e "${NC}"

# 检查 Docker
echo -e "${YELLOW}[1/5] 检查 Docker 环境...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ 错误: Docker 未安装${NC}"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker 已安装${NC}"

# 检查 Docker Compose
echo -e "\n${YELLOW}[2/5] 检查 Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠ Docker Compose 未安装，将使用 docker 命令${NC}"
    USE_COMPOSE=false
else
    echo -e "${GREEN}✓ Docker Compose 已安装${NC}"
    USE_COMPOSE=true
fi

# 选择构建方式
echo -e "\n${YELLOW}[3/5] 选择部署方式:${NC}"
echo "  1) 标准版 (推荐，快速启动)"
echo "  2) 生产优化版 (体积更小)"
echo "  3) GPU 加速版 (需要 NVIDIA GPU)"
read -p "请选择 [1-3, 默认: 1]: " choice
choice=${choice:-1}

case $choice in
    1)
        DOCKERFILE="Dockerfile"
        IMAGE_TAG="latest"
        echo -e "${GREEN}选择: 标准版${NC}"
        ;;
    2)
        DOCKERFILE="Dockerfile.prod"
        IMAGE_TAG="prod"
        echo -e "${GREEN}选择: 生产优化版${NC}"
        ;;
    3)
        DOCKERFILE="Dockerfile.gpu"
        IMAGE_TAG="gpu"
        echo -e "${GREEN}选择: GPU 加速版${NC}"
        # 检查 NVIDIA Docker
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo -e "${RED}❌ GPU 环境检测失败${NC}"
            echo "请确保已安装 NVIDIA 驱动和 nvidia-docker2"
            exit 1
        fi
        echo -e "${GREEN}✓ GPU 环境检测成功${NC}"
        ;;
    *)
        echo -e "${RED}无效选择，使用默认配置${NC}"
        DOCKERFILE="Dockerfile"
        IMAGE_TAG="latest"
        ;;
esac

# 构建镜像
echo -e "\n${YELLOW}[4/5] 构建 Docker 镜像...${NC}"
docker build -f $DOCKERFILE -t ocr-service:$IMAGE_TAG .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 镜像构建失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 镜像构建成功${NC}"

# 启动容器
echo -e "\n${YELLOW}[5/5] 启动服务...${NC}"

# 停止并删除旧容器
docker stop ocr-service 2>/dev/null || true
docker rm ocr-service 2>/dev/null || true

# 根据选择启动
if [ "$choice" = "3" ]; then
    # GPU 版本
    docker run -d \
        --name ocr-service \
        --gpus all \
        -p 8078:8078 \
        --restart unless-stopped \
        ocr-service:$IMAGE_TAG
else
    # CPU 版本
    docker run -d \
        --name ocr-service \
        -p 8078:8078 \
        --restart unless-stopped \
        ocr-service:$IMAGE_TAG
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 容器启动失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 服务启动成功${NC}"

# 等待服务就绪
echo -e "\n${YELLOW}等待服务就绪...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8078/ > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 服务已就绪${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# 显示结果
echo -e "\n${GREEN}"
echo "======================================"
echo "    部署完成！"
echo "======================================"
echo -e "${NC}"
echo -e "服务地址: ${BLUE}http://localhost:8078${NC}"
echo ""
echo "常用命令:"
echo "  查看日志:   docker logs -f ocr-service"
echo "  停止服务:   docker stop ocr-service"
echo "  重启服务:   docker restart ocr-service"
echo "  删除容器:   docker rm -f ocr-service"
echo ""
echo "测试 API:"
echo -e "  ${BLUE}curl -X POST http://localhost:8078/paddle_ocr \\${NC}"
echo -e "    ${BLUE}-H 'secret: 6aac5f82-141b-44a4-817f-369c64b12b19' \\${NC}"
echo -e "    ${BLUE}-F 'file=@your-image.jpg'${NC}"
echo ""
echo -e "${YELLOW}更多信息请查看 DOCKER_README.md${NC}"

