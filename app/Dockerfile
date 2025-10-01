# 更容易拉取的基础镜像
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 基础构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖声明
COPY requirements.txt /app/

# 先装 PyTorch CPU 官方 wheel（不要换源）
RUN pip install --no-cache-dir \
      torch==2.3.0 torchvision==0.18.0 \
      --index-url https://download.pytorch.org/whl/cpu

# 再装其他依赖（这一步可以用国内源加速）
RUN pip install --no-cache-dir -r requirements.txt \
      -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制你的代码和权重
COPY app.py /app/
COPY nets /app/nets
COPY util /app/util
COPY checkpoint /app/checkpoint

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
