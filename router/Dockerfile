FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gcc libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA-compatible torch (bump CUDA suffix when upgrading torch)
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.12.0

# Install laya and its deps without upgrading torch
COPY requirements.txt .
RUN echo "torch==2.12.0" > /tmp/constraint.txt && \
    pip install --no-build-isolation --constraint /tmp/constraint.txt -r requirements.txt

COPY app.py index.html ./

EXPOSE 8010
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010", "--no-access-log"]
