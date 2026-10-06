# Self-hosted Laya decision server behind the Jev-compatible `POST /v1/systemone` endpoint.
# This is the official laya-serve — no custom inference logic.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install "laya[serve]>=0.3.26"

EXPOSE 8000
CMD ["laya-serve"]
