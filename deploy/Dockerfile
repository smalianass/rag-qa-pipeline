# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app/src
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=5

COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.14.1+cpu" "torchvision==0.29.1+cpu"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-api.txt

COPY src ./src


EXPOSE 8000

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
