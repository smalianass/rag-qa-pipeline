FROM python:3.12-slim

WORKDIR /rag

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY rag.py search.py generate.py chat_memory.py ./


EXPOSE 8000

CMD ["uvicorn", "rag:app", "--host", "0.0.0.0", "--port", "8000"]
