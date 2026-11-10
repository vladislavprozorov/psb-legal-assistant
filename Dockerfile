FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 INDEX_CACHE_DIR=/app/cache
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY legal_core.py solution.py app.py ./
COPY static ./static
COPY data/knowledge-base.zip ./data/knowledge-base.zip
RUN mkdir -p /app/cache && useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
