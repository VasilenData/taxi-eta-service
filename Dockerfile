FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
COPY models/current ./models/current
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "tripeta.api:app", "--host", "0.0.0.0", "--port", "8000"]
