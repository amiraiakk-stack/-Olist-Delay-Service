FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/service
WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY src ./src
COPY scripts ./scripts
COPY config ./config
COPY saved_artifacts ./saved_artifacts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]