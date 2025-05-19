FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HOST=0.0.0.0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn \
    --bind $HOST:$PORT \
    --workers 2 \
    --timeout 120 \
    --preload \
    main:app
