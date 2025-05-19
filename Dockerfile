FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    APP_HOME=/app

WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Add HOST ENV variable
ENV HOST=0.0.0.0

# Modify CMD to explicitly bind to 0.0.0.0
CMD exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --timeout 120 \
    main:app

