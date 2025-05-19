# Use Python 3.11 slim base image
FROM python:3.11-slim

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY app.py .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Start the FastAPI app with Gunicorn and Uvicorn workers
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]
