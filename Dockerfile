# Use the official lightweight Python image.
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Set working directory
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy local code to the container image.
COPY . ./

# Install production dependencies.
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run the web service on container startup. Use gunicorn for production.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
