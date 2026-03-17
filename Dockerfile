FROM python:3.11-slim

# System deps for matplotlib rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (pre-built wheels, no compilation needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories for uploads and results
RUN mkdir -p static/uploads static/results

ENV PORT=8080
EXPOSE 8080

CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 300 wsgi:app
