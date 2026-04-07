FROM python:3.11-slim-bookworm

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# HF Spaces default port
EXPOSE 7860

# Single worker — matches single-session assumption and vcpu=2 constraint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
