FROM python:3.11-slim

WORKDIR /app

# Copy app code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set MLflow URI via env var if needed
# ENV MLFLOW_TRACKING_URI=http://host.docker.internal:8080

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
