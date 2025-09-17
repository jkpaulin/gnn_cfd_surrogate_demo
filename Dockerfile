FROM python:3.11-slim

WORKDIR /app

# Copy app code
COPY pre_trained_model ./pre_trained_model
COPY requirements.txt ./requirements.txt
COPY static ./static
COPY main.py ./main.py

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set MLflow URI via env var if needed
# ENV MLFLOW_TRACKING_URI=http://host.docker.internal:8080

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
