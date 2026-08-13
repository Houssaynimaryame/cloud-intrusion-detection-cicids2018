FROM python:3.11-slim

WORKDIR /app

# Dependances Python
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif et artefacts du modele
COPY deployment/app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
