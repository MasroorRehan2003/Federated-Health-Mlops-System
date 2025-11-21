FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only needed deps
COPY requirements_fl.txt .
RUN pip install --upgrade pip && pip install -r requirements_fl.txt

COPY . .

CMD ["python", "src/federated_learning/fl_server.py"]
