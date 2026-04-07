FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend-requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/backend-requirements.txt

COPY . /app

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
