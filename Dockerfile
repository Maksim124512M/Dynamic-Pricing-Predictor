FROM python:3.12

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PORT=8000

CMD sh -c "mlflow server --backend-store-uri $DATABASE_URL --host 0.0.0.0 --port 5000 & uvicorn app:app --host 0.0.0.0 --port $PORT"