FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY ./app /app/app

RUN groupadd -r celeryuser && useradd -r -g celeryuser celeryuser

RUN chown -R celeryuser:celeryuser .

USER celeryuser

CMD ["celery", "-A", "worker_process", "worker"]