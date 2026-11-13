FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    TZ=Europe/Moscow \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 analyst \
    && mkdir -p /tmp/matplotlib \
    && chown analyst:analyst /tmp/matplotlib

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agent ./agent
COPY api ./api
COPY data ./data
COPY knowledge ./knowledge

RUN mkdir -p /app/agent/reports \
    && chown -R analyst:analyst /app

USER analyst

EXPOSE 8000

CMD ["python", "-m", "api"]
