FROM python:3.12.11-slim-bookworm

ARG OPENHANDS_VERSION=1.16.0

RUN python -m pip install --no-cache-dir "openhands==${OPENHANDS_VERSION}"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin openhands \
    && mkdir -p /opt/metarigor /run/metarigor \
    && chown -R openhands:openhands /opt/metarigor /run/metarigor

COPY restricted_sitecustomize.py /opt/metarigor/sitecustomize.py
COPY model_proxy.py /opt/metarigor/model_proxy.py

ENV PYTHONPATH=/opt/metarigor \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENHANDS_SUPPRESS_BANNER=1

USER openhands
WORKDIR /work

ENTRYPOINT []
