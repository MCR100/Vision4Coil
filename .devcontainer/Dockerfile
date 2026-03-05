FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bash \
    libgl1 libglib2.0-0 ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# venv
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

WORKDIR /workspaces