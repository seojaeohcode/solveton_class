FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# libgomp1 is required by common scientific Python wheels; build tools cover
# the rare package that needs a local wheel build on a new architecture.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade \
      pip==24.3.1 \
      setuptools==75.6.0 \
      wheel==0.45.1 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY . ./

RUN python -m compileall -q analysis.py app.py

EXPOSE 7860

CMD ["python", "app.py"]

