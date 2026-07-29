FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY taxonomy.json .

# Download model files from GitHub (LFS objects not available in build context)
RUN mkdir -p model && \
    curl -L -o model/pytorch_model.bin \
    https://github.com/Xavi36772/-tagging-service/raw/master/model/pytorch_model.bin && \
    curl -L -o model/tokenizer.json \
    https://github.com/Xavi36772/-tagging-service/raw/master/model/tokenizer.json && \
    curl -L -o model/tokenizer_config.json \
    https://github.com/Xavi36772/-tagging-service/raw/master/model/tokenizer_config.json && \
    curl -L -o model/thresholds.npy \
    https://github.com/Xavi36772/-tagging-service/raw/master/model/thresholds.npy && \
    curl -L -o model/metrics.json \
    https://github.com/Xavi36772/-tagging-service/raw/master/model/metrics.json

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
