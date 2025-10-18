FROM python:3.11-slim

ARG EXTRAS=0
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true
RUN if [ "$EXTRAS" = "1" ]; then pip install --no-cache-dir pandas pyarrow fastparquet; fi

COPY . .

CMD ["python", "-m", "src.main", "--mode", "baseline", "--dataset", "synthetic", "--steps", "400", "--batch_size", "8", "--seq_len", "128"]
