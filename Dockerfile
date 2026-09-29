# Minimal image for the ForensiQ consumer (used by PlatformDemo's docker-compose).
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
CMD ["sh", "-c", "until [ -f /data/spans.jsonl ]; do sleep 2; done; forensiq ingest --file /data/spans.jsonl; forensiq report --file /data/spans.jsonl --out /data/rca_report.md; tail -f /dev/null"]
