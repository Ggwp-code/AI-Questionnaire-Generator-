# Ingestion with OCR. Build from the repository root:
#   docker build --platform linux/amd64 -f deploy/ingest.Dockerfile -t tribunal-ingest .
#   add --build-arg TESSERACT_LANGS="eng hin tam" to change the OCR languages
FROM python:3.12-slim

ARG TESSERACT_LANGS="eng hin"
RUN apt-get update \
    && apt-get install -y --no-install-recommends ocrmypdf tesseract-ocr \
        $(for l in $TESSERACT_LANGS; do echo "tesseract-ocr-$l"; done) \
    && rm -rf /var/lib/apt/lists/*

COPY backend /src/backend
RUN pip install --no-cache-dir "/src/backend[s3,postgres]" \
    && mkdir -p /app \
    && cp -r /src/backend/config /app/config \
    && rm -rf /src \
    && useradd --system --create-home --uid 10001 tribunal

WORKDIR /app
# APP_PROFILE, DATABASE_URL, secrets and RUNTIME_CONFIG are supplied at start.
ENV HOST=0.0.0.0 PORT=8080
USER tribunal
EXPOSE 8080
CMD ["tribunal", "serve"]
