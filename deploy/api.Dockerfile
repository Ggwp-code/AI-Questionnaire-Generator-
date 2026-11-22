# API and generation jobs. Build from the repository root:
#   docker build --platform linux/amd64 -f deploy/api.Dockerfile -t tribunal-api .
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY backend /src/backend
RUN pip install --no-cache-dir "/src/backend[openai,s3,postgres]" \
    && mkdir -p /app \
    && cp -r /src/backend/config /app/config \
    && rm -rf /src \
    && useradd --system --create-home --uid 10001 tribunal

WORKDIR /app
# APP_PROFILE, DATABASE_URL, secrets and RUNTIME_CONFIG (for example
# /app/config/runtime.example.yaml) are supplied as environment variables at start.
ENV HOST=0.0.0.0 \
    PORT=8080 \
    PDF_FONT_REGULAR=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
    PDF_FONT_BOLD=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
USER tribunal
EXPOSE 8080
CMD ["tribunal", "serve"]
