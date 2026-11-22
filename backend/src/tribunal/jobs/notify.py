"""Best-effort queue wake-up. The database is the source of truth; a lost message is recovered by the cron drain."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

QUEUE_FOR_KIND = {"ingest": "ingestion", "generate_paper": "generation", "regenerate_slot": "generation"}
_API = "https://api.cloudflare.com/client/v4"


class NullNotifier:
    def notify(self, kind: str, job_id: str) -> None:
        pass


class CloudflareQueueNotifier:
    def __init__(self, settings, client: httpx.Client | None = None):
        self._account = settings.cf_account_id
        self._token = settings.cf_api_token.get_secret_value()
        self._queues = {"ingestion": settings.cf_queue_ingestion_id, "generation": settings.cf_queue_generation_id}
        self._client = client or httpx.Client()

    def notify(self, kind: str, job_id: str) -> None:
        try:
            url = f"{_API}/accounts/{self._account}/queues/{self._queues[QUEUE_FOR_KIND[kind]]}/messages"
            response = self._client.post(
                url, headers={"Authorization": f"Bearer {self._token}"}, timeout=5,
                json={"content_type": "json", "body": {"job_id": job_id, "kind": kind}})
            if not response.is_success:
                log.warning("queue wake-up for job %s was refused with status %s", job_id, response.status_code)
        except Exception as exc:  # never fail a committed request; type only, the message may echo request data
            log.warning("queue wake-up for job %s failed: %s", job_id, type(exc).__name__)
