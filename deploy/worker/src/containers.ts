import { DurableObject } from "cloudflare:workers";

export interface Env {
  API_CONTAINER: DurableObjectNamespace<ApiContainer>;
  INGEST_CONTAINER: DurableObjectNamespace<IngestContainer>;
  INTERNAL_DISPATCH_TOKEN?: string;
  API_POOL_SIZE?: string;
}

// Every name tribunal.settings reads (Settings fields upper-cased, plus TRIBUNAL_ENV_FILE), and the
// credential and TRIBUNAL_* names that config/runtime.example.yaml reads. Nothing else reaches a container.
export const CONTAINER_ENV_NAMES = [
  "APP_PROFILE", "APP_NAME", "PUBLIC_BASE_URL", "INSTITUTION_NAME",
  "HOST", "PORT", "ALLOWED_ORIGINS",
  "DATABASE_URL", "DB_POOL_SIZE", "DB_MAX_OVERFLOW",
  "RUNTIME_CONFIG",
  "AUTH_MODE", "OIDC_ISSUER", "OIDC_AUDIENCE", "OIDC_JWKS_URL", "OIDC_REQUIRE_EMAIL_VERIFIED", "AUTH_JWT_HEADER",
  "ADMIN_EMAILS",
  "STORAGE_BACKEND", "STORAGE_LOCAL_ROOT", "S3_ENDPOINT_URL", "S3_REGION", "S3_BUCKET_UPLOADS", "S3_BUCKET_EXPORTS",
  "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "PRESIGN_TTL_SECONDS",
  "SIGNING_SECRET", "INTERNAL_DISPATCH_TOKEN",
  "JOB_NOTIFY", "CF_ACCOUNT_ID", "CF_API_TOKEN", "CF_QUEUE_INGESTION_ID", "CF_QUEUE_GENERATION_ID",
  "PDF_FONT_REGULAR", "PDF_FONT_BOLD", "PDF_FONT_FALLBACKS",
  "SERVE_FRONTEND_DIR", "LOG_LEVEL", "LOG_FORMAT", "ALLOW_STUB_PROVIDER",
  "TRIBUNAL_ENV_FILE",
  "OPENAI_API_KEY",
  "TRIBUNAL_WRITER_MODEL", "TRIBUNAL_REVIEWER_MODEL", "TRIBUNAL_REASONER_MODEL",
  "TRIBUNAL_OCR_LANGUAGE", "TRIBUNAL_WORKER_THREADS",
] as const;

export function containerEnv(env: object, names: readonly string[]): Record<string, string> {
  const source = env as Record<string, unknown>;
  const picked: Record<string, string> = {};
  for (const name of names) {
    const value = source[name];
    if (typeof value === "string") picked[name] = value;
  }
  return picked;
}

const PORT = 8080;
const READY_TIMEOUT_MS = 60_000;
// Longer than the longest job (OCR is capped at 600 s), so a job whose caller was cut off still finishes.
const INACTIVITY_TIMEOUT_MS = 15 * 60 * 1000;

// Follows https://developers.cloudflare.com/containers/get-started/ (Durable Object Container API).
abstract class TribunalContainer extends DurableObject<Env> {
  protected abstract readonly instance: "standard-1" | "standard-2";
  private starting: Promise<void> | undefined;
  private monitoring: Promise<void> | undefined;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const container = ctx.container;
    if (container?.running) {
      // A restarted Durable Object starts without a timeout.
      void ctx.blockConcurrencyWhile(() => container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS));
      this.observeExit();
    }
  }

  async fetch(request: Request): Promise<Response> {
    this.starting ??= this.startAndWaitForPort().finally(() => {
      this.starting = undefined;
    });
    await this.starting;

    const url = new URL(request.url);
    url.protocol = "http:";
    url.host = "container";
    const forwarded = new Request(url, request);
    forwarded.headers.delete("host");
    return this.ctx.container!.getTcpPort(PORT).fetch(forwarded);
  }

  private async startAndWaitForPort(): Promise<void> {
    const container = this.ctx.container!;
    if (!container.running) {
      container.start({
        image: container.images.base,
        instance: this.instance,
        // PostgreSQL, object storage, the identity provider and the model provider are all outside.
        enableInternet: true,
        env: containerEnv(this.env, CONTAINER_ENV_NAMES),
      });
    }
    this.observeExit();
    await container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS);

    const port = container.getTcpPort(PORT);
    const deadline = Date.now() + READY_TIMEOUT_MS;
    while (Date.now() < deadline) {
      try {
        const response = await port.fetch("http://container/healthz", { signal: AbortSignal.timeout(1000) });
        await response.body?.cancel();
        if (response.ok) return;
      } catch {
        // Not listening yet.
      }
      await scheduler.wait(500);
    }
    throw new Error(`container did not answer /healthz on port ${PORT} within ${READY_TIMEOUT_MS / 1000} s`);
  }

  private observeExit(): void {
    if (this.monitoring) return;
    this.monitoring = this.ctx.container!.monitor()
      .then(() => console.log("container exited"))
      .catch((error: unknown) => console.error("container failed:", error))
      .finally(() => {
        this.monitoring = undefined;
      });
    this.ctx.waitUntil(this.monitoring);
  }
}

export class ApiContainer extends TribunalContainer {
  protected readonly instance = "standard-1";
}

export class IngestContainer extends TribunalContainer {
  protected readonly instance = "standard-2";
}
