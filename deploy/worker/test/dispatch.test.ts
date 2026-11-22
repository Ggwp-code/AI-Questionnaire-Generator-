import { describe, expect, it, vi } from "vitest";

// The Workers runtime module does not exist under Node; the container classes are not exercised here.
vi.mock("cloudflare:workers", () => ({ DurableObject: class {} }));

import { CONTAINER_ENV_NAMES, containerEnv, type Env } from "../src/containers";
import { drainRequest, kindsForQueue, route } from "../src/dispatch";
import worker from "../src/index";

type Call = { name: string; request: Request };

function fakeEnv(status = 200) {
  const calls: Call[] = [];
  const namespace = {
    getByName: (name: string) => ({
      fetch: async (request: Request) => {
        calls.push({ name, request });
        return Response.json({ processed: 0 }, { status });
      },
    }),
  };
  const env = { API_CONTAINER: namespace, INGEST_CONTAINER: namespace, INTERNAL_DISPATCH_TOKEN: "test-token" };
  return { env: env as unknown as Env, calls };
}

function fakeBatch(queue: string) {
  return { queue, messages: [], ackAll: vi.fn(), retryAll: vi.fn() };
}

async function runQueue(queue: string, status: number) {
  const { env, calls } = fakeEnv(status);
  const batch = fakeBatch(queue);
  await worker.queue(batch as unknown as MessageBatch, env);
  return { batch, calls };
}

describe("route", () => {
  it("separates internal, api and asset paths", () => {
    expect(route("/internal/drain")).toBe("internal");
    expect(route("/api/papers")).toBe("api");
    expect(route("/")).toBe("assets");
  });
});

describe("fetch", () => {
  it("answers 404 for a public /internal/drain request without reaching a container", async () => {
    const { env, calls } = fakeEnv();
    const request = new Request<unknown, IncomingRequestCfProperties>("https://gateway.test/internal/drain", {
      method: "POST",
      headers: { authorization: "Bearer test-token" },
      body: JSON.stringify({ kinds: ["ingest"], max_seconds: 10 }),
    });
    const response = await worker.fetch(request, env);
    expect(response.status).toBe(404);
    expect(calls).toHaveLength(0);
  });

  it("forwards /api/* to an API container from the pool", async () => {
    const { env, calls } = fakeEnv();
    const request = new Request<unknown, IncomingRequestCfProperties>("https://gateway.test/api/papers");
    const response = await worker.fetch(request, env);
    expect(response.status).toBe(200);
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toMatch(/^api-[01]$/);
    expect(calls[0].request).toBe(request);
  });
});

describe("queue", () => {
  it("acks the batch when the drain succeeds", async () => {
    const { batch, calls } = await runQueue("tribunal-ingestion", 200);
    expect(batch.ackAll).toHaveBeenCalledOnce();
    expect(batch.retryAll).not.toHaveBeenCalled();
    expect(calls.map((c) => c.name)).toEqual(["ingest-0"]);
  });

  it("retries the batch when the drain fails", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const { batch } = await runQueue("tribunal-generation", 500);
    expect(batch.retryAll).toHaveBeenCalledOnce();
    expect(batch.ackAll).not.toHaveBeenCalled();
    expect(error).toHaveBeenCalledWith(expect.stringContaining("500"));
    error.mockRestore();
  });

  it("sends the bearer token and the queue's kinds to the drain route", async () => {
    const { calls } = await runQueue("tribunal-generation", 200);
    expect(calls).toHaveLength(1);
    const { name, request } = calls[0];
    expect(name).toMatch(/^api-[01]$/);
    expect(request.method).toBe("POST");
    expect(request.url).toBe("http://container/internal/drain");
    expect(request.headers.get("authorization")).toBe("Bearer test-token");
    expect(await request.json()).toEqual({ kinds: ["generate_paper", "regenerate_slot"], max_seconds: 180 });
  });
});

describe("scheduled", () => {
  it("drains both kind groups once", async () => {
    const { env, calls } = fakeEnv();
    await worker.scheduled({} as ScheduledController, env);
    const drained = await Promise.all(
      calls.map(async (c) => [c.name.replace(/\d+$/, "N"), ((await c.request.json()) as { kinds: string[] }).kinds]),
    );
    expect(drained).toEqual(
      expect.arrayContaining([
        ["ingest-N", ["ingest"]],
        ["api-N", ["generate_paper", "regenerate_slot"]],
      ]),
    );
    expect(drained).toHaveLength(2);
  });

  it("fails the cron run when a drain fails", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const { env, calls } = fakeEnv(500);
    await expect(worker.scheduled({} as ScheduledController, env)).rejects.toThrow(/cron drain failed/);
    expect(calls).toHaveLength(2);
    expect(error).toHaveBeenCalledTimes(2);
    error.mockRestore();
  });
});

describe("dispatch helpers", () => {
  it("maps each queue to its job kinds and rejects an unknown queue", () => {
    expect(kindsForQueue("tribunal-ingestion")).toEqual(["ingest"]);
    expect(kindsForQueue("tribunal-generation")).toEqual(["generate_paper", "regenerate_slot"]);
    expect(() => kindsForQueue("other")).toThrow(/other/);
  });

  it("builds the drain request", async () => {
    const request = drainRequest(["ingest"], "t", 5);
    expect(request.headers.get("content-type")).toBe("application/json");
    expect(await request.json()).toEqual({ kinds: ["ingest"], max_seconds: 5 });
  });
});

describe("containerEnv", () => {
  it("passes only allow-listed names that are set", () => {
    const env = { DATABASE_URL: "postgresql://db", OPENAI_API_KEY: "k", API_POOL_SIZE: "2", UNLISTED_SECRET: "s" };
    expect(containerEnv(env, CONTAINER_ENV_NAMES)).toEqual({ DATABASE_URL: "postgresql://db", OPENAI_API_KEY: "k" });
    expect(containerEnv(env, ["DATABASE_URL"])).toEqual({ DATABASE_URL: "postgresql://db" });
  });
});
