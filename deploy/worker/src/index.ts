import { ApiContainer, IngestContainer, type Env } from "./containers";
import { drainRequest, kindsForQueue, route } from "./dispatch";

export { ApiContainer, IngestContainer };

// Queue consumers and cron triggers get 15 minutes of wall time. A drain stops starting jobs after
// this many seconds but finishes the one it is running (OCR is capped at 600 s), after up to 60 s of
// container start: 60 + 180 + 600 = 840 s, 60 s inside the limit.
const DRAIN_MAX_SECONDS = 180;
const QUEUES = ["tribunal-ingestion", "tribunal-generation"];

function apiContainer(env: Env) {
  const size = Math.max(1, Math.floor(Number(env.API_POOL_SIZE)) || 2);
  return env.API_CONTAINER.getByName(`api-${Math.floor(Math.random() * size)}`);
}

async function drain(queue: string, env: Env): Promise<boolean> {
  const kinds = kindsForQueue(queue);
  const container = kinds.includes("ingest") ? env.INGEST_CONTAINER.getByName("ingest-0") : apiContainer(env);
  try {
    const response = await container.fetch(drainRequest(kinds, env.INTERNAL_DISPATCH_TOKEN ?? "", DRAIN_MAX_SECONDS));
    await response.body?.cancel();
    if (response.ok) return true;
    console.error(`drain for ${queue} failed with status ${response.status}`);
  } catch (error) {
    console.error(`drain for ${queue} failed:`, error);
  }
  return false;
}

export default {
  async fetch(request, env) {
    if (route(new URL(request.url).pathname) === "api") return apiContainer(env).fetch(request);
    // /internal/* is never public; every other path was already left to Static Assets.
    return new Response("Not found", { status: 404 });
  },

  async queue(batch, env) {
    if (await drain(batch.queue, env)) batch.ackAll();
    else batch.retryAll();
  },

  async scheduled(_controller, env) {
    const results = await Promise.all(QUEUES.map((queue) => drain(queue, env)));
    if (results.includes(false)) throw new Error("cron drain failed; see the log lines above");
  },
} satisfies ExportedHandler<Env>;
