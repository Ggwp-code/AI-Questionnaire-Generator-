// Pure routing and drain helpers; index.ts wires them to bindings.

export function route(pathname: string): "api" | "internal" | "assets" {
  if (pathname === "/internal" || pathname.startsWith("/internal/")) return "internal";
  if (pathname.startsWith("/api/")) return "api";
  return "assets";
}

export function drainRequest(kinds: string[], token: string, maxSeconds: number): Request {
  return new Request("http://container/internal/drain", {
    method: "POST",
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({ kinds, max_seconds: maxSeconds }),
  });
}

export function kindsForQueue(queueName: string): string[] {
  switch (queueName) {
    case "tribunal-ingestion":
      return ["ingest"];
    case "tribunal-generation":
      return ["generate_paper", "regenerate_slot"];
    default:
      // an empty kinds list would drain every kind, so an unknown queue is an error
      throw new Error(`no job kinds are mapped to queue ${queueName}`);
  }
}
