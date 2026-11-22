import { useEffect, useRef } from 'react';

// Sequential timeouts instead of setInterval so a slow call never overlaps the next.
export function usePolling(fn: () => Promise<void>, active: boolean, intervalMs = 3000): void {
  const latest = useRef(fn);
  latest.current = fn;
  useEffect(() => {
    if (!active) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try { await latest.current(); } catch { /* the caller reports its own errors */ }
      if (!stopped) timer = setTimeout(tick, intervalMs);
    };
    timer = setTimeout(tick, intervalMs);
    return () => { stopped = true; clearTimeout(timer); };
  }, [active, intervalMs]);
}
