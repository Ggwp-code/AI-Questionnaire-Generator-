import { useCallback, useEffect, useState } from 'react';

export type Tab = 'library' | 'topics' | 'papers';
export type Route =
  | { view: 'subjects' }
  | { view: 'subject'; sid: string; tab: Tab }
  | { view: 'paper'; pid: string };

const TABS: Tab[] = ['library', 'topics', 'papers'];

function decode(part: string): string | null {
  try {
    return decodeURIComponent(part);
  } catch {
    return null;
  }
}

export function parseHash(hash: string): Route {
  const [root, id, tab] = hash.replace(/^#\/?/, '').split('/');
  const key = id ? decode(id) : null;
  if (key && root === 'subjects' && TABS.includes(tab as Tab)) return { view: 'subject', sid: key, tab: tab as Tab };
  if (key && root === 'papers') return { view: 'paper', pid: key };
  return { view: 'subjects' };
}

export function formatHash(r: Route): string {
  if (r.view === 'subject') return `#/subjects/${encodeURIComponent(r.sid)}/${r.tab}`;
  if (r.view === 'paper') return `#/papers/${encodeURIComponent(r.pid)}`;
  return '#/subjects';
}

export function useRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const go = useCallback((r: Route) => { window.location.hash = formatHash(r); }, []);
  return [route, go];
}
