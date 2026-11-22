import { useCallback, useEffect, useState } from 'react';

export type Tab = 'library' | 'topics' | 'papers';
export type Route =
  | { view: 'subjects' }
  | { view: 'subject'; sid: string; tab: Tab }
  | { view: 'paper'; pid: string };

const TABS: Tab[] = ['library', 'topics', 'papers'];

export function parseHash(hash: string): Route {
  const [root, id, tab] = hash.replace(/^#\/?/, '').split('/');
  if (root === 'subjects' && id && TABS.includes(tab as Tab)) {
    return { view: 'subject', sid: decodeURIComponent(id), tab: tab as Tab };
  }
  if (root === 'papers' && id) return { view: 'paper', pid: decodeURIComponent(id) };
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
