import { describe, expect, it } from 'vitest';
import { formatHash, parseHash } from './route';

describe('route', () => {
  it('round-trips every route shape', () => {
    const routes = [
      { view: 'subjects' },
      { view: 'subject', sid: 's1', tab: 'library' },
      { view: 'subject', sid: 's1', tab: 'topics' },
      { view: 'subject', sid: 's1', tab: 'papers' },
      { view: 'paper', pid: 'p9' },
    ] as const;
    for (const r of routes) expect(parseHash(formatHash(r))).toEqual(r);
    expect(formatHash({ view: 'subjects' })).toBe('#/subjects');
    expect(formatHash({ view: 'subject', sid: 's1', tab: 'papers' })).toBe('#/subjects/s1/papers');
    expect(formatHash({ view: 'paper', pid: 'p9' })).toBe('#/papers/p9');
  });
  it('unknown hashes give subjects', () => {
    for (const h of ['', '#', '#/nope', '#/subjects/s1/bogus', '#/papers']) {
      expect(parseHash(h)).toEqual({ view: 'subjects' });
    }
  });
  it('does not throw on malformed or incomplete hashes', () => {
    for (const h of ['#/subjects/%/library', '#/papers/%E0%A4%A', '#/subjects/', '#/papers/']) {
      expect(parseHash(h)).toEqual({ view: 'subjects' });
    }
    expect(() => parseHash('#/subjects/s1/library/x')).not.toThrow();
  });
});
