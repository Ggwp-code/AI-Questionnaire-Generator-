import { useState } from 'react';
import { ApiError, addMember } from '../../services/tribunalApi';
import { formatHash, type Tab } from './route';
import ErrorNotice from './ErrorNotice';

const TABS: [Tab, string][] = [['library', 'Library'], ['topics', 'Topics'], ['papers', 'Papers']];

export default function SubjectPage({ sid, tab }: { sid: string; tab: Tab }) {
  const [email, setEmail] = useState('');
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [added, setAdded] = useState<string | null>(null);

  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    try {
      await addMember(sid, email.trim());
      setAdded(email.trim());
      setEmail('');
      setError(null);
    } catch (err) {
      setAdded(null);
      setError(err as Error);
    }
  };

  return (
    <section className="space-y-6">
      <a href="#/subjects" className="text-sm text-accent hover:underline">All subjects</a>
      <nav className="flex gap-2" aria-label="Subject sections">
        {TABS.map(([t, label]) => (
          <a key={t} href={formatHash({ view: 'subject', sid, tab: t })}
            aria-current={t === tab ? 'page' : undefined}
            className={`rounded-full px-4 py-1.5 text-sm ${t === tab ? 'bg-accent text-white' : 'bg-surface text-ink'}`}>
            {label}
          </a>
        ))}
      </nav>
      {/* Tasks 4, 5 and 6 replace this placeholder with the real pages */}
      <p className="text-ink/60">The {tab} page is not available yet.</p>
      <form onSubmit={add} className="flex flex-wrap items-end gap-3">
        <label className="text-sm text-ink">
          Colleague email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
            className="mt-1 block rounded-lg border border-warm px-3 py-2" />
        </label>
        <button type="submit" className="rounded-full bg-accent px-5 py-2 text-sm font-medium text-white hover:bg-accent-hover">
          Add colleague
        </button>
      </form>
      <ErrorNotice error={error} />
      {added && <p className="text-sm text-ink">Added {added}.</p>}
    </section>
  );
}
