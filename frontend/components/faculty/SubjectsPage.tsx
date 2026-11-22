import { useEffect, useState } from 'react';
import { ApiError, createSubject, listSubjects } from '../../services/tribunalApi';
import type { Subject } from '../../services/tribunalTypes';
import ErrorNotice from './ErrorNotice';

export default function SubjectsPage() {
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [name, setName] = useState('');

  useEffect(() => {
    listSubjects().then(setSubjects, setError);
  }, []);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    try {
      const s = await createSubject(name.trim());
      setSubjects((prev) => [...prev, s]);
      setName('');
      setError(null);
    } catch (err) {
      setError(err as Error);
    }
  };

  return (
    <section className="space-y-6">
      <h1 className="font-display text-2xl font-bold text-ink">Subjects</h1>
      <ErrorNotice error={error} />
      <ul className="space-y-2">
        {subjects.map((s) => (
          <li key={s.id}>
            <a href={`#/subjects/${encodeURIComponent(s.id)}/library`}
              className="block rounded-xl bg-surface px-4 py-3 text-ink shadow-sm hover:text-accent">
              {s.name} <span className="text-xs text-ink/60">({s.role})</span>
            </a>
          </li>
        ))}
      </ul>
      <form onSubmit={create} className="flex flex-wrap items-end gap-3">
        <label className="text-sm text-ink">
          Subject name
          <input value={name} onChange={(e) => setName(e.target.value)}
            className="mt-1 block rounded-lg border border-warm px-3 py-2" />
        </label>
        <button type="submit" className="rounded-full bg-accent px-5 py-2 text-sm font-medium text-white hover:bg-accent-hover">
          Create subject
        </button>
      </form>
    </section>
  );
}
