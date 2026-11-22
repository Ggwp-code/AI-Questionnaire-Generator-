import { useState } from 'react';
import { setDevUser } from '../../services/tribunalApi';

export default function DevSignIn({ onDone }: { onDone: () => void }) {
  const [email, setEmail] = useState('');
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    setDevUser(email.trim());
    onDone();
  };
  return (
    <form onSubmit={submit} className="mx-auto mt-24 max-w-sm space-y-3 rounded-2xl bg-surface p-6 shadow-sm">
      <h1 className="font-display text-xl font-bold text-ink">Development sign-in</h1>
      <label className="block text-sm text-ink">
        Email
        <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
          className="mt-1 w-full rounded-lg border border-warm px-3 py-2" />
      </label>
      <button type="submit" className="rounded-full bg-accent px-5 py-2 text-sm font-medium text-white hover:bg-accent-hover">
        Sign in
      </button>
    </form>
  );
}
