import React, { useEffect, useState } from 'react';
import { ApiError, getDevUser, me } from './services/tribunalApi';
import type { User } from './services/tribunalTypes';
import { useRoute } from './components/faculty/route';
import DevSignIn from './components/faculty/DevSignIn';
import ErrorNotice from './components/faculty/ErrorNotice';
import SubjectsPage from './components/faculty/SubjectsPage';
import SubjectPage from './components/faculty/SubjectPage';

const App: React.FC = () => {
  const devMode = import.meta.env.VITE_AUTH_MODE === 'dev';
  const [signedIn, setSignedIn] = useState(() => !devMode || getDevUser() !== null);
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [route] = useRoute();

  useEffect(() => {
    if (!signedIn) return;
    me().then(setUser, setError);
  }, [signedIn]);

  let content: React.ReactNode;
  if (!signedIn) content = <DevSignIn onDone={() => setSignedIn(true)} />;
  else if (route.view === 'subject') content = <SubjectPage sid={route.sid} tab={route.tab} />;
  else if (route.view === 'paper') content = <p className="text-ink/60">The paper page is not available yet.</p>;
  else content = <SubjectsPage />;

  return (
    <div className="min-h-screen bg-paper font-sans text-ink">
      <header className="flex items-center justify-between border-b border-warm px-6 py-4">
        <a href="#/subjects" className="font-display text-lg font-bold text-accent">Tribunal</a>
        <span className="text-sm">{user?.email}</span>
      </header>
      <main className="mx-auto max-w-5xl space-y-4 px-6 py-8">
        <ErrorNotice error={error} />
        {content}
      </main>
    </div>
  );
};

export default App;
