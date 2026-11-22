import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./services/tribunalApi', async (orig) => ({
  ...(await orig<typeof import('./services/tribunalApi')>()),
  me: vi.fn(),
  listSubjects: vi.fn(),
  getDevUser: vi.fn(),
  setDevUser: vi.fn(),
}));
import { getDevUser, listSubjects, me } from './services/tribunalApi';
import App from './App';

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(me).mockResolvedValue({ id: 'u1', email: 'me@x.y', is_admin: false });
  vi.mocked(listSubjects).mockResolvedValue([]);
  window.location.hash = '';
});
afterEach(() => { cleanup(); vi.unstubAllEnvs(); });

describe('App', () => {
  it('shows the signed-in email and subjects', async () => {
    vi.stubEnv('VITE_AUTH_MODE', 'prod');
    render(<App />);
    expect(screen.getByText('Tribunal')).toBeTruthy();
    expect(await screen.findByText('me@x.y')).toBeTruthy();
    expect(screen.getByLabelText('Subject name')).toBeTruthy();
  });
  it('never renders DevSignIn in a non-dev build', async () => {
    vi.stubEnv('VITE_AUTH_MODE', 'prod');
    vi.mocked(getDevUser).mockReturnValue(null);
    render(<App />);
    await screen.findByText('me@x.y');
    expect(screen.queryByRole('button', { name: 'Sign in' })).toBeNull();
  });
  it('renders DevSignIn in dev mode without a stored user', async () => {
    vi.stubEnv('VITE_AUTH_MODE', 'dev');
    vi.mocked(getDevUser).mockReturnValue(null);
    render(<App />);
    expect(await screen.findByRole('button', { name: 'Sign in' })).toBeTruthy();
    expect(screen.getByLabelText('Email')).toBeTruthy();
  });
});
