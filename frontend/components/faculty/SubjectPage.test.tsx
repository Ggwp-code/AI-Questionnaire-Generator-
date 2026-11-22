import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../services/tribunalApi', async (orig) => ({
  ...(await orig<typeof import('../../services/tribunalApi')>()),
  addMember: vi.fn(),
}));
import { ApiError, addMember } from '../../services/tribunalApi';
import SubjectPage from './SubjectPage';

beforeEach(() => vi.resetAllMocks());
afterEach(cleanup);

describe('SubjectPage', () => {
  it('has the three tabs', () => {
    render(<SubjectPage sid="s1" tab="topics" />);
    for (const n of ['Library', 'Topics', 'Papers']) {
      expect(screen.getByRole('link', { name: n }).getAttribute('href')).toBe(`#/subjects/s1/${n.toLowerCase()}`);
    }
  });
  it('adds a colleague and shows a refusal', async () => {
    vi.mocked(addMember).mockRejectedValue(new ApiError(403, 'forbidden', 'Not the owner', ['owners only']));
    render(<SubjectPage sid="s1" tab="library" />);
    await userEvent.type(screen.getByLabelText('Colleague email'), 'a@b.c');
    await userEvent.click(screen.getByRole('button', { name: 'Add colleague' }));
    expect(addMember).toHaveBeenCalledWith('s1', 'a@b.c');
    expect(await screen.findByText('Not the owner')).toBeTruthy();
    expect(screen.getByText('owners only')).toBeTruthy();
  });
});
