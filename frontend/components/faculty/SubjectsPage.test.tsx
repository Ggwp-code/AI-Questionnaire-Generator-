import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../services/tribunalApi', async (orig) => ({
  ...(await orig<typeof import('../../services/tribunalApi')>()),
  listSubjects: vi.fn(),
  createSubject: vi.fn(),
}));
import { ApiError, createSubject, listSubjects } from '../../services/tribunalApi';
import SubjectsPage from './SubjectsPage';

beforeEach(() => vi.resetAllMocks());
afterEach(cleanup);

describe('SubjectsPage', () => {
  it('lists subjects and creating one adds it', async () => {
    vi.mocked(listSubjects).mockResolvedValue([{ id: 's1', name: 'Alpha', role: 'owner' }]);
    vi.mocked(createSubject).mockResolvedValue({ id: 's2', name: 'Beta', role: 'owner' });
    render(<SubjectsPage />);
    const link = await screen.findByRole('link', { name: /Alpha/ });
    expect(link.getAttribute('href')).toBe('#/subjects/s1/library');
    await userEvent.type(screen.getByLabelText('Subject name'), 'Beta');
    await userEvent.click(screen.getByRole('button', { name: 'Create subject' }));
    expect(createSubject).toHaveBeenCalledWith('Beta');
    expect(await screen.findByRole('link', { name: /Beta/ })).toBeTruthy();
  });

  it('shows the server error and no rows when listing fails', async () => {
    vi.mocked(listSubjects).mockRejectedValue(new ApiError(500, 'boom', 'Listing failed', ['db down']));
    render(<SubjectsPage />);
    expect(await screen.findByText('Listing failed')).toBeTruthy();
    expect(screen.getByText('db down')).toBeTruthy();
    expect(screen.queryAllByRole('link')).toHaveLength(0);
  });
});
