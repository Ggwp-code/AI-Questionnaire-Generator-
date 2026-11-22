import { act, cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DocumentInfo } from '../../services/tribunalTypes';

vi.mock('../../services/tribunalApi', async (orig) => ({
  ...(await orig<typeof import('../../services/tribunalApi')>()),
  options: vi.fn(),
  listDocuments: vi.fn(),
  uploadDocument: vi.fn(),
  reprocessDocument: vi.fn(),
  fetchDocumentPdf: vi.fn(),
}));
import { ApiError, fetchDocumentPdf, listDocuments, options, reprocessDocument, uploadDocument } from '../../services/tribunalApi';
import LibraryPage from './LibraryPage';

const doc = (o: Partial<DocumentInfo>): DocumentInfo => ({
  id: 'd1', filename: 'book.pdf', status: 'ready', error: null, page_count: 10,
  created_at: '', ready_at: null, searchable_seconds: null, excluded_pages: [], ...o,
});

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(options).mockResolvedValue({
    question_types: [], bloom_levels: [], difficulty_levels: [], default_ocr_languages: ['xx', 'yy'],
  });
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('LibraryPage', () => {
  it('renders status, pages, searchable time, error and excluded pages', async () => {
    vi.mocked(listDocuments).mockResolvedValue([
      doc({ id: 'a', filename: 'a.pdf', searchable_seconds: 12, excluded_pages: [3, 7] }),
      doc({ id: 'b', filename: 'b.pdf', status: 'failed', error: 'No usable text', page_count: null }),
    ]);
    render(<LibraryPage sid="s1" />);
    const a = (await screen.findByText('a.pdf')).closest('tr')!;
    expect(within(a).getByText('ready')).toBeTruthy();
    expect(within(a).getByText('10')).toBeTruthy();
    expect(within(a).getByText('Searchable after 12 s')).toBeTruthy();
    expect(within(a).getByText('Pages kept out of evidence: 3, 7')).toBeTruthy();
    const b = screen.getByText('b.pdf').closest('tr')!;
    expect(within(b).getByText('failed')).toBeTruthy();
    expect(within(b).getByText('No usable text')).toBeTruthy();
  });

  it('shows list errors with details', async () => {
    vi.mocked(listDocuments).mockRejectedValue(new ApiError(500, 'x', 'Boom', ['detail one']));
    render(<LibraryPage sid="s1" />);
    expect(await screen.findByText('Boom')).toBeTruthy();
    expect(screen.getByText('detail one')).toBeTruthy();
  });

  it('reprocesses, and disables it for a processing document', async () => {
    vi.mocked(listDocuments).mockResolvedValue([
      doc({ id: 'a', filename: 'a.pdf' }),
      doc({ id: 'b', filename: 'b.pdf', status: 'processing' }),
    ]);
    vi.mocked(reprocessDocument).mockResolvedValue({ document: doc({}), job_id: 'j' });
    render(<LibraryPage sid="s1" />);
    const a = (await screen.findByText('a.pdf')).closest('tr')!;
    const b = screen.getByText('b.pdf').closest('tr')!;
    expect((within(b).getByRole('button', { name: 'Reprocess' }) as HTMLButtonElement).disabled).toBe(true);
    await userEvent.click(within(a).getByRole('button', { name: 'Reprocess' }));
    expect(reprocessDocument).toHaveBeenCalledWith('a');
  });

  it('opens the pdf in a new tab and revokes the url later', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.mocked(listDocuments).mockResolvedValue([doc({})]);
    const blob = new Blob(['x']);
    vi.mocked(fetchDocumentPdf).mockResolvedValue(blob);
    const create = vi.fn(() => 'blob:u');
    const revoke = vi.fn();
    const open = vi.fn();
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: create, revokeObjectURL: revoke }));
    vi.stubGlobal('open', open);
    render(<LibraryPage sid="s1" />);
    await userEvent.click(await screen.findByRole('button', { name: 'Open' }));
    expect(fetchDocumentPdf).toHaveBeenCalledWith('d1');
    expect(create).toHaveBeenCalledWith(blob);
    expect(open).toHaveBeenCalledWith('blob:u#page=1', '_blank');
    expect(revoke).not.toHaveBeenCalled();
    await act(() => vi.advanceTimersByTimeAsync(70000));
    expect(revoke).toHaveBeenCalledWith('blob:u');
  });

  it('uploads with optional languages, placeholder from options', async () => {
    vi.mocked(listDocuments).mockResolvedValue([]);
    vi.mocked(uploadDocument).mockResolvedValue({ document: doc({}), job_id: 'j' });
    render(<LibraryPage sid="s1" />);
    const lang = await screen.findByLabelText('OCR languages');
    await vi.waitFor(() => expect(lang.getAttribute('placeholder')).toBe('xx, yy'));
    let input = screen.getByLabelText('PDF file') as HTMLInputElement;
    expect(input.accept).toBe('application/pdf');
    const f = new File(['x'], 'n.pdf', { type: 'application/pdf' });
    await userEvent.upload(input, f);
    await userEvent.click(screen.getByRole('button', { name: 'Upload' }));
    expect(uploadDocument).toHaveBeenLastCalledWith('s1', f, undefined);
    await vi.waitFor(() => expect(input.isConnected).toBe(false));
    input = screen.getByLabelText('PDF file') as HTMLInputElement;
    await userEvent.upload(input, f);
    await userEvent.type(lang, 'aa, bb');
    await userEvent.click(screen.getByRole('button', { name: 'Upload' }));
    expect(uploadDocument).toHaveBeenLastCalledWith('s1', f, ['aa', 'bb']);
  });

  it('polls while a document is processing and stops once ready', async () => {
    vi.useFakeTimers();
    vi.mocked(listDocuments)
      .mockResolvedValueOnce([doc({ status: 'processing' })])
      .mockResolvedValue([doc({ status: 'ready' })]);
    render(<LibraryPage sid="s1" />);
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(listDocuments).toHaveBeenCalledTimes(1);
    await act(() => vi.advanceTimersByTimeAsync(3100));
    expect(listDocuments).toHaveBeenCalledTimes(2);
    expect(screen.getByText('ready')).toBeTruthy();
    await act(() => vi.advanceTimersByTimeAsync(10000));
    expect(listDocuments).toHaveBeenCalledTimes(2);
  });
});
