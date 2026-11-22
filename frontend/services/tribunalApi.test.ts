import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, approveSlot, getDevUser, setDevUser, uploadDocument } from './tribunalApi';

const fetchMock = vi.fn();

function reply(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock);
  localStorage.clear();
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('tribunal api client', () => {
  it('approves a slot with the revision id in the body', async () => {
    fetchMock.mockResolvedValue(reply(200, { slot_id: 's', number: 2, status: 'approved', revision: {} }));
    await approveSlot('p1', 2, 'r9');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/papers/p1/slots/2/approve');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ revision_id: 'r9' });
  });

  it('turns an error envelope into an ApiError', async () => {
    const envelope = { error: { code: 'conflict', message: 'the question changed', details: ['a', 'b'] } };
    fetchMock.mockResolvedValue(reply(409, envelope));
    const err = await approveSlot('p1', 2, 'r9').catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.code).toBe('conflict');
    expect(err.message).toBe('the question changed');
    expect(err.details).toEqual(['a', 'b']);
  });

  it('turns a network failure into network_error', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));
    const err = await approveSlot('p1', 2, 'r9').catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
    expect(err.code).toBe('network_error');
    expect(err.message).toBe('The server could not be reached');
  });

  it('sends X-Dev-User in dev mode only', async () => {
    fetchMock.mockImplementation(async () => reply(200, {}));
    setDevUser('a@example.org');
    expect(getDevUser()).toBe('a@example.org');
    vi.stubEnv('VITE_AUTH_MODE', 'dev');
    await approveSlot('p1', 2, 'r9');
    expect(new Headers(fetchMock.mock.calls[0][1].headers).get('X-Dev-User')).toBe('a@example.org');
    vi.stubEnv('VITE_AUTH_MODE', '');
    await approveSlot('p1', 2, 'r9');
    expect(new Headers(fetchMock.mock.calls[1][1].headers).has('X-Dev-User')).toBe(false);
  });

  it('uploads a file as multipart with languages', async () => {
    fetchMock.mockResolvedValue(reply(202, { document: {}, job_id: 'j' }));
    const file = new File(['%PDF-'], 'book.pdf', { type: 'application/pdf' });
    await uploadDocument('s1', file, ['eng', 'hin']);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/subjects/s1/documents');
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get('file')).toBeInstanceOf(File);
    expect((init.body as FormData).get('languages')).toBe('eng,hin');
    expect(new Headers(init.headers).has('Content-Type')).toBe(false);
  });
});
