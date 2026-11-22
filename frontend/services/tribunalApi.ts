import type {
  ApprovalResult, DocumentInfo, ExportDoc, Options, Paper, PaperInput, PaperSummary, Passage,
  RevisionContent, Revision, Subject, Template, TemplateSpec, Topic, UploadResult, User,
} from './tribunalTypes';

const DEV_USER_KEY = 'tribunal.devUser';

export class ApiError extends Error {
  status: number;
  code: string;
  details: string[];
  constructor(status: number, code: string, message: string, details: string[] = []) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export function getDevUser(): string | null {
  try {
    return localStorage.getItem(DEV_USER_KEY);
  } catch {
    return null;
  }
}

export function setDevUser(email: string): void {
  try {
    localStorage.setItem(DEV_USER_KEY, email);
  } catch {
    // storage unavailable: the user stays unset
  }
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  if (import.meta.env.VITE_AUTH_MODE === 'dev') {
    const user = getDevUser();
    if (user) headers.set('X-Dev-User', user);
  }
  let res: Response;
  try {
    res = await fetch(`/api${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, 'network_error', 'The server could not be reached');
  }
  if (!res.ok) {
    let body: any = null;
    try {
      body = await res.json();
    } catch {
      // not JSON: fall through to a generic error
    }
    const e = body?.error;
    throw new ApiError(
      res.status,
      e?.code ?? 'http_error',
      e?.message ?? `Request failed with status ${res.status}`,
      Array.isArray(e?.details) ? e.details.map(String) : [],
    );
  }
  return res;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  return (await request(path, init)).json() as Promise<T>;
}

function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  return json<T>(path, {
    method,
    ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  });
}

export const me = () => json<User>('/me');
export const options = () => json<Options>('/options');

export const listSubjects = () => json<Subject[]>('/subjects');
export const createSubject = (name: string) => send<Subject>('POST', '/subjects', { name });
export const addMember = (sid: string, email: string) =>
  send<{ subject_id: string; email: string }>('POST', `/subjects/${sid}/members`, { email });

export const listDocuments = (sid: string) => json<DocumentInfo[]>(`/subjects/${sid}/documents`);

export function uploadDocument(sid: string, file: File, languages?: string[]): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);
  if (languages && languages.length) form.append('languages', languages.join(','));
  return json<UploadResult>(`/subjects/${sid}/documents`, { method: 'POST', body: form });
}

export const reprocessDocument = (did: string) =>
  send<UploadResult>('POST', `/documents/${did}/reprocess`);

export async function fetchDocumentPdf(did: string): Promise<Blob> {
  return (await request(`/documents/${did}/file`)).blob();
}

export const listTopics = (sid: string) => json<Topic[]>(`/subjects/${sid}/topics`);
export const createTopic = (sid: string, title: string) =>
  send<Topic>('POST', `/subjects/${sid}/topics`, { title });
export const renameTopic = (tid: string, title: string) => send<Topic>('PATCH', `/topics/${tid}`, { title });
export const getPassage = (pid: string) => json<Passage>(`/passages/${pid}`);

export const listTemplates = () => json<Template[]>('/templates');
export const createTemplate = (name: string, spec: TemplateSpec) =>
  send<Template>('POST', '/templates', { name, spec });

export const listPapers = (sid: string) => json<PaperSummary[]>(`/subjects/${sid}/papers`);
export const createPaper = (input: PaperInput) =>
  send<{ paper_id: string; job_id: string }>('POST', '/papers', input);
export const getPaper = (pid: string) => json<Paper>(`/papers/${pid}`);

export const regenerateSlot = (pid: string, n: number) =>
  send<{ job_id: string }>('POST', `/papers/${pid}/slots/${n}/regenerate`);
export const editSlot = (pid: string, n: number, content: RevisionContent) =>
  send<Revision>('PUT', `/papers/${pid}/slots/${n}`, { content });
export const approveSlot = (pid: string, n: number, revisionId: string) =>
  send<ApprovalResult>('POST', `/papers/${pid}/slots/${n}/approve`, { revision_id: revisionId });

export type ExportPartName = 'paper' | 'key';

export const exportJson = (pid: string, part: ExportPartName) =>
  json<ExportDoc>(`/papers/${pid}/export?part=${part}&format=json`);
export async function exportPdf(pid: string, part: ExportPartName): Promise<Blob> {
  return (await request(`/papers/${pid}/export?part=${part}&format=pdf`)).blob();
}
