import { useCallback, useEffect, useState } from 'react';
import { fetchDocumentPdf, listDocuments, options, reprocessDocument, uploadDocument } from '../../services/tribunalApi';
import type { ApiError } from '../../services/tribunalApi';
import type { DocumentInfo } from '../../services/tribunalTypes';
import ErrorNotice from './ErrorNotice';
import { openPdf } from './openPdf';
import { usePolling } from './usePolling';

const busy = (d: DocumentInfo) => d.status === 'queued' || d.status === 'processing';
const button = 'rounded-full bg-accent px-4 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50';

export default function LibraryPage({ sid }: { sid: string }) {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [fileKey, setFileKey] = useState(0); // remounting the input is how a file input is cleared
  const [languages, setLanguages] = useState('');
  const [placeholder, setPlaceholder] = useState('');

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments(sid));
      setError(null);
    } catch (err) {
      setError(err as Error);
    }
  }, [sid]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    options().then((o) => setPlaceholder(o.default_ocr_languages.join(', '))).catch((err) => setError(err as Error));
  }, []);
  usePolling(refresh, docs.some(busy));

  const act = async (run: () => Promise<unknown>) => {
    try {
      await run();
      await refresh();
    } catch (err) {
      setError(err as Error);
    }
  };

  const upload = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    const langs = languages.split(',').map((l) => l.trim()).filter(Boolean);
    void act(async () => {
      await uploadDocument(sid, file, langs.length ? langs : undefined);
      setFile(null);
      setLanguages('');
      setFileKey((k) => k + 1);
    });
  };

  return (
    <div className="space-y-6">
      <form onSubmit={upload} className="flex flex-wrap items-end gap-3">
        <label className="text-sm text-ink">
          PDF file
          <input key={fileKey} type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="mt-1 block text-sm" />
        </label>
        <label className="text-sm text-ink">
          OCR languages
          <input value={languages} placeholder={placeholder} onChange={(e) => setLanguages(e.target.value)}
            className="mt-1 block rounded-lg border border-warm px-3 py-2" />
        </label>
        <button type="submit" disabled={!file} className={button}>Upload</button>
      </form>
      <ErrorNotice error={error} />
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-ink">
          <thead>
            <tr className="border-b border-warm font-display">
              <th className="py-2 pr-4">File</th><th className="pr-4">Status</th>
              <th className="pr-4">Pages</th><th className="pr-4">Notes</th><th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.id} className="border-b border-warm/50 align-top">
                <td className="py-2 pr-4">{d.filename}</td>
                <td className="pr-4">{d.status}</td>
                <td className="pr-4">{d.page_count ?? ''}</td>
                <td className="pr-4 space-y-1">
                  {d.searchable_seconds != null && <p>Searchable after {d.searchable_seconds} s</p>}
                  {d.error && <p className="text-red-800">{d.error}</p>}
                  {d.excluded_pages.length > 0 && <p>Pages kept out of evidence: {d.excluded_pages.join(', ')}</p>}
                </td>
                <td className="space-x-2 whitespace-nowrap">
                  <button type="button" className={button}
                    onClick={() => act(async () => openPdf(await fetchDocumentPdf(d.id)))}>Open</button>
                  <button type="button" className={button} disabled={busy(d)}
                    onClick={() => act(() => reprocessDocument(d.id))}>Reprocess</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {docs.length === 0 && <p className="py-4 text-ink/60">No books uploaded yet.</p>}
      </div>
    </div>
  );
}
