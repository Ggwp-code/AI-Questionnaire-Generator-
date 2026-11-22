import type { ApiError } from '../../services/tribunalApi';

export default function ErrorNotice({ error }: { error: ApiError | Error | null }) {
  if (!error) return null;
  const details = 'details' in error ? (error as ApiError).details : [];
  return (
    <div role="alert" className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
      <p className="font-medium">{error.message}</p>
      {details.length > 0 && (
        <ul className="mt-1 list-disc pl-5">
          {details.map((d, i) => <li key={i}>{d}</li>)}
        </ul>
      )}
    </div>
  );
}
