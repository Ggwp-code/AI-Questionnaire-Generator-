// A plain link cannot send X-Dev-User, so the PDF is fetched as a Blob and opened from an object URL.
export function openPdf(blob: Blob, page = 1): void {
  const url = URL.createObjectURL(blob);
  window.open(`${url}#page=${page}`, '_blank');
  // Revoke late: the new tab may still be loading the URL.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
