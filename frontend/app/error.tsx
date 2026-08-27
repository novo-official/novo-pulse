'use client';

import { useEffect } from 'react';

import { ErrorState } from '@/components/ui/states';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="rounded-2xl border border-line bg-surface shadow-card">
      <ErrorState message={error.message} onRetry={reset} />
    </div>
  );
}
