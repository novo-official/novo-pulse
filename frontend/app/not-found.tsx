import Link from 'next/link';

import { Card } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';

export default function NotFound() {
  return (
    <Card>
      <EmptyState
        title="صفحه پیدا نشد"
        description="آدرس واردشده در این برنامه وجود ندارد."
        action={
          <Link
            href="/dashboard"
            className="rounded-xl bg-brand-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-brand-700"
          >
            بازگشت به داشبورد
          </Link>
        }
      />
    </Card>
  );
}
