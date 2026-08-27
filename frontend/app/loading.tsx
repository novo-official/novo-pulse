import { Card } from '@/components/ui/card';
import { CardSkeleton } from '@/components/ui/states';

export default function Loading() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {Array.from({ length: 6 }).map((_, index) => (
        <Card key={index}>
          <CardSkeleton lines={3} />
        </Card>
      ))}
    </div>
  );
}
