'use client';

import { useQuery } from '@tanstack/react-query';
import { Award, Cpu, Info, Layers, Trophy } from 'lucide-react';

import { CensoringCard, TuningCard } from '@/components/dashboard/censoring-card';
import { Badge } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { PageHeader } from '@/components/ui/page-header';
import { AsyncBoundary, CardSkeleton, NoModelState } from '@/components/ui/states';
import { Td, TableWrap, Th } from '@/components/ui/table';
import { InfoHint } from '@/components/ui/tooltip';
import { api } from '@/lib/api/endpoints';
import { MODEL_KIND_FA, cn, formatDateTime, formatMetric, formatNumber } from '@/lib/utils';

export default function ModelsPage() {
  const leaderboardQuery = useQuery({ queryKey: ['leaderboard'], queryFn: api.leaderboard });
  const modelsQuery = useQuery({ queryKey: ['models'], queryFn: api.models });

  const board = leaderboardQuery.data?.data;
  const registry = modelsQuery.data?.data;
  const metric = board?.primary_metric ?? 'wape';
  const champion = board?.leaderboard.find((row) => row.is_champion);
  const bestBaseline = board?.leaderboard.find((row) => row.is_baseline);

  if (leaderboardQuery.isSuccess && !leaderboardQuery.data?.available) {
    return (
      <Card>
        <NoModelState detail={leaderboardQuery.data?.detailFa} />
      </Card>
    );
  }

  return (
    <main className="page-stack models-page">
      <PageHeader className="models-header" eyebrow="موتور هوشمند" title="مدل‌ها و عملکرد" description="مدل منتخب، رتبه‌بندی و جزئیات فنی را شفاف ببینید؛ انتخاب برنده فقط بر پایه اعتبارسنجی زمانی انجام می‌شود." />

      {/* ------------------------------------------------ champion banner */}
      <AsyncBoundary
        isLoading={leaderboardQuery.isLoading}
        error={leaderboardQuery.error}
        onRetry={() => leaderboardQuery.refetch()}
        skeleton={
          <Card>
            <CardSkeleton lines={3} />
          </Card>
        }
      >
        {champion ? (
          <Card className="models-champion">
            <CardBody className="models-champion__body flex flex-wrap items-center justify-between gap-6">
              <div className="flex items-center gap-4">
                <span className="models-champion__icon flex h-14 w-14 items-center justify-center text-white">
                  <Trophy className="h-6 w-6" />
                </span>
                <div>
                  <p className="models-champion__eyebrow">مدل منتخب</p>
                  <p className="models-champion__name" dir="ltr">{champion.model}</p>
                  <p className="mt-0.5 text-xs text-muted">
                    {MODEL_KIND_FA[champion.is_baseline ? 'baseline' : 'ml']} · آموزش‌دیده روی کل
                    تاریخچه
                  </p>
                </div>
              </div>
              <div className="models-champion__stats flex flex-wrap">
                <div className="models-champion__stat">
                  <p className="text-xs text-muted">{metric.toUpperCase()}</p>
                  <p className="nums mt-1 text-2xl font-semibold text-ink">
                    {formatMetric(champion.primary_value, metric)}
                  </p>
                </div>
                {bestBaseline ? (
                  <div className="models-champion__stat">
                    <p className="text-xs text-muted">بهترین مدل پایه</p>
                    <p className="nums mt-1 text-2xl font-semibold text-muted">
                      {formatMetric(bestBaseline.primary_value, metric)}
                    </p>
                  </div>
                ) : null}
                {champion.improvement_vs_baseline !== null &&
                champion.improvement_vs_baseline !== undefined ? (
                  <div className="models-champion__stat models-champion__stat--positive">
                    <p className="text-xs text-muted">بهبود</p>
                    <p className="nums mt-1 text-2xl font-semibold text-emerald-600">
                      {(champion.improvement_vs_baseline * 100).toFixed(1)}٪
                    </p>
                  </div>
                ) : null}
              </div>
            </CardBody>
          </Card>
        ) : null}
      </AsyncBoundary>

      {/* ---------------------------------------------------- leaderboard */}
      <Card className="models-card models-leaderboard">
        <CardHeader
          icon={<Award className="h-4.5 w-4.5" />}
          title="جدول رتبه‌بندی مدل‌ها"
          subtitle="بر اساس معیار رسمی، به تفکیک افق پیش‌بینی"
          action={
            <InfoHint>
              همه اعداد از اعتبارسنجی متحرک زمانی می‌آیند. ستون «وزن» سهم مدل در ترکیب نهایی است که
              از همین نتایج محاسبه می‌شود، نه از مقادیر ثابت.
            </InfoHint>
          }
        />
        <AsyncBoundary
          isLoading={leaderboardQuery.isLoading}
          error={leaderboardQuery.error}
          onRetry={() => leaderboardQuery.refetch()}
          skeleton={<CardSkeleton lines={8} />}
        >
          <CardBody className="models-leaderboard__body">
            <TableWrap className="models-table" minWidth={1080}>
              <colgroup>
                <col className="models-table__col-rank" />
                <col className="models-table__col-model" />
                <col className="models-table__col-kind" />
                <col className="models-table__col-metric" />
                <col className="models-table__col-value" />
                <col className="models-table__col-value" />
                {(board?.horizon_buckets ?? []).map((bucket) => (
                  <col key={bucket} className="models-table__col-horizon" />
                ))}
                <col className="models-table__col-weight" />
                <col className="models-table__col-improvement" />
              </colgroup>
              <thead>
                <tr>
                  <Th align="center">رتبه</Th>
                  <Th align="right">مدل</Th>
                  <Th align="center">نوع</Th>
                  <Th align="center">{metric.toUpperCase()}</Th>
                  <Th align="center">MAE</Th>
                  <Th align="center">RMSE</Th>
                  {(board?.horizon_buckets ?? []).map((bucket) => (
                    <Th key={bucket} align="center" title={`افق ${bucket} روز`}>
                      {bucket}
                    </Th>
                  ))}
                  <Th align="center">وزن ترکیب</Th>
                  <Th align="center">بهبود</Th>
                </tr>
              </thead>
              <tbody>
                {(board?.leaderboard ?? []).map((row) => (
                  <tr
                    key={row.model}
                    className={cn('models-table__row', row.is_champion && 'models-table__row--champion')}
                  >
                    <Td align="center" className="nums models-table__rank-cell">
                      <span className="models-table__rank">{row.rank}</span>
                    </Td>
                    <Td align="right">
                      <span className="models-table__model flex items-center gap-2">
                        {row.is_champion ? <Trophy className="h-3.5 w-3.5 text-brand-600" /> : null}
                        <span dir="ltr">{row.model}</span>
                      </span>
                    </Td>
                    <Td align="center">
                      <Badge tone={row.is_baseline ? 'neutral' : 'brand'}>
                        {row.is_baseline ? 'پایه' : row.model === 'ensemble' ? 'ترکیبی' : 'یادگیری'}
                      </Badge>
                    </Td>
                    <Td align="center" className="nums">
                      <span className="models-table__primary">
                        {formatMetric(row.primary_value, metric)}
                      </span>
                    </Td>
                    <Td align="center" className="nums text-muted">{formatNumber(row.metrics.mae, 2)}</Td>
                    <Td align="center" className="nums text-muted">{formatNumber(row.metrics.rmse, 2)}</Td>
                    {(board?.horizon_buckets ?? []).map((bucket) => {
                      const entry = row.by_horizon.find((item) => item.bucket === bucket);
                      return (
                        <Td key={bucket} align="center" className="nums text-muted">
                          {entry ? formatMetric(Number(entry[metric]), metric) : '—'}
                        </Td>
                      );
                    })}
                    <Td align="center" className="nums text-muted">
                      {row.ensemble_weight !== null && row.ensemble_weight !== undefined
                        ? `${(row.ensemble_weight * 100).toFixed(0)}٪`
                        : '—'}
                    </Td>
                    <Td
                      align="center"
                      className={cn(
                        'nums models-table__improvement',
                        (row.improvement_vs_baseline ?? 0) > 0 && 'models-table__improvement--positive',
                      )}
                    >
                      {row.improvement_vs_baseline !== null &&
                      row.improvement_vs_baseline !== undefined
                        ? `${(row.improvement_vs_baseline * 100).toFixed(1)}٪`
                        : '—'}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </TableWrap>
          </CardBody>
        </AsyncBoundary>
      </Card>

      {registry?.trained?.tuning?.length ? <TuningCard className="models-card models-tuning" runs={registry.trained.tuning} /> : null}
      {registry?.trained?.censoring?.enabled ? (
        <CensoringCard className="models-card models-censoring" report={registry.trained.censoring} />
      ) : null}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* ------------------------------------------- technical metadata */}
        <Card className="models-card models-meta-card">
          <CardHeader
            icon={<Info className="h-4.5 w-4.5" />}
            title="شفافیت فنی"
            subtitle="مدل جعبه‌سیاه نیست — همه‌چیز قابل بازبینی است"
          />
          <AsyncBoundary
            isLoading={modelsQuery.isLoading}
            error={modelsQuery.error}
            onRetry={() => modelsQuery.refetch()}
            skeleton={<CardSkeleton lines={7} />}
          >
            <CardBody>
              {registry?.trained ? (
                <dl className="models-metadata text-sm">
                  {[
                    ['شناسه اجرا', registry.trained.run_id],
                    ['مدل پیش‌بینی', registry.trained.champion],
                    ['دیتاست', registry.trained.dataset],
                    ['متغیر هدف', registry.trained.target],
                    ['معیار ارزیابی', registry.trained.primary_metric?.toUpperCase()],
                    ['افق پیش‌بینی', `${registry.trained.horizon} روز`],
                    ['فراوانی داده', registry.trained.frequency],
                    ['تعداد ویژگی‌ها', formatNumber(registry.trained.n_features)],
                    [
                      'پنجره آموزش',
                      `${registry.trained.training_window?.start} تا ${registry.trained.training_window?.end}`,
                    ],
                    ['روش عدم‌قطعیت', registry.trained.uncertainty_method],
                    ['پروفایل آموزش', registry.trained.profile],
                    ['آخرین آموزش', formatDateTime(registry.trained.last_trained)],
                  ].map(([label, value]) => (
                    <div key={label} className="models-metadata__row flex justify-between gap-4">
                      <dt className="shrink-0 text-muted">{label}</dt>
                      <dd className="truncate text-left text-ink" dir="auto">
                        {value ?? '—'}
                      </dd>
                    </div>
                  ))}
                </dl>
              ) : null}
              {registry?.trained?.warnings?.length ? (
                <div className="mt-4 space-y-1.5">
                  {registry.trained.warnings.map((warning) => (
                    <p
                      key={warning}
                      dir="ltr"
                      className="rounded-lg bg-amber-50 px-3 py-2 text-left text-xs leading-5 text-amber-800"
                    >
                      {warning}
                    </p>
                  ))}
                </div>
              ) : null}
            </CardBody>
          </AsyncBoundary>
        </Card>

        {/* ---------------------------------------------- model registry */}
        <Card className="models-card models-registry-card">
          <CardHeader
            icon={<Layers className="h-4.5 w-4.5" />}
            title="مدل‌های در دسترس"
            subtitle="مدل‌های اختیاری در صورت نبود وابستگی، بی‌صدا کنار گذاشته می‌شوند"
          />
          <AsyncBoundary
            isLoading={modelsQuery.isLoading}
            error={modelsQuery.error}
            skeleton={<CardSkeleton lines={8} />}
          >
            <CardBody className="models-registry space-y-2">
              {(registry?.registry ?? []).map((model) => (
                <div
                  key={model.name}
                  className="models-registry__item flex flex-wrap items-center justify-between gap-2"
                >
                  <span className="flex items-center gap-2.5">
                    <Cpu
                      className={cn(
                        'h-4 w-4',
                        model.available ? 'text-emerald-500' : 'text-slate-300',
                      )}
                    />
                    <span>
                      <span className="block text-sm text-ink" dir="ltr">
                        {model.label}
                      </span>
                      <span className="block text-[11px] text-muted">
                        {MODEL_KIND_FA[model.kind] ?? model.kind}
                      </span>
                    </span>
                  </span>
                  <Badge
                    tone={model.available ? 'success' : model.optional ? 'neutral' : 'warning'}
                    title={model.status}
                  >
                    {model.available ? 'فعال' : 'غیرفعال'}
                  </Badge>
                </div>
              ))}
            </CardBody>
          </AsyncBoundary>
        </Card>
      </div>
    </main>
  );
}
