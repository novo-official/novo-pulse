'use client';

/**
 * Data Lab - the competition-day workflow:
 *   upload -> profile -> map columns -> validate -> train
 *
 * Nothing here is specific to the synthetic dataset: the schema suggestions
 * come from the backend profiler, and the mapping is written to a data
 * contract that the training pipeline consumes.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  CheckCircle2,
  Columns3,
  Database,
  FileSpreadsheet,
  Loader2,
  Play,
  Table2,
  Upload,
  Wand2,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import { DataQualityCard } from '@/components/dashboard/data-quality';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Select, TextInput } from '@/components/ui/select';
import { EmptyState, Skeleton } from '@/components/ui/states';
import { Td, TableWrap, Th } from '@/components/ui/table';
import { InfoHint } from '@/components/ui/tooltip';
import { api } from '@/lib/api/endpoints';
import type {
  ColumnProfile,
  DatasetProfile,
  TrainingRun,
  ValidationReport,
} from '@/lib/types/api';
import { cn, formatNumber } from '@/lib/utils';

interface Mapping {
  [key: string]: unknown;
  timestamp: string;
  target: string;
  entity_id: string;
  destination: string;
  category: string;
  frequency: string;
  aggregation: string;
  calendar: string;
  future_features: string[];
  historical_features: string[];
  static_features: string[];
  primary_metric: string;
  horizons: number[];
  non_negative: boolean;
  integer: boolean;
  joins: JoinRow[];
}

/** One side table merged onto the main file before the panel is built. */
interface JoinRow {
  dataset_id: number;
  name: string;
  columns: string[];
  on: string[];
}

const EMPTY_MAPPING: Mapping = {
  timestamp: '',
  target: '',
  entity_id: '',
  destination: '',
  category: '',
  frequency: '',
  aggregation: 'sum',
  calendar: 'auto',
  future_features: [],
  historical_features: [],
  static_features: [],
  primary_metric: 'wape',
  horizons: [7, 14, 30, 60, 90],
  non_negative: true,
  integer: false,
  joins: [],
};

const KIND_FA: Record<string, string> = {
  numeric: 'عددی',
  categorical: 'دسته‌ای',
  datetime: 'تاریخ',
  datetime_like: 'تاریخ (متنی)',
  boolean: 'دودویی',
  unknown: 'نامشخص',
};

function Step({
  index,
  title,
  done,
  active,
}: {
  index: number;
  title: string;
  done: boolean;
  active: boolean;
}) {
  return (
    <div className="flex items-center gap-2.5">
      <span
        className={cn(
          'nums flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition',
          done
            ? 'bg-emerald-500 text-white'
            : active
              ? 'bg-brand-600 text-white'
              : 'bg-slate-100 text-slate-400',
        )}
      >
        {done ? <CheckCircle2 className="h-4 w-4" /> : index}
      </span>
      <span
        className={cn(
          'whitespace-nowrap text-xs font-medium',
          active || done ? 'text-ink' : 'text-muted',
        )}
      >
        {title}
      </span>
    </div>
  );
}

function MultiSelect({
  label,
  columns,
  selected,
  onChange,
  hint,
}: {
  label: string;
  columns: ColumnProfile[];
  selected: string[];
  onChange: (next: string[]) => void;
  hint: string;
}) {
  const toggle = (name: string) =>
    onChange(selected.includes(name) ? selected.filter((c) => c !== name) : [...selected, name]);

  return (
    <div>
      <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted">
        {label}
        <InfoHint>{hint}</InfoHint>
      </p>
      <div className="flex max-h-36 flex-wrap gap-1.5 overflow-y-auto rounded-xl border border-line p-2.5">
        {columns.length === 0 ? (
          <span className="text-xs text-muted">ستونی موجود نیست</span>
        ) : null}
        {columns.map((column) => (
          <button
            key={column.name}
            type="button"
            onClick={() => toggle(column.name)}
            dir="ltr"
            className={cn(
              'rounded-lg px-2 py-1 text-[11px] font-medium transition',
              selected.includes(column.name)
                ? 'bg-brand-600 text-white'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200',
            )}
          >
            {column.name}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function DataLabPage() {
  const queryClient = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);

  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [mapping, setMapping] = useState<Mapping>(EMPTY_MAPPING);
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [datasetSearch, setDatasetSearch] = useState('');

  const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: api.datasets });
  const trainingQuery = useQuery({ queryKey: ['training'], queryFn: api.training });

  // Poll while a run is in flight so the progress bar is live.
  const runQuery = useQuery({
    queryKey: ['training-run', run?.run_id],
    queryFn: () => api.trainingRun(run!.run_id),
    enabled: Boolean(run && run.status !== 'succeeded' && run.status !== 'failed'),
    refetchInterval: 2500,
  });

  useEffect(() => {
    const latest = runQuery.data?.data;
    if (latest) {
      setRun(latest);
      if (latest.status === 'succeeded') {
        queryClient.invalidateQueries();
      }
    }
  }, [runQuery.data, queryClient]);

  const applySuggestion = (next: DatasetProfile, keepJoins: JoinRow[] = []) => {
    const suggested = next.suggested_schema;
    setMapping({
      ...EMPTY_MAPPING,
      timestamp: suggested.timestamp ?? '',
      target: suggested.target ?? '',
      entity_id: suggested.entity_id ?? '',
      destination: suggested.destination ?? '',
      category: suggested.category ?? '',
      frequency: next.frequency?.frequency ?? '',
      // The profiler reports "count" when the file is a raw booking log: one
      // row per booking and no demand column to sum.
      aggregation: suggested.aggregation ?? 'sum',
      calendar: suggested.calendar ?? next.calendar?.calendar ?? 'auto',
      future_features: suggested.future_features,
      historical_features: suggested.historical_features,
      static_features: suggested.static_features,
      non_negative: suggested.non_negative,
      integer: suggested.integer,
      // Joins describe *which files* to combine, not how to read this one, so
      // re-running the column suggestion must not throw them away.
      joins: keepJoins,
    });
  };

  const upload = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append('file', file);
      return api.uploadDataset(form);
    },
    onSuccess: (response) => {
      setMessage(null);
      setValidation(null);
      setRun(null);
      if (response.data) {
        setDatasetId(response.data.dataset.id);
        setProfile(response.data.profile);
        applySuggestion(response.data.profile);
      } else {
        setMessage(response.detailFa ?? response.detail ?? 'بارگذاری ناموفق بود.');
      }
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
    },
    onError: (error: Error) => setMessage(error.message),
  });

  // The competition ships demand, accommodation/destination and booking data as
  // separate files. A side file is uploaded like any other dataset, then merged
  // onto the main one by a shared key.
  const addJoin = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append('file', file);
      return api.uploadDataset(form);
    },
    onSuccess: (response) => {
      if (!response.data) {
        setMessage(response.detailFa ?? response.detail ?? 'بارگذاری فایل جانبی ناموفق بود.');
        return;
      }
      const sideColumns = response.data.profile.columns.map((column) => column.name);
      const shared = sideColumns.filter((name) =>
        (profile?.columns ?? []).some((column) => column.name === name),
      );
      if (shared.length === 0) {
        setMessage(
          `«${response.data.dataset.name}» هیچ ستون مشترکی با فایل اصلی ندارد؛ ` +
            'برای اتصال دو فایل باید دست‌کم یک ستون کلید مشترک وجود داشته باشد.',
        );
        return;
      }
      setMessage(null);
      setValidation(null);
      setMapping((current) => ({
        ...current,
        joins: [
          ...current.joins,
          {
            dataset_id: response.data!.dataset.id,
            name: response.data!.dataset.name,
            columns: sideColumns,
            // Default to the single obvious key; the user can change it.
            on: [shared[0]],
          },
        ],
      }));
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
    },
    onError: (error: Error) => setMessage(error.message),
  });

  const loadExisting = useMutation({
    mutationFn: (id: number) => api.profileDataset({ dataset_id: id }),
    onSuccess: (response, id) => {
      setValidation(null);
      setRun(null);
      if (response.data?.profile) {
        setDatasetId(id);
        setProfile(response.data.profile);
        applySuggestion(response.data.profile);
      }
    },
  });

  const saveMapping = useMutation({
    mutationFn: () => api.mapDataset({ dataset_id: datasetId, ...mapping, save_as_active: true }),
    onSuccess: (response) =>
      setMessage(
        response.available
          ? 'نگاشت ستون‌ها ذخیره شد و به‌عنوان قرارداد فعال ثبت گردید.'
          : (response.detail ?? 'ذخیره ناموفق بود.'),
      ),
    onError: (error: Error) => setMessage(error.message),
  });

  const validate = useMutation({
    mutationFn: () => api.validateDataset({ dataset_id: datasetId!, mapping }),
    onSuccess: (response) => {
      setValidation(response.data);
      if (!response.available) setMessage(response.detail ?? 'اعتبارسنجی ناموفق بود.');
    },
    onError: (error: Error) => setMessage(error.message),
  });

  const [profileName, setProfileName] = useState('demo');
  const [horizon, setHorizon] = useState(30);

  const train = useMutation({
    // The mapping on screen is sent with the run, so what you see is what gets
    // trained - no separate "save first" step to forget.
    mutationFn: () =>
      api.startTraining({
        dataset_id: datasetId ?? undefined,
        profile: profileName,
        horizon,
        metric: mapping.primary_metric,
        mapping: { ...mapping, dataset_id: datasetId ?? undefined },
      }),
    onSuccess: (response) => {
      if (response.data) setRun(response.data);
      else setMessage(response.detail ?? 'شروع آموزش ناموفق بود.');
    },
    onError: (error: Error) => setMessage(error.message),
  });

  const allDatasets = useMemo(
    () => datasetsQuery.data?.data?.datasets ?? [],
    [datasetsQuery.data],
  );
  const visibleDatasets = useMemo(() => {
    const needle = datasetSearch.trim().toLowerCase();
    if (!needle) return allDatasets;
    return allDatasets.filter((dataset) => dataset.name.toLowerCase().includes(needle));
  }, [allDatasets, datasetSearch]);

  const columns = useMemo(() => profile?.columns ?? [], [profile]);
  const numeric = useMemo(() => columns.filter((c) => c.kind === 'numeric'), [columns]);
  const nonKey = useMemo(
    () =>
      columns.filter(
        (c) => ![mapping.timestamp, mapping.target, mapping.entity_id].includes(c.name),
      ),
    [columns, mapping.timestamp, mapping.target, mapping.entity_id],
  );

  // A raw booking log has no demand column: the target *is* the row count.
  const counting = mapping.aggregation === 'count';
  const mapped = Boolean(mapping.timestamp && (mapping.target || counting));
  const step = !profile ? 1 : !mapped ? 2 : !validation ? 3 : 4;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="section-title">آزمایشگاه داده</h2>
        <p className="mt-1 text-sm text-muted">
          دیتاست جدید را وارد کنید، ستون‌ها را نگاشت کنید و مدل را آموزش دهید — بدون تغییر در کد.
        </p>
      </div>

      <Card>
        <CardBody className="flex flex-wrap items-center gap-x-6 gap-y-3 pt-5">
          <Step index={1} title="بارگذاری دیتاست" done={step > 1} active={step === 1} />
          <span className="hidden h-px w-8 bg-line sm:block" />
          <Step index={2} title="نگاشت ستون‌ها" done={step > 2} active={step === 2} />
          <span className="hidden h-px w-8 bg-line sm:block" />
          <Step index={3} title="اعتبارسنجی داده" done={step > 3} active={step === 3} />
          <span className="hidden h-px w-8 bg-line sm:block" />
          <Step index={4} title="آموزش مدل" done={run?.status === 'succeeded'} active={step === 4} />
        </CardBody>
      </Card>

      {message ? (
        <div className="rounded-xl border border-brand-200 bg-brand-50 px-4 py-3 text-sm text-brand-800">
          {message}
        </div>
      ) : null}

      {/* --------------------------------------------------------- upload */}
      <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
        <Card>
          <CardHeader
            icon={<Upload className="h-4.5 w-4.5" />}
            title="۱. بارگذاری دیتاست"
            subtitle="CSV، Parquet یا Excel"
          />
          <CardBody className="space-y-4">
            <input
              ref={fileRef}
              type="file"
              accept=".csv,.tsv,.parquet,.pq,.xlsx,.xls,.json"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) upload.mutate(file);
              }}
            />
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={upload.isPending}
              className="flex w-full flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-line px-6 py-8 transition hover:border-brand-400 hover:bg-brand-50/40 disabled:opacity-60"
            >
              {upload.isPending ? (
                <Loader2 className="h-7 w-7 animate-spin text-brand-500" />
              ) : (
                <FileSpreadsheet className="h-7 w-7 text-slate-400" />
              )}
              <span className="text-sm font-medium text-ink">
                {upload.isPending ? 'در حال پردازش...' : 'انتخاب فایل دیتاست'}
              </span>
              <span className="text-xs text-muted">حداکثر ۲ گیگابایت</span>
            </button>

            <div className="border-t border-line pt-4">
              <div className="mb-2 flex items-center justify-between gap-2">
                <p className="text-xs font-medium text-muted">یا دیتاست موجود را انتخاب کنید</p>
                {allDatasets.length > 6 ? (
                  <TextInput
                    aria-label="جست‌وجوی دیتاست"
                    placeholder="جست‌وجو…"
                    className="h-8 w-40 text-xs"
                    value={datasetSearch}
                    onChange={(event) => setDatasetSearch(event.target.value)}
                  />
                ) : null}
              </div>
              {/* Every registered file stays reachable: uploading three
                  competition files plus a few retries must not push the one you
                  need out of the list. */}
              <div className="max-h-72 space-y-1.5 overflow-y-auto pl-1">
                {visibleDatasets.map((dataset) => (
                  <button
                    key={dataset.id}
                    type="button"
                    onClick={() => loadExisting.mutate(dataset.id)}
                    className={cn(
                      'flex w-full items-center justify-between gap-2 rounded-xl border px-3 py-2.5 text-right transition',
                      datasetId === dataset.id
                        ? 'border-brand-300 bg-brand-50'
                        : 'border-line/70 hover:bg-slate-50',
                    )}
                  >
                    <span className="min-w-0">
                      <span className="block truncate text-sm text-ink">{dataset.name}</span>
                      <span className="nums block text-[11px] text-muted">
                        {formatNumber(dataset.n_rows)} سطر · {dataset.n_columns} ستون
                      </span>
                    </span>
                    <Badge tone="neutral">{dataset.source}</Badge>
                  </button>
                ))}
                {datasetsQuery.isLoading ? <Skeleton className="h-12 w-full" /> : null}
                {!datasetsQuery.isLoading && allDatasets.length > 0 && visibleDatasets.length === 0 ? (
                  <p className="px-3 py-3 text-xs text-muted">
                    دیتاستی با این نام پیدا نشد.
                  </p>
                ) : null}
                {!datasetsQuery.isLoading && allDatasets.length === 0 ? (
                  <p className="rounded-xl border border-dashed border-line px-3 py-3 text-xs leading-6 text-muted">
                    هنوز دیتاستی ثبت نشده است. فایل خود را بارگذاری کنید، یا برای ساخت داده
                    نمونه این دستور را اجرا کنید:
                    <code
                      dir="ltr"
                      className="mt-2 block rounded-lg bg-slate-900 px-2.5 py-1.5 text-[11px] text-slate-100"
                    >
                      make seed
                    </code>
                  </p>
                ) : null}
              </div>
            </div>
          </CardBody>
        </Card>

        {/* ------------------------------------------------------ preview */}
        <Card>
          <CardHeader
            icon={<Table2 className="h-4.5 w-4.5" />}
            title="پیش‌نمایش و پروفایل داده"
            subtitle={
              profile
                ? `${formatNumber(profile.rows)} سطر · ${profile.n_columns} ستون · ${profile.memory_mb} مگابایت`
                : 'ابتدا دیتاستی را بارگذاری کنید'
            }
            action={
              profile ? (
                <div className="flex flex-wrap items-center gap-2">
                  {profile.calendar?.calendar === 'jalali' ? (
                    <Badge tone="brand">تقویم شمسی</Badge>
                  ) : null}
                  {profile.transactional?.transactional ? (
                    <Badge tone="brand">جدول خام رزرو</Badge>
                  ) : null}
                  {profile.frequency ? (
                    <Badge tone={profile.frequency.regular ? 'success' : 'warning'}>
                      فراوانی: {profile.frequency.frequency}
                    </Badge>
                  ) : null}
                </div>
              ) : null
            }
          />
          <CardBody>
            {/* Two detections change what the numbers mean, so neither may be
                applied silently: the user has to be able to see and undo them. */}
            {profile?.calendar?.calendar === 'jalali' ||
            profile?.transactional?.transactional ? (
              <ul className="mb-4 space-y-1.5 rounded-xl border border-brand-200 bg-brand-50/60 p-3 text-xs leading-6 text-brand-900">
                {profile?.calendar?.calendar === 'jalali' ? (
                  <li>
                    تاریخ‌ها شمسی تشخیص داده شد و برای مدل‌سازی به میلادی تبدیل می‌شود (
                    {profile.calendar.sample.slice(0, 2).join('، ')}). اگر اشتباه است، تقویم را در
                    بخش نگاشت ستون‌ها دستی انتخاب کنید.
                  </li>
                ) : null}
                {profile?.transactional?.transactional ? (
                  <li>
                    این فایل یک سطر به ازای هر رزرو دارد؛ تقاضا از «شمارش سطرها» ساخته می‌شود، نه از
                    جمع یک ستون ({profile.transactional.reason}).
                  </li>
                ) : null}
              </ul>
            ) : null}
            {!profile ? (
              <EmptyState
                icon={<Database className="h-6 w-6" />}
                title="هنوز دیتاستی انتخاب نشده"
                description="پس از بارگذاری، ستون‌ها، نوع داده، مقادیر گمشده و پیشنهاد نگاشت نمایش داده می‌شود."
              />
            ) : (
              <TableWrap className="max-h-[380px] overflow-y-auto">
                <thead className="sticky top-0 bg-surface">
                  <tr>
                    <Th align="right">ستون</Th>
                    <Th>نوع</Th>
                    <Th>گمشده</Th>
                    <Th>یکتا</Th>
                    <Th align="right">نمونه</Th>
                  </tr>
                </thead>
                <tbody>
                  {columns.map((column) => (
                    <tr key={column.name}>
                      <Td align="right" className="font-medium" >
                        <span dir="ltr">{column.name}</span>
                      </Td>
                      <Td>
                        <Badge tone={column.kind === 'numeric' ? 'brand' : 'neutral'}>
                          {KIND_FA[column.kind] ?? column.kind}
                        </Badge>
                      </Td>
                      <Td
                        className={cn(
                          'nums',
                          column.null_pct > 20 ? 'text-rose-600' : 'text-muted',
                        )}
                      >
                        {column.null_pct}٪
                      </Td>
                      <Td className="nums text-muted">{formatNumber(column.unique_count)}</Td>
                      <Td align="right" className="max-w-[180px] truncate text-xs text-muted">
                        <span dir="ltr">{column.sample_values.slice(0, 3).join(', ')}</span>
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </TableWrap>
            )}
          </CardBody>
        </Card>
      </div>

      {/* -------------------------------------------------------- mapping */}
      {profile ? (
        <Card>
          <CardHeader
            icon={<Columns3 className="h-4.5 w-4.5" />}
            title="۲. نگاشت ستون‌ها"
            subtitle="پیشنهاد خودکار اعمال شده است — در صورت نیاز اصلاح کنید"
            action={
              <Button
                variant="secondary"
                size="sm"
                onClick={() => applySuggestion(profile, mapping.joins)}
              >
                <Wand2 className="h-3.5 w-3.5" />
                بازگشت به پیشنهاد خودکار
              </Button>
            }
          />
          <CardBody className="space-y-5">
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <Select
                label="ستون زمان *"
                value={mapping.timestamp}
                onChange={(event) => setMapping({ ...mapping, timestamp: event.target.value })}
              >
                <option value="">— انتخاب کنید —</option>
                {columns.map((column) => (
                  <option key={column.name} value={column.name}>
                    {column.name}
                  </option>
                ))}
              </Select>
              <Select
                label={counting ? 'متغیر هدف (لازم نیست)' : 'متغیر هدف *'}
                value={mapping.target}
                onChange={(event) => setMapping({ ...mapping, target: event.target.value })}
                disabled={counting}
              >
                <option value="">
                  {counting ? '— شمارش سطرها —' : '— انتخاب کنید —'}
                </option>
                {numeric.map((column) => (
                  <option key={column.name} value={column.name}>
                    {column.name}
                  </option>
                ))}
              </Select>
              <Select
                label="شناسه موجودیت"
                value={mapping.entity_id}
                onChange={(event) => setMapping({ ...mapping, entity_id: event.target.value })}
              >
                <option value="">— بدون شناسه (سری واحد) —</option>
                {columns.map((column) => (
                  <option key={column.name} value={column.name}>
                    {column.name}
                  </option>
                ))}
              </Select>
              <Select
                label="فراوانی"
                value={mapping.frequency}
                onChange={(event) => setMapping({ ...mapping, frequency: event.target.value })}
              >
                <option value="">تشخیص خودکار</option>
                <option value="D">روزانه (D)</option>
                <option value="W">هفتگی (W)</option>
                <option value="M">ماهانه (M)</option>
                <option value="H">ساعتی (H)</option>
              </Select>
              <Select
                label="مقصد (سلسله‌مراتب)"
                value={mapping.destination}
                onChange={(event) => setMapping({ ...mapping, destination: event.target.value })}
              >
                <option value="">— ندارد —</option>
                {columns.map((column) => (
                  <option key={column.name} value={column.name}>
                    {column.name}
                  </option>
                ))}
              </Select>
              <Select
                label="دسته‌بندی (سلسله‌مراتب)"
                value={mapping.category}
                onChange={(event) => setMapping({ ...mapping, category: event.target.value })}
              >
                <option value="">— ندارد —</option>
                {columns.map((column) => (
                  <option key={column.name} value={column.name}>
                    {column.name}
                  </option>
                ))}
              </Select>
              <Select
                label="معیار رسمی مسابقه"
                value={mapping.primary_metric}
                onChange={(event) => setMapping({ ...mapping, primary_metric: event.target.value })}
              >
                {(datasetsQuery.data?.data?.metrics ?? ['wape']).map((metric) => (
                  <option key={metric} value={metric}>
                    {metric.toUpperCase()}
                  </option>
                ))}
              </Select>
              <Select
                label="تجمیع سطرهای تکراری"
                value={mapping.aggregation}
                onChange={(event) =>
                  setMapping({
                    ...mapping,
                    aggregation: event.target.value,
                    // Counting rows *is* the target; any column chosen before
                    // would silently be measuring something else.
                    target: event.target.value === 'count' ? '' : mapping.target,
                  })
                }
              >
                <option value="count">شمارش سطرها (count) — جدول خام رزرو</option>
                <option value="sum">جمع (sum)</option>
                <option value="mean">میانگین (mean)</option>
                <option value="max">بیشینه (max)</option>
                <option value="min">کمینه (min)</option>
                <option value="first">اولین (first)</option>
              </Select>
              <Select
                label="تقویم ستون تاریخ"
                value={mapping.calendar}
                onChange={(event) => setMapping({ ...mapping, calendar: event.target.value })}
              >
                <option value="auto">تشخیص خودکار</option>
                <option value="jalali">شمسی (جلالی)</option>
                <option value="gregorian">میلادی</option>
              </Select>
            </div>

            <JoinEditor
              joins={mapping.joins}
              mainColumns={columns.map((column) => column.name)}
              pending={addJoin.isPending}
              onAdd={(file) => addJoin.mutate(file)}
              onChangeKey={(index, key) =>
                setMapping({
                  ...mapping,
                  joins: mapping.joins.map((join, position) =>
                    position === index ? { ...join, on: [key] } : join,
                  ),
                })
              }
              onRemove={(index) =>
                setMapping({
                  ...mapping,
                  joins: mapping.joins.filter((_, position) => position !== index),
                })
              }
            />

            <div className="grid gap-4 lg:grid-cols-3">
              <MultiSelect
                label="متغیرهای آینده‌معلوم"
                columns={nonKey}
                selected={mapping.future_features}
                onChange={(next) => setMapping({ ...mapping, future_features: next })}
                hint="مقادیری که در زمان پیش‌بینی برای روزهای آینده هم می‌دانیم: قیمت برنامه‌ریزی‌شده، تعطیلات، رویدادها."
              />
              <MultiSelect
                label="متغیرهای تاریخی"
                columns={nonKey}
                selected={mapping.historical_features}
                onChange={(next) => setMapping({ ...mapping, historical_features: next })}
                hint="فقط برای گذشته در دسترس‌اند؛ مدل تنها از وقفه (lag) آن‌ها استفاده می‌کند: جست‌وجو، بازدید، کلیک."
              />
              <MultiSelect
                label="ویژگی‌های ثابت"
                columns={nonKey}
                selected={mapping.static_features}
                onChange={(next) => setMapping({ ...mapping, static_features: next })}
                hint="ثابت به ازای هر موجودیت: ظرفیت، دسته‌بندی، امتیاز."
              />
            </div>

            <div className="flex flex-wrap gap-3 border-t border-line pt-4">
              <Button onClick={() => saveMapping.mutate()} disabled={!mapped || saveMapping.isPending}>
                {saveMapping.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                ذخیره نگاشت
              </Button>
              <Button
                variant="secondary"
                onClick={() => validate.mutate()}
                disabled={!mapped || validate.isPending}
              >
                {validate.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                اعتبارسنجی داده
              </Button>
            </div>
          </CardBody>
        </Card>
      ) : null}

      {/* ----------------------------------------------------- validation */}
      {validation ? <DataQualityCard report={validation} /> : null}

      {/* -------------------------------------------------------- training */}
      {profile ? (
        <Card>
          <CardHeader
            icon={<Play className="h-4.5 w-4.5" />}
            title="۴. آموزش مدل"
            subtitle="آموزش به‌صورت پس‌زمینه اجرا می‌شود؛ نیازی به Redis یا Celery نیست"
          />
          <CardBody className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-3">
              <Select
                label="پروفایل آموزش"
                value={profileName}
                onChange={(event) => setProfileName(event.target.value)}
              >
                {Object.entries(trainingQuery.data?.data?.profiles ?? { demo: { description: '' } }).map(
                  ([name, info]) => (
                    <option key={name} value={name}>
                      {name} — {(info as { description: string }).description}
                    </option>
                  ),
                )}
              </Select>
              <TextInput
                label="افق پیش‌بینی (روز)"
                type="number"
                min={1}
                max={365}
                value={horizon}
                onChange={(event) => setHorizon(Number(event.target.value))}
              />
              <div className="flex items-end">
                <Button
                  onClick={() => train.mutate()}
                  disabled={!mapped || train.isPending || run?.status === 'running'}
                  className="w-full"
                >
                  {train.isPending || run?.status === 'running' ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                  شروع آموزش
                </Button>
              </div>
            </div>

            {run ? (
              <div className="rounded-xl border border-line/70 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="nums text-sm font-medium text-ink" dir="ltr">
                    {run.run_id}
                  </span>
                  <Badge
                    tone={
                      run.status === 'succeeded'
                        ? 'success'
                        : run.status === 'failed'
                          ? 'danger'
                          : 'info'
                    }
                  >
                    {run.status === 'succeeded'
                      ? 'موفق'
                      : run.status === 'failed'
                        ? 'ناموفق'
                        : run.stage || 'در حال اجرا'}
                  </Badge>
                </div>
                <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={cn(
                      'h-full rounded-full transition-all duration-500',
                      run.status === 'failed' ? 'bg-rose-500' : 'bg-brand-600',
                    )}
                    style={{ width: `${Math.round((run.progress ?? 0) * 100)}%` }}
                  />
                </div>
                {run.status === 'succeeded' ? (
                  <p className="nums mt-3 text-sm text-ink">
                    مدل منتخب: <span dir="ltr">{run.champion_model}</span> ·{' '}
                    {run.primary_metric.toUpperCase()}={run.champion_score?.toFixed(4)}
                    {run.improvement !== null
                      ? ` · بهبود ${(run.improvement * 100).toFixed(1)}٪ نسبت به مدل پایه`
                      : ''}
                  </p>
                ) : null}
                {run.error ? (
                  <pre
                    dir="ltr"
                    className="mt-3 max-h-32 overflow-auto rounded-lg bg-rose-50 p-3 text-[11px] text-rose-800"
                  >
                    {run.error}
                  </pre>
                ) : null}
                {run.warnings?.length ? (
                  <div className="mt-3 space-y-1">
                    {run.warnings.map((warning) => (
                      <p
                        key={warning}
                        dir="ltr"
                        className="text-left text-xs text-amber-700"
                      >
                        • {warning}
                      </p>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}
          </CardBody>
        </Card>
      ) : null}
    </div>
  );
}

/**
 * Side tables. The hackathon supplies demand, accommodation/destination and
 * booking data as separate files; this merges them on a shared key before the
 * panel is built, so nothing has to be pre-joined outside the product.
 */
function JoinEditor({
  joins,
  mainColumns,
  pending,
  onAdd,
  onChangeKey,
  onRemove,
}: {
  joins: JoinRow[];
  mainColumns: string[];
  pending: boolean;
  onAdd: (file: File) => void;
  onChangeKey: (index: number, key: string) => void;
  onRemove: (index: number) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="rounded-xl border border-line bg-slate-50/60 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-ink">
            فایل‌های جانبی
            <InfoHint>
              اگر دیتاست مسابقه چند فایل جداست — رزروها، اطلاعات اقامتگاه و مقصد، تقویم و
              تعطیلات — همه را اینجا اضافه کنید تا روی یک ستون کلید مشترک به فایل اصلی متصل
              شوند.
            </InfoHint>
          </p>
          <p className="mt-0.5 text-xs text-muted">
            {joins.length === 0
              ? 'فایل اصلی به‌تنهایی کافی است؛ در صورت وجود فایل‌های جداگانه آن‌ها را اضافه کنید.'
              : `${joins.length} فایل جانبی به فایل اصلی متصل می‌شود.`}
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept=".csv,.tsv,.parquet,.pq,.xlsx,.xls,.json"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onAdd(file);
            event.target.value = '';
          }}
        />
        <Button
          variant="secondary"
          size="sm"
          onClick={() => inputRef.current?.click()}
          disabled={pending}
        >
          {pending ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Upload className="h-3.5 w-3.5" />
          )}
          افزودن فایل جانبی
        </Button>
      </div>

      {joins.length > 0 ? (
        <div className="mt-3 space-y-2">
          {joins.map((join, index) => {
            const shared = join.columns.filter((name) => mainColumns.includes(name));
            return (
              <div
                key={`${join.dataset_id}-${index}`}
                className="flex flex-wrap items-end gap-3 rounded-lg border border-line bg-surface p-3"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-medium text-ink">{join.name}</p>
                  <p className="nums text-[11px] text-muted">{join.columns.length} ستون</p>
                </div>
                <Select
                  label="ستون کلید"
                  className="h-9 w-52"
                  value={join.on[0] ?? ''}
                  onChange={(event) => onChangeKey(index, event.target.value)}
                >
                  {shared.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </Select>
                <Button variant="ghost" size="sm" onClick={() => onRemove(index)}>
                  حذف
                </Button>
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
