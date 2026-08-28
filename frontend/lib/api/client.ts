/**
 * Thin fetch wrapper around the Django API.
 *
 * Unwraps the availability envelope so callers deal with `T | null` plus an
 * explicit `available` flag, and turns HTTP failures into typed errors the
 * error boundaries can render.
 */
import type { ApiEnvelope } from '@/lib/types/api';

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '') ?? 'http://localhost:8000/api/v1';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export interface ApiResult<T> {
  available: boolean;
  demoMode: boolean;
  data: T | null;
  detail?: string;
  detailFa?: string;
  extra: Record<string, unknown>;
}

type Query = Record<string, string | number | boolean | null | undefined>;

function buildUrl(path: string, query?: Query): string {
  const url = new URL(`${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== null && value !== undefined && value !== '') {
        url.searchParams.set(key, String(value));
      }
    }
  }
  return url.toString();
}

async function request<T>(path: string, init?: RequestInit, query?: Query): Promise<T> {
  let response: Response;
  try {
    response = await fetch(buildUrl(path, query), {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...init?.headers,
      },
      cache: 'no-store',
    });
  } catch {
    throw new ApiError('ارتباط با سرور برقرار نشد. آیا بک‌اند در حال اجراست؟', 0);
  }

  const text = await response.text();
  let payload: unknown = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = null;
  }

  if (!response.ok) {
    // Prefer the Persian message; the English one is for logs and API users.
    const body = payload as { detail?: string; detail_fa?: string } | null;
    const detail =
      body?.detail_fa ?? body?.detail ?? `درخواست با خطای ${response.status} مواجه شد`;
    throw new ApiError(detail, response.status);
  }
  return payload as T;
}

/** GET an envelope-wrapped endpoint. */
export async function getEnvelope<T>(path: string, query?: Query): Promise<ApiResult<T>> {
  const payload = await request<ApiEnvelope<T>>(path, { method: 'GET' }, query);
  const { available, demo_mode, data, detail, detail_fa, ...extra } = payload;
  return {
    available: Boolean(available),
    demoMode: Boolean(demo_mode),
    data: (data ?? null) as T | null,
    detail,
    detailFa: detail_fa,
    extra: extra as Record<string, unknown>,
  };
}

/** GET a bare (non-enveloped) endpoint such as /health/. */
export function getRaw<T>(path: string, query?: Query): Promise<T> {
  return request<T>(path, { method: 'GET' }, query);
}

export async function postEnvelope<T>(path: string, body: unknown): Promise<ApiResult<T>> {
  const payload = await request<ApiEnvelope<T>>(path, {
    method: 'POST',
    body: JSON.stringify(body),
  });
  const { available, demo_mode, data, detail, detail_fa, ...extra } = payload;
  return {
    available: Boolean(available),
    demoMode: Boolean(demo_mode),
    data: (data ?? null) as T | null,
    detail,
    detailFa: detail_fa,
    extra: extra as Record<string, unknown>,
  };
}

export async function postForm<T>(path: string, form: FormData): Promise<ApiResult<T>> {
  const payload = await request<ApiEnvelope<T>>(path, { method: 'POST', body: form });
  const { available, demo_mode, data, detail, detail_fa, ...extra } = payload;
  return {
    available: Boolean(available),
    demoMode: Boolean(demo_mode),
    data: (data ?? null) as T | null,
    detail,
    detailFa: detail_fa,
    extra: extra as Record<string, unknown>,
  };
}
