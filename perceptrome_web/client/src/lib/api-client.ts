export type ApiErrorDetail = string | Record<string, unknown> | unknown[] | null;

export type FrontendApiError = Error & {
  status: number;
  code?: string;
  retryAfterSeconds?: number;
  detail?: ApiErrorDetail;
};

type RequestOptions = Omit<RequestInit, "body"> & {
  body?: BodyInit | Record<string, unknown> | null;
};

type ParsedErrorPayload = {
  message: string;
  code?: string;
  retryAfterSeconds?: number;
  detail?: ApiErrorDetail;
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseRetryAfterSeconds(response: Response, body: Record<string, unknown> | null): number | undefined {
  const fromBody = body?.retry_after_seconds;
  if (typeof fromBody === "number" && Number.isFinite(fromBody)) return fromBody;
  const header = response.headers.get("Retry-After");
  if (!header) return undefined;
  const parsed = Number(header);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function normalizeMessage(detail: ApiErrorDetail, fallbackMessage: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (isPlainObject(item)) {
          const itemMessage = item.msg ?? item.message ?? item.detail;
          return typeof itemMessage === "string" ? itemMessage : JSON.stringify(item);
        }
        return String(item);
      })
      .join("; ");
  }
  if (isPlainObject(detail)) {
    const objectMessage = detail.message ?? detail.msg ?? detail.reason ?? detail.error;
    if (typeof objectMessage === "string" && objectMessage.trim()) return objectMessage;
    return fallbackMessage;
  }
  return fallbackMessage;
}

async function parseErrorPayload(response: Response): Promise<ParsedErrorPayload> {
  const fallbackMessage = `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`;
  const contentType = response.headers.get("content-type") ?? "";
  const rawText = await response.text();

  if (!rawText) {
    return {
      message: fallbackMessage,
      retryAfterSeconds: parseRetryAfterSeconds(response, null),
    };
  }

  if (!contentType.includes("application/json")) {
    return {
      message: rawText.trim() || fallbackMessage,
      retryAfterSeconds: parseRetryAfterSeconds(response, null),
      detail: rawText,
    };
  }

  try {
    const parsed = JSON.parse(rawText) as unknown;
    if (!isPlainObject(parsed)) {
      return {
        message: fallbackMessage,
        retryAfterSeconds: parseRetryAfterSeconds(response, null),
        detail: parsed as ApiErrorDetail,
      };
    }

    const detail = (parsed.detail ?? parsed.message ?? null) as ApiErrorDetail;
    const message = normalizeMessage(detail, typeof parsed.message === "string" ? parsed.message : fallbackMessage);
    const code = typeof parsed.code === "string" ? parsed.code : isPlainObject(detail) && typeof detail.code === "string" ? detail.code : undefined;

    return {
      message,
      code,
      retryAfterSeconds: parseRetryAfterSeconds(response, parsed),
      detail,
    };
  } catch {
    return {
      message: rawText.trim() || fallbackMessage,
      retryAfterSeconds: parseRetryAfterSeconds(response, null),
      detail: rawText,
    };
  }
}

export function isFrontendApiError(error: unknown): error is FrontendApiError {
  return error instanceof Error && typeof (error as FrontendApiError).status === "number";
}

export async function parseApiResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await parseErrorPayload(response);
    const error = new Error(payload.message) as FrontendApiError;
    error.status = response.status;
    error.code = payload.code;
    error.retryAfterSeconds = payload.retryAfterSeconds;
    error.detail = payload.detail;
    throw error;
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const rawText = await response.text();
  if (!rawText) return {} as T;

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return JSON.parse(rawText) as T;
  }

  return rawText as T;
}

export async function apiRequest<T>(path: string, init: RequestOptions = {}): Promise<T> {
  const { body, headers, ...rest } = init;
  const isObjectBody = body != null && !(body instanceof FormData) && !(body instanceof URLSearchParams) && typeof body !== "string" && !(body instanceof Blob) && !(body instanceof ArrayBuffer);
  const response = await fetch(path, {
    credentials: "include",
    ...rest,
    headers: {
      ...(isObjectBody ? { "Content-Type": "application/json" } : {}),
      ...(headers ?? {}),
    },
    body: isObjectBody ? JSON.stringify(body) : (body as BodyInit | null | undefined),
  });
  return parseApiResponse<T>(response);
}
