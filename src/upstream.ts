import type {
  ChatClientLike,
  ChatCompletion,
  ChatCompletionChunk,
} from "./types.js";
import { ToolCapabilityError, UpstreamError } from "./errors.js";

export interface FetchClientOptions {
  /** Base URL of the OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1`. */
  baseURL: string;
  /** Bearer token sent as `Authorization: Bearer <apiKey>` (optional). */
  apiKey?: string;
  /** Extra headers merged into every request. */
  headers?: Record<string, string>;
  /** Custom fetch implementation (defaults to global `fetch`). */
  fetch?: typeof fetch;
}

/** Normalizes SSE text (CRLF → LF) and splits an event boundary. */
async function* parseSSE(
  response: Response,
): AsyncIterable<ChatCompletionChunk> {
  const body = response.body;
  if (!body) return;
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let sep: number;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const event = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        for (const line of event.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") return;
          if (!data) continue;
          try {
            yield JSON.parse(data) as ChatCompletionChunk;
          } catch {
            // ignore malformed keep-alive / comment lines
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * A minimal OpenAI-compatible client backed by `fetch` — no `openai` SDK
 * required. Talks to `${baseURL}/chat/completions`, returning a parsed
 * `ChatCompletion` or, when `stream: true`, an async iterable of
 * `ChatCompletionChunk`s decoded from the SSE response. Suitable as the
 * `client` argument to `wrapToolSupport` / `createToolRunner`.
 */
export function createFetchClient(options: FetchClientOptions): ChatClientLike {
  const doFetch = options.fetch ?? globalThis.fetch;
  if (typeof doFetch !== "function") {
    throw new ToolCapabilityError(
      "No fetch implementation available; pass options.fetch or use Node >= 18.",
    );
  }
  const base = options.baseURL.replace(/\/+$/, "");

  return {
    chat: {
      completions: {
        async create(body: any, requestOptions?: any) {
          const headers: Record<string, string> = {
            "content-type": "application/json",
            accept: body?.stream ? "text/event-stream" : "application/json",
            ...options.headers,
            // Explicit apiKey wins over any authorization in `headers`.
            ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}),
          };
          const response = await doFetch(`${base}/chat/completions`, {
            method: "POST",
            headers,
            body: JSON.stringify(body),
            signal: requestOptions?.signal,
          });
          if (!response.ok) {
            const text = await response.text().catch(() => "");
            throw new UpstreamError(response.status, response.statusText, text);
          }
          if (body?.stream) {
            return parseSSE(response);
          }
          return (await response.json()) as ChatCompletion;
        },
      },
    },
  };
}
