import http from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { createHash, timingSafeEqual } from "node:crypto";
import type { ChatCompletion, ChatCompletionChunk, ToolCapableClient } from "./types.js";
import { wrapToolSupport, type WrapOptions } from "./client.js";
import { createFetchClient } from "./upstream.js";
import { UpstreamError } from "./errors.js";

const DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024;

export interface ProxyOptions extends WrapOptions {
  /** Upstream OpenAI-compatible base URL, e.g. `http://localhost:11434/v1`. */
  upstreamBaseURL: string;
  /** Bearer token for the upstream endpoint. */
  upstreamApiKey?: string;
  /** Extra headers forwarded to the upstream endpoint. */
  upstreamHeaders?: Record<string, string>;
  /** Route prefix the proxy serves on. Default: `/v1`. */
  basePath?: string;
  /**
   * If set, clients must send `Authorization: Bearer <apiKey>`. Leave unset to
   * accept unauthenticated requests (fine for localhost).
   */
  apiKey?: string;
  /** Send permissive (wildcard) CORS headers. Default: `false` (secure-by-default). */
  cors?: boolean;
  /** Max accepted request body size in bytes. Default: 10 MiB. */
  maxBodySize?: number;
  /** Custom fetch implementation. */
  fetch?: typeof fetch;
}

function readRawBody(req: IncomingMessage, limitBytes: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let done = false;
    req.on("data", (c: Buffer) => {
      if (done) return; // keep draining (no buffering) so the socket closes cleanly
      size += c.length;
      if (size > limitBytes) {
        done = true;
        chunks.length = 0;
        reject(new Error("Request body too large"));
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => {
      if (done) return;
      done = true;
      resolve(Buffer.concat(chunks));
    });
    req.on("error", (err) => {
      if (!done) {
        done = true;
        reject(err);
      }
    });
  });
}

async function readJsonBody(req: IncomingMessage, limitBytes: number): Promise<any> {
  const raw = (await readRawBody(req, limitBytes)).toString("utf8");
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error("Invalid JSON body");
  }
}

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  res.writeHead(status, { "content-type": "application/json" });
  res.end(body);
}

/** True for auth-class statuses whose error bodies may echo our credentials. */
function isAuthStatus(status: number): boolean {
  return status === 401 || status === 403;
}

/** Strips credential-shaped tokens from an upstream error body before relaying. */
function sanitizeUpstreamBody(body: string): string {
  return body
    .replace(/Bearer\s+[A-Za-z0-9_\-.=]+/gi, "Bearer [redacted]")
    .replace(/\bsk-[A-Za-z0-9_-]{6,}/g, "sk-[redacted]")
    .replace(
      /("?(?:api[_-]?key|authorization|access[_-]?token|token|secret)"?\s*[:=]\s*"?)[A-Za-z0-9_\-.=]+/gi,
      "$1[redacted]",
    );
}

/**
 * Relays an upstream error to the client: a generic message for auth-class
 * failures (which may reflect our token and aren't client-actionable), and the
 * sanitized upstream body otherwise (so context-length / rate-limit / validation
 * errors stay actionable).
 */
function relayUpstreamError(
  res: ServerResponse,
  status: number,
  body: string,
  contentType = "application/json",
): void {
  if (res.headersSent) {
    res.end();
    return;
  }
  if (isAuthStatus(status)) {
    sendJson(res, status, {
      error: { message: "Upstream authentication failed.", type: "upstream_error" },
    });
    return;
  }
  const trimmed = body.trim();
  if (!trimmed) {
    sendJson(res, status, {
      error: { message: `Upstream returned ${status}.`, type: "upstream_error" },
    });
    return;
  }
  res.writeHead(status, { "content-type": contentType });
  res.end(sanitizeUpstreamBody(body));
}

function sendError(res: ServerResponse, status: number, message: string, type = "invalid_request_error"): void {
  if (res.headersSent) {
    res.end();
    return;
  }
  sendJson(res, status, { error: { message, type } });
}

/**
 * Builds a Node HTTP request handler that exposes an OpenAI-compatible
 * `chat/completions` endpoint and transparently adds tool-calling support on
 * top of a tool-less upstream model. Point any OpenAI client's `baseURL` at it.
 */
export function createProxyHandler(
  options: ProxyOptions,
): (req: IncomingMessage, res: ServerResponse) => void {
  const basePath = (options.basePath ?? "/v1").replace(/\/+$/, "");
  const cors = options.cors ?? false;
  const maxBodySize = options.maxBodySize ?? DEFAULT_MAX_BODY_BYTES;
  const upstream = createFetchClient({
    baseURL: options.upstreamBaseURL,
    apiKey: options.upstreamApiKey,
    headers: options.upstreamHeaders,
    fetch: options.fetch,
  });
  const wrapped: ToolCapableClient = wrapToolSupport(upstream, options);

  const setCors = (res: ServerResponse) => {
    if (!cors) return;
    res.setHeader("access-control-allow-origin", "*");
    res.setHeader("access-control-allow-methods", "GET, POST, OPTIONS");
    res.setHeader("access-control-allow-headers", "authorization, content-type");
  };

  const authOk = (req: IncomingMessage): boolean => {
    if (!options.apiKey) return true;
    const header = req.headers["authorization"];
    if (typeof header !== "string") return false;
    // Constant-time compare over fixed-length digests to avoid timing leaks.
    const a = createHash("sha256").update(header).digest();
    const b = createHash("sha256").update(`Bearer ${options.apiKey}`).digest();
    return timingSafeEqual(a, b);
  };

  return (req, res) => {
    void handle(req, res).catch((err) => {
      // An upstream API error (non-2xx with a body) is the legitimate response
      // the client needs — relay its status and body verbatim. Reserve the
      // generic 502 for unexpected failures (bugs, network errors to upstream).
      if (err instanceof UpstreamError) {
        // Full detail (incl. any token the upstream echoed) stays server-side only.
        console.error(`[llm-tool-proxy] upstream ${err.status}: ${err.body.slice(0, 500)}`);
        relayUpstreamError(res, err.status, err.body);
        return;
      }
      console.error("[llm-tool-proxy] request error:", err);
      sendError(res, 502, "Proxy request failed; see server logs.", "api_error");
    });
  };

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    setCors(res);
    const method = req.method ?? "GET";
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname.replace(/\/+$/, "") || "/";

    if (method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    if (method === "GET" && (path === "/" || path === "/health")) {
      sendJson(res, 200, { status: "ok", service: "llm-tool-capability proxy" });
      return;
    }

    if (!authOk(req)) {
      sendError(res, 401, "Missing or invalid API key.", "authentication_error");
      return;
    }

    // Passthrough for model listing.
    if (method === "GET" && path === `${basePath}/models`) {
      const base = options.upstreamBaseURL.replace(/\/+$/, "");
      const doFetch = options.fetch ?? globalThis.fetch;
      const upstreamRes = await doFetch(`${base}/models`, {
        headers: {
          ...options.upstreamHeaders,
          ...(options.upstreamApiKey
            ? { authorization: `Bearer ${options.upstreamApiKey}` }
            : {}),
        },
      });
      const modelsCt = upstreamRes.headers.get("content-type") ?? "application/json";
      if (!upstreamRes.ok) {
        console.error(`[llm-tool-proxy] upstream /models returned ${upstreamRes.status}`);
        relayUpstreamError(res, upstreamRes.status, await upstreamRes.text(), modelsCt);
        return;
      }
      res.writeHead(200, { "content-type": modelsCt });
      res.end(await upstreamRes.text());
      return;
    }

    if (method === "POST" && path === `${basePath}/chat/completions`) {
      let body: any;
      try {
        body = await readJsonBody(req, maxBodySize);
      } catch (err) {
        sendError(res, 400, err instanceof Error ? err.message : "Bad request");
        return;
      }
      if (!body || typeof body !== "object" || typeof body.model !== "string") {
        sendError(res, 400, "Request must include a 'model' and 'messages'.");
        return;
      }

      const controller = new AbortController();
      res.on("close", () => controller.abort());

      if (body.stream) {
        const stream = (await wrapped.chat.completions.create(body, {
          signal: controller.signal,
        })) as unknown as AsyncIterable<ChatCompletionChunk>;
        res.writeHead(200, {
          "content-type": "text/event-stream",
          "cache-control": "no-cache",
          connection: "keep-alive",
        });
        // Honor backpressure: pause when the socket buffer is full.
        const write = (s: string): Promise<void> => {
          if (res.write(s)) return Promise.resolve();
          return new Promise((resolve) => res.once("drain", resolve));
        };
        try {
          for await (const chunk of stream) {
            await write(`data: ${JSON.stringify(chunk)}\n\n`);
          }
        } catch (err) {
          console.error("[llm-tool-proxy] stream error:", err);
          if (!controller.signal.aborted && !res.writableEnded) {
            try {
              res.write(
                `data: ${JSON.stringify({ error: { message: "Upstream request failed." } })}\n\n`,
              );
            } catch {
              // socket already gone
            }
          }
        } finally {
          if (!res.writableEnded) {
            if (!controller.signal.aborted) {
              try {
                res.write("data: [DONE]\n\n");
              } catch {
                // socket already gone
              }
            }
            res.end();
          }
        }
        return;
      }

      const result = (await wrapped.chat.completions.create(body, {
        signal: controller.signal,
      })) as ChatCompletion;
      sendJson(res, 200, result);
      return;
    }

    // Transparent passthrough for other endpoints under basePath
    // (e.g. /completions, /embeddings, /rerank) — forwarded to the upstream
    // verbatim, with no tool injection (those endpoints have no tools).
    if (path === basePath || path.startsWith(`${basePath}/`)) {
      await passthrough(req, res, path.slice(basePath.length) || "/");
      return;
    }

    sendError(res, 404, `Not found: ${method} ${path}`, "not_found");
  }

  async function passthrough(
    req: IncomingMessage,
    res: ServerResponse,
    subpath: string,
  ): Promise<void> {
    const base = options.upstreamBaseURL.replace(/\/+$/, "");
    const doFetch = options.fetch ?? globalThis.fetch;
    const method = req.method ?? "GET";

    let bodyBuf: Buffer | undefined;
    if (method !== "GET" && method !== "HEAD") {
      try {
        bodyBuf = await readRawBody(req, maxBodySize);
      } catch (err) {
        sendError(res, 400, err instanceof Error ? err.message : "Bad request");
        return;
      }
    }

    const controller = new AbortController();
    res.on("close", () => controller.abort());

    const headers: Record<string, string> = {
      ...options.upstreamHeaders,
      ...(options.upstreamApiKey
        ? { authorization: `Bearer ${options.upstreamApiKey}` }
        : {}),
    };
    const ct = req.headers["content-type"];
    if (typeof ct === "string") headers["content-type"] = ct;
    const accept = req.headers["accept"];
    if (typeof accept === "string") headers["accept"] = accept;

    const upstreamRes = await doFetch(`${base}${subpath}`, {
      method,
      headers,
      body: bodyBuf && bodyBuf.length > 0 ? bodyBuf : undefined,
      signal: controller.signal,
    });

    const ptCt = upstreamRes.headers.get("content-type") ?? "application/octet-stream";
    if (!upstreamRes.ok) {
      console.error(`[llm-tool-proxy] upstream ${subpath} returned ${upstreamRes.status}`);
      relayUpstreamError(res, upstreamRes.status, await upstreamRes.text(), ptCt);
      return;
    }

    res.writeHead(upstreamRes.status, { "content-type": ptCt });

    const upstreamBody = upstreamRes.body;
    if (!upstreamBody) {
      res.end();
      return;
    }
    const reader = upstreamBody.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value && !res.write(Buffer.from(value))) {
          await new Promise<void>((resolve) => res.once("drain", resolve));
        }
      }
    } finally {
      reader.releaseLock();
      if (!res.writableEnded) res.end();
    }
  }
}

/** Creates (but does not start) an HTTP server wrapping {@link createProxyHandler}. */
export function createProxyServer(options: ProxyOptions): http.Server {
  return http.createServer(createProxyHandler(options));
}
