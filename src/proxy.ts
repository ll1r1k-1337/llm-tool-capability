import http from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { createHash, timingSafeEqual } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type {
  ChatClientLike,
  ChatCompletion,
  ChatCompletionChunk,
  ToolCapableClient,
} from "./types.js";
import { wrapToolSupport, type WrapOptions } from "./client.js";
import { createFetchClient } from "./upstream.js";
import { UpstreamError } from "./errors.js";

/** Appends one JSON line per event to a debug log file (opt-in via logFile). */
type LogFn = (entry: Record<string, unknown>) => void;

/** Wraps an upstream client so the (post-transform) outbound body is logged. */
function withUpstreamLogging(client: ChatClientLike, log: LogFn): ChatClientLike {
  return {
    chat: {
      completions: {
        create(body: any, opts?: any) {
          log({ type: "upstream_request", stream: !!body?.stream, body });
          return client.chat.completions.create(body, opts);
        },
      },
    },
  };
}

const DEFAULT_MAX_LOG_BYTES = 100 * 1024 * 1024;
const LOG_BACKPRESSURE_BYTES = 8 * 1024 * 1024;

interface FileLogger {
  log: LogFn;
  close: () => void;
}

/**
 * Bounded JSON-lines file logger for the proxy: caps total size, drops entries
 * under write backpressure (to bound memory), stops on stream error, and never
 * throws into the request path.
 */
function createFileLogger(filePath: string, maxBytes: number): FileLogger {
  try {
    fs.mkdirSync(path.dirname(path.resolve(filePath)), { recursive: true });
  } catch {
    // best effort; createWriteStream's 'error' handler reports a bad path
  }
  const stream = fs.createWriteStream(filePath, { flags: "a" });
  let errored = false;
  let written = 0;
  let warnedCap = false;
  let warnedDrop = false;

  stream.on("error", (err) => {
    errored = true;
    console.error("[llm-tool-proxy] log file write error:", err);
  });

  const log: LogFn = (entry) => {
    if (errored) return;
    if (written >= maxBytes) {
      if (!warnedCap) {
        warnedCap = true;
        console.warn(
          `[llm-tool-proxy] log file reached ${maxBytes} bytes; further logging disabled.`,
        );
      }
      return;
    }
    if (stream.writableLength > LOG_BACKPRESSURE_BYTES) {
      if (!warnedDrop) {
        warnedDrop = true;
        console.warn("[llm-tool-proxy] log write backpressure; dropping entries.");
      }
      return;
    }
    let line: string;
    try {
      line = JSON.stringify({ ts: new Date().toISOString(), ...entry }) + "\n";
    } catch (err) {
      console.warn(
        "[llm-tool-proxy] failed to serialize log entry:",
        err instanceof Error ? err.message : String(err),
      );
      return;
    }
    written += Buffer.byteLength(line);
    stream.write(line);
  };

  const close = () => {
    try {
      stream.end();
    } catch {
      // already closed
    }
  };

  return { log, close };
}

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
  /**
   * If set, append a JSON-lines debug log to this file: the request received
   * from the client, the (transformed) request sent upstream, and the response.
   * Request/response bodies are logged in full; headers (and thus tokens) are
   * never logged. Verbose — intended for debugging only.
   */
  logFile?: string;
  /** Stop logging once the log file reaches this many bytes. Default: 100 MiB. */
  maxLogBytes?: number;
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

/**
 * Strips credential-shaped tokens from an upstream error body before relaying.
 * Token bodies are matched up to a whitespace/quote/structural boundary (not a
 * fixed character class) so JWT/base64 tokens containing `/`, `+`, `.` are
 * redacted in full rather than truncated.
 */
function sanitizeUpstreamBody(body: string): string {
  return body
    .replace(/Bearer\s+[^\s"']+/gi, "Bearer [redacted]")
    .replace(/\bsk-[^\s"',}]+/g, "sk-[redacted]")
    .replace(
      /("?(?:api[_-]?key|authorization|access[_-]?token|token|secret)"?\s*[:=]\s*"?)[^\s"',}]+/gi,
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

/** A proxy request handler with a `closeLog()` to flush/close the debug log. */
export interface ProxyHandler {
  (req: IncomingMessage, res: ServerResponse): void;
  /** Flush and close the debug log stream (no-op if logging is disabled). */
  closeLog(): void;
}

/**
 * Builds a Node HTTP request handler that exposes an OpenAI-compatible
 * `chat/completions` endpoint and transparently adds tool-calling support on
 * top of a tool-less upstream model. Point any OpenAI client's `baseURL` at it.
 */
export function createProxyHandler(options: ProxyOptions): ProxyHandler {
  const basePath = (options.basePath ?? "/v1").replace(/\/+$/, "");
  const cors = options.cors ?? false;
  const maxBodySize = options.maxBodySize ?? DEFAULT_MAX_BODY_BYTES;

  const logger = options.logFile
    ? createFileLogger(options.logFile, options.maxLogBytes ?? DEFAULT_MAX_LOG_BYTES)
    : null;
  const log: LogFn = (entry) => logger?.log(entry);

  const rawUpstream = createFetchClient({
    baseURL: options.upstreamBaseURL,
    apiKey: options.upstreamApiKey,
    headers: options.upstreamHeaders,
    fetch: options.fetch,
  });
  const upstream = logger ? withUpstreamLogging(rawUpstream, log) : rawUpstream;
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

  const handler: ProxyHandler = (req, res) => {
    void handle(req, res).catch((err) => {
      // An upstream API error (non-2xx with a body) is the legitimate response
      // the client needs — relay its status and body verbatim. Reserve the
      // generic 502 for unexpected failures (bugs, network errors to upstream).
      if (err instanceof UpstreamError) {
        // Full detail (incl. any token the upstream echoed) stays server-side only.
        console.error(`[llm-tool-proxy] upstream ${err.status}: ${err.body.slice(0, 500)}`);
        log({ type: "upstream_error", status: err.status, body: sanitizeUpstreamBody(err.body) });
        relayUpstreamError(res, err.status, err.body);
        return;
      }
      console.error("[llm-tool-proxy] request error:", err);
      log({ type: "error", message: err instanceof Error ? err.message : String(err) });
      sendError(res, 502, "Proxy request failed; see server logs.", "api_error");
    });
  };
  handler.closeLog = () => logger?.close();
  return handler;

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
      if (body.stream !== undefined && typeof body.stream !== "boolean") {
        sendError(res, 400, "'stream' must be a boolean.");
        return;
      }

      log({ type: "client_request", method, path, body });

      const controller = new AbortController();
      res.on("close", () => controller.abort());

      if (body.stream === true) {
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
        log({ type: "response", path, status: 200, stream: true });
        return;
      }

      const result = (await wrapped.chat.completions.create(body, {
        signal: controller.signal,
      })) as ChatCompletion;
      log({ type: "response", path, status: 200, body: result });
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

    if (logger) {
      const raw = bodyBuf ? bodyBuf.toString("utf8") : "";
      let parsed: unknown = raw;
      try {
        if (raw) parsed = JSON.parse(raw);
      } catch {
        /* keep raw string */
      }
      log({ type: "client_request", method, path: subpath, passthrough: true, body: parsed });
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

/** An http.Server that also flushes/closes the proxy's debug log on `closeLog()`. */
export interface ProxyServer extends http.Server {
  /** Flush and close the debug log stream (no-op if logging is disabled). */
  closeLog(): void;
}

/** Creates (but does not start) an HTTP server wrapping {@link createProxyHandler}. */
export function createProxyServer(options: ProxyOptions): ProxyServer {
  const handler = createProxyHandler(options);
  const server = http.createServer(handler) as ProxyServer;
  server.closeLog = handler.closeLog;
  return server;
}
