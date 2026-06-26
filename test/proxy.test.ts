import { describe, it, expect, vi, afterEach } from "vitest";
import type { AddressInfo } from "node:net";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createProxyServer, type ProxyServer } from "../src/proxy.js";
import type { ChatCompletionTool } from "../src/index.js";
import { makeCompletion } from "./helpers.js";

const weatherTool: ChatCompletionTool = {
  type: "function",
  function: {
    name: "get_weather",
    parameters: { type: "object", properties: { city: { type: "string" } } },
  },
};

const toolCallText =
  '```tool_call\n{"name":"get_weather","arguments":{"city":"Paris"}}\n```';

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function sseResponse(events: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const enc = new TextEncoder();
      for (const e of events) controller.enqueue(enc.encode(e));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

function listen(server: ProxyServer): Promise<string> {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve(`http://127.0.0.1:${port}`);
    });
  });
}

const servers: ProxyServer[] = [];
afterEach(() => {
  for (const s of servers.splice(0)) {
    s.closeLog();
    s.close();
  }
});

async function startProxy(fetchMock: any, extra: Record<string, unknown> = {}) {
  const server = createProxyServer({
    upstreamBaseURL: "http://upstream/v1",
    fetch: fetchMock,
    ...extra,
  });
  servers.push(server);
  const base = await listen(server);
  return base;
}

describe("proxy server", () => {
  it("serves a health check", async () => {
    const base = await startProxy(vi.fn());
    const res = await fetch(`${base}/health`);
    expect(res.status).toBe(200);
    expect(((await res.json()) as any).status).toBe("ok");
  });

  it("adds tool calling on top of a tool-less upstream (non-stream)", async () => {
    const upstream = vi.fn(async (_url: string, _init?: any) =>
      jsonResponse(makeCompletion(toolCallText)),
    );
    const base = await startProxy(upstream);

    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "m",
        messages: [{ role: "user", content: "weather?" }],
        tools: [weatherTool],
      }),
    });
    const json: any = await res.json();
    expect(res.status).toBe(200);
    expect(json.choices[0].message.tool_calls[0].function.name).toBe("get_weather");
    expect(json.choices[0].message.tool_calls[0].function.arguments).toBe(
      '{"city":"Paris"}',
    );

    // The upstream must NOT have received the tools param.
    const sentBody = JSON.parse((upstream.mock.calls[0]![1] as any).body);
    expect(sentBody.tools).toBeUndefined();
    expect(sentBody.messages[0].role).toBe("system");
  });

  it("streams tool calls as SSE ending with [DONE]", async () => {
    const events = [
      `data: ${JSON.stringify({
        id: "c",
        object: "chat.completion.chunk",
        created: 0,
        model: "m",
        choices: [{ index: 0, delta: { content: toolCallText }, finish_reason: null }],
      })}\n\n`,
      "data: [DONE]\n\n",
    ];
    const upstream = vi.fn(async () => sseResponse(events));
    const base = await startProxy(upstream);

    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "m",
        messages: [{ role: "user", content: "weather?" }],
        tools: [weatherTool],
        stream: true,
      }),
    });
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const text = await res.text();
    expect(text).toContain("data: [DONE]");

    const toolCalls = text
      .split("\n\n")
      .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
      .map((l) => JSON.parse(l.slice(6)))
      .flatMap((c: any) => c.choices?.[0]?.delta?.tool_calls ?? []);
    expect(toolCalls[0]?.function?.name).toBe("get_weather");
  });

  it("requires an API key when configured", async () => {
    const upstream = vi.fn(async () => jsonResponse(makeCompletion("x")));
    const base = await startProxy(upstream, { apiKey: "secret" });

    const unauth = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [] }),
    });
    expect(unauth.status).toBe(401);

    const authed = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer secret" },
      body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(authed.status).toBe(200);
  });

  it("returns 400 for a malformed request", async () => {
    const base = await startProxy(vi.fn());
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{ not json",
    });
    expect(res.status).toBe(400);
  });

  it("passes through /models", async () => {
    const upstream = vi.fn(async (url: string) =>
      url.endsWith("/models")
        ? jsonResponse({ object: "list", data: [{ id: "m1" }] })
        : jsonResponse(makeCompletion("x")),
    );
    const base = await startProxy(upstream);
    const res = await fetch(`${base}/v1/models`);
    expect(res.status).toBe(200);
    expect(((await res.json()) as any).data[0].id).toBe("m1");
  });

  it("relays upstream /models errors (status + body) to the client", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const upstream = vi.fn(async () => new Response(JSON.stringify({ error: "boom" }), { status: 503 }));
    const base = await startProxy(upstream);
    const res = await fetch(`${base}/v1/models`);
    expect(res.status).toBe(503);
    expect(((await res.json()) as any).error).toBe("boom");
  });

  it("relays upstream chat errors (status + body) so clients can react", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    // Mirrors issue #2: upstream rejects an over-long prompt.
    const upstream = vi.fn(
      async () =>
        new Response(
          JSON.stringify({ error: "context length exceeded", error_type: "Generation Error" }),
          { status: 424 },
        ),
    );
    const base = await startProxy(upstream);
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(424);
    expect(((await res.json()) as any).error).toBe("context length exceeded");
  });

  it("masks unexpected (non-HTTP) failures behind a generic 502", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const upstream = vi.fn(async () => {
      throw new Error("CONNECTION-INTERNALS-secret");
    });
    const base = await startProxy(upstream);
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(502);
    expect(await res.text()).not.toContain("INTERNALS");
  });

  it("returns a generic message for upstream auth failures (no token echo)", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const upstream = vi.fn(
      async () =>
        new Response(JSON.stringify({ error: "invalid key: Bearer sk-LEAKED12345" }), {
          status: 401,
        }),
    );
    const base = await startProxy(upstream);
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(401);
    const text = await res.text();
    expect(text).not.toContain("LEAKED");
    expect(text).not.toContain("sk-");
  });

  it("scrubs credential tokens from relayed (non-auth) upstream error bodies", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const upstream = vi.fn(
      async () =>
        new Response(
          JSON.stringify({ error: "bad request: Bearer sk-SHOULDNOTLEAK99 rejected" }),
          { status: 400 },
        ),
    );
    const base = await startProxy(upstream);
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(400);
    const text = await res.text();
    expect(text).not.toContain("SHOULDNOTLEAK");
    expect(text).toContain("redacted");
  });

  it("passes through legacy /v1/completions to the upstream", async () => {
    const upstream = vi.fn(async (_url: string, _init?: any) =>
      jsonResponse({ id: "cmpl-x", object: "text_completion", choices: [{ text: "hi" }] }),
    );
    const base = await startProxy(upstream);
    const res = await fetch(`${base}/v1/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", prompt: "hi" }),
    });
    expect(res.status).toBe(200);
    expect(((await res.json()) as any).object).toBe("text_completion");
    expect(String(upstream.mock.calls[0]![0])).toBe("http://upstream/v1/completions");
  });

  it("writes a JSON-lines debug log of client and upstream requests", async () => {
    const logPath = path.join(
      os.tmpdir(),
      `lcap-log-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`,
    );
    const upstream = vi.fn(async (_url: string, _init?: any) =>
      jsonResponse(makeCompletion(toolCallText)),
    );
    const base = await startProxy(upstream, { logFile: logPath });
    await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "m",
        messages: [{ role: "user", content: "weather?" }],
        tools: [weatherTool],
      }),
    });

    // Flush deterministically by closing the log stream, then poll as a safety net.
    servers[servers.length - 1]!.closeLog();
    let content = "";
    for (let i = 0; i < 100 && !content.includes('"response"'); i++) {
      try {
        content = fs.readFileSync(logPath, "utf8");
      } catch {
        /* not created yet */
      }
      if (content.includes('"response"')) break;
      await new Promise((r) => setTimeout(r, 20));
    }

    const lines = content
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((l) => JSON.parse(l));
    const byType = (t: string) => lines.find((l) => l.type === t);

    expect(lines.map((l) => l.type)).toEqual(
      expect.arrayContaining(["client_request", "upstream_request", "response"]),
    );
    // Client request keeps the original user message + tools.
    expect(byType("client_request").body.messages[0].role).toBe("user");
    expect(byType("client_request").body.tools).toHaveLength(1);
    // Upstream request is transformed: tools injected as a system message, stripped from params.
    expect(byType("upstream_request").body.messages[0].role).toBe("system");
    expect(byType("upstream_request").body.tools).toBeUndefined();

    try {
      fs.rmSync(logPath, { force: true });
    } catch {
      /* stream may still hold the file open on Windows */
    }
  });

  it("sanitizes upstream error bodies written to the log file", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const logPath = path.join(
      os.tmpdir(),
      `lcap-log-sec-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`,
    );
    const upstream = vi.fn(
      async () =>
        new Response(JSON.stringify({ error: "invalid: Bearer sk-LOGLEAK123" }), {
          status: 401,
        }),
    );
    const base = await startProxy(upstream, { logFile: logPath });
    await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    });

    servers[servers.length - 1]!.closeLog();
    let content = "";
    for (let i = 0; i < 100 && !content.includes("upstream_error"); i++) {
      try {
        content = fs.readFileSync(logPath, "utf8");
      } catch {
        /* not created yet */
      }
      if (content.includes("upstream_error")) break;
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(content).toContain("upstream_error");
    expect(content).not.toContain("LOGLEAK");

    try {
      fs.rmSync(logPath, { force: true });
    } catch {
      /* stream may still hold the file open */
    }
  });

  it("rejects a non-boolean stream parameter", async () => {
    const base = await startProxy(vi.fn());
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [], stream: "true" }),
    });
    expect(res.status).toBe(400);
  });

  it("fully redacts JWT/base64 tokens (with / and +) from relayed errors", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const upstream = vi.fn(
      async () =>
        new Response(JSON.stringify({ error: "bad: Bearer eyJhbGc.AAA/BBB+CCC end" }), {
          status: 400,
        }),
    );
    const base = await startProxy(upstream);
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status).toBe(400);
    const text = await res.text();
    expect(text).not.toContain("AAA");
    expect(text).not.toContain("BBB");
    expect(text).not.toContain("CCC");
  });

  it("rejects oversized request bodies", async () => {
    const base = await startProxy(vi.fn(), { maxBodySize: 200 });
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "x".repeat(1000) }] }),
    });
    expect(res.status).toBe(400);
  });
});
