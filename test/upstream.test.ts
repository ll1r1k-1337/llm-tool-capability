import { describe, it, expect, vi } from "vitest";
import { createFetchClient } from "../src/index.js";
import type { ChatCompletion, ChatCompletionChunk } from "../src/index.js";
import { makeCompletion, collect } from "./helpers.js";

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
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function chunk(content: string): ChatCompletionChunk {
  return {
    id: "c",
    object: "chat.completion.chunk",
    created: 0,
    model: "m",
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  };
}

describe("createFetchClient", () => {
  it("posts to /chat/completions and parses a non-stream response", async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: any) =>
      jsonResponse(makeCompletion("hi")),
    );
    const client = createFetchClient({
      baseURL: "http://x/v1/",
      apiKey: "secret",
      fetch: fetchMock as any,
    });
    const res = (await client.chat.completions.create({
      model: "m",
      messages: [{ role: "user", content: "hi" }],
    })) as ChatCompletion;

    expect(res.choices[0]!.message.content).toBe("hi");
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://x/v1/chat/completions");
    expect((init as any).method).toBe("POST");
    expect((init as any).headers.authorization).toBe("Bearer secret");
  });

  it("decodes an SSE stream into chunks", async () => {
    const events = [
      `data: ${JSON.stringify(chunk("Hello "))}\n\n`,
      `data: ${JSON.stringify(chunk("world"))}\n\n`,
      "data: [DONE]\n\n",
    ];
    const fetchMock = vi.fn(async () => sseResponse(events));
    const client = createFetchClient({ baseURL: "http://x/v1", fetch: fetchMock as any });
    const stream = (await client.chat.completions.create({
      model: "m",
      messages: [],
      stream: true,
    })) as AsyncIterable<ChatCompletionChunk>;

    const chunks = await collect(stream);
    expect(chunks).toHaveLength(2);
    expect(chunks.map((c) => c.choices[0]!.delta.content).join("")).toBe("Hello world");
  });

  it("handles CRLF SSE framing", async () => {
    const events = [`data: ${JSON.stringify(chunk("x"))}\r\n\r\n`, "data: [DONE]\r\n\r\n"];
    const fetchMock = vi.fn(async () => sseResponse(events));
    const client = createFetchClient({ baseURL: "http://x/v1", fetch: fetchMock as any });
    const stream = (await client.chat.completions.create({
      model: "m",
      messages: [],
      stream: true,
    })) as AsyncIterable<ChatCompletionChunk>;
    const chunks = await collect(stream);
    expect(chunks).toHaveLength(1);
  });

  it("throws on a non-2xx upstream response", async () => {
    const fetchMock = vi.fn(async () => new Response("boom", { status: 500 }));
    const client = createFetchClient({ baseURL: "http://x/v1", fetch: fetchMock as any });
    await expect(
      client.chat.completions.create({ model: "m", messages: [] }),
    ).rejects.toThrow(/500/);
  });
});
