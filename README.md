# llm-tool-capability

Drop-in **OpenAI-compatible tool calling for LLMs that don't support function
calling natively.** It injects your tools into the prompt, parses the model's
text back into OpenAI-shaped `tool_calls`, and can run the whole agentic loop for
you — with streaming.

Works with any OpenAI-compatible endpoint: **Ollama, vLLM, LM Studio,
llama.cpp**, text-generation-webui, and others.

```
npm install llm-tool-capability
```

`openai` is an optional peer dependency — install it if you want to wrap a real
OpenAI client (you can also pass any OpenAI-compatible client object).

---

## Why

Lots of open models are great at following instructions but expose **no
`tools` parameter** — the server rejects it or silently ignores it. This package
makes `tools` / `tool_choice` work anyway by:

1. Rendering your tool schemas + a calling contract into a system prompt.
2. Asking the model to emit calls as fenced ` ```tool_call ` JSON blocks.
3. Parsing those blocks back into **exactly** OpenAI's
   `message.tool_calls` shape (`{ id, type: "function", function: { name, arguments } }`,
   where `arguments` is a JSON **string**).

You can use it **three ways**: as a zero-code **proxy** (run a server, point your
OpenAI client at it), as a **drop-in client** (wrap your client in code), or as
an **agentic runner** (it runs the tool loop for you).

---

## Proxy mode (no code changes)

Run a local OpenAI-compatible proxy in front of your tool-less model. Any OpenAI
client just needs its `baseURL` pointed at the proxy — no other changes.

```bash
npx llm-tool-proxy --upstream http://localhost:11434/v1 --port 8787
# llm-tool-proxy listening on http://127.0.0.1:8787/v1
#   → upstream: http://localhost:11434/v1
```

Now point any OpenAI client at it and pass `tools` as usual:

```ts
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8787/v1", apiKey: "unused" });

const res = await client.chat.completions.create({
  model: "qwen2.5:7b",
  messages: [{ role: "user", content: "What's the weather in Paris?" }],
  tools: [/* … */],            // ← works, even though the model has no native tools
});
// res.choices[0].message.tool_calls → populated, OpenAI shape (streaming too)
```

The proxy forwards everything to the upstream, injects the tool contract, parses
tool calls back, and streams via SSE — identical wire format to OpenAI.

**CLI flags:** `--upstream <url>` (required), `--upstream-key`, `--port`,
`--host`, `--api-key` (require a bearer token from clients), `--base-path`,
`--tag`, `--no-examples`, `--system-injection merge|prepend`, `--xml-tool-calls`
(also parse native `<toolName>…</toolName>` tags — see *Reasoning & native
formats* below), `--no-reasoning` / `--reasoning-tag <tag>` (control
`<think>` extraction), `--cors` (enable wildcard CORS — off by default),
`--max-body-size <bytes>` (default 10 MiB), `--log-file <path>` (append a
JSON-lines debug log of the client request, the transformed upstream request,
and the response — verbose; bodies are logged but headers/tokens never are),
`--max-log-size <bytes>` (cap the log file; default 100 MiB). Key flags have env
equivalents (`UPSTREAM_BASE_URL`, `PORT`, `PROXY_API_KEY`, `PROXY_LOG_FILE`, …).

> **Security:** the proxy binds to `127.0.0.1` and disables CORS by default.
> Before exposing it beyond localhost (`--host 0.0.0.0`), set `--api-key` so
> clients must authenticate. Upstream API errors (status + body — e.g.
> context-length or rate-limit) are relayed to the client so it can react;
> unexpected internal failures are masked behind a generic 502 and logged.

Embed the proxy in your own server instead of the CLI:

```ts
import { createProxyServer } from "llm-tool-capability/proxy";

createProxyServer({
  upstreamBaseURL: "http://localhost:11434/v1",
  apiKey: process.env.PROXY_API_KEY, // optional client auth
}).listen(8787);
```

**Endpoints:** `POST /v1/chat/completions` (with tool support) and `GET /health`.
Every other route under the base path — `/v1/completions`, `/v1/embeddings`,
`/v1/models`, etc. — is **transparently passed through** to the upstream
unchanged (no tool injection; those endpoints have no tools), so the proxy is a
full drop-in, not just a chat endpoint.

> The proxy does **not** execute tools — it returns `tool_calls` to the caller,
> exactly like OpenAI. Your client runs the tools and sends results back. For
> server-side tool *execution*, use the agentic runner (Layer B) in code.

---

## Layer A — drop-in client

`wrapToolSupport(client)` returns a client whose
`chat.completions.create` is a drop-in for OpenAI's. Pass `tools` as usual; get
`tool_calls` back as usual. When you pass no `tools`, it's completely
transparent.

```ts
import OpenAI from "openai";
import { wrapToolSupport } from "llm-tool-capability";

const openai = new OpenAI({ baseURL: "http://localhost:11434/v1", apiKey: "ollama" });
const client = wrapToolSupport(openai);

const res = await client.chat.completions.create({
  model: "llama3.1",
  messages: [{ role: "user", content: "What's the weather in Paris?" }],
  tools: [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get the current weather for a city.",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"],
        },
      },
    },
  ],
});

const toolCalls = res.choices[0].message.tool_calls;
// [{ id: "call_…", type: "function",
//    function: { name: "get_weather", arguments: '{"city":"Paris"}' } }]
```

You drive the loop yourself: execute the call, append a `role: "tool"` message
(with `tool_call_id`), and call again. The wrapper automatically rewrites those
native tool roles back into the prompt contract — so a normal OpenAI tool-calling
loop just works.

### Streaming (layer A)

```ts
const stream = await client.chat.completions.create({
  model: "llama3.1",
  messages,
  tools,
  stream: true,
});

for await (const chunk of stream) {
  const delta = chunk.choices[0]?.delta;
  if (delta?.content) process.stdout.write(delta.content);     // prose, token by token
  if (delta?.tool_calls) handleToolCallDelta(delta.tool_calls); // OpenAI chunk deltas
}
```

Prose streams token-by-token. Each tool call is emitted **atomically** when its
block closes (full `arguments` in one delta) — this avoids ever surfacing
partial/invalid JSON mid-stream. Accumulate by `index` exactly as you would with
OpenAI.

---

## Layer B — agentic runner

`createToolRunner` does the loop for you: ask → parse → run handlers → feed
results back → repeat, until the model answers without calling a tool.

```ts
import OpenAI from "openai";
import { createToolRunner, defineTool } from "llm-tool-capability";

const openai = new OpenAI({ baseURL: "http://localhost:11434/v1", apiKey: "ollama" });

const runner = createToolRunner(openai, {
  tools: [
    defineTool({
      name: "get_weather",
      description: "Get the current weather for a city.",
      parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
      },
      handler: async ({ city }) => {
        const r = await fetch(`https://api.example.com/weather?city=${city}`);
        return r.json();
      },
    }),
  ],
  maxIterations: 8,
});

const result = await runner.run({
  model: "llama3.1",
  messages: [{ role: "user", content: "Is it raining in Paris?" }],
});

console.log(result.content);          // final answer
console.log(result.toolExecutions);   // every tool call + result, in order
console.log(result.messages);         // full transcript
```

Pass the **raw** client to `createToolRunner` — it wraps it internally.

### Streaming events (layer B)

```ts
for await (const ev of runner.runStream({ model: "llama3.1", messages })) {
  switch (ev.type) {
    case "text":        process.stdout.write(ev.delta); break;
    case "tool_call":   console.log("→ calling", ev.toolCall.function.name); break;
    case "tool_result": console.log("← result", ev.execution.content); break;
    case "final":       console.log("\ndone:", ev.content); break;
  }
}
```

### Error feedback

Unknown tools, JSON-schema-invalid arguments, malformed JSON, and handlers that
throw are **not** fatal: the error is fed back to the model as the tool result so
it can correct itself on the next turn. Each is recorded in
`result.toolExecutions[i]` with `isError: true`.

---

## How it works

| Concern | Behavior |
| --- | --- |
| Call format | ` ```tool_call ` block with `{"name", "arguments"}` (arguments is a JSON object). Configurable tag. |
| Multiple calls | Several blocks back-to-back, or an array inside one block. |
| Malformed JSON | Light repair (trailing commas, comments); falls back to the raw string. |
| Lenient parsing | If no tagged block is found, ` ```json `/untagged blocks that look like a call are accepted (toggle with `lenientFences`). |
| History | Native `assistant.tool_calls` and `role: "tool"` messages are flattened back into the contract automatically. |
| Validation | Arguments validated against each tool's JSON Schema via `ajv` (toggle with `validate`). |
| `tool_choice` | `auto` (default), `required`, `{ function: { name } }`, and `none` are honored via prompt instructions. |
| Reasoning | `<think>…</think>` is split out of `content` into `reasoning_content` (and an upstream `reasoning_content` field is forwarded); on by default, toggle with `reasoning`. |
| Native XML calls | Opt-in (`xmlToolCalls`): an own-line `<toolName>…</toolName>` whose tag matches a tool is parsed as a call, for models that emit XML instead of the fence. |
| Loop safety | `maxIterations` ceiling (default 10); returns `finishReason: "max_iterations"` or throws with `throwOnMaxIterations`. |

## Options

`wrapToolSupport(client, options)` / `createToolRunner(client, options)` share:

- `toolCallTag` / `toolResultTag` — fence labels (default `tool_call` / `tool_result`).
- `includeExamples` — include a few-shot example (default `true`; weak models benefit).
- `template` — fully customize the instruction block.
- `systemInjection` — `"merge"` (append to existing system message, default) or `"prepend"`.
- `lenientFences` — accept ` ```json `/untagged look-alikes (default `true`).
- `reasoning` — split `<think>…</think>` into `reasoning_content` (default `true`).
- `reasoningTag` — the reasoning tag to split (default `think`).
- `xmlToolCalls` — also parse native `<toolName>…</toolName>` tags as calls (default `false`).
- `generateId` — custom tool-call id generator.

Runner-only: `tools`, `maxIterations`, `validate`, `throwOnMaxIterations`,
`onToolCall`, `onToolResult`.

### Reasoning & native formats

Some models (e.g. DeepSeek-R1 variants) emit a `<think>…</think>` reasoning
block and/or request tools as **XML tags** rather than the ` ```tool_call `
fence. Two behaviors handle this (in both streaming and non-streaming, and
through the proxy):

- **Reasoning (`reasoning`, on by default).** `<think>…</think>` is removed from
  `content` and surfaced as `message.reasoning_content` (streamed as
  `reasoning_content` deltas). A separate upstream `reasoning_content` field is
  forwarded as-is. Tag matching is **case-sensitive** (default `think`).
  - *Non-streaming* only strips matched open/close pairs, so a forgotten
    `</think>` never swallows the answer (the unterminated tag stays in `content`).
  - *Streaming* must commit as tokens arrive: once `<think>` is seen, text routes
    to `reasoning_content` until the matching `</think>` (or end of stream). A
    model that opens `<think>` and never closes it therefore streams the
    remainder as reasoning rather than content. In practice R1-style models
    always close the tag.

- **Native XML tool calls (`xmlToolCalls`, off by default).** When enabled, an
  **own-line** `<toolName>…</toolName>` block whose tag matches one of the
  request's tools is parsed into a tool call. The inner payload maps to
  arguments: a JSON object is used directly; a bare array/scalar (or plain text)
  is wrapped as the tool's single required parameter — e.g.
  `<question>[…]</question>` → `{ "questions": […] }`. A block that can't be
  mapped (a bare value on a multi-parameter tool) is left as `content`, not
  dropped. It's off by default because an always-on `<tag>` scan risks false
  positives on models that don't use this format. The non-streaming parser skips
  tags inside fenced code blocks; the streaming parser is **best-effort** here
  and may match an own-line `<toolName>` inside a non-`tool_call` code fence, so
  enable `xmlToolCalls` only for models that actually emit this format.

## Building blocks

The internals are exported for custom pipelines: `buildToolPrompt`,
`parseToolCalls`, `ToolCallStreamParser`, `ToolValidator`, `flattenMessages`,
`extractFencedBlocks`, `tryParseJson`, `extractReasoning`, `mapXmlToolCall`,
`ReasoningStreamParser`.

## Limitations

- **Tool-call arguments stream atomically**, not token-by-token (prose does
  stream token-by-token). This is a deliberate trade-off for robustness.
- Quality depends on the model's instruction-following. Small models do better
  with `includeExamples: true` and a short, clear tool list.
- Only `function` tools are supported (matching OpenAI's function tools); the
  newer `custom` tools are out of scope.
- **Streaming processes the first choice only.** `n > 1` is rejected together
  with `tools` (prompted tool calling parses a single completion; OpenAI forbids
  `n > 1` with tools too). Non-streaming `n > 1` without tools passes through.

## Security

- **Tool definitions are trusted input.** Each tool's JSON Schema (including any
  `pattern`) is compiled and executed by `ajv` during argument validation in the
  runner. Don't pass untrusted/user-authored tool schemas without vetting them —
  a malicious `pattern` could cause catastrophic regex backtracking (ReDoS). Tool
  names/descriptions are sanitized before they reach the prompt.
- **Parser limits.** The streaming parser caps its internal buffer
  (`maxBufferBytes`, default 1 MiB) and scans for the closing fence in linear
  time; JSON "repair" is skipped on very large inputs — both bound CPU/memory on
  malformed or unterminated model output.
- **Proxy.** Binds to `127.0.0.1` with CORS off by default; set `--api-key`
  before exposing it. Client request bodies are size-capped; upstream API errors
  are relayed but credential-shaped tokens are stripped first (and `401/403`
  bodies are replaced with a generic message); the debug log never records
  headers/tokens and is size-bounded.

## License

MIT
