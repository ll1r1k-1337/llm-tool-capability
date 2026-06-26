/**
 * Compile-time proof (not executed) that a real `openai` client and its types
 * satisfy our structural interfaces — i.e. that `wrapToolSupport` is genuinely
 * drop-in. `npm run typecheck` fails if a future `openai` version drifts.
 *
 * This file is intentionally NOT a `*.test.ts`, so vitest skips it; it is only
 * type-checked by `tsc`.
 */
import OpenAI from "openai";
import type {
  ChatCompletionTool as OAITool,
  ChatCompletionMessageParam as OAIMessage,
} from "openai/resources/chat/completions";
import { wrapToolSupport, createToolRunner, defineTool } from "../src/index.js";
import type { ChatCompletionTool, ChatCompletionMessageParam } from "../src/index.js";

// A real OpenAI client satisfies ChatClientLike.
const openai = new OpenAI({ apiKey: "test", baseURL: "http://localhost:11434/v1" });
const client = wrapToolSupport(openai);

// Our tool/message types are a deliberate subset of OpenAI's: this package only
// supports `function` tools (not the newer `custom` tools). The casts below
// acknowledge that narrowing — a `custom` tool/tool-call from OpenAI is not
// representable here by design (see README "Limitations").
declare const oaTool: OAITool;
declare const oaMessage: OAIMessage;
const _tool: ChatCompletionTool = oaTool as ChatCompletionTool;
const _msg: ChatCompletionMessageParam = oaMessage as ChatCompletionMessageParam;

async function _demo() {
  // Non-streaming overload returns a (synthetic) ChatCompletion.
  const res = await client.chat.completions.create({
    model: "llama3",
    messages: [{ role: "user", content: "hi" }],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      },
    ],
  });
  const _calls = res.choices[0]?.message.tool_calls;

  // Streaming overload returns an async iterable of chunks.
  const stream = await client.chat.completions.create({
    model: "llama3",
    messages: [{ role: "user", content: "hi" }],
    stream: true,
  });
  for await (const chunk of stream) void chunk.choices[0]?.delta;

  // Runner accepts the raw client too.
  const runner = createToolRunner(openai, {
    tools: [
      defineTool({
        name: "echo",
        parameters: { type: "object", properties: { text: { type: "string" } } },
        handler: (args: { text: string }) => args.text,
      }),
    ],
  });
  await runner.run({ model: "llama3", messages: [{ role: "user", content: "hi" }] });
}

void _demo;
void _tool;
void _msg;
