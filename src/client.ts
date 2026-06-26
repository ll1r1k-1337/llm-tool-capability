import type {
  ChatClientLike,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionChunkDelta,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessage,
  ChatCompletionMessageParam,
  ChatCompletionContentPart,
  ToolCapableClient,
} from "./types.js";
import {
  buildToolPrompt,
  buildToolChoiceInstruction,
  DEFAULT_TOOL_CALL_TAG,
  DEFAULT_TOOL_RESULT_TAG,
  type PromptOptions,
} from "./prompt.js";
import { parseToolCalls, tryParseJson } from "./parser.js";
import { ToolCallStreamParser } from "./stream-parser.js";
import { ToolCapabilityError } from "./errors.js";

export interface WrapOptions extends PromptOptions {
  /** Generates the `id` for parsed tool calls (e.g. for deterministic tests). */
  generateId?: (index: number) => string;
  /** Accept ```json / untagged look-alike blocks as a fallback. Default `true`. */
  lenientFences?: boolean;
  /**
   * How tool instructions combine with an existing system message:
   * `"merge"` appends to the first system/developer message; `"prepend"` always
   * inserts a fresh system message at the front. Default: `"merge"`.
   */
  systemInjection?: "merge" | "prepend";
}

/** Extracts plain text from string-or-content-part-array message content. */
function textFromContent(
  content: string | ChatCompletionContentPart[] | null | undefined,
): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p) => p && p.type === "text" && typeof (p as any).text === "string")
      .map((p) => (p as any).text as string)
      .join("\n");
  }
  return "";
}

/**
 * Collapses a content-part array to a plain string **iff every part is text**.
 * Many tool-less upstreams reject OpenAI's array content form (some tokenize it
 * as 0 tokens and reject the request). Arrays with non-text parts (images, etc.)
 * are returned unchanged so multimodal requests keep working.
 */
function collapseTextContent(
  content: string | ChatCompletionContentPart[] | null | undefined,
): string | ChatCompletionContentPart[] | null | undefined {
  if (!Array.isArray(content) || content.length === 0) return content;
  const allText = content.every(
    (p) => p && p.type === "text" && typeof (p as any).text === "string",
  );
  if (!allText) return content;
  return content.map((p) => (p as any).text as string).join("\n");
}

/** Returns the message with text-only array content collapsed to a string. */
function normalizeOutboundMessage(
  msg: ChatCompletionMessageParam,
): ChatCompletionMessageParam {
  if ("content" in msg && Array.isArray(msg.content)) {
    const collapsed = collapseTextContent(msg.content);
    if (collapsed !== msg.content) {
      return { ...msg, content: collapsed } as ChatCompletionMessageParam;
    }
  }
  return msg;
}

function renderToolCallBlock(tag: string, name: string, argsString: string): string {
  const parsedArgs = tryParseJson(argsString);
  const argsValue = parsedArgs === undefined ? argsString : parsedArgs;
  const payload = JSON.stringify({ name, arguments: argsValue });
  return "```" + tag + "\n" + payload + "\n```";
}

function renderToolResultBlock(
  tag: string,
  name: string,
  id: string,
  result: string,
): string {
  const parsedResult = tryParseJson(result);
  const resultValue = parsedResult === undefined ? result : parsedResult;
  const payload = JSON.stringify({ name, tool_call_id: id, result: resultValue });
  return "```" + tag + "\n" + payload + "\n```";
}

/**
 * Rewrites a message history that uses native tool roles (assistant
 * `tool_calls`, `role: "tool"`) into plain text that follows the prompt
 * contract — because the underlying model understands neither. Consecutive
 * tool results are merged into one user message.
 */
export function flattenMessages(
  messages: ChatCompletionMessageParam[],
  opts: { toolCallTag: string; toolResultTag: string },
): ChatCompletionMessageParam[] {
  const out: ChatCompletionMessageParam[] = [];
  const idToName = new Map<string, string>();
  let pendingResults: string[] = [];

  // Pre-pass: map every tool_call id -> name first, so a tool result that
  // appears before its assistant message still renders the real tool name.
  for (const msg of messages) {
    if (msg.role === "assistant" && msg.tool_calls) {
      for (const tc of msg.tool_calls) idToName.set(tc.id, tc.function.name);
    }
  }

  const flushResults = () => {
    if (pendingResults.length > 0) {
      out.push({ role: "user", content: pendingResults.join("\n\n") });
      pendingResults = [];
    }
  };

  for (const msg of messages) {
    if (msg.role === "tool") {
      const name = idToName.get(msg.tool_call_id) ?? "tool";
      pendingResults.push(
        renderToolResultBlock(
          opts.toolResultTag,
          name,
          msg.tool_call_id,
          textFromContent(msg.content),
        ),
      );
      continue;
    }

    if (msg.role === "function") {
      // Deprecated function-role result — render like a tool result.
      pendingResults.push(
        renderToolResultBlock(
          opts.toolResultTag,
          msg.name,
          msg.name,
          textFromContent(msg.content),
        ),
      );
      continue;
    }

    flushResults();

    if (msg.role === "assistant" && msg.tool_calls && msg.tool_calls.length > 0) {
      const blocks: string[] = [];
      const text = textFromContent(msg.content);
      for (const tc of msg.tool_calls) {
        idToName.set(tc.id, tc.function.name);
        blocks.push(
          renderToolCallBlock(opts.toolCallTag, tc.function.name, tc.function.arguments),
        );
      }
      const content = [text, ...blocks].filter((s) => s.length > 0).join("\n\n");
      out.push({ role: "assistant", content });
      continue;
    }

    out.push(normalizeOutboundMessage(msg));
  }

  flushResults();
  return out;
}

/** Inserts the tool-instruction text as/into a system message. */
function injectSystemPrompt(
  messages: ChatCompletionMessageParam[],
  instruction: string,
  mode: "merge" | "prepend",
): ChatCompletionMessageParam[] {
  if (mode === "prepend") {
    return [{ role: "system", content: instruction }, ...messages];
  }
  const idx = messages.findIndex(
    (m) => m.role === "system" || m.role === "developer",
  );
  if (idx === -1) {
    return [{ role: "system", content: instruction }, ...messages];
  }
  const target = messages[idx]!;
  // Preserve multimodal array content (images etc.) by appending a text part
  // rather than flattening the whole message to a string.
  const newContent = Array.isArray(target.content)
    ? [...target.content, { type: "text", text: instruction } as ChatCompletionContentPart]
    : textFromContent(target.content) + "\n\n" + instruction;
  const merged = { ...target, content: newContent } as ChatCompletionMessageParam;
  const copy = messages.slice();
  copy[idx] = merged;
  return copy;
}

/** Builds a synthetic non-streamed response from the model's raw text. */
function transformResponse(
  res: ChatCompletion,
  opts: { toolCallTag: string; generateId?: (i: number) => string; lenientFences: boolean },
): ChatCompletion {
  const choices = res.choices.map((choice) => {
    const raw = choice.message?.content;
    if (typeof raw !== "string") return choice;
    const { content, toolCalls } = parseToolCalls(raw, {
      toolCallTag: opts.toolCallTag,
      generateId: opts.generateId,
      lenientFences: opts.lenientFences,
    });
    const message: ChatCompletionMessage = {
      role: "assistant",
      content: toolCalls.length > 0 ? content : (content ?? raw),
      refusal: choice.message.refusal ?? null,
    };
    if (toolCalls.length > 0) message.tool_calls = toolCalls;
    return {
      ...choice,
      message,
      finish_reason: toolCalls.length > 0 ? ("tool_calls" as const) : choice.finish_reason,
    };
  });
  return { ...res, choices };
}

function makeChunk(
  template: ChatCompletionChunk | undefined,
  delta: ChatCompletionChunkDelta,
  finishReason: ChatCompletionChunk["choices"][number]["finish_reason"] = null,
  usage?: ChatCompletionChunk["usage"],
): ChatCompletionChunk {
  return {
    id: template?.id ?? "chatcmpl-tool-shim",
    object: "chat.completion.chunk",
    created: template?.created ?? 0,
    model: template?.model ?? "",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
    ...(usage !== undefined ? { usage } : {}),
  };
}

async function* transformStream(
  source: AsyncIterable<ChatCompletionChunk>,
  opts: { toolCallTag: string; generateId?: (i: number) => string },
): AsyncIterable<ChatCompletionChunk> {
  const parser = new ToolCallStreamParser({
    toolCallTag: opts.toolCallTag,
    generateId: opts.generateId,
  });
  let template: ChatCompletionChunk | undefined;
  let started = false;

  for await (const chunk of source) {
    template = chunk;
    if (!started) {
      yield makeChunk(chunk, { role: "assistant" });
      started = true;
    }
    const delta = chunk.choices?.[0]?.delta;
    const text = delta?.content;
    if (typeof text === "string" && text.length > 0) {
      for (const d of parser.push(text)) yield makeChunk(chunk, d);
    }
  }

  if (!started) {
    // Empty upstream — still emit a well-formed role + stop pair.
    yield makeChunk(template, { role: "assistant" });
  }
  for (const d of parser.flush()) yield makeChunk(template, d);
  yield makeChunk(
    template,
    {},
    parser.toolCallCount > 0 ? "tool_calls" : "stop",
    template?.usage ?? undefined,
  );
}

/**
 * Wraps an OpenAI-compatible client so that `tools` / `tool_choice` work even
 * when the underlying model has no native function calling. The returned
 * client's `chat.completions.create` is a drop-in for OpenAI's: it accepts the
 * same params and returns OpenAI-shaped `tool_calls` (and streamed tool-call
 * deltas). When no `tools` are passed it is fully transparent.
 */
export function wrapToolSupport(
  client: ChatClientLike,
  options: WrapOptions = {},
): ToolCapableClient {
  const toolCallTag = options.toolCallTag ?? DEFAULT_TOOL_CALL_TAG;
  const toolResultTag = options.toolResultTag ?? DEFAULT_TOOL_RESULT_TAG;
  const lenientFences = options.lenientFences ?? true;
  const systemInjection = options.systemInjection ?? "merge";

  const promptOptions: PromptOptions = {
    toolCallTag,
    toolResultTag,
    includeExamples: options.includeExamples,
    template: options.template,
  };

  function create(
    body: ChatCompletionCreateParamsNonStreaming,
    requestOptions?: unknown,
  ): Promise<ChatCompletion>;
  function create(
    body: ChatCompletionCreateParamsStreaming,
    requestOptions?: unknown,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  function create(
    body: ChatCompletionCreateParams,
    requestOptions?: unknown,
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>>;
  async function create(
    body: ChatCompletionCreateParams,
    requestOptions?: unknown,
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>> {
    const { tools, tool_choice, parallel_tool_calls, messages, stream, ...rest } = body;

    // Always flatten native tool roles — the model can't read them otherwise.
    let outMessages = flattenMessages(messages, { toolCallTag, toolResultTag });

    const useTools = !!tools && tools.length > 0 && tool_choice !== "none";
    if (useTools) {
      // Prompted tool calling parses a single completion; n>1 would silently
      // drop the alternatives (and OpenAI forbids n>1 with tools anyway).
      const n = (rest as { n?: unknown }).n;
      if (typeof n === "number" && n > 1) {
        throw new ToolCapabilityError("n > 1 is not supported together with tools.");
      }
    }
    if (useTools) {
      const instruction =
        buildToolPrompt(tools!, promptOptions) +
        buildToolChoiceInstruction(tool_choice);
      outMessages = injectSystemPrompt(outMessages, instruction, systemInjection);
    }

    const outBound: ChatCompletionCreateParams = {
      ...rest,
      messages: outMessages,
      ...(stream !== undefined ? { stream } : {}),
    };

    if (stream) {
      const source = (await client.chat.completions.create(
        outBound,
        requestOptions,
      )) as AsyncIterable<ChatCompletionChunk>;
      if (!useTools) return source;
      return transformStream(source, { toolCallTag, generateId: options.generateId });
    }

    const res = (await client.chat.completions.create(
      outBound,
      requestOptions,
    )) as ChatCompletion;
    if (!useTools) return res;
    return transformResponse(res, {
      toolCallTag,
      generateId: options.generateId,
      lenientFences,
    });
  }

  return { chat: { completions: { create } } };
}
