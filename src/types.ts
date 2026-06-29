/**
 * Minimal, structurally OpenAI-compatible types.
 *
 * We deliberately define our own shapes instead of importing them from the
 * `openai` package: `openai` is an *optional* peer dependency, and the exact
 * type names/locations have shifted across v4/v5/v6. These shapes are a strict
 * structural subset of the official ones, so a real `OpenAI` client and its
 * responses remain assignable to/from them. See `test/type-compat.ts` for a
 * compile-time proof against the installed `openai` version.
 */

/** A JSON Schema object (loosely typed — we hand it to ajv as-is). */
export type JSONSchema = Record<string, unknown>;

/** `function` definition, matching OpenAI's `FunctionDefinition`. */
export interface FunctionDefinition {
  name: string;
  description?: string;
  /** JSON Schema describing the arguments. Omitting it means "no arguments". */
  parameters?: JSONSchema;
  strict?: boolean | null;
}

/** A function tool, matching OpenAI's `ChatCompletionFunctionTool`. */
export interface ChatCompletionTool {
  type: "function";
  function: FunctionDefinition;
}

/** Forces, allows, or forbids tool use — matching OpenAI's `tool_choice`. */
export type ChatCompletionToolChoiceOption =
  | "none"
  | "auto"
  | "required"
  | { type: "function"; function: { name: string } };

/** The `function` variant of a tool call (the only kind we emit). */
export interface ChatCompletionMessageToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    /** Arguments as a JSON **string** (may be invalid JSON — validate before use). */
    arguments: string;
  };
}

/** A piece of multimodal content. We pass these through untouched. */
export type ChatCompletionContentPart = { type: string; [key: string]: unknown };

export interface ChatCompletionSystemMessageParam {
  role: "system";
  content: string | ChatCompletionContentPart[];
  name?: string;
}

export interface ChatCompletionDeveloperMessageParam {
  role: "developer";
  content: string | ChatCompletionContentPart[];
  name?: string;
}

export interface ChatCompletionUserMessageParam {
  role: "user";
  content: string | ChatCompletionContentPart[];
  name?: string;
}

export interface ChatCompletionAssistantMessageParam {
  role: "assistant";
  content?: string | ChatCompletionContentPart[] | null;
  name?: string;
  tool_calls?: ChatCompletionMessageToolCall[];
}

export interface ChatCompletionToolMessageParam {
  role: "tool";
  content: string | ChatCompletionContentPart[];
  tool_call_id: string;
}

/** Deprecated `function` role message (kept for OpenAI assignability). */
export interface ChatCompletionFunctionMessageParam {
  role: "function";
  content: string | null;
  name: string;
}

export type ChatCompletionMessageParam =
  | ChatCompletionSystemMessageParam
  | ChatCompletionDeveloperMessageParam
  | ChatCompletionUserMessageParam
  | ChatCompletionAssistantMessageParam
  | ChatCompletionToolMessageParam
  | ChatCompletionFunctionMessageParam;

/** Assistant message as it appears in a (non-streamed) response. */
export interface ChatCompletionMessage {
  role: "assistant";
  content: string | null;
  refusal: string | null;
  tool_calls?: ChatCompletionMessageToolCall[];
  /**
   * Chain-of-thought text, separated from `content`. Populated from a model's
   * `<think>…</think>` block (or forwarded from an upstream `reasoning_content`
   * field). Matches the de-facto DeepSeek/OpenAI-compatible reasoning field.
   */
  reasoning_content?: string | null;
}

export type ChatCompletionFinishReason =
  | "stop"
  | "length"
  | "tool_calls"
  | "content_filter"
  | "function_call";

export interface ChatCompletionChoice {
  index: number;
  message: ChatCompletionMessage;
  finish_reason: ChatCompletionFinishReason;
  logprobs: unknown | null;
}

export interface CompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  [key: string]: unknown;
}

export interface ChatCompletion {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage?: CompletionUsage | null;
  [key: string]: unknown;
}

/** A streamed tool-call delta — accumulated by `index` on the consumer side. */
export interface ChatCompletionChunkToolCall {
  index: number;
  id?: string;
  type?: "function";
  function?: {
    name?: string;
    arguments?: string;
  };
}

export interface ChatCompletionChunkDelta {
  role?: "assistant" | "system" | "user" | "tool" | "developer";
  content?: string | null;
  refusal?: string | null;
  tool_calls?: ChatCompletionChunkToolCall[];
  /** Streamed chain-of-thought, separated from `content`. See {@link ChatCompletionMessage.reasoning_content}. */
  reasoning_content?: string | null;
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: ChatCompletionChunkDelta;
  finish_reason: ChatCompletionFinishReason | null;
  logprobs?: unknown | null;
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
  usage?: CompletionUsage | null;
  [key: string]: unknown;
}

/** The subset of OpenAI's `chat.completions.create` params we read or forward. */
export interface ChatCompletionCreateParams {
  model: string;
  messages: ChatCompletionMessageParam[];
  tools?: ChatCompletionTool[];
  tool_choice?: ChatCompletionToolChoiceOption;
  parallel_tool_calls?: boolean;
  stream?: boolean | null;
  /** Any other provider params (temperature, stop, etc.) are forwarded as-is. */
  [key: string]: unknown;
}

export type ChatCompletionCreateParamsNonStreaming = ChatCompletionCreateParams & {
  stream?: false | null;
};

export type ChatCompletionCreateParamsStreaming = ChatCompletionCreateParams & {
  stream: true;
};

/**
 * The minimal surface we need from an OpenAI-compatible client. The input
 * `create` is intentionally permissive so that any `openai` major version (and
 * compatible clients) satisfies it; the wrapper we return is precisely typed.
 */
export interface ChatClientLike {
  chat: {
    completions: {
      create(body: any, options?: any): any;
    };
  };
}

/** The precisely-typed `create` exposed by the wrapped client. */
export interface ToolCapableCompletions {
  create(
    body: ChatCompletionCreateParamsNonStreaming,
    options?: unknown,
  ): Promise<ChatCompletion>;
  create(
    body: ChatCompletionCreateParamsStreaming,
    options?: unknown,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  create(
    body: ChatCompletionCreateParams,
    options?: unknown,
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>>;
}

export interface ToolCapableClient {
  chat: {
    completions: ToolCapableCompletions;
  };
}
