import type {
  ChatClientLike,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParams,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
  FunctionDefinition,
  JSONSchema,
} from "./types.js";
import { wrapToolSupport, type WrapOptions } from "./client.js";
import { ToolValidator, type ToolValidatorOptions } from "./validate.js";
import { tryParseJson } from "./parser.js";
import { MaxIterationsError, ToolCapabilityError } from "./errors.js";

/** Context handed to a tool handler on each invocation. */
export interface ToolHandlerContext {
  toolCall: ChatCompletionMessageToolCall;
  /** The current transcript so far (do not mutate). */
  messages: ReadonlyArray<ChatCompletionMessageParam>;
  /** 1-based loop iteration in which this call was made. */
  iteration: number;
}

export type ToolHandler = (
  args: any,
  ctx: ToolHandlerContext,
) => unknown | Promise<unknown>;

/** A tool bundled with its executor. */
export interface DefinedTool {
  type: "function";
  function: FunctionDefinition;
  handler: ToolHandler;
}

/** Bundles a JSON-Schema tool definition with the function that runs it. */
export function defineTool(def: {
  name: string;
  description?: string;
  parameters?: JSONSchema;
  strict?: boolean | null;
  handler: ToolHandler;
}): DefinedTool {
  const fn: FunctionDefinition = { name: def.name };
  if (def.description !== undefined) fn.description = def.description;
  if (def.parameters !== undefined) fn.parameters = def.parameters;
  if (def.strict !== undefined) fn.strict = def.strict;
  return { type: "function", function: fn, handler: def.handler };
}

/** Record of a single tool execution within a run. */
export interface ToolExecution {
  id: string;
  name: string;
  /** Raw arguments JSON string as produced by the model. */
  arguments: string;
  /** Parsed arguments (undefined if parsing/validation failed). */
  args: unknown;
  /** The handler's return value (undefined on error). */
  result?: unknown;
  /** The string fed back to the model as the tool result. */
  content: string;
  isError: boolean;
  error?: string;
}

export interface ToolRunnerOptions extends WrapOptions {
  /** Tools (with handlers) available to the model. */
  tools: DefinedTool[];
  /** Hard ceiling on model<->tool round trips. Default: `10`. */
  maxIterations?: number;
  /**
   * Validate arguments against each tool's JSON Schema before calling it.
   * `false` disables schema checks (JSON parsing is still enforced).
   * Default: `true`.
   */
  validate?: boolean | ToolValidatorOptions;
  /** Throw `MaxIterationsError` instead of returning when the ceiling is hit. */
  throwOnMaxIterations?: boolean;
  /** Observability hooks. */
  onToolCall?: (call: ChatCompletionMessageToolCall, iteration: number) => void;
  onToolResult?: (execution: ToolExecution, iteration: number) => void;
}

export interface RunParams {
  model: string;
  messages: ChatCompletionMessageParam[];
  tool_choice?: ChatCompletionToolChoiceOption;
  /** Abort signal applied to every underlying request in the loop. */
  signal?: AbortSignal;
  /** Any other provider params (temperature, top_p, stop, …) are forwarded. */
  [key: string]: unknown;
}

export interface RunResult {
  /** Final assistant text once no more tools are requested. */
  content: string | null;
  /** Full transcript: input + assistant turns + tool results. */
  messages: ChatCompletionMessageParam[];
  /** Number of model calls made. */
  iterations: number;
  finishReason: "stop" | "max_iterations";
  /** Every tool execution performed during the run, in order. */
  toolExecutions: ToolExecution[];
}

export type RunnerEvent =
  | { type: "text"; delta: string }
  | { type: "tool_call"; toolCall: ChatCompletionMessageToolCall; iteration: number }
  | { type: "tool_result"; execution: ToolExecution; iteration: number }
  | { type: "iteration_end"; iteration: number; finishReason: "stop" | "tool_calls" }
  | {
      type: "final";
      content: string | null;
      messages: ChatCompletionMessageParam[];
      iterations: number;
      finishReason: "stop" | "max_iterations";
    };

function stringifyResult(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === undefined) return "null";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export interface ToolRunner {
  run(params: RunParams): Promise<RunResult>;
  runStream(params: RunParams): AsyncIterable<RunnerEvent>;
}

/**
 * Creates an agentic runner over an OpenAI-compatible client. It wraps the
 * client with tool support, then loops: ask the model -> parse tool calls ->
 * run handlers -> feed results back -> repeat, until the model answers without
 * calling a tool (or `maxIterations` is reached). Invalid/unknown/throwing
 * calls are fed back to the model as error text so it can recover.
 */
export function createToolRunner(
  client: ChatClientLike,
  options: ToolRunnerOptions,
): ToolRunner {
  const wrapped = wrapToolSupport(client, options);
  const maxIterations = options.maxIterations ?? 10;
  const schemaTools: ChatCompletionTool[] = options.tools.map((t) => ({
    type: "function",
    function: t.function,
  }));
  const handlers = new Map<string, ToolHandler>(
    options.tools.map((t) => [t.function.name, t.handler]),
  );

  const validationEnabled = options.validate !== false;
  const validatorOpts: ToolValidatorOptions =
    typeof options.validate === "object" ? options.validate : {};
  const validator = validationEnabled
    ? new ToolValidator(schemaTools, validatorOpts)
    : undefined;

  async function execute(
    toolCall: ChatCompletionMessageToolCall,
    messages: ChatCompletionMessageParam[],
    iteration: number,
  ): Promise<ToolExecution> {
    const name = toolCall.function.name;
    const argsString = toolCall.function.arguments;
    const base: ToolExecution = {
      id: toolCall.id,
      name,
      arguments: argsString,
      args: undefined,
      content: "",
      isError: false,
    };

    const handler = handlers.get(name);
    if (!handler) {
      const available = [...handlers.keys()].join(", ") || "(none)";
      base.isError = true;
      base.error = `Unknown tool "${name}".`;
      base.content = `Error: unknown tool "${name}". Available tools: ${available}.`;
      return base;
    }

    let args: unknown;
    if (validator) {
      const v = validator.validate(name, argsString);
      if (!v.valid) {
        base.isError = true;
        base.error = v.errors.join("; ");
        base.content = `Error: invalid arguments for "${name}": ${base.error}. Please fix the arguments and call the tool again.`;
        return base;
      }
      args = v.value;
    } else {
      const parsed = tryParseJson(argsString === "" ? "{}" : argsString);
      if (parsed === undefined) {
        base.isError = true;
        base.error = "Arguments are not valid JSON.";
        base.content = `Error: arguments for "${name}" are not valid JSON. Please resend a valid JSON object.`;
        return base;
      }
      args = parsed;
    }
    base.args = args;

    try {
      const result = await handler(args, { toolCall, messages, iteration });
      base.result = result;
      base.content = stringifyResult(result);
      return base;
    } catch (err) {
      base.isError = true;
      base.error = err instanceof Error ? err.message : String(err);
      base.content = `Error: tool "${name}" failed: ${base.error}`;
      return base;
    }
  }

  async function run(params: RunParams): Promise<RunResult> {
    const messages: ChatCompletionMessageParam[] = [...params.messages];
    const toolExecutions: ToolExecution[] = [];
    const { messages: _ignored, tool_choice, signal, ...restParams } = params;
    const reqOpts = signal ? { signal } : undefined;

    for (let iteration = 1; iteration <= maxIterations; iteration++) {
      const res = (await wrapped.chat.completions.create(
        {
          ...restParams,
          messages,
          tools: schemaTools,
          ...(tool_choice !== undefined ? { tool_choice } : {}),
          stream: false,
        },
        reqOpts,
      )) as ChatCompletion;

      if (!res.choices || res.choices.length === 0) {
        throw new ToolCapabilityError(
          "Model response contained no choices (likely an upstream API error).",
        );
      }
      const msg = res.choices[0]!.message;
      const toolCalls = msg?.tool_calls ?? [];
      const assistant: ChatCompletionMessageParam = {
        role: "assistant",
        content: msg?.content ?? null,
      };
      if (toolCalls.length > 0) assistant.tool_calls = toolCalls;
      messages.push(assistant);

      if (toolCalls.length === 0) {
        const finalText =
          typeof msg?.content === "string" && msg.content.length > 0
            ? msg.content
            : null;
        return {
          content: finalText,
          messages,
          iterations: iteration,
          finishReason: "stop",
          toolExecutions,
        };
      }

      for (const toolCall of toolCalls) {
        options.onToolCall?.(toolCall, iteration);
        const execution = await execute(toolCall, messages, iteration);
        toolExecutions.push(execution);
        options.onToolResult?.(execution, iteration);
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: execution.content,
        });
      }
    }

    if (options.throwOnMaxIterations) throw new MaxIterationsError(maxIterations);
    const last = [...messages].reverse().find((m) => m.role === "assistant");
    return {
      content: last && typeof last.content === "string" ? last.content : null,
      messages,
      iterations: maxIterations,
      finishReason: "max_iterations",
      toolExecutions,
    };
  }

  async function* runStream(params: RunParams): AsyncIterable<RunnerEvent> {
    const messages: ChatCompletionMessageParam[] = [...params.messages];
    const { messages: _ignored, tool_choice, signal, ...restParams } = params;
    const reqOpts = signal ? { signal } : undefined;

    for (let iteration = 1; iteration <= maxIterations; iteration++) {
      const stream = (await wrapped.chat.completions.create(
        {
          ...restParams,
          messages,
          tools: schemaTools,
          ...(tool_choice !== undefined ? { tool_choice } : {}),
          stream: true,
        },
        reqOpts,
      )) as AsyncIterable<ChatCompletionChunk>;

      let text = "";
      const acc = new Map<number, ChatCompletionMessageToolCall>();

      for await (const chunk of stream) {
        const delta = chunk.choices?.[0]?.delta;
        if (!delta) continue;
        if (typeof delta.content === "string" && delta.content.length > 0) {
          text += delta.content;
          yield { type: "text", delta: delta.content };
        }
        for (const tc of delta.tool_calls ?? []) {
          const existing = acc.get(tc.index) ?? {
            id: tc.id ?? "",
            type: "function" as const,
            function: { name: "", arguments: "" },
          };
          if (tc.id) existing.id = tc.id;
          if (tc.function?.name) existing.function.name += tc.function.name;
          if (tc.function?.arguments) existing.function.arguments += tc.function.arguments;
          acc.set(tc.index, existing);
        }
      }

      const toolCalls = [...acc.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([, v]) => v);

      const assistant: ChatCompletionMessageParam = {
        role: "assistant",
        content: text.length > 0 ? text : null,
      };
      if (toolCalls.length > 0) assistant.tool_calls = toolCalls;
      messages.push(assistant);

      if (toolCalls.length === 0) {
        yield { type: "iteration_end", iteration, finishReason: "stop" };
        yield {
          type: "final",
          content: text.length > 0 ? text : null,
          messages,
          iterations: iteration,
          finishReason: "stop",
        };
        return;
      }

      yield { type: "iteration_end", iteration, finishReason: "tool_calls" };

      for (const toolCall of toolCalls) {
        options.onToolCall?.(toolCall, iteration);
        yield { type: "tool_call", toolCall, iteration };
        const execution = await execute(toolCall, messages, iteration);
        options.onToolResult?.(execution, iteration);
        yield { type: "tool_result", execution, iteration };
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: execution.content,
        });
      }
    }

    if (options.throwOnMaxIterations) throw new MaxIterationsError(maxIterations);
    const last = [...messages].reverse().find((m) => m.role === "assistant");
    yield {
      type: "final",
      content: last && typeof last.content === "string" ? last.content : null,
      messages,
      iterations: maxIterations,
      finishReason: "max_iterations",
    };
  }

  return { run, runStream };
}
