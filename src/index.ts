// Layer A — drop-in OpenAI-compatible client with prompted tool calling.
export { wrapToolSupport, flattenMessages, type WrapOptions } from "./client.js";

// Layer B — agentic runner with handlers, validation, and error feedback.
export {
  createToolRunner,
  defineTool,
  type ToolRunner,
  type ToolRunnerOptions,
  type ToolHandler,
  type ToolHandlerContext,
  type DefinedTool,
  type RunParams,
  type RunResult,
  type RunnerEvent,
  type ToolExecution,
} from "./runner.js";

// Minimal fetch-based OpenAI-compatible client (no `openai` SDK needed).
export { createFetchClient, type FetchClientOptions } from "./upstream.js";

// Building blocks (for advanced use and custom pipelines).
export {
  buildToolPrompt,
  buildToolChoiceInstruction,
  DEFAULT_TOOL_CALL_TAG,
  DEFAULT_TOOL_RESULT_TAG,
  type PromptOptions,
} from "./prompt.js";
export {
  parseToolCalls,
  extractNameArgs,
  extractFencedBlocks,
  tryParseJson,
  randomToolCallId,
  extractReasoning,
  mapXmlToolCall,
  escapeRegExp,
  DEFAULT_REASONING_TAG,
  type ParseOptions,
  type ParseResult,
  type ExtractReasoningOptions,
  type ReasoningResult,
} from "./parser.js";
export {
  ToolCallStreamParser,
  ReasoningStreamParser,
  type StreamParserOptions,
  type ReasoningStreamOptions,
  type ReasoningSplit,
} from "./stream-parser.js";
export {
  ToolValidator,
  type ToolValidatorOptions,
  type ValidationResult,
} from "./validate.js";

// Errors.
export {
  ToolCapabilityError,
  MaxIterationsError,
  UnknownToolError,
  ToolArgumentsError,
} from "./errors.js";

// Types.
export type {
  JSONSchema,
  FunctionDefinition,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
  ChatCompletionMessageToolCall,
  ChatCompletionMessageParam,
  ChatCompletionFunctionMessageParam,
  ChatCompletionMessage,
  ChatCompletionFinishReason,
  ChatCompletionChoice,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionChunkChoice,
  ChatCompletionChunkDelta,
  ChatCompletionChunkToolCall,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatClientLike,
  ToolCapableClient,
} from "./types.js";
