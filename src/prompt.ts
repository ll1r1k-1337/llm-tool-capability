import type {
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from "./types.js";

export interface PromptOptions {
  /** Fence label the model uses to emit a call. Default: `tool_call`. */
  toolCallTag?: string;
  /** Fence label used when feeding results back. Default: `tool_result`. */
  toolResultTag?: string;
  /** Include a worked few-shot example in the prompt. Default: `true`. */
  includeExamples?: boolean;
  /**
   * Fully replace the generated instruction block. Receives the rendered tool
   * list and the resolved tags so you can build your own wording around them.
   */
  template?: (ctx: {
    tools: ChatCompletionTool[];
    renderedTools: string;
    toolCallTag: string;
    toolResultTag: string;
  }) => string;
}

export const DEFAULT_TOOL_CALL_TAG = "tool_call";
export const DEFAULT_TOOL_RESULT_TAG = "tool_result";

/** Neutralizes backticks/newlines in a tool name so it can't break markdown. */
function safeName(name: string): string {
  return name.replace(/[`\r\n]+/g, " ").trim();
}

/**
 * Breaks runs of >=3 backticks so a schema value / description can't close the
 * surrounding code fence. Inserts a zero-width space (U+200B, explicit escape so
 * a formatter can't silently strip it) between each backtick.
 */
function neutralizeFences(s: string): string {
  return s.replace(/`{3,}/g, (run) => run.split("").join("\u200b"));
}

/** Renders a single tool as a human/LLM-readable schema block. */
function renderTool(tool: ChatCompletionTool): string {
  const { name, description, parameters } = tool.function;
  const lines: string[] = [`### ${safeName(name)}`];
  if (description) lines.push(neutralizeFences(description));
  const schema = parameters ?? { type: "object", properties: {} };
  lines.push(
    "Parameters (JSON Schema):",
    "```json",
    neutralizeFences(JSON.stringify(schema, null, 2)),
    "```",
  );
  return lines.join("\n");
}

/** Renders the few-shot example used to anchor the output format. */
function renderExample(toolCallTag: string, toolResultTag: string): string {
  return [
    "Example exchange:",
    "",
    "User: What's the weather in Paris?",
    "",
    "Assistant:",
    "```" + toolCallTag,
    '{"name": "get_weather", "arguments": {"location": "Paris"}}',
    "```",
    "",
    "(The runtime executes the tool and replies with:)",
    "```" + toolResultTag,
    '{"name": "get_weather", "tool_call_id": "call_abc123", "result": {"temp_c": 18, "conditions": "cloudy"}}',
    "```",
    "",
    "Assistant: It's currently 18°C and cloudy in Paris.",
  ].join("\n");
}

/**
 * Builds the system-prompt instruction block that teaches a model without
 * native function calling how to request tools via fenced JSON blocks.
 */
export function buildToolPrompt(
  tools: ChatCompletionTool[],
  options: PromptOptions = {},
): string {
  const toolCallTag = options.toolCallTag ?? DEFAULT_TOOL_CALL_TAG;
  const toolResultTag = options.toolResultTag ?? DEFAULT_TOOL_RESULT_TAG;
  const includeExamples = options.includeExamples ?? true;

  const renderedTools = tools.map(renderTool).join("\n\n");

  if (options.template) {
    return options.template({ tools, renderedTools, toolCallTag, toolResultTag });
  }

  const parts: string[] = [
    "# Tool use",
    "",
    "You have access to the following tools. When a tool helps answer the user, " +
      "call it instead of guessing.",
    "",
    "## Available tools",
    "",
    renderedTools,
    "",
    "## How to call a tool",
    "",
    `To call one or more tools, emit a fenced code block tagged \`${toolCallTag}\` for ` +
      "each call, containing a single JSON object with this exact shape:",
    "",
    "```" + toolCallTag,
    '{"name": "<tool name>", "arguments": { /* JSON object matching the tool\'s schema */ }}',
    "```",
    "",
    "Rules:",
    `- The block MUST be valid JSON with exactly two keys: "name" and "arguments".`,
    `- "arguments" MUST be a JSON object (use {} if the tool takes no arguments).`,
    "- To call several tools at once, output several blocks back to back.",
    "- After emitting tool-call blocks, STOP. Do not invent or predict the tool's output.",
    `- Tool results are returned to you in \`${toolResultTag}\` blocks. Use them to continue.`,
    "- When you have enough information, reply normally with your final answer and NO " +
      `\`${toolCallTag}\` block.`,
    "- Only call tools from the list above, with exactly their listed names.",
  ];

  if (includeExamples) {
    parts.push("", "## Example", "", renderExample(toolCallTag, toolResultTag));
  }

  return parts.join("\n");
}

/**
 * Extra instruction appended to the prompt to honour a `tool_choice` of
 * `required` or a specific named function. Returns `""` for `auto`/`none`.
 */
export function buildToolChoiceInstruction(
  toolChoice: ChatCompletionToolChoiceOption | undefined,
): string {
  if (!toolChoice || toolChoice === "auto" || toolChoice === "none") return "";
  if (toolChoice === "required") {
    return "\n\nIMPORTANT: You MUST call at least one tool now. Do not answer directly.";
  }
  if (typeof toolChoice === "object" && toolChoice.type === "function") {
    return `\n\nIMPORTANT: You MUST call the tool \`${safeName(toolChoice.function.name)}\` now. Do not answer directly and do not call any other tool.`;
  }
  return "";
}
