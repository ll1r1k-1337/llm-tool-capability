import type { ChatCompletionMessageToolCall } from "./types.js";
import { DEFAULT_TOOL_CALL_TAG } from "./prompt.js";

export interface ParseOptions {
  /** Fence label that marks a tool call. Default: `tool_call`. */
  toolCallTag?: string;
  /** Generates the `id` for the Nth parsed tool call. */
  generateId?: (index: number) => string;
  /**
   * Also accept ```json or untagged fences whose JSON has `name` + `arguments`.
   * Conservative — only triggers when no correctly-tagged block is present.
   * Default: `true`.
   */
  lenientFences?: boolean;
}

export interface ParseResult {
  /** Prose with all tool-call blocks stripped, or `null` if nothing remains. */
  content: string | null;
  toolCalls: ChatCompletionMessageToolCall[];
}

interface FencedBlock {
  infoString: string;
  content: string;
  start: number;
  end: number;
}

const ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/** Generates an OpenAI-style `call_...` id. Not used when `generateId` is given. */
export function randomToolCallId(): string {
  let suffix = "";
  for (let i = 0; i < 24; i++) {
    suffix += ID_ALPHABET[Math.floor(Math.random() * ID_ALPHABET.length)];
  }
  return `call_${suffix}`;
}

/**
 * Scans text for fenced code blocks, returning each block's info string,
 * inner content, and span. Tolerates an unterminated final fence (treats the
 * remainder of the string as its content).
 */
export function extractFencedBlocks(text: string): FencedBlock[] {
  const blocks: FencedBlock[] = [];
  // Opening fence: >=3 backticks at a line start, then an optional info string.
  const fenceRe = /(^|\n)([ \t]*)(`{3,})([^\n`]*)\n?/g;
  let m: RegExpExecArray | null;
  while ((m = fenceRe.exec(text)) !== null) {
    const ticks = m[3]!;
    const infoString = (m[4] ?? "").trim();
    const contentStart = m.index + m[0]!.length;
    // Closing fence: same-or-more backticks at a line start (CRLF-tolerant).
    const closeRe = new RegExp(`\\n[ \\t]*\`{${ticks.length},}[ \\t]*\\r?(?=\\n|$)`, "g");
    closeRe.lastIndex = contentStart;
    const close = closeRe.exec(text);
    if (close) {
      blocks.push({
        infoString,
        content: text.slice(contentStart, close.index),
        start: m.index + (m[1] ? 1 : 0),
        end: close.index + close[0]!.length,
      });
      fenceRe.lastIndex = close.index + close[0]!.length;
    } else {
      // Unterminated fence — consume to end of string.
      blocks.push({
        infoString,
        content: text.slice(contentStart),
        start: m.index + (m[1] ? 1 : 0),
        end: text.length,
      });
      break;
    }
  }
  return blocks;
}

/** Best-effort JSON parse with light repair (trailing commas, // and /* comments). */
export function tryParseJson(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  try {
    return JSON.parse(trimmed);
  } catch {
    // fall through to repair
  }
  let repaired = trimmed
    // strip /* ... */ block comments
    .replace(/\/\*[\s\S]*?\*\//g, "")
    // strip // line comments — only at line start or after whitespace/structural
    // punctuation, so that "//cdn" inside a JSON string value is left intact.
    .replace(/(^|[\s,{[])\/\/[^\n]*/g, "$1")
    // strip trailing commas before } or ]
    .replace(/,(\s*[}\]])/g, "$1");
  try {
    return JSON.parse(repaired);
  } catch {
    return undefined;
  }
}

/**
 * Extracts a `{ name, arguments }` pair from a parsed candidate object, with
 * `arguments` normalized to a JSON string. Returns `null` if it isn't a call.
 * Shared by the batch parser and the streaming parser.
 */
export function extractNameArgs(
  parsed: unknown,
): { name: string; arguments: string } | null {
  if (parsed == null || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const obj = parsed as Record<string, unknown>;
  const name = obj.name;
  if (typeof name !== "string" || name.length === 0) return null;

  const rawArgs = "arguments" in obj ? obj.arguments : obj.parameters;
  let argString: string;
  if (rawArgs == null) {
    argString = "{}";
  } else if (typeof rawArgs === "string") {
    // Model already produced a JSON string; re-normalize if possible.
    const reparsed = tryParseJson(rawArgs);
    argString = reparsed === undefined ? rawArgs : JSON.stringify(reparsed);
  } else {
    argString = JSON.stringify(rawArgs);
  }

  return { name, arguments: argString };
}

/** Normalizes a parsed candidate into a tool call (or `null` if not one). */
function toToolCall(
  parsed: unknown,
  index: number,
  generateId: (index: number) => string,
): ChatCompletionMessageToolCall | null {
  const na = extractNameArgs(parsed);
  if (!na) return null;
  return { id: generateId(index), type: "function", function: na };
}

/** Expands a parsed block value into zero or more tool calls. */
function collectFromValue(
  parsed: unknown,
  startIndex: number,
  generateId: (index: number) => string,
): ChatCompletionMessageToolCall[] {
  if (Array.isArray(parsed)) {
    const out: ChatCompletionMessageToolCall[] = [];
    for (const item of parsed) {
      const call = toToolCall(item, startIndex + out.length, generateId);
      if (call) out.push(call);
    }
    return out;
  }
  const single = toToolCall(parsed, startIndex, generateId);
  return single ? [single] : [];
}

/**
 * Parses a model's text response into OpenAI-shaped tool calls plus the
 * remaining prose. Handles multiple calls, prose mixed with calls, malformed
 * JSON (light repair), and arrays of calls inside one block.
 */
export function parseToolCalls(text: string, options: ParseOptions = {}): ParseResult {
  const tag = (options.toolCallTag ?? DEFAULT_TOOL_CALL_TAG).toLowerCase();
  const generateId = options.generateId ?? randomToolCallId;
  const lenient = options.lenientFences ?? true;

  const blocks = extractFencedBlocks(text);
  const toolCalls: ChatCompletionMessageToolCall[] = [];
  const removedSpans: Array<[number, number]> = [];

  // Pass 1: correctly-tagged blocks.
  for (const block of blocks) {
    const info = block.infoString.toLowerCase();
    if (info === tag || info.startsWith(tag + " ")) {
      const parsed = tryParseJson(block.content);
      const calls = collectFromValue(parsed, toolCalls.length, generateId);
      if (calls.length > 0) {
        toolCalls.push(...calls);
        removedSpans.push([block.start, block.end]);
      }
    }
  }

  // Pass 2 (fallback): only if nothing matched, accept json/untagged look-alikes.
  if (toolCalls.length === 0 && lenient) {
    for (const block of blocks) {
      const info = block.infoString.toLowerCase();
      if (info !== "" && info !== "json") continue;
      const parsed = tryParseJson(block.content);
      const calls = collectFromValue(parsed, toolCalls.length, generateId);
      if (calls.length > 0) {
        toolCalls.push(...calls);
        removedSpans.push([block.start, block.end]);
      }
    }
  }

  // Build the cleaned prose by removing the consumed spans.
  let content = text;
  if (removedSpans.length > 0) {
    removedSpans.sort((a, b) => b[0] - a[0]);
    for (const [start, end] of removedSpans) {
      content = content.slice(0, start) + content.slice(end);
    }
  }
  content = content.trim();

  return { content: content.length > 0 ? content : null, toolCalls };
}
