import type { ChatCompletionMessageToolCall, ChatCompletionTool } from "./types.js";
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
  /**
   * Also parse native XML-style tool tags — an own-line `<toolName>…</toolName>`
   * block whose tag matches one of {@link ParseOptions.tools} — into tool calls.
   * Off by default (would risk false positives on models that don't use it).
   */
  xmlToolCalls?: boolean;
  /** Tool definitions, used to scope and map XML tags. Required for `xmlToolCalls`. */
  tools?: ChatCompletionTool[];
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

/** Escapes a string for safe interpolation into a RegExp source. */
export function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export const DEFAULT_REASONING_TAG = "think";

export interface ExtractReasoningOptions {
  /** Tag whose contents are reasoning. Default: `think`. */
  reasoningTag?: string;
}

export interface ReasoningResult {
  /** Concatenated reasoning text, or `null` if no reasoning block was found. */
  reasoning: string | null;
  /** The input with every reasoning block removed (and trimmed). */
  content: string;
}

/**
 * Splits `<think>…</think>` reasoning blocks out of a model's text, returning
 * the concatenated reasoning and the remaining content. Only matched open/close
 * pairs are removed — an unterminated `<think>` is left untouched, so a
 * forgotten closing tag can never swallow a tool call or the whole answer. Tag
 * matching is case-sensitive (matching the streaming `ReasoningStreamParser`).
 *
 * Implemented with `indexOf` (not a lazy regex) so work stays linear in the
 * input length — a `<think>`-heavy input with no closers can't backtrack.
 */
export function extractReasoning(
  text: string,
  options: ExtractReasoningOptions = {},
): ReasoningResult {
  const tag = options.reasoningTag ?? DEFAULT_REASONING_TAG;
  const open = `<${tag}>`;
  const close = `</${tag}>`;
  const reasoning: string[] = [];
  let content = "";
  let pos = 0;
  for (;;) {
    const start = text.indexOf(open, pos);
    if (start === -1) break;
    const end = text.indexOf(close, start + open.length);
    if (end === -1) break; // unterminated: leave the rest (incl. the open tag) intact
    content += text.slice(pos, start);
    const inner = text.slice(start + open.length, end).trim();
    if (inner.length > 0) reasoning.push(inner);
    pos = end + close.length;
  }
  if (pos === 0) return { reasoning: null, content: text };
  content += text.slice(pos);
  return {
    reasoning: reasoning.length > 0 ? reasoning.join("\n\n") : null,
    content: content.trim(),
  };
}

/**
 * Determines which single parameter an XML tag's bare (non-object) payload maps
 * to: the sole required parameter, else the sole parameter. Returns `null` when
 * the mapping is ambiguous (zero or several candidates).
 */
function targetXmlParam(tool: ChatCompletionTool | undefined): string | null {
  const schema = tool?.function.parameters as Record<string, unknown> | undefined;
  if (!schema || typeof schema !== "object") return null;
  const required = schema.required;
  if (Array.isArray(required) && required.length === 1 && typeof required[0] === "string") {
    return required[0];
  }
  const props = schema.properties;
  if (props && typeof props === "object") {
    const keys = Object.keys(props as Record<string, unknown>);
    if (keys.length === 1) return keys[0]!;
  }
  return null;
}

/**
 * Maps a native XML tool tag (`<name>inner</name>`) to a tool call. The tag name
 * is the tool; the inner payload becomes the arguments:
 * - a JSON **object** is taken as the arguments object directly (a `{name,
 *   arguments}` envelope is unwrapped);
 * - a JSON **array/scalar** or non-JSON text is wrapped as the tool's single
 *   target parameter (e.g. `<question>[…]</question>` → `{ "questions": […] }`).
 *
 * Returns `null` when the payload can't be mapped (ambiguous multi-param tool).
 */
export function mapXmlToolCall(
  name: string,
  inner: string,
  tool: ChatCompletionTool | undefined,
): { name: string; arguments: string } | null {
  const parsed = tryParseJson(inner);
  if (parsed != null && typeof parsed === "object" && !Array.isArray(parsed)) {
    const obj = parsed as Record<string, unknown>;
    if (typeof obj.name === "string" && ("arguments" in obj || "parameters" in obj)) {
      const na = extractNameArgs(obj);
      if (na) return { name, arguments: na.arguments };
    }
    return { name, arguments: JSON.stringify(parsed) };
  }
  const trimmed = inner.trim();
  if (trimmed === "") return { name, arguments: "{}" };
  const param = targetXmlParam(tool);
  if (!param) return null;
  const value = parsed === undefined ? trimmed : parsed;
  return { name, arguments: JSON.stringify({ [param]: value }) };
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
/** Max input length eligible for regex-based repair (bounds ReDoS risk). */
const MAX_REPAIR_LENGTH = 256 * 1024;

export function tryParseJson(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  try {
    return JSON.parse(trimmed);
  } catch {
    // fall through to repair
  }
  // Valid JSON is handled above; the repair regexes can backtrack quadratically
  // on adversarial input, so only attempt them on reasonably small strings.
  if (trimmed.length > MAX_REPAIR_LENGTH) return undefined;
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

  // Pass 3 (opt-in): native XML tool tags `<name>…</name>` (own-line), scoped to
  // the provided tool names and skipped inside fenced code blocks. Walked
  // line-by-line (not one lazy regex) so work stays linear in the input length —
  // a `<name>`-heavy input with no closers can't trigger quadratic backtracking.
  if (options.xmlToolCalls && options.tools && options.tools.length > 0) {
    const toolByName = new Map(options.tools.map((t) => [t.function.name, t]));
    const names = [...toolByName.keys()].filter((n) => n.length > 0);
    if (names.length > 0) {
      const alt = names.map(escapeRegExp).join("|");
      const openLineRe = new RegExp(`^[ \\t]*<(${alt})>[ \\t]*\\r?$`);
      const closeLineRe = new RegExp(`^[ \\t]*<\\/(${alt})>[ \\t]*\\r?$`);
      let pendingName: string | null = null;
      let blockStart = 0;
      let innerStart = 0;
      let pos = 0;
      for (;;) {
        const nl = text.indexOf("\n", pos);
        const lineEnd = nl === -1 ? text.length : nl;
        const line = text.slice(pos, lineEnd);
        if (pendingName === null) {
          const m = openLineRe.exec(line);
          // Skip openers that fall inside a fenced code block (false positives).
          if (m && !blocks.some((b) => pos < b.end && b.start < lineEnd)) {
            pendingName = m[1]!;
            blockStart = pos;
            innerStart = nl === -1 ? text.length : nl + 1;
          }
        } else {
          const cm = closeLineRe.exec(line);
          if (cm && cm[1] === pendingName) {
            const na = mapXmlToolCall(
              pendingName,
              text.slice(innerStart, pos),
              toolByName.get(pendingName),
            );
            if (na) {
              toolCalls.push({ id: generateId(toolCalls.length), type: "function", function: na });
              removedSpans.push([blockStart, lineEnd]);
            }
            pendingName = null;
          }
        }
        if (nl === -1) break;
        pos = nl + 1;
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
