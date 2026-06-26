import type { ChatCompletionChunkDelta } from "./types.js";
import { DEFAULT_TOOL_CALL_TAG } from "./prompt.js";
import { extractNameArgs, tryParseJson, randomToolCallId } from "./parser.js";

export interface StreamParserOptions {
  /** Fence label that marks a tool call. Default: `tool_call`. */
  toolCallTag?: string;
  /** Generates the `id` for the Nth (global) streamed tool call. */
  generateId?: (index: number) => string;
}

type State = "text" | "in_tool";

/**
 * Incrementally converts a stream of raw text deltas into OpenAI-shaped chunk
 * deltas. Prose is forwarded token-by-token as `content`; each tool-call block
 * is buffered until its closing fence, then emitted atomically as a complete
 * `tool_calls` delta (id + name + full arguments string). Emitting the
 * arguments in one piece avoids surfacing partial/invalid JSON mid-stream.
 */
export class ToolCallStreamParser {
  private readonly tag: string;
  private readonly generateId: (index: number) => string;

  private buf = "";
  private state: State = "text";
  private atLineStart = true;
  private openTicks = 3;
  private nextIndex = 0;
  private _toolCallCount = 0;

  constructor(options: StreamParserOptions = {}) {
    this.tag = (options.toolCallTag ?? DEFAULT_TOOL_CALL_TAG).toLowerCase();
    this.generateId = options.generateId ?? randomToolCallId;
  }

  get toolCallCount(): number {
    return this._toolCallCount;
  }

  /** Feed a raw text delta; returns zero or more chunk deltas to emit. */
  push(textDelta: string): ChatCompletionChunkDelta[] {
    if (textDelta) this.buf += textDelta;
    return this.drain(false);
  }

  /** Signal end of stream; flushes any buffered prose or unterminated block. */
  flush(): ChatCompletionChunkDelta[] {
    return this.drain(true);
  }

  private drain(final: boolean): ChatCompletionChunkDelta[] {
    const out: ChatCompletionChunkDelta[] = [];
    // Each step returns true only after consuming buffer, so this terminates.
    let progress = true;
    while (progress) {
      progress =
        this.state === "text" ? this.stepText(out, final) : this.stepTool(out, final);
    }
    return out;
  }

  // --- TEXT state ---------------------------------------------------------

  private stepText(out: ChatCompletionChunkDelta[], final: boolean): boolean {
    if (this.buf === "") return false;

    // At a line start, the buffer might be (the prefix of) an opening fence.
    if (this.atLineStart) {
      const open = this.matchOpenFence(this.buf);
      if (open > 0) {
        this.buf = this.buf.slice(open);
        this.state = "in_tool";
        this.atLineStart = true;
        return true;
      }
      if (open === -1) {
        // Viable but incomplete opening fence: hold unless the stream ended.
        if (!final) return false;
        // On flush, an incomplete fence is just leftover text — emit it.
        out.push({ content: this.buf });
        this.buf = "";
        this.atLineStart = false;
        return false;
      }
      // open === 0 -> not a fence; fall through to prose handling.
    }

    const nl = this.buf.indexOf("\n");
    if (nl === -1) {
      // No newline yet: the trailing text is the current (partial) line.
      if (this.atLineStart && this.isOpenFenceViablePrefix(this.buf)) {
        // Could still become an opening fence — hold for more input.
        if (!final) return false;
      }
      if (this.buf.length === 0) return false;
      out.push({ content: this.buf });
      this.buf = "";
      this.atLineStart = false;
      return false;
    }

    // We have a complete line ending at `nl`.
    const line = this.buf.slice(0, nl + 1);
    out.push({ content: line });
    this.buf = this.buf.slice(nl + 1);
    this.atLineStart = true;
    return true;
  }

  // --- IN_TOOL state ------------------------------------------------------

  private stepTool(out: ChatCompletionChunkDelta[], final: boolean): boolean {
    const close = this.matchCloseFence(this.buf, this.openTicks);
    if (close) {
      const inner = this.buf.slice(0, close.start);
      this.emitToolCalls(inner, out);
      this.buf = this.buf.slice(close.end);
      this.state = "text";
      this.atLineStart = true;
      return true;
    }
    if (final) {
      // Unterminated block at end of stream: drop a dangling closing fence (one
      // with no trailing newline) if present, then best-effort parse the rest.
      const trailing = new RegExp(
        `(^|\\n)[ \\t]*\`{${this.openTicks},}[ \\t]*\\r?\\n?\\s*$`,
      );
      const inner = this.buf.replace(trailing, "");
      this.emitToolCalls(inner, out);
      this.buf = "";
      this.state = "text";
      return false;
    }
    // Wait for the closing fence.
    return false;
  }

  private emitToolCalls(inner: string, out: ChatCompletionChunkDelta[]): void {
    const parsed = tryParseJson(inner);
    if (parsed === undefined) return;
    const items = Array.isArray(parsed) ? parsed : [parsed];
    for (const item of items) {
      const na = extractNameArgs(item);
      if (!na) continue;
      const index = this.nextIndex++;
      out.push({
        tool_calls: [
          {
            index,
            id: this.generateId(index),
            type: "function",
            function: { name: na.name, arguments: na.arguments },
          },
        ],
      });
      this._toolCallCount++;
    }
  }

  // --- Fence matching helpers --------------------------------------------

  /**
   * If `text` begins with a complete opening fence line, returns the number of
   * characters to consume (through the trailing newline). Returns -1 if `text`
   * is a viable but incomplete prefix of one, or 0 if it cannot be one.
   */
  private matchOpenFence(text: string): number {
    const re = new RegExp(
      `^[ \\t]*(\`{3,})[ \\t]*${escapeRegExp(this.tag)}(?![\\w-])[^\\n]*\\r?\\n`,
    );
    const m = re.exec(text);
    if (m) {
      this.openTicks = m[1]!.length;
      return m[0].length;
    }
    return this.isOpenFenceViablePrefix(text) ? -1 : 0;
  }

  /** True if `p` (no newline reached) could still grow into an opening fence. */
  private isOpenFenceViablePrefix(p: string): boolean {
    const rest = p.replace(/^[ \t]*/, "");
    if (rest === "") return true;
    const ticks = /^`+/.exec(rest);
    const tickCount = ticks ? ticks[0].length : 0;
    if (tickCount === 0) return false;
    let rest2 = rest.slice(tickCount);
    if (tickCount < 3) {
      // Need more backticks; only viable if nothing follows them yet.
      return rest2 === "";
    }
    rest2 = rest2.replace(/^[ \t]*/, "");
    if (rest2 === "") return true;
    // rest2 must be a prefix of the tag, or the tag (+ trailing info string).
    if (this.tag.startsWith(rest2)) return true;
    if (rest2.startsWith(this.tag)) {
      const after = rest2[this.tag.length];
      // A word/dash char right after the tag would make it a different tag.
      return after === undefined || !/[\w-]/.test(after);
    }
    return false;
  }

  /**
   * Finds a closing fence (>= `ticks` backticks alone on a line) terminated by
   * a newline. Returns the span to remove, or `null` if none is complete yet.
   */
  private matchCloseFence(
    text: string,
    ticks: number,
  ): { start: number; end: number } | null {
    const re = new RegExp(`(^|\\n)[ \\t]*\`{${ticks},}[ \\t]*\\r?\\n`, "g");
    const m = re.exec(text);
    if (!m) return null;
    const lead = m[1] === "\n" ? 1 : 0;
    return { start: m.index + lead, end: m.index + m[0].length };
  }
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
