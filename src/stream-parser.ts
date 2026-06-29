import type { ChatCompletionChunkDelta, ChatCompletionTool } from "./types.js";
import { DEFAULT_TOOL_CALL_TAG } from "./prompt.js";
import {
  extractNameArgs,
  tryParseJson,
  randomToolCallId,
  mapXmlToolCall,
  escapeRegExp,
  DEFAULT_REASONING_TAG,
} from "./parser.js";

export interface ReasoningStreamOptions {
  /** Tag whose contents are reasoning. Default: `think`. */
  reasoningTag?: string;
}

export interface ReasoningSplit {
  /** Text classified as ordinary content in this step. */
  content: string;
  /** Text classified as reasoning (inside `<think>…</think>`) in this step. */
  reasoning: string;
}

/**
 * Streaming splitter that routes `<think>…</think>` text to `reasoning` and
 * everything else to `content`, holding back only a partial marker that may be
 * split across deltas. Linear time; the held buffer is bounded by the marker
 * length. Pairs with {@link ToolCallStreamParser}: feed its `content` output
 * into the tool parser.
 */
export class ReasoningStreamParser {
  private readonly open: string;
  private readonly close: string;
  private hold = "";
  private inThink = false;

  constructor(options: ReasoningStreamOptions = {}) {
    const tag = options.reasoningTag ?? DEFAULT_REASONING_TAG;
    this.open = `<${tag}>`;
    this.close = `</${tag}>`;
  }

  /** Feed a raw text delta; returns the content/reasoning classified so far. */
  push(text: string): ReasoningSplit {
    return this.process(this.hold + text, false);
  }

  /** End of stream: flush any held partial marker as its current class. */
  flush(): ReasoningSplit {
    return this.process(this.hold, true);
  }

  private process(buf: string, final: boolean): ReasoningSplit {
    let content = "";
    let reasoning = "";
    for (;;) {
      const marker = this.inThink ? this.close : this.open;
      const i = buf.indexOf(marker);
      if (i !== -1) {
        const head = buf.slice(0, i);
        if (this.inThink) reasoning += head;
        else content += head;
        buf = buf.slice(i + marker.length);
        this.inThink = !this.inThink;
        continue;
      }
      // No complete marker: emit everything except a possible split-marker tail.
      const hold = final ? "" : trailingPartial(buf, marker);
      const head = buf.slice(0, buf.length - hold.length);
      if (this.inThink) reasoning += head;
      else content += head;
      this.hold = hold;
      return { content, reasoning };
    }
  }
}

/** Longest suffix of `buf` that is a prefix of `marker` (a possible split marker). */
function trailingPartial(buf: string, marker: string): string {
  const max = Math.min(buf.length, marker.length - 1);
  for (let n = max; n > 0; n--) {
    if (marker.startsWith(buf.slice(buf.length - n))) return buf.slice(buf.length - n);
  }
  return "";
}

export interface StreamParserOptions {
  /** Fence label that marks a tool call. Default: `tool_call`. */
  toolCallTag?: string;
  /** Generates the `id` for the Nth (global) streamed tool call. */
  generateId?: (index: number) => string;
  /**
   * Max bytes buffered while holding a partial fence / tool block. If exceeded,
   * the buffer is flushed as plain text and parsing resumes — this bounds memory
   * and CPU on malformed or unterminated input. Default: 1 MiB.
   */
  maxBufferBytes?: number;
  /**
   * Also parse native XML-style tool tags — an own-line `<toolName>…</toolName>`
   * block whose tag matches one of {@link StreamParserOptions.tools}. Off by
   * default. Tags appearing inside foreign code fences are not guarded against
   * in streaming mode (the non-streaming parser does guard); this is a
   * best-effort, opt-in path for models that emit tool calls as XML.
   */
  xmlToolCalls?: boolean;
  /** Tool definitions, used to scope and map XML tags. Required for `xmlToolCalls`. */
  tools?: ChatCompletionTool[];
}

type State = "text" | "in_tool";
type BlockKind = "fence" | "xml";

const DEFAULT_MAX_BUFFER = 1024 * 1024;

/**
 * Incrementally converts a stream of raw text deltas into OpenAI-shaped chunk
 * deltas. Prose is forwarded token-by-token as `content`; each tool-call block
 * is buffered until its closing marker, then emitted atomically as a complete
 * `tool_calls` delta (id + name + full arguments string). Emitting the
 * arguments in one piece avoids surfacing partial/invalid JSON mid-stream.
 *
 * Two block syntaxes are recognized, both as own-line markers: the ` ```tool_call `
 * fence (always) and, when `xmlToolCalls` is set, `<toolName>…</toolName>` tags.
 *
 * Closing-marker scanning is offset-tracked (each delta only re-scans the new
 * tail plus a small overlap), so total work is linear in the stream length, and
 * the buffer is size-capped to bound memory on unterminated/oversized blocks.
 */
export class ToolCallStreamParser {
  private readonly tag: string;
  private readonly generateId: (index: number) => string;
  private readonly maxBuffer: number;
  private readonly openRe: RegExp;
  private readonly xmlOpenRe: RegExp | null;
  private readonly xmlNames: string[];
  private readonly toolByName: Map<string, ChatCompletionTool>;

  private buf = "";
  private state: State = "text";
  private atLineStart = true;
  private openTicks = 3;
  private nextIndex = 0;
  private _toolCallCount = 0;
  private closeRe: RegExp | null = null;
  private flushStripRe: RegExp | null = null;
  private closeRewind = 0;
  private toolScanFrom = 0;
  private blockKind: BlockKind = "fence";
  private blockName = "";
  private blockOpenText = "";

  constructor(options: StreamParserOptions = {}) {
    this.tag = (options.toolCallTag ?? DEFAULT_TOOL_CALL_TAG).toLowerCase();
    this.generateId = options.generateId ?? randomToolCallId;
    this.maxBuffer = options.maxBufferBytes ?? DEFAULT_MAX_BUFFER;
    this.openRe = new RegExp(
      `^[ \\t]*(\`{3,})[ \\t]*${escapeRegExp(this.tag)}(?![\\w-])[^\\n]*\\r?\\n`,
    );
    this.toolByName = new Map(
      (options.xmlToolCalls && options.tools ? options.tools : []).map((t) => [
        t.function.name,
        t,
      ]),
    );
    this.xmlNames = [...this.toolByName.keys()].filter((n) => n.length > 0);
    this.xmlOpenRe =
      this.xmlNames.length > 0
        ? new RegExp(`^[ \\t]*<(${this.xmlNames.map(escapeRegExp).join("|")})>[ \\t]*\\r?\\n`)
        : null;
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

  /** Flushes the whole buffer as plain text (overflow / give-up path). */
  private flushBufAsText(out: ChatCompletionChunkDelta[]): void {
    if (this.buf.length > 0) out.push({ content: this.buf });
    this.buf = "";
    this.atLineStart = false;
  }

  /** Sets up the close/flush regexes for a fence block of `ticks` backticks. */
  private enterFenceBlock(ticks: number): void {
    this.state = "in_tool";
    this.blockKind = "fence";
    this.openTicks = ticks;
    this.closeRe = new RegExp(`(^|\\n)[ \\t]*\`{${ticks},}[ \\t]*\\r?\\n`, "g");
    this.flushStripRe = new RegExp(`(^|\\n)[ \\t]*\`{${ticks},}[ \\t]*\\r?\\n?\\s*$`);
    this.closeRewind = ticks + 8;
    this.atLineStart = true;
    this.toolScanFrom = 0;
  }

  /** Sets up the close/flush regexes for an XML `<name>…</name>` block. */
  private enterXmlBlock(name: string, openText: string): void {
    this.state = "in_tool";
    this.blockKind = "xml";
    this.blockName = name;
    this.blockOpenText = openText;
    const close = `<\\/${escapeRegExp(name)}>`;
    this.closeRe = new RegExp(`(^|\\n)[ \\t]*${close}[ \\t]*\\r?\\n`, "g");
    this.flushStripRe = new RegExp(`(^|\\n)[ \\t]*${close}[ \\t]*\\r?\\n?\\s*$`);
    this.closeRewind = name.length + 10;
    this.atLineStart = true;
    this.toolScanFrom = 0;
  }

  private exitBlock(): void {
    this.state = "text";
    this.atLineStart = true;
    this.closeRe = null;
    this.flushStripRe = null;
    this.toolScanFrom = 0;
  }

  // --- TEXT state ---------------------------------------------------------

  private stepText(out: ChatCompletionChunkDelta[], final: boolean): boolean {
    if (this.buf === "") return false;

    // At a line start, the buffer might be (the prefix of) an opening marker.
    if (this.atLineStart) {
      const open = this.matchOpenFence(this.buf);
      if (open > 0) {
        this.buf = this.buf.slice(open);
        this.enterFenceBlock(this.openTicks);
        return true;
      }
      if (open === -1) return this.holdOrFlush(out, final);

      if (this.xmlOpenRe) {
        const xopen = this.matchXmlOpen(this.buf);
        if (xopen > 0) {
          const openText = this.buf.slice(0, xopen);
          this.buf = this.buf.slice(xopen);
          this.enterXmlBlock(this.blockName, openText);
          return true;
        }
        if (xopen === -1) return this.holdOrFlush(out, final);
      }
      // Neither a fence nor an XML opener -> prose.
    }

    const nl = this.buf.indexOf("\n");
    if (nl === -1) {
      // No newline yet: the trailing text is the current (partial) line.
      if (
        this.atLineStart &&
        this.buf.length <= this.maxBuffer &&
        (this.isOpenFenceViablePrefix(this.buf) || this.isXmlOpenViablePrefix(this.buf))
      ) {
        // Could still become an opening marker — hold for more input.
        if (!final) return false;
      }
      this.flushBufAsText(out);
      return false;
    }

    // We have a complete line ending at `nl`.
    const line = this.buf.slice(0, nl + 1);
    out.push({ content: line });
    this.buf = this.buf.slice(nl + 1);
    this.atLineStart = true;
    return true;
  }

  /** Holds a viable-but-incomplete opening marker, or flushes it as text. */
  private holdOrFlush(out: ChatCompletionChunkDelta[], final: boolean): boolean {
    if (!final && this.buf.length <= this.maxBuffer) return false;
    this.flushBufAsText(out);
    return false;
  }

  // --- IN_TOOL state ------------------------------------------------------

  private stepTool(out: ChatCompletionChunkDelta[], final: boolean): boolean {
    const close = this.matchClose();
    if (close) {
      const inner = this.buf.slice(0, close.start);
      const closeText = this.buf.slice(close.start, close.end);
      this.emitBlock(inner, closeText, out);
      this.buf = this.buf.slice(close.end);
      this.exitBlock();
      return true;
    }
    if (final) {
      // Unterminated block at end of stream: drop a dangling closing marker (one
      // with no trailing newline) if present, then best-effort parse the rest.
      const inner = this.flushStripRe ? this.buf.replace(this.flushStripRe, "") : this.buf;
      this.emitBlock(inner, "", out);
      this.buf = "";
      this.exitBlock();
      return false;
    }
    // Oversized unterminated block: give up and emit it as text to bound memory.
    if (this.buf.length > this.maxBuffer) {
      this.flushBufAsText(out);
      this.exitBlock();
      return false;
    }
    // Wait for the closing marker.
    return false;
  }

  /** Emits zero or more tool calls for the just-closed block's inner text. */
  private emitBlock(inner: string, closeText: string, out: ChatCompletionChunkDelta[]): void {
    if (this.blockKind === "xml") {
      const na = mapXmlToolCall(this.blockName, inner, this.toolByName.get(this.blockName));
      if (na) {
        this.pushToolCall(na.name, na.arguments, out);
      } else if (this.blockOpenText.length + inner.length + closeText.length > 0) {
        // Unmappable XML block (e.g. a bare value on a multi-param tool): emit it
        // verbatim as content instead of dropping it — parity with the
        // non-streaming parser, which leaves such blocks in `content`.
        out.push({ content: this.blockOpenText + inner + closeText });
      }
      return;
    }
    const parsed = tryParseJson(inner);
    if (parsed === undefined) return;
    const items = Array.isArray(parsed) ? parsed : [parsed];
    for (const item of items) {
      const na = extractNameArgs(item);
      if (na) this.pushToolCall(na.name, na.arguments, out);
    }
  }

  private pushToolCall(
    name: string,
    args: string,
    out: ChatCompletionChunkDelta[],
  ): void {
    const index = this.nextIndex++;
    out.push({
      tool_calls: [
        {
          index,
          id: this.generateId(index),
          type: "function",
          function: { name, arguments: args },
        },
      ],
    });
    this._toolCallCount++;
  }

  // --- Fence/XML opener matching helpers ---------------------------------

  /**
   * If `text` begins with a complete opening fence line, returns the number of
   * characters to consume (through the trailing newline) and records the tick
   * count. Returns -1 if `text` is a viable but incomplete prefix of one, or 0
   * if it cannot be one.
   */
  private matchOpenFence(text: string): number {
    const m = this.openRe.exec(text);
    if (m) {
      this.openTicks = m[1]!.length;
      return m[0].length;
    }
    return this.isOpenFenceViablePrefix(text) ? -1 : 0;
  }

  /**
   * If `text` begins with a complete `<name>` opener line, returns the chars to
   * consume and records the name. Returns -1 for a viable prefix, 0 otherwise.
   */
  private matchXmlOpen(text: string): number {
    if (!this.xmlOpenRe) return 0;
    const m = this.xmlOpenRe.exec(text);
    if (m) {
      this.blockName = m[1]!;
      return m[0].length;
    }
    return this.isXmlOpenViablePrefix(text) ? -1 : 0;
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

  /** True if `p` (no newline reached) could still grow into an `<name>` opener. */
  private isXmlOpenViablePrefix(p: string): boolean {
    if (!this.xmlOpenRe) return false;
    const rest = p.replace(/^[ \t]*/, "");
    if (rest === "") return false; // covered by the fence prefix check
    if (rest[0] !== "<") return false;
    for (const name of this.xmlNames) {
      const token = `<${name}>`;
      if (token.startsWith(rest)) return true; // e.g. "<que" prefix of "<question>"
      if (rest.startsWith(token)) {
        // Full "<name>" seen — viable while only whitespace follows (await \n).
        if (/^[ \t]*$/.test(rest.slice(token.length))) return true;
      }
    }
    return false;
  }

  /**
   * Finds the closing marker in `this.buf`, scanning only the new tail (plus a
   * small overlap) since the last call so total work stays linear. Returns the
   * span to remove, or `null` if none is complete yet.
   */
  private matchClose(): { start: number; end: number } | null {
    const re = this.closeRe!;
    // Re-scan a small overlap so a marker split across deltas isn't missed. A
    // marker indented past the slack AND split exactly within that indent could
    // be missed mid-stream, but flush() then recovers it (its trailing regex
    // allows indentation), so the call is still emitted — just at end of stream.
    re.lastIndex = Math.max(0, this.toolScanFrom - this.closeRewind);
    const m = re.exec(this.buf);
    if (!m) {
      this.toolScanFrom = this.buf.length;
      return null;
    }
    const lead = m[1] === "\n" ? 1 : 0;
    return { start: m.index + lead, end: m.index + m[0].length };
  }
}
