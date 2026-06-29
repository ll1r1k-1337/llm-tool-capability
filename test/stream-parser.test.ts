import { describe, it, expect } from "vitest";
import { ToolCallStreamParser, ReasoningStreamParser } from "../src/index.js";
import type {
  ChatCompletionChunkDelta,
  ChatCompletionTool,
  StreamParserOptions,
} from "../src/index.js";
import { seqIds } from "./helpers.js";

function run(deltas: string[], options: Partial<StreamParserOptions> = {}) {
  const p = new ToolCallStreamParser({ generateId: seqIds(), ...options });
  const emitted: ChatCompletionChunkDelta[] = [];
  for (const d of deltas) emitted.push(...p.push(d));
  emitted.push(...p.flush());
  const content = emitted.map((e) => e.content ?? "").join("");
  const toolCalls = emitted.flatMap((e) => e.tool_calls ?? []);
  return { content, toolCalls, emitted, count: p.toolCallCount };
}

function runReasoning(deltas: string[]) {
  const p = new ReasoningStreamParser();
  let content = "";
  let reasoning = "";
  for (const d of deltas) {
    const r = p.push(d);
    content += r.content;
    reasoning += r.reasoning;
  }
  const f = p.flush();
  content += f.content;
  reasoning += f.reasoning;
  return { content, reasoning };
}

const block = '```tool_call\n{"name":"get_weather","arguments":{"city":"Paris"}}\n```';

const questionTool: ChatCompletionTool = {
  type: "function",
  function: {
    name: "question",
    parameters: {
      type: "object",
      properties: { questions: { type: "array", items: { type: "object" } } },
      required: ["questions"],
    },
  },
};

const editTool: ChatCompletionTool = {
  type: "function",
  function: {
    name: "edit",
    parameters: {
      type: "object",
      properties: {
        filePath: { type: "string" },
        oldString: { type: "string" },
        newString: { type: "string" },
      },
      required: ["filePath", "oldString", "newString"],
    },
  },
};

describe("ToolCallStreamParser", () => {
  it("parses a single block fed in one push", () => {
    const { content, toolCalls } = run([block]);
    expect(content).toBe("");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]).toEqual({
      index: 0,
      id: "call_0",
      type: "function",
      function: { name: "get_weather", arguments: '{"city":"Paris"}' },
    });
  });

  it("produces the same result fed character by character", () => {
    const { content, toolCalls } = run(block.split(""));
    expect(content).toBe("");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function).toEqual({
      name: "get_weather",
      arguments: '{"city":"Paris"}',
    });
  });

  it("handles the opening fence split across deltas", () => {
    const { toolCalls } = run([
      "``",
      "`tool_",
      'call\n{"name":"a",',
      '"arguments":{}}\n',
      "```",
    ]);
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function!.name).toBe("a");
  });

  it("streams prose token-by-token and keeps it as content", () => {
    const { content, toolCalls, emitted } = run("Hello world".split(""));
    expect(content).toBe("Hello world");
    expect(toolCalls).toEqual([]);
    // Prose is forwarded incrementally, not buffered into one piece.
    expect(emitted.filter((e) => e.content).length).toBeGreaterThan(1);
  });

  it("handles prose interleaved with a tool call", () => {
    const { content, toolCalls } = run([
      "Let me check.\n",
      block,
      "\nAll done.",
    ]);
    expect(toolCalls).toHaveLength(1);
    expect(content).toContain("Let me check.");
    expect(content).toContain("All done.");
  });

  it("emits multiple tool calls with increasing index", () => {
    const two =
      '```tool_call\n{"name":"a","arguments":{}}\n```\n' +
      '```tool_call\n{"name":"b","arguments":{}}\n```';
    const { toolCalls } = run([two]);
    expect(toolCalls.map((t) => t.index)).toEqual([0, 1]);
    expect(toolCalls.map((t) => t.function!.name)).toEqual(["a", "b"]);
  });

  it("best-effort parses an unterminated block at flush", () => {
    const { toolCalls } = run(['```tool_call\n{"name":"a","arguments":{"x":1}}']);
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function!.arguments).toBe('{"x":1}');
  });

  it("parses a block whose closing fence has no trailing newline", () => {
    const { toolCalls } = run(['```tool_call\n{"name":"a","arguments":{}}\n```']);
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function!.name).toBe("a");
  });

  it("passes a foreign code block through as content", () => {
    const { content, toolCalls } = run(["before\n```js\nconst x = 1\n```\nafter"]);
    expect(toolCalls).toEqual([]);
    expect(content).toContain("const x = 1");
    expect(content).toContain("```js");
  });

  it("reports the tool-call count", () => {
    const { count } = run([block]);
    expect(count).toBe(1);
  });

  it("bounds memory on an oversized unterminated tool block", () => {
    const p = new ToolCallStreamParser({ generateId: seqIds(), maxBufferBytes: 64 });
    const emitted: ChatCompletionChunkDelta[] = [];
    emitted.push(...p.push("```tool_call\n"));
    emitted.push(...p.push("x".repeat(500))); // exceeds cap, no closing fence
    emitted.push(...p.flush());
    const content = emitted.map((e) => e.content ?? "").join("");
    // Degrades to plain text rather than buffering unboundedly / hanging.
    expect(content).toContain("x");
    expect(p.toolCallCount).toBe(0);
  });
});

describe("ToolCallStreamParser — native XML tags (opt-in)", () => {
  const xmlOpts = { xmlToolCalls: true, tools: [questionTool, editTool] };

  it("parses an own-line <question> block in one push", () => {
    const text = "<question>\n[{\"question\":\"a\"}]\n</question>";
    const { toolCalls, content } = run([text], xmlOpts);
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function!.name).toBe("question");
    expect(JSON.parse(toolCalls[0]!.function!.arguments!)).toEqual({
      questions: [{ question: "a" }],
    });
    expect(content).toBe("");
  });

  it("produces the same result fed character by character", () => {
    const text = "Sure.\n<question>\n[{\"question\":\"a\"}]\n</question>\n";
    const { toolCalls, content } = run(text.split(""), xmlOpts);
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function!.name).toBe("question");
    expect(content).toContain("Sure.");
  });

  it("handles the opening tag split across deltas", () => {
    const { toolCalls } = run(
      ["<que", "stion>\n", "[{\"question\":\"a\"}]\n", "</question>\n"],
      xmlOpts,
    );
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function!.name).toBe("question");
  });

  it("does not treat XML tags as calls when xmlToolCalls is off", () => {
    const text = "<question>\n[{\"question\":\"a\"}]\n</question>";
    const { toolCalls, content } = run([text]);
    expect(toolCalls).toEqual([]);
    expect(content).toContain("<question>");
  });

  it("leaves a same-line <question> mention as prose (own-line only)", () => {
    const { toolCalls, content } = run(["use the <question> tool wisely\n"], xmlOpts);
    expect(toolCalls).toEqual([]);
    expect(content).toContain("<question>");
  });

  it("does not match an unknown tag name", () => {
    const { toolCalls, content } = run(["<unknown>\nhi\n</unknown>\n"], xmlOpts);
    expect(toolCalls).toEqual([]);
    expect(content).toContain("<unknown>");
  });

  it("best-effort parses an unterminated XML block at flush", () => {
    const { toolCalls } = run(["<question>\n[{\"question\":\"a\"}]\n"], xmlOpts);
    expect(toolCalls).toHaveLength(1);
    expect(JSON.parse(toolCalls[0]!.function!.arguments!)).toEqual({
      questions: [{ question: "a" }],
    });
  });

  it("keeps an unmappable XML block as content instead of dropping it", () => {
    // A bare value on a multi-param tool can't be mapped — must not be lost.
    const { toolCalls, content } = run(["<edit>\nbare text\n</edit>\n"], xmlOpts);
    expect(toolCalls).toEqual([]);
    expect(content).toContain("<edit>");
    expect(content).toContain("bare text");
    expect(content).toContain("</edit>");
  });
});

describe("ReasoningStreamParser", () => {
  it("routes <think> content to reasoning and the rest to content", () => {
    const { content, reasoning } = runReasoning([
      "<think>planning</think>",
      "the answer",
    ]);
    expect(reasoning).toBe("planning");
    expect(content).toBe("the answer");
  });

  it("produces the same split fed character by character", () => {
    const { content, reasoning } = runReasoning(
      "<think>abc</think>hello".split(""),
    );
    expect(reasoning).toBe("abc");
    expect(content).toBe("hello");
  });

  it("handles the open marker split across deltas", () => {
    const { content, reasoning } = runReasoning(["<thi", "nk>r", "</think>c"]);
    expect(reasoning).toBe("r");
    expect(content).toBe("c");
  });

  it("handles the close marker split across deltas", () => {
    const { content, reasoning } = runReasoning(["<think>r</thi", "nk>c"]);
    expect(reasoning).toBe("r");
    expect(content).toBe("c");
  });

  it("passes through plain text with no reasoning", () => {
    const { content, reasoning } = runReasoning(["hello ", "world"]);
    expect(content).toBe("hello world");
    expect(reasoning).toBe("");
  });

  it("does not mistake a '<' in prose for a marker", () => {
    const { content, reasoning } = runReasoning(["a < b and c > d"]);
    expect(content).toBe("a < b and c > d");
    expect(reasoning).toBe("");
  });

  it("matches the <think> tag case-sensitively (parity with extractReasoning)", () => {
    const { content, reasoning } = runReasoning(["<THINK>x</THINK>answer"]);
    expect(reasoning).toBe("");
    expect(content).toBe("<THINK>x</THINK>answer");
  });

  it("streams an unterminated <think> as reasoning (documented streaming contract)", () => {
    // Streaming commits as tokens arrive: once <think> opens, text routes to
    // reasoning until close or end of stream. (Non-streaming would keep it.)
    const { content, reasoning } = runReasoning(["<think>", "no close here"]);
    expect(reasoning).toBe("no close here");
    expect(content).toBe("");
  });
});
