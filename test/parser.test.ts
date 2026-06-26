import { describe, it, expect } from "vitest";
import {
  parseToolCalls,
  extractFencedBlocks,
  tryParseJson,
  randomToolCallId,
} from "../src/index.js";
import { seqIds } from "./helpers.js";

const opts = { generateId: seqIds() };

describe("parseToolCalls", () => {
  it("parses a single tool-call block into OpenAI shape", () => {
    const text = '```tool_call\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```';
    const { content, toolCalls } = parseToolCalls(text, opts);
    expect(content).toBeNull();
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]).toEqual({
      id: "call_0",
      type: "function",
      function: { name: "get_weather", arguments: '{"city":"Paris"}' },
    });
  });

  it("keeps surrounding prose as content", () => {
    const text =
      'Let me check.\n```tool_call\n{"name":"t","arguments":{}}\n```\nDone.';
    const { content, toolCalls } = parseToolCalls(text, opts);
    expect(content).toBe("Let me check.\n\nDone.");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function.arguments).toBe("{}");
  });

  it("parses multiple blocks into multiple calls", () => {
    const text =
      '```tool_call\n{"name":"a","arguments":{"x":1}}\n```\n' +
      '```tool_call\n{"name":"b","arguments":{"y":2}}\n```';
    const { toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls.map((t) => t.function.name)).toEqual(["a", "b"]);
    expect(toolCalls.map((t) => t.id)).toEqual(["call_0", "call_1"]);
  });

  it("expands an array of calls inside one block", () => {
    const text =
      '```tool_call\n[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]\n```';
    const { toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls.map((t) => t.function.name)).toEqual(["a", "b"]);
  });

  it("repairs trailing commas in arguments", () => {
    const text = '```tool_call\n{"name":"a","arguments":{"x":1,}}\n```';
    const { toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls[0]!.function.arguments).toBe('{"x":1}');
  });

  it("defaults missing arguments to {}", () => {
    const text = '```tool_call\n{"name":"a"}\n```';
    const { toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls[0]!.function.arguments).toBe("{}");
  });

  it("normalizes arguments given as a JSON string", () => {
    const text = '```tool_call\n{"name":"a","arguments":"{\\"x\\": 1}"}\n```';
    const { toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls[0]!.function.arguments).toBe('{"x":1}');
  });

  it("returns plain text untouched when there is no tool call", () => {
    const { content, toolCalls } = parseToolCalls("Just a normal answer.", opts);
    expect(content).toBe("Just a normal answer.");
    expect(toolCalls).toEqual([]);
  });

  it("does not treat a foreign code block as a tool call", () => {
    const text = "Here is code:\n```js\nconsole.log(1)\n```";
    const { content, toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls).toEqual([]);
    expect(content).toContain("console.log(1)");
  });

  it("falls back to a json-tagged block when no tool_call tag is present", () => {
    const text = '```json\n{"name":"a","arguments":{"x":1}}\n```';
    const { toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function.name).toBe("a");
  });

  it("does not use the json fallback when a real tag is present", () => {
    const text =
      '```tool_call\n{"name":"real","arguments":{}}\n```\n' +
      '```json\n{"name":"ignored","arguments":{}}\n```';
    const { toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls.map((t) => t.function.name)).toEqual(["real"]);
  });

  it("ignores a json block that is not a tool call", () => {
    const text = '```json\n{"foo": "bar"}\n```';
    const { content, toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls).toEqual([]);
    expect(content).toContain('"foo": "bar"');
  });

  it("supports a custom tag", () => {
    const text = '```call\n{"name":"a","arguments":{}}\n```';
    const { toolCalls } = parseToolCalls(text, { ...opts, toolCallTag: "call" });
    expect(toolCalls).toHaveLength(1);
  });

  it("parses blocks with CRLF (Windows) line endings", () => {
    const text =
      '```tool_call\r\n{"name":"get_weather","arguments":{"city":"Paris"}}\r\n```';
    const { toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function.name).toBe("get_weather");
  });

  it("does not corrupt protocol-relative URLs when repairing JSON", () => {
    // Trailing comma forces the repair path; the // inside the string must survive.
    const text = '```tool_call\n{"name":"a","arguments":{"u":"//cdn.example.com/x",}}\n```';
    const { toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls[0]!.function.arguments).toBe('{"u":"//cdn.example.com/x"}');
  });
});

describe("extractFencedBlocks", () => {
  it("handles an unterminated final fence", () => {
    const blocks = extractFencedBlocks("```tool_call\n{partial");
    expect(blocks).toHaveLength(1);
    expect(blocks[0]!.infoString).toBe("tool_call");
    expect(blocks[0]!.content).toBe("{partial");
  });
});

describe("tryParseJson", () => {
  it("parses valid json", () => {
    expect(tryParseJson('{"a":1}')).toEqual({ a: 1 });
  });
  it("repairs trailing commas and comments", () => {
    expect(tryParseJson('{"a":1, /* c */ "b":2,}')).toEqual({ a: 1, b: 2 });
  });
  it("returns undefined for hopeless input", () => {
    expect(tryParseJson("not json at all {")).toBeUndefined();
  });
});

describe("randomToolCallId", () => {
  it("produces call_-prefixed ids", () => {
    expect(randomToolCallId()).toMatch(/^call_[A-Za-z0-9]{24}$/);
  });
});
