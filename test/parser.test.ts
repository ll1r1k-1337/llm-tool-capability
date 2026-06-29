import { describe, it, expect } from "vitest";
import {
  parseToolCalls,
  extractFencedBlocks,
  tryParseJson,
  randomToolCallId,
  extractReasoning,
  mapXmlToolCall,
} from "../src/index.js";
import type { ChatCompletionTool } from "../src/index.js";
import { seqIds } from "./helpers.js";

const opts = { generateId: seqIds() };

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

const bashTool: ChatCompletionTool = {
  type: "function",
  function: {
    name: "bash",
    parameters: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
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

  it("skips regex repair on very large malformed input (ReDoS guard)", () => {
    const big = "{" + "/*".repeat(200_000); // >256 KB, malformed, many '/*'
    const t0 = Date.now();
    expect(tryParseJson(big)).toBeUndefined();
    expect(Date.now() - t0).toBeLessThan(1000);
  });

  it("still parses large VALID json (guard only affects repair)", () => {
    const big = JSON.stringify({ a: "y".repeat(300_000) });
    expect(tryParseJson(big)).toEqual({ a: "y".repeat(300_000) });
  });
});

describe("randomToolCallId", () => {
  it("produces call_-prefixed ids", () => {
    expect(randomToolCallId()).toMatch(/^call_[A-Za-z0-9]{24}$/);
  });
});

describe("extractReasoning", () => {
  it("splits a <think> block out of content", () => {
    const { reasoning, content } = extractReasoning(
      "<think>let me plan this</think>\n\nHere is the answer.",
    );
    expect(reasoning).toBe("let me plan this");
    expect(content).toBe("Here is the answer.");
  });

  it("returns null reasoning and untouched content when no block is present", () => {
    const { reasoning, content } = extractReasoning("just a plain answer");
    expect(reasoning).toBeNull();
    expect(content).toBe("just a plain answer");
  });

  it("leaves an unterminated <think> untouched (never swallows the answer)", () => {
    const { reasoning, content } = extractReasoning("<think>oops no close\nanswer");
    expect(reasoning).toBeNull();
    expect(content).toBe("<think>oops no close\nanswer");
  });

  it("keeps a tool_call fence intact while removing reasoning", () => {
    const text =
      "<think>reasoning</think>\nLet me check.\n" +
      '```tool_call\n{"name":"glob","arguments":{"pattern":"*"}}\n```';
    const { reasoning, content } = extractReasoning(text);
    expect(reasoning).toBe("reasoning");
    expect(content).toContain("```tool_call");
    expect(content).toContain("Let me check.");
  });

  it("concatenates multiple reasoning blocks", () => {
    const { reasoning, content } = extractReasoning("<think>a</think>x<think>b</think>y");
    expect(reasoning).toBe("a\n\nb");
    expect(content).toBe("xy");
  });

  it("supports a custom reasoning tag", () => {
    const { reasoning, content } = extractReasoning(
      "<reasoning>why</reasoning>answer",
      { reasoningTag: "reasoning" },
    );
    expect(reasoning).toBe("why");
    expect(content).toBe("answer");
  });

  it("matches the tag case-sensitively (parity with the streaming parser)", () => {
    const { reasoning, content } = extractReasoning("<THINK>x</THINK>answer");
    expect(reasoning).toBeNull();
    expect(content).toBe("<THINK>x</THINK>answer");
  });

  it("stays linear on many unterminated openers (ReDoS guard)", () => {
    const big = "<think>".repeat(200_000); // openers, no closers
    const t0 = Date.now();
    const { reasoning, content } = extractReasoning(big);
    expect(reasoning).toBeNull();
    expect(content).toBe(big);
    expect(Date.now() - t0).toBeLessThan(1000);
  });
});

describe("mapXmlToolCall", () => {
  it("wraps a bare array as the tool's single required parameter", () => {
    const na = mapXmlToolCall("question", '[{"question":"a"}]', questionTool);
    expect(na).toEqual({ name: "question", arguments: '{"questions":[{"question":"a"}]}' });
  });

  it("wraps a bare scalar string as the single parameter", () => {
    const na = mapXmlToolCall("bash", "ls -la", bashTool);
    expect(na).toEqual({ name: "bash", arguments: '{"command":"ls -la"}' });
  });

  it("takes a JSON object as the arguments directly", () => {
    const na = mapXmlToolCall("bash", '{"command":"ls"}', bashTool);
    expect(na).toEqual({ name: "bash", arguments: '{"command":"ls"}' });
  });

  it("unwraps a {name, arguments} envelope but keeps the tag name", () => {
    const na = mapXmlToolCall(
      "bash",
      '{"name":"ignored","arguments":{"command":"pwd"}}',
      bashTool,
    );
    expect(na).toEqual({ name: "bash", arguments: '{"command":"pwd"}' });
  });

  it("returns null for a bare value on an ambiguous multi-param tool", () => {
    expect(mapXmlToolCall("edit", "some text", editTool)).toBeNull();
  });

  it("still maps a JSON object for a multi-param tool", () => {
    const na = mapXmlToolCall(
      "edit",
      '{"filePath":"a.ts","oldString":"x","newString":"y"}',
      editTool,
    );
    expect(na!.name).toBe("edit");
    expect(JSON.parse(na!.arguments)).toEqual({
      filePath: "a.ts",
      oldString: "x",
      newString: "y",
    });
  });
});

describe("parseToolCalls — native XML tags (opt-in)", () => {
  const xmlOpts = { ...opts, xmlToolCalls: true, tools: [questionTool, bashTool] };

  it("parses an own-line <question> block (issue #4 reproduction)", () => {
    const text =
      "Please answer these questions:\n" +
      "<question>\n" +
      '[\n  {"question":"frontend or backend?","header":"Location"}\n]\n' +
      "</question>\n" +
      "Once you answer I can continue.";
    const { content, toolCalls } = parseToolCalls(text, xmlOpts);
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function.name).toBe("question");
    expect(JSON.parse(toolCalls[0]!.function.arguments)).toEqual({
      questions: [{ question: "frontend or backend?", header: "Location" }],
    });
    expect(content).toContain("Please answer these questions:");
    expect(content).not.toContain("<question>");
  });

  it("is off by default — XML tags stay as content", () => {
    const text = "<question>\n[{\"question\":\"a\"}]\n</question>";
    const { content, toolCalls } = parseToolCalls(text, opts);
    expect(toolCalls).toEqual([]);
    expect(content).toContain("<question>");
  });

  it("does not parse an XML tag inside a fenced code block", () => {
    const text =
      "Here is XML:\n```xml\n<question>\n[{\"q\":1}]\n</question>\n```";
    const { toolCalls } = parseToolCalls(text, xmlOpts);
    expect(toolCalls).toEqual([]);
  });

  it("ignores XML tags whose name is not a known tool", () => {
    const text = "<unknown>\nhello\n</unknown>";
    const { toolCalls } = parseToolCalls(text, xmlOpts);
    expect(toolCalls).toEqual([]);
  });

  it("parses fence and XML tool calls together", () => {
    const text =
      '```tool_call\n{"name":"bash","arguments":{"command":"ls"}}\n```\n' +
      "<question>\n[{\"question\":\"ok?\"}]\n</question>";
    const { toolCalls } = parseToolCalls(text, xmlOpts);
    expect(toolCalls.map((t) => t.function.name)).toEqual(["bash", "question"]);
  });

  it("handles CRLF line endings around an XML tag", () => {
    const text = "<question>\r\n[{\"question\":\"a\"}]\r\n</question>\r\n";
    const { toolCalls } = parseToolCalls(text, xmlOpts);
    expect(toolCalls).toHaveLength(1);
    expect(JSON.parse(toolCalls[0]!.function.arguments)).toEqual({
      questions: [{ question: "a" }],
    });
  });
});
