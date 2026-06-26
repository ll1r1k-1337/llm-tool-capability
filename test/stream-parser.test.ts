import { describe, it, expect } from "vitest";
import { ToolCallStreamParser } from "../src/index.js";
import type { ChatCompletionChunkDelta } from "../src/index.js";
import { seqIds } from "./helpers.js";

function run(deltas: string[], options: { toolCallTag?: string } = {}) {
  const p = new ToolCallStreamParser({ generateId: seqIds(), ...options });
  const emitted: ChatCompletionChunkDelta[] = [];
  for (const d of deltas) emitted.push(...p.push(d));
  emitted.push(...p.flush());
  const content = emitted.map((e) => e.content ?? "").join("");
  const toolCalls = emitted.flatMap((e) => e.tool_calls ?? []);
  return { content, toolCalls, emitted, count: p.toolCallCount };
}

const block = '```tool_call\n{"name":"get_weather","arguments":{"city":"Paris"}}\n```';

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
});
