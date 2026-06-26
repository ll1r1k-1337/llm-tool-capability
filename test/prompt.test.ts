import { describe, it, expect } from "vitest";
import {
  buildToolPrompt,
  buildToolChoiceInstruction,
  type ChatCompletionTool,
} from "../src/index.js";

const tools: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the weather for a city.",
      parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
      },
    },
  },
];

describe("buildToolPrompt", () => {
  it("includes the tool name, description and schema", () => {
    const p = buildToolPrompt(tools);
    expect(p).toContain("get_weather");
    expect(p).toContain("Get the weather for a city.");
    expect(p).toContain('"required"');
    expect(p).toContain("tool_call");
  });

  it("includes a few-shot example by default and can omit it", () => {
    expect(buildToolPrompt(tools)).toContain("Example");
    expect(buildToolPrompt(tools, { includeExamples: false })).not.toContain(
      "## Example",
    );
  });

  it("respects a custom tag", () => {
    const p = buildToolPrompt(tools, { toolCallTag: "invoke" });
    expect(p).toContain("```invoke");
  });

  it("supports a custom template", () => {
    const p = buildToolPrompt(tools, {
      template: ({ renderedTools, toolCallTag }) => `T:${toolCallTag}\n${renderedTools}`,
    });
    expect(p.startsWith("T:tool_call")).toBe(true);
    expect(p).toContain("get_weather");
  });
});

describe("buildToolChoiceInstruction", () => {
  it("returns empty for auto/none/undefined", () => {
    expect(buildToolChoiceInstruction(undefined)).toBe("");
    expect(buildToolChoiceInstruction("auto")).toBe("");
    expect(buildToolChoiceInstruction("none")).toBe("");
  });

  it("requires a tool for 'required'", () => {
    expect(buildToolChoiceInstruction("required")).toContain("MUST call at least one");
  });

  it("names a specific tool", () => {
    const out = buildToolChoiceInstruction({
      type: "function",
      function: { name: "get_weather" },
    });
    expect(out).toContain("get_weather");
  });
});
