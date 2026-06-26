import { describe, it, expect } from "vitest";
import { ToolValidator, type ChatCompletionTool } from "../src/index.js";

const tools: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      parameters: {
        type: "object",
        properties: { city: { type: "string" }, days: { type: "integer" } },
        required: ["city"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: { name: "no_schema" },
  },
];

describe("ToolValidator", () => {
  const v = new ToolValidator(tools);

  it("accepts valid arguments", () => {
    const r = v.validate("get_weather", '{"city":"Paris","days":3}');
    expect(r.valid).toBe(true);
    expect(r.value).toEqual({ city: "Paris", days: 3 });
  });

  it("rejects missing required fields", () => {
    const r = v.validate("get_weather", '{"days":3}');
    expect(r.valid).toBe(false);
    expect(r.errors.join(" ")).toMatch(/city|required/i);
  });

  it("rejects wrong types", () => {
    const r = v.validate("get_weather", '{"city":123}');
    expect(r.valid).toBe(false);
  });

  it("rejects invalid json", () => {
    const r = v.validate("get_weather", "{not json");
    expect(r.valid).toBe(false);
    expect(r.errors[0]).toMatch(/json/i);
  });

  it("passes tools without a schema", () => {
    const r = v.validate("no_schema", '{"anything":true}');
    expect(r.valid).toBe(true);
  });

  it("treats unknown tools as schema-less (valid)", () => {
    const r = v.validate("does_not_exist", "{}");
    expect(r.valid).toBe(true);
  });

  it("coerces types when enabled", () => {
    const vc = new ToolValidator(tools, { coerceTypes: true });
    const r = vc.validate("get_weather", '{"city":"Paris","days":"3"}');
    expect(r.valid).toBe(true);
    expect(r.value).toEqual({ city: "Paris", days: 3 });
  });

  it("reports known tools", () => {
    expect(v.hasTool("get_weather")).toBe(true);
    expect(v.hasTool("nope")).toBe(false);
  });

  it("enforces validation even when two tools share a $id", () => {
    const dup = new ToolValidator([
      {
        type: "function",
        function: {
          name: "a",
          parameters: {
            $id: "http://example.com/shared",
            type: "object",
            properties: { q: { type: "string" } },
            required: ["q"],
            additionalProperties: false,
          },
        },
      },
      {
        type: "function",
        function: {
          name: "b",
          parameters: {
            $id: "http://example.com/shared",
            type: "object",
            properties: { z: { type: "number" } },
            required: ["z"],
            additionalProperties: false,
          },
        },
      },
    ]);
    expect(dup.validate("a", "{}").valid).toBe(false);
    // Would be true (validator silently dropped) without the $id fix:
    expect(dup.validate("b", "{}").valid).toBe(false);
    expect(dup.validate("b", '{"z":1}').valid).toBe(true);
  });
});
