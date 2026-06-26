import { describe, it, expect, vi } from "vitest";
import {
  createToolRunner,
  defineTool,
  MaxIterationsError,
  type RunnerEvent,
} from "../src/index.js";
import { mockClient, collect, seqIds } from "./helpers.js";

const weather = defineTool({
  name: "get_weather",
  description: "weather",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  handler: async (a: { city: string }) => ({ temp: 18, city: a.city }),
});

const call = (name: string, args: unknown) =>
  "```tool_call\n" + JSON.stringify({ name, arguments: args }) + "\n```";

const ask = [{ role: "user" as const, content: "weather in Paris?" }];

describe("createToolRunner.run", () => {
  it("runs the full loop: call a tool, feed the result, then answer", async () => {
    const client = mockClient([
      call("get_weather", { city: "Paris" }),
      "The weather in Paris is 18°C.",
    ]);
    const runner = createToolRunner(client, { tools: [weather], generateId: seqIds() });
    const res = await runner.run({ model: "m", messages: ask });

    expect(res.content).toBe("The weather in Paris is 18°C.");
    expect(res.iterations).toBe(2);
    expect(res.finishReason).toBe("stop");
    expect(res.toolExecutions).toHaveLength(1);
    expect(res.toolExecutions[0]!.result).toEqual({ temp: 18, city: "Paris" });
    expect(res.toolExecutions[0]!.isError).toBe(false);

    const roles = res.messages.map((m) => m.role);
    expect(roles).toEqual(["user", "assistant", "tool", "assistant"]);
  });

  it("feeds back an error for an unknown tool and keeps going", async () => {
    const client = mockClient([call("nope", {}), "I cannot do that."]);
    const runner = createToolRunner(client, { tools: [weather] });
    const res = await runner.run({ model: "m", messages: ask });

    expect(res.toolExecutions[0]!.isError).toBe(true);
    expect(res.toolExecutions[0]!.error).toMatch(/unknown tool/i);
    expect(res.toolExecutions[0]!.content).toMatch(/get_weather/); // lists available
    expect(res.content).toBe("I cannot do that.");
  });

  it("feeds back a validation error for bad arguments", async () => {
    const client = mockClient([call("get_weather", {}), "done"]);
    const runner = createToolRunner(client, { tools: [weather] });
    const res = await runner.run({ model: "m", messages: ask });

    expect(res.toolExecutions[0]!.isError).toBe(true);
    expect(res.toolExecutions[0]!.error).toMatch(/city|required/i);
    expect(res.toolExecutions[0]!.content).toMatch(/invalid arguments/i);
  });

  it("feeds back an error when a handler throws", async () => {
    const boom = defineTool({
      name: "boom",
      parameters: { type: "object", properties: {} },
      handler: () => {
        throw new Error("kaboom");
      },
    });
    const client = mockClient([call("boom", {}), "recovered"]);
    const runner = createToolRunner(client, { tools: [boom] });
    const res = await runner.run({ model: "m", messages: ask });

    expect(res.toolExecutions[0]!.isError).toBe(true);
    expect(res.toolExecutions[0]!.error).toBe("kaboom");
    expect(res.toolExecutions[0]!.content).toMatch(/kaboom/);
    expect(res.content).toBe("recovered");
  });

  it("stops at maxIterations and reports it", async () => {
    const client = mockClient([
      call("get_weather", { city: "P" }),
      call("get_weather", { city: "P" }),
    ]);
    const runner = createToolRunner(client, { tools: [weather], maxIterations: 2 });
    const res = await runner.run({ model: "m", messages: ask });
    expect(res.finishReason).toBe("max_iterations");
    expect(res.iterations).toBe(2);
  });

  it("throws MaxIterationsError when configured to", async () => {
    const client = mockClient([
      call("get_weather", { city: "P" }),
      call("get_weather", { city: "P" }),
    ]);
    const runner = createToolRunner(client, {
      tools: [weather],
      maxIterations: 2,
      throwOnMaxIterations: true,
    });
    await expect(runner.run({ model: "m", messages: ask })).rejects.toBeInstanceOf(
      MaxIterationsError,
    );
  });

  it("invokes observability hooks", async () => {
    const onToolCall = vi.fn();
    const onToolResult = vi.fn();
    const client = mockClient([call("get_weather", { city: "Paris" }), "ok"]);
    const runner = createToolRunner(client, {
      tools: [weather],
      onToolCall,
      onToolResult,
    });
    await runner.run({ model: "m", messages: ask });
    expect(onToolCall).toHaveBeenCalledTimes(1);
    expect(onToolResult).toHaveBeenCalledTimes(1);
  });

  it("throws when the API returns no choices", async () => {
    const client = {
      chat: {
        completions: {
          create: async () => ({
            id: "x",
            object: "chat.completion",
            created: 0,
            model: "m",
            choices: [],
          }),
        },
      },
    };
    const runner = createToolRunner(client as any, { tools: [weather] });
    await expect(runner.run({ model: "m", messages: ask })).rejects.toThrow(
      /no choices/i,
    );
  });

  it("forwards the abort signal to the underlying client", async () => {
    const seen: any[] = [];
    const client = {
      chat: {
        completions: {
          create: async (_body: any, opts: any) => {
            seen.push(opts);
            return {
              id: "x",
              object: "chat.completion",
              created: 0,
              model: "m",
              choices: [
                {
                  index: 0,
                  message: { role: "assistant", content: "done", refusal: null },
                  finish_reason: "stop",
                  logprobs: null,
                },
              ],
            };
          },
        },
      },
    };
    const ac = new AbortController();
    const runner = createToolRunner(client as any, { tools: [weather] });
    await runner.run({ model: "m", messages: ask, signal: ac.signal });
    expect(seen[0]?.signal).toBe(ac.signal);
  });

  it("rejects n>1 together with tools", async () => {
    const client = mockClient(["unused"]);
    const runner = createToolRunner(client, { tools: [weather] });
    await expect(
      runner.run({ model: "m", messages: ask, n: 2 } as any),
    ).rejects.toThrow(/n > 1/);
  });

  it("disables schema validation when validate:false", async () => {
    const client = mockClient([call("get_weather", {}), "done"]);
    const runner = createToolRunner(client, { tools: [weather], validate: false });
    const res = await runner.run({ model: "m", messages: ask });
    // No schema check -> handler runs with {} (city undefined).
    expect(res.toolExecutions[0]!.isError).toBe(false);
    expect(res.toolExecutions[0]!.result).toEqual({ temp: 18, city: undefined });
  });
});

describe("createToolRunner.runStream", () => {
  it("emits text, tool_call, tool_result and final events", async () => {
    const client = mockClient([
      call("get_weather", { city: "Paris" }),
      "Paris is 18°C.",
    ]);
    const runner = createToolRunner(client, { tools: [weather], generateId: seqIds() });
    const events = await collect<RunnerEvent>(
      runner.runStream({ model: "m", messages: ask }),
    );

    const types = events.map((e) => e.type);
    expect(types).toContain("tool_call");
    expect(types).toContain("tool_result");
    expect(types).toContain("final");

    const toolCall = events.find((e) => e.type === "tool_call");
    expect(toolCall && toolCall.type === "tool_call" && toolCall.toolCall.function.name).toBe(
      "get_weather",
    );

    const result = events.find((e) => e.type === "tool_result");
    expect(
      result && result.type === "tool_result" && result.execution.result,
    ).toEqual({ temp: 18, city: "Paris" });

    const text = events
      .filter((e): e is Extract<RunnerEvent, { type: "text" }> => e.type === "text")
      .map((e) => e.delta)
      .join("");
    expect(text).toContain("Paris is 18°C.");

    const final = events.find((e) => e.type === "final");
    expect(final && final.type === "final" && final.content).toBe("Paris is 18°C.");
    expect(final && final.type === "final" && final.finishReason).toBe("stop");
  });
});
