import { describe, it, expect } from "vitest";
import { wrapToolSupport, flattenMessages } from "../src/index.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionTool,
  ChatCompletionMessageParam,
} from "../src/index.js";
import { mockClient, collect, seqIds } from "./helpers.js";

const tools: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      parameters: { type: "object", properties: { city: { type: "string" } } },
    },
  },
];

const userMsg: ChatCompletionMessageParam[] = [{ role: "user", content: "weather?" }];

describe("wrapToolSupport (non-streaming)", () => {
  it("turns a tool_call block into OpenAI-shaped tool_calls", async () => {
    const client = mockClient([
      '```tool_call\n{"name":"get_weather","arguments":{"city":"Paris"}}\n```',
    ]);
    const wrapped = wrapToolSupport(client, { generateId: seqIds() });
    const res = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools,
    })) as ChatCompletion;

    const msg = res.choices[0]!.message;
    expect(res.choices[0]!.finish_reason).toBe("tool_calls");
    expect(msg.tool_calls).toEqual([
      {
        id: "call_0",
        type: "function",
        function: { name: "get_weather", arguments: '{"city":"Paris"}' },
      },
    ]);
    expect(msg.content).toBeNull();
  });

  it("injects a system instruction and strips tool params from the request", async () => {
    const client = mockClient(["plain answer"]);
    const wrapped = wrapToolSupport(client);
    await wrapped.chat.completions.create({ model: "m", messages: userMsg, tools });

    const sent = client.calls[0];
    expect(sent.tools).toBeUndefined();
    expect(sent.tool_choice).toBeUndefined();
    expect(sent.messages[0].role).toBe("system");
    expect(sent.messages[0].content).toContain("get_weather");
  });

  it("is transparent when no tools are passed", async () => {
    const client = mockClient(["hello there"]);
    const wrapped = wrapToolSupport(client);
    const res = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
    })) as ChatCompletion;
    expect(res.choices[0]!.message.content).toBe("hello there");
    expect(client.calls[0].messages[0].role).toBe("user");
  });

  it("does not inject or parse when tool_choice is 'none'", async () => {
    const client = mockClient([
      '```tool_call\n{"name":"x","arguments":{}}\n```',
    ]);
    const wrapped = wrapToolSupport(client);
    const res = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools,
      tool_choice: "none",
    })) as ChatCompletion;
    // No injection happened...
    expect(client.calls[0].messages[0].role).toBe("user");
    // ...and the raw text is returned untouched (not parsed into tool_calls).
    expect(res.choices[0]!.message.tool_calls).toBeUndefined();
  });

  it("merges instructions into an existing system message by default", async () => {
    const client = mockClient(["ok"]);
    const wrapped = wrapToolSupport(client);
    await wrapped.chat.completions.create({
      model: "m",
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "hi" },
      ],
      tools,
    });
    const sys = client.calls[0].messages[0];
    expect(sys.role).toBe("system");
    expect(sys.content).toContain("You are helpful.");
    expect(sys.content).toContain("get_weather");
    expect(client.calls[0].messages).toHaveLength(2);
  });

  it("prepends a system message when systemInjection is 'prepend'", async () => {
    const client = mockClient(["ok"]);
    const wrapped = wrapToolSupport(client, { systemInjection: "prepend" });
    await wrapped.chat.completions.create({
      model: "m",
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "hi" },
      ],
      tools,
    });
    expect(client.calls[0].messages).toHaveLength(3);
    expect(client.calls[0].messages[0].content).toContain("get_weather");
    expect(client.calls[0].messages[1].content).toBe("You are helpful.");
  });

  it("preserves multimodal array content when merging the instruction", async () => {
    const client = mockClient(["ok"]);
    const wrapped = wrapToolSupport(client);
    await wrapped.chat.completions.create({
      model: "m",
      messages: [
        {
          role: "system",
          content: [
            { type: "text", text: "You are helpful." },
            { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
          ],
        },
        { role: "user", content: "hi" },
      ],
      tools,
    });
    const sys = client.calls[0].messages[0];
    expect(Array.isArray(sys.content)).toBe(true);
    expect(sys.content.some((p: any) => p.type === "image_url")).toBe(true);
    expect(
      sys.content.some((p: any) => p.type === "text" && p.text.includes("get_weather")),
    ).toBe(true);
  });

  it("forwards extra params untouched", async () => {
    const client = mockClient(["ok"]);
    const wrapped = wrapToolSupport(client);
    await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools,
      temperature: 0.2,
      stop: ["X"],
    } as any);
    expect(client.calls[0].temperature).toBe(0.2);
    expect(client.calls[0].stop).toEqual(["X"]);
  });
});

describe("flattenMessages", () => {
  it("renders assistant tool_calls and tool results back into text", () => {
    const flat = flattenMessages(
      [
        { role: "user", content: "weather?" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "get_weather", arguments: '{"city":"Paris"}' },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: '{"temp":18}' },
      ],
      { toolCallTag: "tool_call", toolResultTag: "tool_result" },
    );

    expect(flat).toHaveLength(3);
    expect(flat[1]!.role).toBe("assistant");
    expect((flat[1] as any).content).toContain("```tool_call");
    expect((flat[1] as any).content).toContain("get_weather");
    expect((flat[1] as any).tool_calls).toBeUndefined();
    // tool result becomes a user message referencing the tool name + id.
    expect(flat[2]!.role).toBe("user");
    expect((flat[2] as any).content).toContain("```tool_result");
    expect((flat[2] as any).content).toContain("get_weather");
    expect((flat[2] as any).content).toContain("call_1");
  });

  it("merges consecutive tool results into one user message", () => {
    const flat = flattenMessages(
      [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "a", type: "function", function: { name: "x", arguments: "{}" } },
            { id: "b", type: "function", function: { name: "y", arguments: "{}" } },
          ],
        },
        { role: "tool", tool_call_id: "a", content: "1" },
        { role: "tool", tool_call_id: "b", content: "2" },
      ],
      { toolCallTag: "tool_call", toolResultTag: "tool_result" },
    );
    const results = flat.filter((m) => m.role === "user");
    expect(results).toHaveLength(1);
    expect((results[0] as any).content).toContain('"x"');
    expect((results[0] as any).content).toContain('"y"');
  });

  it("renders a deprecated function-role message as a result", () => {
    const flat = flattenMessages(
      [{ role: "function", name: "get_weather", content: '{"temp":18}' }],
      { toolCallTag: "tool_call", toolResultTag: "tool_result" },
    );
    expect(flat).toHaveLength(1);
    expect(flat[0]!.role).toBe("user");
    expect((flat[0] as any).content).toContain("```tool_result");
    expect((flat[0] as any).content).toContain("get_weather");
  });

  it("collapses text-only array content to a string (issue #3)", () => {
    const flat = flattenMessages(
      [
        {
          role: "user",
          content: [
            { type: "text", text: "hello" },
            { type: "text", text: "world" },
          ] as any,
        },
      ],
      { toolCallTag: "tool_call", toolResultTag: "tool_result" },
    );
    expect(flat[0]!.content).toBe("hello\nworld");
  });

  it("renders the real tool name when a tool result precedes its assistant message", () => {
    const flat = flattenMessages(
      [
        { role: "tool", tool_call_id: "call_1", content: '{"x":1}' },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "get_weather", arguments: "{}" } },
          ],
        },
      ],
      { toolCallTag: "tool_call", toolResultTag: "tool_result" },
    );
    const result = flat.find((m) => m.role === "user");
    expect((result as any).content).toContain("get_weather");
    expect((result as any).content).not.toContain('"name":"tool"');
  });

  it("preserves array content that contains non-text parts (multimodal)", () => {
    const flat = flattenMessages(
      [
        {
          role: "user",
          content: [
            { type: "text", text: "look" },
            { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
          ] as any,
        },
      ],
      { toolCallTag: "tool_call", toolResultTag: "tool_result" },
    );
    expect(Array.isArray(flat[0]!.content)).toBe(true);
    expect(flat[0]!.content).toHaveLength(2);
  });
});

describe("wrapToolSupport (streaming)", () => {
  it("streams prose then emits a tool-call delta and a finish chunk", async () => {
    const client = mockClient([
      ["Sure.\n", '```tool_call\n{"name":"get_weather","arguments":{"city":"Paris"}}\n```'],
    ]);
    const wrapped = wrapToolSupport(client, { generateId: seqIds() });
    const stream = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools,
      stream: true,
    })) as AsyncIterable<ChatCompletionChunk>;

    const chunks = await collect(stream);
    const content = chunks.map((c) => c.choices[0]?.delta.content ?? "").join("");
    const toolCalls = chunks.flatMap((c) => c.choices[0]?.delta.tool_calls ?? []);
    const finish = chunks[chunks.length - 1]!.choices[0]!.finish_reason;

    expect(chunks[0]!.choices[0]!.delta.role).toBe("assistant");
    expect(content).toContain("Sure.");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function).toEqual({
      name: "get_weather",
      arguments: '{"city":"Paris"}',
    });
    expect(finish).toBe("tool_calls");
  });

  it("ends with finish_reason 'stop' when no tool is called", async () => {
    const client = mockClient([["Hello ", "world"]]);
    const wrapped = wrapToolSupport(client);
    const stream = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools,
      stream: true,
    })) as AsyncIterable<ChatCompletionChunk>;
    const chunks = await collect(stream);
    const content = chunks.map((c) => c.choices[0]?.delta.content ?? "").join("");
    expect(content).toBe("Hello world");
    expect(chunks[chunks.length - 1]!.choices[0]!.finish_reason).toBe("stop");
  });

  it("passes the stream through untouched when no tools are used", async () => {
    const client = mockClient([["a", "b"]]);
    const wrapped = wrapToolSupport(client);
    const stream = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      stream: true,
    })) as AsyncIterable<ChatCompletionChunk>;
    const chunks = await collect(stream);
    // Raw upstream shape: role chunk + 2 content chunks + stop chunk.
    const content = chunks.map((c) => c.choices[0]?.delta.content ?? "").join("");
    expect(content).toBe("ab");
  });
});

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

describe("wrapToolSupport — reasoning (<think>) handling, issue #4", () => {
  it("splits <think> out of content into reasoning_content (non-streaming)", async () => {
    const client = mockClient([
      "<think>I should list files</think>\nLet me check.\n" +
        '```tool_call\n{"name":"get_weather","arguments":{"city":"Paris"}}\n```',
    ]);
    const wrapped = wrapToolSupport(client, { generateId: seqIds() });
    const res = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools,
    })) as ChatCompletion;

    const msg = res.choices[0]!.message;
    expect(msg.reasoning_content).toBe("I should list files");
    expect(msg.content).not.toContain("<think>");
    expect(msg.content).toContain("Let me check.");
    expect(msg.tool_calls).toHaveLength(1);
  });

  it("streams reasoning as reasoning_content deltas, clean content (streaming)", async () => {
    const client = mockClient([
      ["<think>plan", "ning</think>", "Hello ", "world"],
    ]);
    const wrapped = wrapToolSupport(client);
    const stream = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools,
      stream: true,
    })) as AsyncIterable<ChatCompletionChunk>;
    const chunks = await collect(stream);
    const content = chunks.map((c) => c.choices[0]?.delta.content ?? "").join("");
    const reasoning = chunks
      .map((c) => (c.choices[0]?.delta as any).reasoning_content ?? "")
      .join("");
    expect(reasoning).toBe("planning");
    expect(content).toBe("Hello world");
    expect(content).not.toContain("<think>");
  });

  it("streams <think> then a tool_call fence (glob comment scenario)", async () => {
    const client = mockClient([
      [
        "<think>list the files</think>\n",
        "Let me check.\n",
        '```tool_call\n{"name":"get_weather","arguments":{"city":"Paris"}}\n```',
      ],
    ]);
    const wrapped = wrapToolSupport(client, { generateId: seqIds() });
    const stream = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools,
      stream: true,
    })) as AsyncIterable<ChatCompletionChunk>;
    const chunks = await collect(stream);
    const content = chunks.map((c) => c.choices[0]?.delta.content ?? "").join("");
    const reasoning = chunks
      .map((c) => (c.choices[0]?.delta as any).reasoning_content ?? "")
      .join("");
    const toolCalls = chunks.flatMap((c) => c.choices[0]?.delta.tool_calls ?? []);
    expect(reasoning).toBe("list the files");
    expect(content).toContain("Let me check.");
    expect(content).not.toContain("<think>");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function!.name).toBe("get_weather");
    expect(chunks[chunks.length - 1]!.choices[0]!.finish_reason).toBe("tool_calls");
  });

  it("leaves <think> in content when reasoning is disabled", async () => {
    const client = mockClient(["<think>x</think>answer"]);
    const wrapped = wrapToolSupport(client, { reasoning: false });
    const res = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools,
    })) as ChatCompletion;
    expect(res.choices[0]!.message.content).toContain("<think>");
    expect(res.choices[0]!.message.reasoning_content).toBeUndefined();
  });
});

describe("wrapToolSupport — native XML tool calls (opt-in), issue #4", () => {
  it("turns an own-line <question> block into tool_calls (non-streaming)", async () => {
    const client = mockClient([
      "<think>need to clarify</think>\nPlease answer:\n" +
        "<question>\n[{\"question\":\"frontend or backend?\"}]\n</question>",
    ]);
    const wrapped = wrapToolSupport(client, {
      generateId: seqIds(),
      xmlToolCalls: true,
    });
    const res = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools: [questionTool],
    })) as ChatCompletion;

    const msg = res.choices[0]!.message;
    expect(res.choices[0]!.finish_reason).toBe("tool_calls");
    expect(msg.reasoning_content).toBe("need to clarify");
    expect(msg.tool_calls).toHaveLength(1);
    expect(msg.tool_calls![0]!.function.name).toBe("question");
    expect(JSON.parse(msg.tool_calls![0]!.function.arguments)).toEqual({
      questions: [{ question: "frontend or backend?" }],
    });
  });

  it("turns an own-line <question> block into a tool_call delta (streaming)", async () => {
    const client = mockClient([
      ["<question>\n", "[{\"question\":\"ok?\"}]\n", "</question>\n"],
    ]);
    const wrapped = wrapToolSupport(client, {
      generateId: seqIds(),
      xmlToolCalls: true,
    });
    const stream = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools: [questionTool],
      stream: true,
    })) as AsyncIterable<ChatCompletionChunk>;
    const chunks = await collect(stream);
    const toolCalls = chunks.flatMap((c) => c.choices[0]?.delta.tool_calls ?? []);
    const finish = chunks[chunks.length - 1]!.choices[0]!.finish_reason;
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.function!.name).toBe("question");
    expect(finish).toBe("tool_calls");
  });

  it("does not parse XML tags by default (opt-in only)", async () => {
    const client = mockClient([
      "<question>\n[{\"question\":\"a\"}]\n</question>",
    ]);
    const wrapped = wrapToolSupport(client);
    const res = (await wrapped.chat.completions.create({
      model: "m",
      messages: userMsg,
      tools: [questionTool],
    })) as ChatCompletion;
    expect(res.choices[0]!.message.tool_calls).toBeUndefined();
    expect(res.choices[0]!.message.content).toContain("<question>");
  });
});
