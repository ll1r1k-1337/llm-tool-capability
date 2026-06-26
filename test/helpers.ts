import type {
  ChatClientLike,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionChunkDelta,
} from "../src/index.js";

export function makeCompletion(content: string | null, model = "mock"): ChatCompletion {
  return {
    id: "cmpl-mock",
    object: "chat.completion",
    created: 1000,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content, refusal: null },
        finish_reason: "stop",
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

function chunk(
  delta: ChatCompletionChunkDelta,
  finish: ChatCompletionChunk["choices"][number]["finish_reason"] = null,
  model = "mock",
): ChatCompletionChunk {
  return {
    id: "cmpl-mock",
    object: "chat.completion.chunk",
    created: 1000,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

/** An upstream stream that emits the given text deltas, like a raw provider. */
export async function* makeChunkStream(
  deltas: string[],
  model = "mock",
): AsyncIterable<ChatCompletionChunk> {
  yield chunk({ role: "assistant" }, null, model);
  for (const d of deltas) yield chunk({ content: d }, null, model);
  yield chunk({}, "stop", model);
}

export type Script = string | string[];

export interface MockClient extends ChatClientLike {
  calls: any[];
}

/**
 * A scripted OpenAI-compatible client. Each entry is consumed per `create`
 * call: a string is returned as a (non-stream) completion or, if `stream` is
 * set, as a single-delta stream; an array streams its elements as deltas (or,
 * for non-stream calls, joins them).
 */
export function mockClient(script: Script[]): MockClient {
  let i = 0;
  const calls: any[] = [];
  return {
    calls,
    chat: {
      completions: {
        create(body: any) {
          calls.push(body);
          const entry = script[i++];
          if (entry === undefined) throw new Error("mockClient: ran out of script");
          if (body.stream) {
            const deltas = Array.isArray(entry) ? entry : [entry];
            return Promise.resolve(makeChunkStream(deltas, body.model));
          }
          const text = Array.isArray(entry) ? entry.join("") : entry;
          return Promise.resolve(makeCompletion(text, body.model));
        },
      },
    },
  };
}

/** Drains an async iterable of runner/stream events or chunks into an array. */
export async function collect<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of it) out.push(x);
  return out;
}

/** Deterministic id generator for assertions. */
export function seqIds(): (i: number) => string {
  return (i: number) => `call_${i}`;
}
