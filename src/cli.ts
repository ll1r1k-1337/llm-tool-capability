#!/usr/bin/env node
import { createProxyServer, type ProxyOptions } from "./proxy.js";

interface ParsedArgs {
  [key: string]: string | boolean;
}

function parseArgs(argv: string[]): ParsedArgs {
  const out: ParsedArgs = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    if (key.startsWith("no-")) {
      out[key.slice(3)] = false;
      continue;
    }
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

const HELP = `llm-tool-proxy — OpenAI-compatible proxy that adds tool calling to tool-less models

Usage:
  llm-tool-proxy --upstream <url> [options]

Options:
  --upstream <url>        Upstream OpenAI-compatible base URL (e.g. http://localhost:11434/v1)
                          [env: UPSTREAM_BASE_URL]
  --upstream-key <key>    Bearer token for the upstream [env: UPSTREAM_API_KEY]
  --port <n>              Port to listen on (default: 8787) [env: PORT]
  --host <host>           Host to bind (default: 127.0.0.1) [env: HOST]
  --api-key <key>         Require this bearer token from clients [env: PROXY_API_KEY]
  --base-path <path>      Route prefix (default: /v1)
  --tag <label>           Tool-call fence label (default: tool_call)
  --no-examples           Omit the few-shot example from the injected prompt
  --system-injection <m>  'merge' (default) or 'prepend'
  --xml-tool-calls        Also parse native XML tool tags (<toolName>…</toolName>)
                          into tool calls — for models that emit calls as XML
                          instead of the tool_call fence (off by default)
  --no-reasoning          Don't split <think>…</think> into reasoning_content
                          (reasoning extraction is on by default)
  --reasoning-tag <tag>   Reasoning tag to split out (default: think)
  --cors                  Enable permissive (wildcard) CORS headers (off by default)
  --max-body-size <bytes> Max request body size (default: 10485760 = 10 MiB)
  --log-file <path>       Append a JSON-lines debug log (client request, the
                          transformed upstream request, and the response) to
                          <path> [env: PROXY_LOG_FILE]. Verbose; headers/tokens
                          are never logged.
  --max-log-size <bytes>  Stop logging once the log file hits this size
                          (default: 104857600 = 100 MiB)
  -h, --help              Show this help

Then point any OpenAI client at  http://<host>:<port>/v1  and pass tools as usual.`;

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || args.h) {
    console.log(HELP);
    return;
  }

  const upstreamBaseURL =
    (args.upstream as string) || process.env.UPSTREAM_BASE_URL || "";
  if (!upstreamBaseURL) {
    console.error("Error: --upstream <url> (or UPSTREAM_BASE_URL) is required.\n");
    console.error(HELP);
    process.exitCode = 1;
    return;
  }

  const port = Number(args.port ?? process.env.PORT ?? 8787);
  const host = (args.host as string) || process.env.HOST || "127.0.0.1";

  const options: ProxyOptions = {
    upstreamBaseURL,
    upstreamApiKey: (args["upstream-key"] as string) || process.env.UPSTREAM_API_KEY,
    apiKey: (args["api-key"] as string) || process.env.PROXY_API_KEY,
    basePath: (args["base-path"] as string) || "/v1",
    toolCallTag: (args.tag as string) || undefined,
    includeExamples: args.examples === false ? false : undefined,
    systemInjection:
      args["system-injection"] === "prepend" ? "prepend" : undefined,
    xmlToolCalls: args["xml-tool-calls"] === true ? true : undefined,
    reasoning: args.reasoning === false ? false : undefined,
    reasoningTag: (args["reasoning-tag"] as string) || undefined,
    cors: args.cors === true ? true : undefined,
    maxBodySize: args["max-body-size"]
      ? Number(args["max-body-size"])
      : undefined,
    logFile: (args["log-file"] as string) || process.env.PROXY_LOG_FILE,
    maxLogBytes: args["max-log-size"] ? Number(args["max-log-size"]) : undefined,
  };

  const server = createProxyServer(options);
  server.listen(port, host, () => {
    console.log(`llm-tool-proxy listening on http://${host}:${port}${options.basePath}`);
    console.log(`  → upstream: ${upstreamBaseURL}`);
    if (options.apiKey) console.log("  → client auth: required");
    if (options.logFile) console.log(`  → debug log: ${options.logFile}`);
  });

  const shutdown = () => {
    server.closeLog();
    server.close(() => process.exit(0));
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main();
