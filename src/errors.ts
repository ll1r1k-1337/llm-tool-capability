/** Base class for all errors thrown by this package. */
export class ToolCapabilityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ToolCapabilityError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** Thrown when the upstream endpoint returns a non-2xx HTTP response. */
export class UpstreamError extends ToolCapabilityError {
  /** The upstream HTTP status code. */
  readonly status: number;
  /** The upstream response body (so a proxy can relay it to the caller). */
  readonly body: string;
  constructor(status: number, statusText: string, body: string) {
    super(`Upstream returned ${status} ${statusText}: ${body.slice(0, 500)}`);
    this.name = "UpstreamError";
    this.status = status;
    this.body = body;
  }
}

/** Thrown when the agentic runner hits its iteration ceiling without finishing. */
export class MaxIterationsError extends ToolCapabilityError {
  readonly iterations: number;
  constructor(iterations: number) {
    super(
      `Tool runner exceeded maxIterations (${iterations}) without producing a final answer.`,
    );
    this.name = "MaxIterationsError";
    this.iterations = iterations;
  }
}

/** Thrown when the model calls a tool that has no registered handler. */
export class UnknownToolError extends ToolCapabilityError {
  readonly toolName: string;
  constructor(toolName: string) {
    super(`No handler registered for tool "${toolName}".`);
    this.name = "UnknownToolError";
    this.toolName = toolName;
  }
}

/** Thrown when a tool's arguments fail JSON parsing or schema validation. */
export class ToolArgumentsError extends ToolCapabilityError {
  readonly toolName: string;
  readonly rawArguments: string;
  readonly details: string;
  constructor(toolName: string, rawArguments: string, details: string) {
    super(`Invalid arguments for tool "${toolName}": ${details}`);
    this.name = "ToolArgumentsError";
    this.toolName = toolName;
    this.rawArguments = rawArguments;
    this.details = details;
  }
}
