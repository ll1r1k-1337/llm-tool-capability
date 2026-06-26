import Ajv from "ajv";
import type { ValidateFunction } from "ajv";
import type { ChatCompletionTool } from "./types.js";
import { tryParseJson } from "./parser.js";

export interface ValidationResult {
  valid: boolean;
  /** The parsed (and possibly coerced/defaulted) arguments object. */
  value: unknown;
  /** Human-readable error messages, empty when `valid`. */
  errors: string[];
}

export interface ToolValidatorOptions {
  /** Coerce scalar types (e.g. "5" -> 5) before validating. Default: `false`. */
  coerceTypes?: boolean;
  /** Apply `default` values from the schema. Default: `true`. */
  useDefaults?: boolean;
}

/**
 * Compiles each tool's parameter schema once and validates parsed arguments
 * against it. Unknown tools and tools without a schema validate trivially.
 */
export class ToolValidator {
  private readonly ajv: Ajv;
  private readonly validators = new Map<string, ValidateFunction>();
  private readonly known = new Set<string>();

  constructor(tools: ChatCompletionTool[], options: ToolValidatorOptions = {}) {
    this.ajv = new Ajv({
      allErrors: true,
      strict: false,
      coerceTypes: options.coerceTypes ?? false,
      useDefaults: options.useDefaults ?? true,
    });
    for (const tool of tools) {
      const { name, parameters } = tool.function;
      this.known.add(name);
      if (parameters && typeof parameters === "object") {
        // Strip $id/$ref-ish identity keys so two tools that reuse the same $id
        // don't collide in ajv's registry (which would silently drop the second
        // validator and let that tool validate trivially).
        const schema: Record<string, unknown> = { ...(parameters as object) };
        delete schema.$id;
        try {
          this.validators.set(name, this.ajv.compile(schema));
        } catch {
          // An un-compilable schema shouldn't break the whole runner; treat the
          // tool as "no schema" (validates trivially) rather than throwing here.
          this.validators.delete(name);
        }
      }
    }
  }

  hasTool(name: string): boolean {
    return this.known.has(name);
  }

  /** Validates a JSON-string of arguments against the named tool's schema. */
  validate(toolName: string, argumentsString: string): ValidationResult {
    const parsed = tryParseJson(argumentsString === "" ? "{}" : argumentsString);
    if (parsed === undefined) {
      return {
        valid: false,
        value: undefined,
        errors: ["Arguments are not valid JSON."],
      };
    }

    const validator = this.validators.get(toolName);
    if (!validator) {
      // No schema (or unknown tool) — nothing to check against.
      return { valid: true, value: parsed, errors: [] };
    }

    const ok = validator(parsed);
    if (ok) {
      return { valid: true, value: parsed, errors: [] };
    }
    const errors = (validator.errors ?? []).map((e) => {
      const path = e.instancePath || "(root)";
      return `${path} ${e.message ?? "is invalid"}`.trim();
    });
    return {
      valid: false,
      value: parsed,
      errors: errors.length > 0 ? errors : ["Arguments failed schema validation."],
    };
  }
}
