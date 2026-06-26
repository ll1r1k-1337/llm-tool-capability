import { extractNameArgs, tryParseJson } from "./dist/index.js";

// Scenario 1: Model outputs arguments as empty string
const parsed1 = JSON.parse('{"name": "test", "arguments": ""}');
const result1 = extractNameArgs(parsed1);
console.log("Scenario 1 - Model outputs empty string arguments:");
console.log("  extractNameArgs result:", result1);
console.log("  Arguments value:", result1?.arguments);

// Scenario 2: Without the empty string check, what happens?
const argsString = result1?.arguments || "";
console.log("\nWithout empty string check:");
console.log("  tryParseJson('') returns:", tryParseJson(""));

console.log("\nWith empty string check:");
console.log("  tryParseJson(argsString === '' ? '{}' : argsString) returns:", 
  tryParseJson(argsString === "" ? "{}" : argsString));
