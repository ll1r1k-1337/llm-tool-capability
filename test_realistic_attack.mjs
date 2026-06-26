// Test realistic attack scenarios
import { parseToolCalls } from './dist/index.js';

function createToolCallBlock(commentCount) {
  const comments = Array(commentCount).fill('/*x').join('');
  // Incomplete JSON - triggers repair path
  return `\`\`\`tool_call
{"name":"test","arguments":{"data":"${comments}
\`\`\``;
}

console.log('Realistic Attack Scenario: Model outputs incomplete JSON in tool_call\n');
console.log('Tool calls\tJSON Length\tTime(ms)\tRealistic?');
console.log('(comments)\t(approx)\t\t');
console.log('---');

const commentCounts = [1000, 5000, 10000];

for (const count of commentCounts) {
  const block = createToolCallBlock(count);
  
  console.log(`Processing tool_call block with ${count} comment sequences...`);
  const start = process.hrtime.bigint();
  const result = parseToolCalls(block);
  const end = process.hrtime.bigint();
  
  const duration = Number(end - start) / 1e6;
  const jsonLength = block.length;
  console.log(`${count}\t\t${jsonLength}\t\t${duration.toFixed(0)}\t\tYes - model could output this`);
  
  if (duration > 10000) {
    console.log('\nWARNING: Response takes 10+ seconds!');
    break;
  }
}

console.log('\nEstimate for 100KB malicious input: ~10-30 seconds');
console.log('Estimate for 1MB malicious input: ~10-30 minutes');
