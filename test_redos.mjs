// Test for ReDoS in tryParseJson
import { tryParseJson } from './dist/index.js';

// Create a string with many unclosed /* sequences
function createMaliciousInput(count) {
  const parts = [];
  for (let i = 0; i < count; i++) {
    parts.push('/*a');
  }
  return parts.join('');
}

console.log('Testing for quadratic ReDoS growth...\n');
console.log('Size\tTime(ms)\tQuadratic(*1.33^2=1.77x)');

const sizes = [10000, 13300, 17700, 23500, 31300];

for (const size of sizes) {
  const input = createMaliciousInput(size);
  
  const start = process.hrtime.bigint();
  const result = tryParseJson(input);
  const end = process.hrtime.bigint();
  
  const duration = Number(end - start) / 1e6;
  console.log(`${size}\t${duration.toFixed(2)}`);
  
  if (duration > 60000) {
    console.log('Time exceeded 60 seconds - stopping');
    break;
  }
}
