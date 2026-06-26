// Test the regex directly
const regex = /\/\*[\s\S]*?\*\//g;

function createMaliciousInput(count) {
  const parts = [];
  for (let i = 0; i < count; i++) {
    parts.push('/*a');
  }
  return parts.join('');
}

console.log('Testing regex directly: /\/\*[\s\S]*?\*\//g\n');
console.log('Size\tTime(ms)\tGrowth');

let prevTime = null;

const sizes = [5000, 10000, 15000, 20000];

for (const size of sizes) {
  const input = createMaliciousInput(size);
  
  const start = process.hrtime.bigint();
  const result = input.replace(regex, '');  // This is what tryParseJson does
  const end = process.hrtime.bigint();
  
  const duration = Number(end - start) / 1e6;
  let growth = prevTime !== null ? (duration / prevTime).toFixed(2) + 'x' : '';
  
  console.log(`${size}\t${duration.toFixed(2)}\t${growth}`);
  prevTime = duration;
  
  if (duration > 30000) break;
}
