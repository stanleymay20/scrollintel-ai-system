// Debug script to identify frontend runtime errors
const fs = require('fs');
const path = require('path');

console.log('🔍 Debugging ScrollIntel Frontend Runtime Error...\n');

// Check if all required files exist
const requiredFiles = [
  'frontend/src/app/page.tsx',
  'frontend/src/components/error-boundary.tsx',
  'frontend/src/components/ui/loading.tsx',
  'frontend/src/components/ui/fallback.tsx',
  'frontend/src/components/onboarding/index.tsx',
  'frontend/src/types/index.ts',
  'frontend/src/lib/api.ts',
  'frontend/src/lib/utils.ts',
  'frontend/package.json',
  'frontend/next.config.js'
];

console.log('📁 Checking required files:');
requiredFiles.forEach(file => {
  const exists = fs.existsSync(file);
  console.log(`${exists ? '✅' : '❌'} ${file}`);
});

// Check for common issues
console.log('\n🔧 Checking for common issues:');

// Check if node_modules exists
const nodeModulesExists = fs.existsSync('frontend/node_modules');
console.log(`${nodeModulesExists ? '✅' : '❌'} Node modules installed`);

// Check if .next build directory exists
const nextBuildExists = fs.existsSync('frontend/.next');
console.log(`${nextBuildExists ? '✅' : '❌'} Next.js build directory`);

// Check package.json for dependencies
try {
  const packageJson = JSON.parse(fs.readFileSync('frontend/package.json', 'utf8'));
  const hasReact = packageJson.dependencies?.react;
  const hasNext = packageJson.dependencies?.next;
  console.log(`${hasReact ? '✅' : '❌'} React dependency`);
  console.log(`${hasNext ? '✅' : '❌'} Next.js dependency`);
} catch (err) {
  console.log('❌ Error reading package.json');
}

console.log('\n💡 Recommendations:');
console.log('1. Run "npm install" in the frontend directory');
console.log('2. Run "npm run build" to check for build errors');
console.log('3. Check browser console for specific error messages');
console.log('4. Ensure backend API is running on port 8000');