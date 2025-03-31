const { exec } = require('child_process');

// Run the test script
console.log('Running Enhanced Valuation Test...');
exec('python3 test_enhanced_valuation.py', (error, stdout, stderr) => {
  if (error) {
    console.error(`Error: ${error.message}`);
    console.error(`Stderr: ${stderr}`);
    return;
  }
  
  console.log('Test Results:');
  console.log(stdout);
});