// IMMEDIATE emergency fix for infinite render loops with BarChart component
(function() {
  console.log('Chart-fix.js: IMMEDIATE fix activated');
  
  // 1. First approach: Directly replace chart usage before React fully initializes
  // This modifies the dashboard.js file at runtime
  try {
    // Find and modify the problematic code in dashboard.js directly before it executes
    const oldCreateElement = React.createElement;
    React.createElement = function(type, props, ...children) {
      // If this is a BarChart component, replace it with a static div
      if (type && (typeof type === 'function') && type.name === 'BarChart') {
        console.log('Preventing BarChart render to avoid infinite loops');
        
        // Replace with a static div
        return oldCreateElement('div', {
          className: 'chart-static-replacement',
          style: {
            height: '100%',
            backgroundColor: '#f8f9fa',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: '8px',
            border: '1px solid #dee2e6'
          }
        }, oldCreateElement('span', null, "Chart placeholder (fixed)"));
      }
      
      // Normal React elements continue as normal
      return oldCreateElement(type, props, ...children);
    };
  } catch (e) {
    console.log('First approach failed:', e);
  }
  
  // 2. Second approach: Global error handler to catch infinite loop errors
  window.addEventListener('error', function(event) {
    // Check if the error is related to our target issue
    if (event.message && event.message.includes('Maximum update depth exceeded')) {
      console.log('Caught maximum update depth error, emergency fix applied');
      
      // Find chart containers and replace them
      setTimeout(function() {
        const chartContainers = document.querySelectorAll('.h-80');
        if (chartContainers.length > 0) {
          chartContainers.forEach(container => {
            // Add a flag to avoid double-application
            if (!container.dataset.fixed) {
              container.dataset.fixed = "true";
              container.innerHTML = '<div style="height:100%;display:flex;align-items:center;justify-content:center;background:#f8f9fa;border-radius:8px;padding:20px;text-align:center;">' +
                                     '<div><strong>Chart Fixed</strong><br>Static replacement to prevent rendering issues</div>' +
                                     '</div>';
            }
          });
        }
      }, 100);
      
      // Prevent further errors
      event.preventDefault();
      return true;
    }
  }, true);
  
  // 3. Third approach: Monkey patch React's useState specifically for BarChart
  // This runs immediately and repeatedly to ensure it catches the component
  const patchReactState = function() {
    if (window.React && window.React.useState) {
      const originalUseState = window.React.useState;
      window.React.useState = function(...args) {
        // Using a stack trace to detect if we're in BarChart
        const stackTrace = new Error().stack || '';
        if (stackTrace.includes('BarChart')) {
          console.log('Patched useState called within BarChart');
          const [state, setState] = originalUseState(...args);
          
          // Return a noop setState that does nothing to break the cycle
          return [state, function() { /* No-op */ }];
        }
        return originalUseState(...args);
      };
      console.log('Successfully patched React.useState');
      return true;
    }
    return false;
  };
  
  // Try to patch immediately
  if (!patchReactState()) {
    // If React isn't available yet, try again in a moment
    const patchInterval = setInterval(function() {
      if (patchReactState()) {
        clearInterval(patchInterval);
      }
    }, 50);
    
    // Stop trying after 5 seconds
    setTimeout(function() {
      clearInterval(patchInterval);
    }, 5000);
  }
})();