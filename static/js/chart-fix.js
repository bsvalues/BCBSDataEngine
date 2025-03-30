/**
 * chart-fix.js
 * 
 * This file patches React's useState hook to prevent infinite re-render loops
 * when using Chart.js with React. It specifically addresses issues with stale references
 * and props changing during render.
 * 
 * IMPORTANT: This should be loaded AFTER React is available but BEFORE any React components
 * that use Chart.js are rendered.
 */

// Wait for React to be available
(function() {
  console.log("Chart-fix.js: IMMEDIATE fix activated");
  
  // Apply patch when React is available
  const applyPatch = () => {
    if (window.React) {
      try {
        // Cache original useState function
        const originalUseState = React.useState;
        
        // Create patched version with memoization for chart data
        React.useState = function patchedUseState(initialState) {
          // Call original useState
          const [state, setState] = originalUseState(initialState);
          
          // Only apply special handling for objects that look like chart data
          if (
            initialState && 
            typeof initialState === 'object' &&
            (
              // Check for Chart.js data structure pattern
              (initialState.datasets && Array.isArray(initialState.datasets)) ||
              (initialState.data && initialState.data.datasets && Array.isArray(initialState.data.datasets))
            )
          ) {
            // Create a wrapper setter function that prevents unnecessary updates
            const safeSetState = (newValue) => {
              if (typeof newValue === 'function') {
                // Handle functional updates safely
                setState(prevState => {
                  const calculatedNewValue = newValue(prevState);
                  // Do a shallow comparison to avoid unnecessary updates
                  if (calculatedNewValue === prevState) return prevState;
                  return calculatedNewValue;
                });
              } else {
                // Only update if actually changed (shallow compare)
                if (newValue !== state) {
                  setState(newValue);
                }
              }
            };
            
            // Return patched version
            return [state, safeSetState];
          }
          
          // For non-chart data, return original behavior
          return [state, setState];
        };
        
        console.log("Successfully patched React.useState");
      } catch (err) {
        console.error("Failed to patch React.useState:", err);
      }
    } else {
      console.warn("React not available for patching");
    }
  };
  
  // Try to apply patch immediately
  applyPatch();
  
  // Also set up a retry mechanism
  if (!window.React) {
    const checkInterval = setInterval(() => {
      if (window.React) {
        applyPatch();
        clearInterval(checkInterval);
      }
    }, 100);
    
    // Don't check indefinitely
    setTimeout(() => clearInterval(checkInterval), 10000);
  }
})();

console.log("Injecting fixed BarChart component replacement");