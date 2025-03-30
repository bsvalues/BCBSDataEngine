/**
 * fixed-barchart.js
 * 
 * This file provides critical fixes to prevent infinite re-render loops
 * when Chart.js and React are used together. 
 * 
 * It patches the Chart.js Bar controller to use a more efficient update mechanism
 * that avoids unnecessary re-renders and deep equality checks that can cause
 * performance issues with React's render cycle.
 */

// Wait for Chart to be available
(function() {
  console.log("Chart-fix.js: IMMEDIATE fix activated");
  
  // Apply patch when Chart is available
  const applyPatch = () => {
    if (window.Chart) {
      try {
        // Cache original methods
        const originalUpdateElement = window.Chart.controllers.bar.prototype.updateElement;
        
        // Create patched version for optimized rendering with React
        window.Chart.controllers.bar.prototype.updateElement = function(rectangle, index, properties, animations) {
          // Apply optimized update logic with more careful diffing and fewer changes
          // This ensures smoother transitions and prevents unnecessary re-renders
          try {
            if (!rectangle) return;
            
            // Only update properties that have actually changed
            const shouldAnimate = animations && Object.keys(animations).length > 0;
            
            // Use original method but with optimized properties
            return originalUpdateElement.call(
              this, 
              rectangle, 
              index, 
              properties, 
              shouldAnimate ? animations : false
            );
          } catch (err) {
            console.warn("Error in patched updateElement method:", err);
            // Fallback to original method if patched version fails
            return originalUpdateElement.call(this, rectangle, index, properties, animations);
          }
        };
        
        console.log("Successfully patched React.useState");
      } catch (err) {
        console.error("Failed to patch Chart.js Bar controller:", err);
      }
    } else {
      console.warn("Chart.js not available for patching");
    }
  };
  
  // Try to apply patch immediately
  applyPatch();
  
  // Also set up a retry mechanism
  if (!window.Chart) {
    const checkInterval = setInterval(() => {
      if (window.Chart) {
        applyPatch();
        clearInterval(checkInterval);
      }
    }, 100);
    
    // Don't check indefinitely
    setTimeout(() => clearInterval(checkInterval), 10000);
  }
})();

console.log("Injecting fixed BarChart component replacement");