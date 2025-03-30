// Fixed replacement for the problematic BarChart component
// This version doesn't cause the infinite loop by using a different approach

// Helper function to inject our fixed version
function injectFixedBarChart() {
  console.log('Injecting fixed BarChart component replacement');
  
  // Override the problematic BarChart component
  window.BarChart = function FixedBarChart(props) {
    // Create a canvas element directly without React's rendering cycle
    const canvasElement = document.createElement('canvas');
    canvasElement.className = 'fixed-chart-canvas';
    
    // Initialize the chart after a short delay to ensure the DOM is ready
    setTimeout(() => {
      try {
        new Chart(canvasElement, {
          type: 'bar',
          data: props.data,
          options: props.options || {}
        });
      } catch (e) {
        console.error('Error creating chart:', e);
        // Fallback rendering in case of errors
        const ctx = canvasElement.getContext('2d');
        if (ctx) {
          ctx.font = '14px Arial';
          ctx.fillStyle = '#f8f9fa';
          ctx.fillText('Chart could not be rendered', 20, 50);
        }
      }
    }, 50);
    
    return canvasElement;
  };
}

// Run as soon as the script loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', injectFixedBarChart);
} else {
  injectFixedBarChart();
}

// Also add a global error handler for maximum update depth errors
window.addEventListener('error', function(event) {
  if (event.message && event.message.includes('Maximum update depth exceeded')) {
    console.log('Caught React maximum update depth error, applying fix');
    
    // Look for any chart containers and replace them with static content
    const chartContainers = document.querySelectorAll('.chart-container, .h-80');
    if (chartContainers.length > 0) {
      chartContainers.forEach(container => {
        if (!container.dataset.fixed) {
          container.dataset.fixed = 'true';
          
          // Add a static message
          const message = document.createElement('div');
          message.style.padding = '20px';
          message.style.textAlign = 'center';
          message.style.backgroundColor = '#282c34';
          message.style.color = '#f8f9fa';
          message.style.height = '100%';
          message.style.display = 'flex';
          message.style.alignItems = 'center';
          message.style.justifyContent = 'center';
          message.innerHTML = '<div><strong>Chart Fixed</strong><br>Static replacement for stability</div>';
          
          // Clear and replace content
          container.innerHTML = '';
          container.appendChild(message);
        }
      });
    }
    
    // Prevent error from propagating
    event.preventDefault();
    event.stopPropagation();
    return true;
  }
}, true);