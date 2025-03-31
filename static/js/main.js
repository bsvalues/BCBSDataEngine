/**
 * Main JavaScript file for BCBS Values application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    
    // Initialize all popovers
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    const popoverList = [...popoverTriggerList].map(popoverTriggerEl => new bootstrap.Popover(popoverTriggerEl));
    
    // Handle active nav links
    setActiveNavLink();
    
    // Format currency values
    formatCurrencyValues();
    
    // Initialize any charts on the page
    initializeCharts();
});

/**
 * Set the active navigation link based on the current URL
 */
function setActiveNavLink() {
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        // Get the href attribute
        const href = link.getAttribute('href');
        
        // Check if the current location matches or starts with the link href
        if (currentLocation === href || 
            (href !== '/' && currentLocation.startsWith(href))) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

/**
 * Format currency values marked with the currency-value class
 */
function formatCurrencyValues() {
    const currencyElements = document.querySelectorAll('.currency-value');
    
    currencyElements.forEach(element => {
        const value = parseFloat(element.textContent);
        if (!isNaN(value)) {
            element.textContent = formatCurrency(value);
        }
    });
}

/**
 * Format a number as USD currency
 * @param {number} value - The value to format
 * @returns {string} - Formatted currency string
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

/**
 * Initialize any charts on the page using Chart.js
 */
function initializeCharts() {
    // Check if the valuationHistoryChart element exists
    const valuationHistoryChart = document.getElementById('valuationHistoryChart');
    if (valuationHistoryChart) {
        initializeValuationHistoryChart(valuationHistoryChart);
    }
    
    // Check if the propertyTypeChart element exists
    const propertyTypeChart = document.getElementById('propertyTypeChart');
    if (propertyTypeChart) {
        initializePropertyTypeChart(propertyTypeChart);
    }
}

/**
 * Initialize a chart showing valuation history
 * @param {HTMLElement} chartElement - The canvas element for the chart
 */
function initializeValuationHistoryChart(chartElement) {
    // This would typically come from an API or be rendered server-side
    // For now, we'll use sample data
    const chartData = {
        labels: chartElement.dataset.labels ? JSON.parse(chartElement.dataset.labels) : [],
        values: chartElement.dataset.values ? JSON.parse(chartElement.dataset.values) : []
    };
    
    if (chartData.labels.length === 0 || chartData.values.length === 0) {
        return;
    }
    
    new Chart(chartElement, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: 'Property Value',
                data: chartData.values,
                borderColor: '#3f51b5',
                backgroundColor: 'rgba(63, 81, 181, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return '$' + context.parsed.y.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

/**
 * Initialize a chart showing property type distribution
 * @param {HTMLElement} chartElement - The canvas element for the chart
 */
function initializePropertyTypeChart(chartElement) {
    // This would typically come from an API or be rendered server-side
    // For now, we'll use sample data
    const chartData = {
        labels: chartElement.dataset.labels ? JSON.parse(chartElement.dataset.labels) : [],
        values: chartElement.dataset.values ? JSON.parse(chartElement.dataset.values) : []
    };
    
    if (chartData.labels.length === 0 || chartData.values.length === 0) {
        return;
    }
    
    new Chart(chartElement, {
        type: 'doughnut',
        data: {
            labels: chartData.labels,
            datasets: [{
                data: chartData.values,
                backgroundColor: [
                    '#3f51b5',
                    '#f44336',
                    '#4caf50',
                    '#ff9800',
                    '#9c27b0',
                    '#607d8b'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
}