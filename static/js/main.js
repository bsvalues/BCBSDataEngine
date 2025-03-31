/**
 * BCBS Property Valuation - Main JavaScript
 * This file contains client-side functionality for the BCBS Property Valuation application.
 */

// Wait for DOM content to be loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Handle dismissible alerts
    const alertList = document.querySelectorAll('.alert-dismissible');
    alertList.forEach(function(alert) {
        const closeButton = alert.querySelector('.btn-close');
        if (closeButton) {
            closeButton.addEventListener('click', function() {
                alert.classList.add('fade');
                setTimeout(function() {
                    alert.remove();
                }, 150);
            });
        }
    });

    // Handle API key copy functionality
    const apiKeyElements = document.querySelectorAll('.api-key');
    apiKeyElements.forEach(function(element) {
        const copyButton = element.nextElementSibling;
        if (copyButton && copyButton.classList.contains('btn-copy')) {
            copyButton.addEventListener('click', function() {
                const keyText = element.textContent.trim();
                navigator.clipboard.writeText(keyText).then(function() {
                    // Change button text and icon temporarily
                    const originalHTML = copyButton.innerHTML;
                    copyButton.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    copyButton.classList.add('btn-success');
                    copyButton.classList.remove('btn-outline-secondary');
                    
                    setTimeout(function() {
                        copyButton.innerHTML = originalHTML;
                        copyButton.classList.remove('btn-success');
                        copyButton.classList.add('btn-outline-secondary');
                    }, 2000);
                });
            });
        }
    });

    // Handle confirmation dialogs
    const confirmButtons = document.querySelectorAll('[data-confirm]');
    confirmButtons.forEach(function(button) {
        button.addEventListener('click', function(event) {
            const message = button.getAttribute('data-confirm');
            if (message && !confirm(message)) {
                event.preventDefault();
            }
        });
    });

    // Initialize date pickers
    const datePickers = document.querySelectorAll('.datepicker');
    datePickers.forEach(function(input) {
        // If you want to add a date picker library, you would initialize it here
        // For now, we'll just use native date inputs
        input.type = 'date';
    });

    // Format currency values
    const currencyElements = document.querySelectorAll('.currency');
    currencyElements.forEach(function(element) {
        const value = parseFloat(element.textContent.trim());
        if (!isNaN(value)) {
            element.textContent = formatCurrency(value);
        }
    });

    // Format percentage values
    const percentageElements = document.querySelectorAll('.percentage');
    percentageElements.forEach(function(element) {
        const value = parseFloat(element.textContent.trim());
        if (!isNaN(value)) {
            element.textContent = formatPercentage(value);
        }
    });

    // Handle confidence level indicators
    const confidenceIndicators = document.querySelectorAll('.confidence-indicator');
    confidenceIndicators.forEach(function(indicator) {
        const value = parseFloat(indicator.getAttribute('data-value'));
        if (!isNaN(value)) {
            const confidenceLevel = indicator.querySelector('.confidence-level');
            if (confidenceLevel) {
                confidenceLevel.style.width = (value * 100) + '%';
                
                // Set color based on confidence level
                if (value >= 0.9) {
                    confidenceLevel.style.backgroundColor = 'var(--success-color)';
                } else if (value >= 0.7) {
                    confidenceLevel.style.backgroundColor = 'var(--info-color)';
                } else if (value >= 0.5) {
                    confidenceLevel.style.backgroundColor = 'var(--warning-color)';
                } else {
                    confidenceLevel.style.backgroundColor = 'var(--danger-color)';
                }
            }
        }
    });

    // Toggle password visibility
    const togglePasswordButtons = document.querySelectorAll('.toggle-password');
    togglePasswordButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            const passwordField = document.querySelector(button.getAttribute('data-target'));
            if (passwordField) {
                const type = passwordField.getAttribute('type') === 'password' ? 'text' : 'password';
                passwordField.setAttribute('type', type);
                
                const icon = button.querySelector('i');
                if (icon) {
                    if (type === 'password') {
                        icon.classList.remove('fa-eye-slash');
                        icon.classList.add('fa-eye');
                    } else {
                        icon.classList.remove('fa-eye');
                        icon.classList.add('fa-eye-slash');
                    }
                }
            }
        });
    });

    // Initialize property search form
    const propertySearchForm = document.getElementById('property-search-form');
    if (propertySearchForm) {
        // Clear filters button
        const clearFiltersButton = document.getElementById('clear-filters');
        if (clearFiltersButton) {
            clearFiltersButton.addEventListener('click', function(event) {
                event.preventDefault();
                
                // Reset all form inputs
                const inputs = propertySearchForm.querySelectorAll('input, select');
                inputs.forEach(function(input) {
                    if (input.type === 'text' || input.type === 'number' || input.tagName === 'SELECT') {
                        input.value = '';
                    }
                });
                
                // Submit the form
                propertySearchForm.submit();
            });
        }
    }

    // Initialize charts if Chart.js is available and there are chart containers
    if (typeof Chart !== 'undefined') {
        initializeCharts();
    }
});

/**
 * Initialize charts for the dashboard
 */
function initializeCharts() {
    // Property Type Distribution Chart
    const propertyTypeChart = document.getElementById('property-type-chart');
    if (propertyTypeChart) {
        const ctx = propertyTypeChart.getContext('2d');
        const data = JSON.parse(propertyTypeChart.getAttribute('data-chart'));
        
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: [
                        'rgba(13, 110, 253, 0.8)',
                        'rgba(25, 135, 84, 0.8)',
                        'rgba(13, 202, 240, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(220, 53, 69, 0.8)'
                    ],
                    borderColor: [
                        'rgba(13, 110, 253, 1)',
                        'rgba(25, 135, 84, 1)',
                        'rgba(13, 202, 240, 1)',
                        'rgba(255, 193, 7, 1)',
                        'rgba(220, 53, 69, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Valuation Method Distribution Chart
    const valuationMethodChart = document.getElementById('valuation-method-chart');
    if (valuationMethodChart) {
        const ctx = valuationMethodChart.getContext('2d');
        const data = JSON.parse(valuationMethodChart.getAttribute('data-chart'));
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Valuations',
                    data: data.values,
                    backgroundColor: 'rgba(13, 110, 253, 0.7)',
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // Neighborhood Distribution Chart
    const neighborhoodChart = document.getElementById('neighborhood-chart');
    if (neighborhoodChart) {
        const ctx = neighborhoodChart.getContext('2d');
        const data = JSON.parse(neighborhoodChart.getAttribute('data-chart'));
        
        new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Properties',
                    data: data.values,
                    backgroundColor: 'rgba(13, 202, 240, 0.7)',
                    borderColor: 'rgba(13, 202, 240, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // Agent Status Chart
    const agentStatusChart = document.getElementById('agent-status-chart');
    if (agentStatusChart) {
        const ctx = agentStatusChart.getContext('2d');
        const data = JSON.parse(agentStatusChart.getAttribute('data-chart'));
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: [
                        'rgba(25, 135, 84, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(220, 53, 69, 0.8)'
                    ],
                    borderColor: [
                        'rgba(25, 135, 84, 1)',
                        'rgba(255, 193, 7, 1)',
                        'rgba(220, 53, 69, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                    }
                }
            }
        });
    }

    // Property valuation history chart
    const valuationHistoryChart = document.getElementById('valuation-history-chart');
    if (valuationHistoryChart) {
        const ctx = valuationHistoryChart.getContext('2d');
        const data = JSON.parse(valuationHistoryChart.getAttribute('data-chart'));
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Estimated Value',
                    data: data.values,
                    borderColor: 'rgba(13, 110, 253, 1)',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
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
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'Value: ' + formatCurrency(context.raw);
                            }
                        }
                    }
                }
            }
        });
    }
}

/**
 * Format a number as currency (USD)
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
 * Format a date in a readable format
 * @param {Date} date - The date to format
 * @returns {string} - Formatted date string
 */
function formatDate(date) {
    if (!(date instanceof Date)) {
        date = new Date(date);
    }
    
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    }).format(date);
}

/**
 * Format a percentage value
 * @param {number} value - Value to format as percentage (0-1)
 * @returns {string} - Formatted percentage string
 */
function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(value);
}

/**
 * Format a number with thousands separators
 * @param {number} value - The value to format
 * @returns {string} - Formatted number string
 */
function formatNumber(value) {
    return new Intl.NumberFormat('en-US').format(value);
}

/**
 * Get a color class based on a value (for visual indicators)
 * @param {number} value - The value (typically 0-1)
 * @param {boolean} inverse - If true, higher values get "danger" colors
 * @returns {string} - Bootstrap color class
 */
function getColorClass(value, inverse = false) {
    if (inverse) {
        value = 1 - value;
    }
    
    if (value >= 0.8) {
        return 'success';
    } else if (value >= 0.6) {
        return 'info';
    } else if (value >= 0.4) {
        return 'warning';
    } else {
        return 'danger';
    }
}

/**
 * Debounce function to limit how often a function can be called
 * @param {Function} func - The function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} - Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}