// BCBS Values - Main JavaScript file

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    const popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Handle password confirmation validation
    const passwordField = document.getElementById('password');
    const confirmPasswordField = document.getElementById('confirm_password');
    
    if (passwordField && confirmPasswordField) {
        confirmPasswordField.addEventListener('input', function() {
            if (passwordField.value !== confirmPasswordField.value) {
                confirmPasswordField.setCustomValidity("Passwords don't match");
            } else {
                confirmPasswordField.setCustomValidity('');
            }
        });
        
        passwordField.addEventListener('input', function() {
            if (passwordField.value !== confirmPasswordField.value) {
                confirmPasswordField.setCustomValidity("Passwords don't match");
            } else {
                confirmPasswordField.setCustomValidity('');
            }
        });
    }
    
    // Add animation to progress bars
    const progressBars = document.querySelectorAll('.animate-progress-bar');
    progressBars.forEach(bar => {
        setTimeout(() => {
            const targetWidth = bar.getAttribute('data-width');
            bar.style.width = targetWidth + '%';
        }, 200);
    });
    
    // Handle property search form
    const searchForm = document.getElementById('property-search-form');
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            const searchInput = document.getElementById('search-input');
            if (searchInput && searchInput.value.trim() === '') {
                e.preventDefault();
                searchInput.classList.add('is-invalid');
            }
        });
    }
    
    // Add behavior for collapsible property details
    const propertyDetails = document.querySelectorAll('.property-details-toggle');
    propertyDetails.forEach(detail => {
        detail.addEventListener('click', function() {
            const target = document.querySelector(this.getAttribute('data-target'));
            if (target) {
                target.classList.toggle('show');
                
                // Update the toggle icon
                const icon = this.querySelector('i');
                if (icon) {
                    if (target.classList.contains('show')) {
                        icon.classList.remove('fa-chevron-down');
                        icon.classList.add('fa-chevron-up');
                    } else {
                        icon.classList.remove('fa-chevron-up');
                        icon.classList.add('fa-chevron-down');
                    }
                }
            }
        });
    });
    
    // Handle modal property details
    const propertyModal = document.getElementById('propertyDetailsModal');
    if (propertyModal) {
        propertyModal.addEventListener('show.bs.modal', function(event) {
            const button = event.relatedTarget;
            const propertyId = button.getAttribute('data-property-id');
            const propertyTitle = button.getAttribute('data-property-title');
            
            const modalTitle = propertyModal.querySelector('.modal-title');
            if (modalTitle && propertyTitle) {
                modalTitle.textContent = propertyTitle;
            }
            
            // Here you would normally fetch property details via AJAX
            // For demonstration, we'll just show a loading message
            const modalBody = propertyModal.querySelector('.modal-body');
            if (modalBody) {
                modalBody.innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-3">Loading property details...</p></div>';
                
                // Simulate loading data
                setTimeout(() => {
                    if (propertyId) {
                        // In a real application, this would be an AJAX call to get property data
                        fetchPropertyDetails(propertyId, modalBody);
                    }
                }, 1000);
            }
        });
    }
    
    // Function to fetch property details (simulated)
    function fetchPropertyDetails(propertyId, container) {
        // In a real application, this would be an AJAX call
        // For demo purposes, we're just setting some HTML content
        container.innerHTML = '<div class="alert alert-info">This would show detailed information for property #' + propertyId + '. In a real application, this data would be fetched from the server.</div>';
    }
    
    // Add event listeners for dashboard filters
    const filterInputs = document.querySelectorAll('.dashboard-filter');
    filterInputs.forEach(input => {
        input.addEventListener('change', function() {
            // In a real application, this would trigger a filter action
            console.log('Filter changed:', this.id, this.value);
        });
    });
});

// Function to format currency values
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

// Function to initialize charts (if Chart.js is included)
function initializeCharts() {
    if (typeof Chart !== 'undefined') {
        // Property Value Distribution Chart
        const valueDistributionCtx = document.getElementById('valueDistributionChart');
        if (valueDistributionCtx) {
            new Chart(valueDistributionCtx, {
                type: 'bar',
                data: {
                    labels: ['$100k-$200k', '$200k-$300k', '$300k-$400k', '$400k-$500k', '$500k+'],
                    datasets: [{
                        label: 'Properties',
                        data: [12, 19, 8, 5, 2],
                        backgroundColor: 'rgba(13, 110, 253, 0.7)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
        
        // Valuation Models Comparison Chart
        const valModelComparisonCtx = document.getElementById('valModelComparisonChart');
        if (valModelComparisonCtx) {
            new Chart(valModelComparisonCtx, {
                type: 'radar',
                data: {
                    labels: ['Accuracy', 'Processing Speed', 'Feature Coverage', 'Spatial Analysis', 'Market Trend Integration'],
                    datasets: [{
                        label: 'Basic Model',
                        data: [65, 90, 55, 30, 40],
                        backgroundColor: 'rgba(13, 202, 240, 0.2)',
                        borderColor: 'rgba(13, 202, 240, 1)',
                        pointBackgroundColor: 'rgba(13, 202, 240, 1)'
                    }, {
                        label: 'Advanced Model',
                        data: [85, 70, 80, 75, 85],
                        backgroundColor: 'rgba(25, 135, 84, 0.2)',
                        borderColor: 'rgba(25, 135, 84, 1)',
                        pointBackgroundColor: 'rgba(25, 135, 84, 1)'
                    }, {
                        label: 'GIS Enhanced',
                        data: [90, 60, 90, 95, 80],
                        backgroundColor: 'rgba(220, 53, 69, 0.2)',
                        borderColor: 'rgba(220, 53, 69, 1)',
                        pointBackgroundColor: 'rgba(220, 53, 69, 1)'
                    }]
                }
            });
        }
    }
}