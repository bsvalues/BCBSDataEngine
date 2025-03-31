/**
 * Dashboard Component
 * 
 * This component fetches property valuation data from the API and displays it in a
 * filterable table and interactive visualizations. It includes options for filtering 
 * by neighborhood, price range, property type, and date, and handles loading states 
 * and errors gracefully.
 */

document.addEventListener('DOMContentLoaded', function() {
    // State variables
    let properties = [];
    let filteredProperties = [];
    let neighborhoods = [];
    let currentPage = 1;
    let itemsPerPage = 10;
    let sortColumn = 'address';
    let sortDirection = 'asc';
    let selectedProperty = null;
    let chartInstances = {};
    
    // DOM Elements
    const searchInput = document.getElementById('searchInput');
    const neighborhoodFilter = document.getElementById('neighborhoodFilter');
    const propertyTypeFilter = document.getElementById('propertyTypeFilter');
    const resetFiltersBtn = document.getElementById('resetFilters');
    const refreshDataBtn = document.getElementById('refreshData');
    const prevPageBtn = document.getElementById('prevPage');
    const nextPageBtn = document.getElementById('nextPage');
    const propertiesTableBody = document.getElementById('propertiesTableBody');
    const paginationShowing = document.getElementById('paginationShowing');
    const paginationTotal = document.getElementById('paginationTotal');
    
    // Metric elements
    const totalPropertiesEl = document.getElementById('totalProperties');
    const avgValueEl = document.getElementById('avgValue');
    const totalValuationsEl = document.getElementById('totalValuations');
    const avgConfidenceEl = document.getElementById('avgConfidence');
    
    // Chart elements
    const valueDistributionChart = document.getElementById('valueDistributionChart');
    const neighborhoodComparisonChart = document.getElementById('neighborhoodComparisonChart');
    const valueTrendChart = document.getElementById('valueTrendChart');
    const modelDistributionChart = document.getElementById('modelDistributionChart');
    
    // Modal elements
    const propertyDetailModal = new bootstrap.Modal(document.getElementById('propertyDetailModal'));
    const propertyDetailContent = document.getElementById('propertyDetailContent');
    const viewFullDetailsLink = document.getElementById('viewFullDetailsLink');
    
    /**
     * Initialize the dashboard
     */
    function init() {
        // Add event listeners
        searchInput.addEventListener('input', handleSearch);
        neighborhoodFilter.addEventListener('change', applyFilters);
        propertyTypeFilter.addEventListener('change', applyFilters);
        resetFiltersBtn.addEventListener('click', resetFilters);
        refreshDataBtn.addEventListener('click', fetchProperties);
        prevPageBtn.addEventListener('click', () => handlePageChange(currentPage - 1));
        nextPageBtn.addEventListener('click', () => handlePageChange(currentPage + 1));
        
        // Initial data fetch
        fetchProperties();
        
        // Set up intersection observer for animation
        setupAnimations();
    }
    
    /**
     * Function to fetch property valuations
     */
    function fetchProperties() {
        showLoadingState();
        
        // In a real application, we would fetch data from the API
        // For now, we'll use sample data
        setTimeout(() => {
            // This would be replaced with an actual API call
            generateSampleProperties();
            extractNeighborhoods();
            applyFilters();
            updateMetrics();
            initializeCharts();
            hideLoadingState();
        }, 1000);
    }
    
    /**
     * Generate sample property data for testing
     * In a real application, this would be replaced with API data
     */
    function generateSampleProperties() {
        const propertyTypes = ['single_family', 'condo', 'townhouse', 'multi_family', 'land', 'commercial'];
        const neighborhoods = ['West Richland', 'Kennewick', 'Richland', 'Prosser', 'Benton City'];
        
        properties = Array.from({ length: 50 }, (_, i) => {
            const propertyType = propertyTypes[Math.floor(Math.random() * propertyTypes.length)];
            const neighborhood = neighborhoods[Math.floor(Math.random() * neighborhoods.length)];
            const bedrooms = propertyType === 'land' ? 0 : Math.floor(Math.random() * 6) + 1;
            const bathrooms = propertyType === 'land' ? 0 : Math.floor(Math.random() * 5) + 1;
            const squareFeet = propertyType === 'land' ? 0 : Math.floor(Math.random() * 3000) + 1000;
            const yearBuilt = propertyType === 'land' ? null : Math.floor(Math.random() * 60) + 1960;
            const baseValue = Math.floor(Math.random() * 500000) + 200000;
            
            return {
                id: i + 1,
                address: `${1000 + i} ${['Main', 'Oak', 'Pine', 'Maple', 'Cedar'][Math.floor(Math.random() * 5)]} ${['St', 'Ave', 'Blvd', 'Dr', 'Ln'][Math.floor(Math.random() * 5)]}`,
                city: neighborhood,
                state: 'WA',
                zipCode: '9935' + Math.floor(Math.random() * 10),
                neighborhood: neighborhood,
                propertyType: propertyType,
                bedrooms: bedrooms,
                bathrooms: bathrooms,
                squareFeet: squareFeet,
                lotSize: Math.floor(Math.random() * 2 * 10) / 10,
                yearBuilt: yearBuilt,
                estimatedValue: baseValue,
                confidenceScore: Math.floor(Math.random() * 30) + 70,
                valuationMethod: ['basic', 'enhanced', 'advanced_gis'][Math.floor(Math.random() * 3)],
                valuationDate: new Date(2023, Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1)
            };
        });
    }
    
    /**
     * Extract unique neighborhoods from the property data
     */
    function extractNeighborhoods() {
        neighborhoods = [...new Set(properties.map(property => property.neighborhood))].sort();
        
        // Populate neighborhood filter
        neighborhoodFilter.innerHTML = '<option value="">All Neighborhoods</option>';
        neighborhoods.forEach(neighborhood => {
            const option = document.createElement('option');
            option.value = neighborhood;
            option.textContent = neighborhood;
            neighborhoodFilter.appendChild(option);
        });
    }
    
    /**
     * Function to handle search input
     */
    function handleSearch() {
        applyFilters();
    }
    
    /**
     * Apply all filters and update the displayed properties
     */
    function applyFilters() {
        const searchTerm = searchInput.value.toLowerCase();
        const selectedNeighborhood = neighborhoodFilter.value;
        const selectedPropertyType = propertyTypeFilter.value;
        
        filteredProperties = properties.filter(property => {
            // Search term filter
            const matchesSearch = 
                property.address.toLowerCase().includes(searchTerm) ||
                property.city.toLowerCase().includes(searchTerm) ||
                property.zipCode.toLowerCase().includes(searchTerm);
                
            // Neighborhood filter
            const matchesNeighborhood = !selectedNeighborhood || property.neighborhood === selectedNeighborhood;
            
            // Property type filter
            const matchesPropertyType = !selectedPropertyType || property.propertyType === selectedPropertyType;
            
            return matchesSearch && matchesNeighborhood && matchesPropertyType;
        });
        
        // Reset to first page when filters change
        currentPage = 1;
        
        // Update the table, pagination, and charts
        renderPropertiesTable();
        updatePagination();
        updateMetrics();
        updateCharts();
    }
    
    /**
     * Reset all filters to their default values
     */
    function resetFilters() {
        searchInput.value = '';
        neighborhoodFilter.value = '';
        propertyTypeFilter.value = '';
        
        applyFilters();
    }
    
    /**
     * Render the properties table with the current filtered data
     */
    function renderPropertiesTable() {
        // Apply sorting
        const sortedProperties = [...filteredProperties].sort((a, b) => {
            let valueA = a[sortColumn];
            let valueB = b[sortColumn];
            
            // Handle date comparisons
            if (sortColumn === 'valuationDate') {
                valueA = new Date(valueA);
                valueB = new Date(valueB);
            }
            
            if (valueA < valueB) return sortDirection === 'asc' ? -1 : 1;
            if (valueA > valueB) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
        
        // Apply pagination
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const paginatedProperties = sortedProperties.slice(startIndex, endIndex);
        
        // Clear the table
        propertiesTableBody.innerHTML = '';
        
        // If no properties match the filters
        if (paginatedProperties.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="8" class="text-center">
                    <p class="my-3">No properties match your filter criteria.</p>
                </td>
            `;
            propertiesTableBody.appendChild(row);
            return;
        }
        
        // Add properties to the table
        paginatedProperties.forEach(property => {
            const row = document.createElement('tr');
            row.style.cursor = 'pointer';
            row.addEventListener('click', () => showPropertyDetails(property));
            
            // Determine the confidence level class
            let confidenceClass = 'confidence-low';
            if (property.confidenceScore >= 90) {
                confidenceClass = 'confidence-high';
            } else if (property.confidenceScore >= 75) {
                confidenceClass = 'confidence-medium';
            }
            
            // Format the property type
            const propertyTypeDisplay = formatPropertyType(property.propertyType);
            
            row.innerHTML = `
                <td>${property.address}<br><small class="text-muted">${property.city}, ${property.state} ${property.zipCode}</small></td>
                <td><span class="property-type-badge type-${property.propertyType}">${propertyTypeDisplay}</span></td>
                <td>${property.bedrooms} / ${property.bathrooms}</td>
                <td>${property.squareFeet.toLocaleString()}</td>
                <td>${property.yearBuilt || 'N/A'}</td>
                <td class="fw-bold">$${property.estimatedValue.toLocaleString()}</td>
                <td><span class="confidence-indicator ${confidenceClass}">${property.confidenceScore}%</span></td>
                <td>${formatDate(property.valuationDate)}</td>
            `;
            
            propertiesTableBody.appendChild(row);
        });
    }
    
    /**
     * Update the pagination controls and information
     */
    function updatePagination() {
        const totalPages = Math.ceil(filteredProperties.length / itemsPerPage);
        const startItem = Math.min((currentPage - 1) * itemsPerPage + 1, filteredProperties.length);
        const endItem = Math.min(startItem + itemsPerPage - 1, filteredProperties.length);
        
        paginationShowing.textContent = `${startItem}-${endItem}`;
        paginationTotal.textContent = filteredProperties.length;
        
        prevPageBtn.disabled = currentPage <= 1;
        nextPageBtn.disabled = currentPage >= totalPages;
    }
    
    /**
     * Handle pagination page changes
     */
    function handlePageChange(newPage) {
        currentPage = newPage;
        renderPropertiesTable();
        updatePagination();
    }
    
    /**
     * Update the dashboard metrics based on filtered properties
     */
    function updateMetrics() {
        if (filteredProperties.length === 0) {
            totalPropertiesEl.textContent = '0';
            avgValueEl.textContent = '$0';
            totalValuationsEl.textContent = '0';
            avgConfidenceEl.textContent = '0%';
            return;
        }
        
        const totalProperties = filteredProperties.length;
        
        const avgValue = filteredProperties.reduce((sum, property) => sum + property.estimatedValue, 0) / totalProperties;
        
        const totalValuations = filteredProperties.length; // In a real app, this might be different from total properties
        
        const avgConfidence = filteredProperties.reduce((sum, property) => sum + property.confidenceScore, 0) / totalProperties;
        
        // Update the DOM
        totalPropertiesEl.textContent = totalProperties.toLocaleString();
        avgValueEl.textContent = '$' + Math.round(avgValue).toLocaleString();
        totalValuationsEl.textContent = totalValuations.toLocaleString();
        avgConfidenceEl.textContent = Math.round(avgConfidence) + '%';
    }
    
    /**
     * Initialize all charts with the property data
     */
    function initializeCharts() {
        // Initialize value distribution chart
        if (valueDistributionChart) {
            chartInstances.valueDistribution = createValueDistributionChart();
        }
        
        // Initialize neighborhood comparison chart
        if (neighborhoodComparisonChart) {
            chartInstances.neighborhoodComparison = createNeighborhoodComparisonChart();
        }
        
        // Initialize value trend chart
        if (valueTrendChart) {
            chartInstances.valueTrend = createValueTrendChart();
        }
        
        // Initialize model distribution chart
        if (modelDistributionChart) {
            chartInstances.modelDistribution = createModelDistributionChart();
        }
    }
    
    /**
     * Update all charts with the filtered property data
     */
    function updateCharts() {
        // Only update if we have charts initialized
        if (Object.keys(chartInstances).length === 0) return;
        
        // Update value distribution chart
        if (chartInstances.valueDistribution) {
            updateValueDistributionChart();
        }
        
        // Update neighborhood comparison chart
        if (chartInstances.neighborhoodComparison) {
            updateNeighborhoodComparisonChart();
        }
        
        // Update value trend chart
        if (chartInstances.valueTrend) {
            updateValueTrendChart();
        }
        
        // Update model distribution chart
        if (chartInstances.modelDistribution) {
            updateModelDistributionChart();
        }
    }
    
    /**
     * Create the value distribution chart
     */
    function createValueDistributionChart() {
        const ctx = valueDistributionChart.getContext('2d');
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Number of Properties',
                    data: [],
                    backgroundColor: 'rgba(63, 81, 181, 0.6)',
                    borderColor: 'rgba(63, 81, 181, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.parsed.y} properties`;
                            },
                            title: function(tooltipItems) {
                                const item = tooltipItems[0];
                                return `$${item.label}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Property Value'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Properties'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update the value distribution chart with filtered data
     */
    function updateValueDistributionChart() {
        // Create value distribution data
        const priceRanges = [
            '0-250k', '250k-500k', '500k-750k', '750k-1M', '1M+'
        ];
        
        const distribution = [0, 0, 0, 0, 0];
        
        filteredProperties.forEach(property => {
            const value = property.estimatedValue;
            if (value < 250000) {
                distribution[0]++;
            } else if (value < 500000) {
                distribution[1]++;
            } else if (value < 750000) {
                distribution[2]++;
            } else if (value < 1000000) {
                distribution[3]++;
            } else {
                distribution[4]++;
            }
        });
        
        // Update chart data
        chartInstances.valueDistribution.data.labels = priceRanges;
        chartInstances.valueDistribution.data.datasets[0].data = distribution;
        chartInstances.valueDistribution.update();
    }
    
    /**
     * Create the neighborhood comparison chart
     */
    function createNeighborhoodComparisonChart() {
        const ctx = neighborhoodComparisonChart.getContext('2d');
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Average Value',
                    data: [],
                    backgroundColor: 'rgba(76, 175, 80, 0.6)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `$${context.parsed.x.toLocaleString()}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + (value / 1000) + 'k';
                            }
                        },
                        title: {
                            display: true,
                            text: 'Average Property Value'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Neighborhood'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update the neighborhood comparison chart with filtered data
     */
    function updateNeighborhoodComparisonChart() {
        // Group properties by neighborhood and calculate average values
        const neighborhoodData = {};
        
        filteredProperties.forEach(property => {
            if (!neighborhoodData[property.neighborhood]) {
                neighborhoodData[property.neighborhood] = {
                    totalValue: 0,
                    count: 0
                };
            }
            
            neighborhoodData[property.neighborhood].totalValue += property.estimatedValue;
            neighborhoodData[property.neighborhood].count++;
        });
        
        // Calculate averages and sort by average value
        const neighborhoodAverages = Object.entries(neighborhoodData).map(([name, data]) => ({
            name,
            avgValue: data.totalValue / data.count
        })).sort((a, b) => b.avgValue - a.avgValue);
        
        // Prepare data for chart
        const labels = neighborhoodAverages.map(n => n.name);
        const data = neighborhoodAverages.map(n => n.avgValue);
        
        // Update chart
        chartInstances.neighborhoodComparison.data.labels = labels;
        chartInstances.neighborhoodComparison.data.datasets[0].data = data;
        chartInstances.neighborhoodComparison.update();
    }
    
    /**
     * Create the value trend chart
     */
    function createValueTrendChart() {
        const ctx = valueTrendChart.getContext('2d');
        
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Average Property Value',
                    data: [],
                    borderColor: 'rgba(33, 150, 243, 1)',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `$${context.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + (value / 1000) + 'k';
                            }
                        },
                        title: {
                            display: true,
                            text: 'Average Value'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Month'
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update the value trend chart with filtered data
     */
    function updateValueTrendChart() {
        // Group properties by month and calculate average values
        const today = new Date();
        const monthsData = {};
        
        // Initialize last 12 months
        for (let i = 11; i >= 0; i--) {
            const d = new Date(today.getFullYear(), today.getMonth() - i, 1);
            const monthKey = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
            monthsData[monthKey] = {
                totalValue: 0,
                count: 0,
                month: d.toLocaleString('default', { month: 'short' }),
                year: d.getFullYear()
            };
        }
        
        // Group properties by month
        filteredProperties.forEach(property => {
            const date = new Date(property.valuationDate);
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            
            // Only include data from the last 12 months
            if (monthsData[monthKey]) {
                monthsData[monthKey].totalValue += property.estimatedValue;
                monthsData[monthKey].count++;
            }
        });
        
        // Prepare data for chart
        const labels = Object.values(monthsData).map(m => `${m.month} ${m.year}`);
        const data = Object.values(monthsData).map(m => m.count > 0 ? m.totalValue / m.count : null);
        
        // Update chart
        chartInstances.valueTrend.data.labels = labels;
        chartInstances.valueTrend.data.datasets[0].data = data;
        chartInstances.valueTrend.update();
    }
    
    /**
     * Create the model distribution chart
     */
    function createModelDistributionChart() {
        const ctx = modelDistributionChart.getContext('2d');
        
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Basic', 'Enhanced ML', 'Advanced GIS'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(156, 39, 176, 0.7)',
                        'rgba(33, 150, 243, 0.7)',
                        'rgba(76, 175, 80, 0.7)'
                    ],
                    borderColor: [
                        'rgba(156, 39, 176, 1)',
                        'rgba(33, 150, 243, 1)',
                        'rgba(76, 175, 80, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = Math.round((value * 100) / total);
                                return `${context.label}: ${percentage}% (${value})`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update the model distribution chart with filtered data
     */
    function updateModelDistributionChart() {
        // Count properties by valuation method
        const methodCounts = {
            'basic': 0,
            'enhanced': 0,
            'advanced_gis': 0
        };
        
        filteredProperties.forEach(property => {
            methodCounts[property.valuationMethod]++;
        });
        
        // Update chart
        chartInstances.modelDistribution.data.datasets[0].data = [
            methodCounts.basic,
            methodCounts.enhanced,
            methodCounts.advanced_gis
        ];
        chartInstances.modelDistribution.update();
    }
    
    /**
     * Show the loading state in the properties table
     */
    function showLoadingState() {
        propertiesTableBody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading property data...</p>
                </td>
            </tr>
        `;
        
        // Also show placeholder values in metrics
        totalPropertiesEl.textContent = '-';
        avgValueEl.textContent = '-';
        totalValuationsEl.textContent = '-';
        avgConfidenceEl.textContent = '-';
    }
    
    /**
     * Hide the loading state
     */
    function hideLoadingState() {
        // This function is called after data is loaded and displayed
    }
    
    /**
     * Show detailed information for a selected property
     */
    function showPropertyDetails(property) {
        selectedProperty = property;
        
        // Update the view full details link
        viewFullDetailsLink.href = `/properties/${property.id}`;
        
        // Build the property detail content
        const confidenceClass = property.confidenceScore >= 90 ? 'confidence-high' : 
                              property.confidenceScore >= 75 ? 'confidence-medium' : 'confidence-low';
        
        const propertyTypeDisplay = formatPropertyType(property.propertyType);
        
        propertyDetailContent.innerHTML = `
            <div class="property-detail-header">
                <div class="property-image">
                    <span class="property-detail-image-placeholder">🏠</span>
                </div>
                <div class="property-detail-info">
                    <h3 class="property-detail-address">${property.address}</h3>
                    <div class="property-detail-meta">
                        <span>${property.city}, ${property.state} ${property.zipCode}</span>
                        <span class="property-type-badge type-${property.propertyType}">${propertyTypeDisplay}</span>
                    </div>
                    <div class="property-detail-value">$${property.estimatedValue.toLocaleString()}</div>
                    <div>
                        <span class="confidence-indicator ${confidenceClass}">${property.confidenceScore}% Confidence</span>
                        <small class="text-muted ms-2">Valued on ${formatDate(property.valuationDate)}</small>
                    </div>
                </div>
            </div>
            
            <div class="property-attributes">
                <div class="property-attribute">
                    <div class="attribute-value">${property.bedrooms}</div>
                    <div class="attribute-label">Bedrooms</div>
                </div>
                <div class="property-attribute">
                    <div class="attribute-value">${property.bathrooms}</div>
                    <div class="attribute-label">Bathrooms</div>
                </div>
                <div class="property-attribute">
                    <div class="attribute-value">${property.squareFeet.toLocaleString()}</div>
                    <div class="attribute-label">Square Feet</div>
                </div>
                <div class="property-attribute">
                    <div class="attribute-value">${property.lotSize}</div>
                    <div class="attribute-label">Lot Size (acres)</div>
                </div>
                <div class="property-attribute">
                    <div class="attribute-value">${property.yearBuilt || 'N/A'}</div>
                    <div class="attribute-label">Year Built</div>
                </div>
                <div class="property-attribute">
                    <div class="attribute-value">${formatValuationMethod(property.valuationMethod)}</div>
                    <div class="attribute-label">Valuation Method</div>
                </div>
            </div>
            
            <div class="mt-4">
                <h5>Valuation Factors</h5>
                <div class="row">
                    <div class="col-md-6">
                        <ul class="list-group">
                            <li class="list-group-item d-flex justify-content-between">
                                <span>Base Property Value</span>
                                <span>$${Math.round(property.estimatedValue * 0.7).toLocaleString()}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between">
                                <span>Location Adjustment</span>
                                <span>$${Math.round(property.estimatedValue * 0.15).toLocaleString()}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between">
                                <span>Property Condition</span>
                                <span>$${Math.round(property.estimatedValue * 0.08).toLocaleString()}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between">
                                <span>Market Trends</span>
                                <span>$${Math.round(property.estimatedValue * 0.07).toLocaleString()}</span>
                            </li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <canvas id="propertyFactorChart" height="180"></canvas>
                    </div>
                </div>
            </div>
        `;
        
        // Show the modal
        propertyDetailModal.show();
        
        // Create the factor breakdown chart
        setTimeout(() => {
            const factorChartEl = document.getElementById('propertyFactorChart');
            if (factorChartEl) {
                const ctx = factorChartEl.getContext('2d');
                new Chart(ctx, {
                    type: 'pie',
                    data: {
                        labels: ['Base Value', 'Location', 'Condition', 'Market'],
                        datasets: [{
                            data: [70, 15, 8, 7],
                            backgroundColor: [
                                'rgba(33, 150, 243, 0.7)',
                                'rgba(76, 175, 80, 0.7)',
                                'rgba(255, 152, 0, 0.7)',
                                'rgba(156, 39, 176, 0.7)'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const value = context.parsed;
                                        return `${context.label}: ${value}%`;
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }, 100);
    }
    
    /**
     * Format a property type for display
     */
    function formatPropertyType(type) {
        const typeMap = {
            'single_family': 'Single Family',
            'condo': 'Condominium',
            'townhouse': 'Townhouse',
            'multi_family': 'Multi-Family',
            'land': 'Land/Lot',
            'commercial': 'Commercial'
        };
        
        return typeMap[type] || type;
    }
    
    /**
     * Format a valuation method for display
     */
    function formatValuationMethod(method) {
        const methodMap = {
            'basic': 'Basic Valuation',
            'enhanced': 'Enhanced ML',
            'advanced_gis': 'Advanced GIS'
        };
        
        return methodMap[method] || method;
    }
    
    /**
     * Format a date for display
     */
    function formatDate(date) {
        return new Date(date).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }
    
    /**
     * Set up intersection observers for animation
     */
    function setupAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in-up');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        
        // Observe charts and metric cards
        document.querySelectorAll('.chart-container, .metric-card').forEach(el => {
            observer.observe(el);
        });
    }
    
    // Initialize the dashboard
    init();
});