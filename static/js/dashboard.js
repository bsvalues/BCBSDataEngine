/**
 * BCBS Values - Dashboard JavaScript
 * 
 * This file contains the JavaScript functionality for the dashboard page.
 */

// Global variables
let propertyData = [];
let neighborhoods = [];
let propertyTypes = [];
let currentPage = 1;
let itemsPerPage = 10;
let totalProperties = 0;
let sortColumn = 'estimated_value';
let sortDirection = 'desc';
let filters = {
    search: '',
    neighborhood: '',
    propertyType: '',
    minPrice: '',
    maxPrice: '',
    bedrooms: '',
    lastUpdated: ''
};
let charts = {};
let autoRefreshInterval = null;
let autoRefreshEnabled = false;

// DOM elements
const propertyTableBody = document.getElementById('property-table-body');
const pagination = document.querySelector('.pagination');
const totalPropertiesElement = document.getElementById('total-properties');
const avgValuationElement = document.getElementById('avg-valuation');
const medianValuationElement = document.getElementById('median-valuation');
const filteredCountElement = document.getElementById('filtered-count');
const searchInput = document.getElementById('search');
const neighborhoodSelect = document.getElementById('neighborhood');
const propertyTypeSelect = document.getElementById('propertyType');
const minPriceInput = document.getElementById('minPrice');
const maxPriceInput = document.getElementById('maxPrice');
const bedroomButtons = document.querySelectorAll('input[name="bedrooms"]');
const lastUpdatedSelect = document.getElementById('lastUpdated');
const applyFiltersButton = document.getElementById('apply-filters');
const resetFiltersButton = document.getElementById('reset-filters');
const refreshButton = document.getElementById('refresh-btn');
const autoRefreshToggle = document.getElementById('auto-refresh-toggle');
const pageSizeSelect = document.getElementById('page-size');
const prevPageButton = document.getElementById('prev-page');
const nextPageButton = document.getElementById('next-page');
const sortableHeaders = document.querySelectorAll('th.sortable');
const etlPipelineList = document.getElementById('etl-pipeline-list');
const etlJobTable = document.getElementById('etl-job-table');
const agentTable = document.getElementById('agent-table');
const propertyDetailModal = new bootstrap.Modal(document.getElementById('propertyDetailModal'));
const propertyDetailContent = document.getElementById('property-detail-content');
const viewFullDetailsButton = document.getElementById('view-full-details');
const agentDetailModal = new bootstrap.Modal(document.getElementById('agentDetailModal'));
const agentDetailContent = document.getElementById('agent-detail-content');
const restartAgentButton = document.getElementById('restart-agent');
const filterToggleButton = document.getElementById('filter-toggle');
const filterSidebar = document.getElementById('filter-sidebar');
const dashboardTabs = document.querySelectorAll('#dashboardTabs button');
const exportCSVButton = document.getElementById('export-csv');
const exportJSONButton = document.getElementById('export-json');
const exportExcelButton = document.getElementById('export-excel');

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Load initial data
    loadDashboardData();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize charts
    initializeCharts();
});

/**
 * Load all dashboard data
 */
function loadDashboardData() {
    loadProperties();
    // These will be loaded when the respective tabs are clicked
    dashboardTabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(e) {
            if (e.target.id === 'etl-tab') {
                loadETLStatus();
            } else if (e.target.id === 'agents-tab') {
                loadAgentStatus();
            } else if (e.target.id === 'analytics-tab') {
                loadAnalyticsData();
            }
        });
    });
}

/**
 * Set up event listeners for user interactions
 */
function setupEventListeners() {
    // Filter form submission
    applyFiltersButton.addEventListener('click', function() {
        collectFilters();
        currentPage = 1;
        loadProperties();
    });
    
    // Reset filters
    resetFiltersButton.addEventListener('click', function() {
        resetFilters();
        loadProperties();
    });
    
    // Refresh button
    refreshButton.addEventListener('click', function() {
        loadDashboardData();
    });
    
    // Auto-refresh toggle
    autoRefreshToggle.addEventListener('click', function() {
        toggleAutoRefresh();
    });
    
    // Items per page change
    pageSizeSelect.addEventListener('change', function() {
        itemsPerPage = parseInt(this.value);
        currentPage = 1;
        loadProperties();
    });
    
    // Pagination controls
    prevPageButton.parentElement.addEventListener('click', function(e) {
        e.preventDefault();
        if (!prevPageButton.parentElement.classList.contains('disabled')) {
            currentPage--;
            loadProperties();
        }
    });
    
    nextPageButton.parentElement.addEventListener('click', function(e) {
        e.preventDefault();
        if (!nextPageButton.parentElement.classList.contains('disabled')) {
            currentPage++;
            loadProperties();
        }
    });
    
    // Sortable table headers
    sortableHeaders.forEach(header => {
        header.addEventListener('click', function() {
            const column = this.dataset.sort;
            
            // If clicking the same column, toggle direction
            if (column === sortColumn) {
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                sortColumn = column;
                sortDirection = 'asc';
            }
            
            // Update UI
            sortableHeaders.forEach(h => {
                h.classList.remove('asc', 'desc');
            });
            this.classList.add(sortDirection);
            
            // Reload properties with new sort
            loadProperties();
        });
    });
    
    // Filter toggle for mobile
    if (filterToggleButton) {
        filterToggleButton.addEventListener('click', function() {
            if (filterSidebar) {
                filterSidebar.classList.toggle('d-none');
            }
        });
    }
    
    // Export buttons
    if (exportCSVButton) {
        exportCSVButton.addEventListener('click', function(e) {
            e.preventDefault();
            exportData('csv');
        });
    }
    
    if (exportJSONButton) {
        exportJSONButton.addEventListener('click', function(e) {
            e.preventDefault();
            exportData('json');
        });
    }
    
    if (exportExcelButton) {
        exportExcelButton.addEventListener('click', function(e) {
            e.preventDefault();
            exportData('excel');
        });
    }
    
    // Handle debounced search
    if (searchInput) {
        searchInput.addEventListener('input', debounce(function() {
            filters.search = this.value.trim();
            currentPage = 1;
            loadProperties();
        }, 500));
    }
}

/**
 * Load properties from the API
 */
function loadProperties() {
    showLoadingState();
    
    // Build query parameters
    const params = new URLSearchParams();
    params.append('page', currentPage);
    params.append('per_page', itemsPerPage);
    params.append('sort_by', sortColumn);
    params.append('sort_dir', sortDirection);
    
    // Add filters
    if (filters.search) params.append('search', filters.search);
    if (filters.neighborhood) params.append('neighborhood', filters.neighborhood);
    if (filters.propertyType) params.append('property_type', filters.propertyType);
    if (filters.minPrice) params.append('min_price', filters.minPrice);
    if (filters.maxPrice) params.append('max_price', filters.maxPrice);
    if (filters.bedrooms) params.append('bedrooms', filters.bedrooms);
    if (filters.lastUpdated) {
        const date = new Date();
        date.setDate(date.getDate() - parseInt(filters.lastUpdated));
        params.append('updated_since', date.toISOString().split('T')[0]);
    }
    
    // Fetch properties from API
    fetch(`/api/properties?${params.toString()}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch properties');
            }
            return response.json();
        })
        .then(data => {
            propertyData = data.properties || [];
            totalProperties = data.total || 0;
            
            // Update UI
            updatePropertyTable();
            updatePagination();
            updateSummaryMetrics();
            updateCharts();
            
            // Load filter options if they're not already loaded
            if (neighborhoods.length === 0) {
                loadNeighborhoods();
            }
            
            if (propertyTypes.length === 0) {
                loadPropertyTypes();
            }
            
            hideLoadingState();
        })
        .catch(error => {
            console.error('Error loading properties:', error);
            showErrorState('Failed to load properties. Please try again later.');
            hideLoadingState();
        });
}

/**
 * Load neighborhoods for the filter dropdown
 */
function loadNeighborhoods() {
    fetch('/api/neighborhoods')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch neighborhoods');
            }
            return response.json();
        })
        .then(data => {
            neighborhoods = data.neighborhoods || [];
            
            // Clear existing options except the first one
            while (neighborhoodSelect.options.length > 1) {
                neighborhoodSelect.remove(1);
            }
            
            // Add new options
            neighborhoods.forEach(neighborhood => {
                const option = document.createElement('option');
                option.value = neighborhood;
                option.textContent = neighborhood;
                neighborhoodSelect.appendChild(option);
            });
            
            // Set the selected value if there's a filter
            if (filters.neighborhood) {
                neighborhoodSelect.value = filters.neighborhood;
            }
        })
        .catch(error => {
            console.error('Error loading neighborhoods:', error);
        });
}

/**
 * Load property types for the filter dropdown
 */
function loadPropertyTypes() {
    fetch('/api/property-types')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch property types');
            }
            return response.json();
        })
        .then(data => {
            propertyTypes = data.property_types || [];
            
            // Clear existing options except the first one
            while (propertyTypeSelect.options.length > 1) {
                propertyTypeSelect.remove(1);
            }
            
            // Add new options
            propertyTypes.forEach(type => {
                const option = document.createElement('option');
                option.value = type;
                option.textContent = type;
                propertyTypeSelect.appendChild(option);
            });
            
            // Set the selected value if there's a filter
            if (filters.propertyType) {
                propertyTypeSelect.value = filters.propertyType;
            }
        })
        .catch(error => {
            console.error('Error loading property types:', error);
        });
}

/**
 * Load ETL status data
 */
function loadETLStatus() {
    // Show loading state
    etlPipelineList.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mb-0">Loading ETL data...</p>
        </div>
    `;
    etlJobTable.innerHTML = '';
    
    // Fetch ETL status from API
    fetch('/api/etl-status')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch ETL status');
            }
            return response.json();
        })
        .then(data => {
            const etlJobs = data.etl_jobs || [];
            
            // Group jobs by job name to get pipelines
            const pipelines = {};
            etlJobs.forEach(job => {
                if (!pipelines[job.job_name]) {
                    pipelines[job.job_name] = {
                        name: job.job_name,
                        lastRun: job.start_time,
                        status: job.status,
                        recordsProcessed: job.records_processed
                    };
                }
            });
            
            // Update ETL pipeline list
            updateETLPipelineList(Object.values(pipelines));
            
            // Update ETL job table
            updateETLJobTable(etlJobs);
        })
        .catch(error => {
            console.error('Error loading ETL status:', error);
            etlPipelineList.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    Failed to load ETL data. Please try again later.
                </div>
            `;
        });
}

/**
 * Update the ETL pipeline list
 */
function updateETLPipelineList(pipelines) {
    if (pipelines.length === 0) {
        etlPipelineList.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-database-slash fs-1 d-block mb-3 text-secondary"></i>
                <p class="mb-0">No ETL pipelines found</p>
            </div>
        `;
        return;
    }
    
    etlPipelineList.innerHTML = '';
    
    pipelines.forEach(pipeline => {
        const lastRunDate = new Date(pipeline.lastRun);
        const statusClass = getStatusClass(pipeline.status);
        const statusIcon = getStatusIcon(pipeline.status);
        
        const item = document.createElement('a');
        item.href = '#';
        item.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
        item.innerHTML = `
            <div>
                <h6 class="mb-1">${pipeline.name}</h6>
                <p class="text-secondary mb-0 small">Last run: ${lastRunDate.toLocaleString()}</p>
            </div>
            <div class="d-flex align-items-center">
                <span class="badge ${statusClass} me-2">
                    ${statusIcon} ${pipeline.status}
                </span>
                <span class="badge bg-secondary">${pipeline.recordsProcessed} records</span>
            </div>
        `;
        
        etlPipelineList.appendChild(item);
    });
}

/**
 * Update the ETL job table
 */
function updateETLJobTable(jobs) {
    if (jobs.length === 0) {
        etlJobTable.innerHTML = `
            <tr>
                <td colspan="5" class="text-center py-4">
                    <i class="bi bi-database-slash fs-1 d-block mb-3 text-secondary"></i>
                    <p class="mb-0">No ETL jobs found</p>
                </td>
            </tr>
        `;
        return;
    }
    
    etlJobTable.innerHTML = '';
    
    jobs.forEach(job => {
        const startTime = new Date(job.start_time);
        const endTime = job.end_time ? new Date(job.end_time) : null;
        const duration = endTime ? ((endTime - startTime) / 1000).toFixed(1) + 's' : 'Running...';
        const statusClass = getStatusClass(job.status);
        const statusIcon = getStatusIcon(job.status);
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${job.job_name}</td>
            <td>${startTime.toLocaleString()}</td>
            <td>${duration}</td>
            <td><span class="badge ${statusClass}">${statusIcon} ${job.status}</span></td>
            <td>${job.records_processed}</td>
        `;
        
        etlJobTable.appendChild(row);
    });
}

/**
 * Load agent status data
 */
function loadAgentStatus() {
    // Show loading state
    agentTable.innerHTML = `
        <tr class="agent-loading-placeholder">
            <td colspan="7" class="text-center py-5">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mb-0">Loading agent data...</p>
            </td>
        </tr>
    `;
    
    // Fetch agent status from API
    fetch('/api/agent-status')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch agent status');
            }
            return response.json();
        })
        .then(data => {
            const agents = data.agents || [];
            
            // Update agent table
            updateAgentTable(agents);
            
            // Update agent performance chart
            updateAgentPerformanceChart(agents);
        })
        .catch(error => {
            console.error('Error loading agent status:', error);
            agentTable.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center">
                        <div class="alert alert-danger" role="alert">
                            Failed to load agent data. Please try again later.
                        </div>
                    </td>
                </tr>
            `;
        });
}

/**
 * Update the agent table
 */
function updateAgentTable(agents) {
    if (agents.length === 0) {
        agentTable.innerHTML = `
            <tr>
                <td colspan="7" class="text-center py-4">
                    <i class="bi bi-robot fs-1 d-block mb-3 text-secondary"></i>
                    <p class="mb-0">No agents found</p>
                </td>
            </tr>
        `;
        return;
    }
    
    agentTable.innerHTML = '';
    
    agents.forEach(agent => {
        const lastHeartbeat = agent.last_heartbeat ? new Date(agent.last_heartbeat) : null;
        const statusClass = getAgentStatusClass(agent.status);
        const statusIcon = getAgentStatusIcon(agent.status);
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${agent.agent_name}</td>
            <td><span class="agent-status ${statusClass}">${statusIcon} ${agent.status}</span></td>
            <td>${lastHeartbeat ? lastHeartbeat.toLocaleString() : 'Never'}</td>
            <td>${agent.current_task || 'None'}</td>
            <td>
                <div class="d-flex align-items-center">
                    <div class="progress flex-grow-1 me-2">
                        <div class="progress-bar" role="progressbar" style="width: ${Math.min(100, (agent.queue_size / 10) * 100)}%" aria-valuenow="${agent.queue_size}" aria-valuemin="0" aria-valuemax="10"></div>
                    </div>
                    <span>${agent.queue_size}</span>
                </div>
            </td>
            <td>
                <div class="d-flex align-items-center">
                    <div class="progress flex-grow-1 me-2">
                        <div class="progress-bar bg-success" role="progressbar" style="width: ${agent.success_rate}%" aria-valuenow="${agent.success_rate}" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                    <span>${agent.success_rate}%</span>
                </div>
            </td>
            <td>
                <button class="btn btn-sm btn-outline-primary view-agent-details" data-agent-id="${agent.agent_id}">
                    <i class="bi bi-info-circle"></i>
                </button>
            </td>
        `;
        
        // Add event listener for viewing agent details
        const viewDetailsButton = row.querySelector('.view-agent-details');
        viewDetailsButton.addEventListener('click', function() {
            viewAgentDetails(agent.agent_id);
        });
        
        agentTable.appendChild(row);
    });
}

/**
 * View agent details in a modal
 */
function viewAgentDetails(agentId) {
    // Show loading state
    agentDetailContent.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Loading agent details...</p>
        </div>
    `;
    
    // Disable restart button
    restartAgentButton.disabled = true;
    
    // Show the modal
    agentDetailModal.show();
    
    // Fetch agent details from API
    fetch(`/api/agent-status/${agentId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch agent details');
            }
            return response.json();
        })
        .then(data => {
            const agent = data.agent;
            const logs = data.logs || [];
            
            // Update agent details in modal
            updateAgentDetailsModal(agent, logs);
            
            // Enable restart button
            restartAgentButton.disabled = false;
        })
        .catch(error => {
            console.error('Error loading agent details:', error);
            agentDetailContent.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    Failed to load agent details. Please try again later.
                </div>
            `;
        });
}

/**
 * Update the agent details modal
 */
function updateAgentDetailsModal(agent, logs) {
    const lastHeartbeat = agent.last_heartbeat ? new Date(agent.last_heartbeat).toLocaleString() : 'Never';
    const statusClass = getAgentStatusClass(agent.status);
    const statusIcon = getAgentStatusIcon(agent.status);
    
    agentDetailContent.innerHTML = `
        <div class="agent-summary mb-4">
            <div class="d-flex justify-content-between align-items-start mb-3">
                <h5>${agent.agent_name}</h5>
                <span class="agent-status ${statusClass}">${statusIcon} ${agent.status}</span>
            </div>
            
            <div class="row mb-3">
                <div class="col-md-6">
                    <div class="detail-item">
                        <div class="detail-label">Agent ID</div>
                        <div class="detail-value">${agent.agent_id}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Last Heartbeat</div>
                        <div class="detail-value">${lastHeartbeat}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Current Task</div>
                        <div class="detail-value">${agent.current_task || 'None'}</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="detail-item">
                        <div class="detail-label">Queue Size</div>
                        <div class="detail-value">${agent.queue_size}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Success Rate</div>
                        <div class="detail-value">${agent.success_rate}%</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Error Count</div>
                        <div class="detail-value">${agent.error_count}</div>
                    </div>
                </div>
            </div>
        </div>
        
        <h6 class="mb-3">Agent Logs</h6>
        <div class="agent-log-container custom-scrollbar">
            ${logs.length > 0 ? renderAgentLogs(logs) : '<p class="text-center text-secondary my-3">No logs available</p>'}
        </div>
    `;
    
    // Update modal title
    document.getElementById('agentDetailModalLabel').textContent = `Agent: ${agent.agent_name}`;
    
    // Add event listener for restart button
    restartAgentButton.addEventListener('click', function() {
        restartAgent(agent.agent_id);
    });
}

/**
 * Render agent logs
 */
function renderAgentLogs(logs) {
    let logHtml = '';
    
    logs.forEach(log => {
        const timestamp = new Date(log.timestamp).toLocaleString();
        const levelClass = getLevelClass(log.level);
        
        logHtml += `
            <div class="log-entry">
                <div class="log-timestamp">${timestamp}</div>
                <div class="log-message ${levelClass}">${log.message}</div>
            </div>
        `;
    });
    
    return logHtml;
}

/**
 * Restart an agent
 */
function restartAgent(agentId) {
    // Show loading state
    restartAgentButton.disabled = true;
    restartAgentButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Restarting...';
    
    // Call API to restart agent
    fetch(`/api/agent-status/${agentId}/restart`, {
        method: 'POST'
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to restart agent');
            }
            return response.json();
        })
        .then(data => {
            // Show success message
            showAlert('Agent restarted successfully', 'success');
            
            // Close the modal
            agentDetailModal.hide();
            
            // Reload agent status
            loadAgentStatus();
        })
        .catch(error => {
            console.error('Error restarting agent:', error);
            showAlert('Failed to restart agent. Please try again later.', 'danger');
            
            // Reset button
            restartAgentButton.disabled = false;
            restartAgentButton.innerHTML = 'Restart Agent';
        });
}

/**
 * Update the agent performance chart
 */
function updateAgentPerformanceChart(agents) {
    // Create data for chart
    const labels = agents.map(agent => agent.agent_name);
    const successRates = agents.map(agent => agent.success_rate);
    const errorCounts = agents.map(agent => agent.error_count);
    
    // If chart exists, update it
    if (charts.agentPerformance) {
        charts.agentPerformance.data.labels = labels;
        charts.agentPerformance.data.datasets[0].data = successRates;
        charts.agentPerformance.data.datasets[1].data = errorCounts;
        charts.agentPerformance.update();
        return;
    }
    
    // Otherwise, create the chart
    const ctx = document.getElementById('agentPerformanceChart').getContext('2d');
    charts.agentPerformance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Success Rate (%)',
                    data: successRates,
                    backgroundColor: 'rgba(25, 135, 84, 0.7)',
                    borderColor: 'rgba(25, 135, 84, 1)',
                    borderWidth: 1,
                    yAxisID: 'y'
                },
                {
                    label: 'Error Count',
                    data: errorCounts,
                    backgroundColor: 'rgba(220, 53, 69, 0.7)',
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: 'white'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Success Rate (%)',
                        color: 'white'
                    },
                    ticks: {
                        color: 'white'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    min: 0,
                    max: 100
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Error Count',
                        color: 'white'
                    },
                    ticks: {
                        color: 'white'
                    },
                    grid: {
                        drawOnChartArea: false
                    },
                    min: 0
                },
                x: {
                    ticks: {
                        color: 'white'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
}

/**
 * Load analytics data
 */
function loadAnalyticsData() {
    // This is a placeholder for analytics data loading
    // In a real implementation, this would fetch data from the API
    // and update the analytics charts
}

/**
 * Initialize charts on the dashboard
 */
function initializeCharts() {
    // Value Distribution Chart
    const valueDistributionCtx = document.getElementById('valueDistributionChart').getContext('2d');
    charts.valueDistribution = new Chart(valueDistributionCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Property Count',
                data: [],
                backgroundColor: 'rgba(13, 110, 253, 0.7)',
                borderColor: 'rgba(13, 110, 253, 1)',
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
                            const value = context.parsed.y;
                            return `Properties: ${value}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: 'white'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: 'white'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
    
    // Neighborhood Comparison Chart
    const neighborhoodComparisonCtx = document.getElementById('neighborhoodComparisonChart').getContext('2d');
    charts.neighborhoodComparison = new Chart(neighborhoodComparisonCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Average Valuation',
                data: [],
                backgroundColor: 'rgba(13, 202, 240, 0.7)',
                borderColor: 'rgba(13, 202, 240, 1)',
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
                            const value = context.parsed.y;
                            return `Avg. Value: ${formatCurrency(value)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: 'white',
                        callback: function(value) {
                            return formatCurrencyShort(value);
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: 'white'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
    
    // Valuation Trend Chart
    const valuationTrendCtx = document.getElementById('valuationTrendChart').getContext('2d');
    charts.valuationTrend = new Chart(valuationTrendCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Average Valuation',
                data: [],
                fill: false,
                borderColor: 'rgba(25, 135, 84, 1)',
                tension: 0.1,
                pointBackgroundColor: 'rgba(25, 135, 84, 1)',
                pointBorderColor: '#fff',
                pointRadius: 4
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
                            const value = context.parsed.y;
                            return `Avg. Value: ${formatCurrency(value)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: 'white',
                        callback: function(value) {
                            return formatCurrencyShort(value);
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: 'white'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
    
    // Model Distribution Chart
    const modelDistributionCtx = document.getElementById('modelDistributionChart').getContext('2d');
    charts.modelDistribution = new Chart(modelDistributionCtx, {
        type: 'pie',
        data: {
            labels: ['Linear Regression', 'Ridge Regression', 'Gradient Boosting', 'Elastic Net', 'Lasso'],
            datasets: [{
                data: [30, 25, 20, 15, 10],
                backgroundColor: [
                    'rgba(13, 110, 253, 0.7)',
                    'rgba(25, 135, 84, 0.7)',
                    'rgba(13, 202, 240, 0.7)',
                    'rgba(220, 53, 69, 0.7)',
                    'rgba(255, 193, 7, 0.7)'
                ],
                borderColor: [
                    'rgba(13, 110, 253, 1)',
                    'rgba(25, 135, 84, 1)',
                    'rgba(13, 202, 240, 1)',
                    'rgba(220, 53, 69, 1)',
                    'rgba(255, 193, 7, 1)'
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
                    labels: {
                        color: 'white'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value * 100) / total);
                            return `${percentage}% (${value} properties)`;
                        }
                    }
                }
            }
        }
    });
    
    // Feature Importance Chart
    const featureImportanceCtx = document.getElementById('featureImportanceChart').getContext('2d');
    charts.featureImportance = new Chart(featureImportanceCtx, {
        type: 'bar',
        data: {
            labels: ['Square Feet', 'Bedrooms', 'Bathrooms', 'Year Built', 'Lot Size', 'Neighborhood', 'Proximity Score', 'School Rating'],
            datasets: [{
                label: 'Importance',
                data: [0.35, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05],
                backgroundColor: 'rgba(13, 110, 253, 0.7)',
                borderColor: 'rgba(13, 110, 253, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.x;
                            return `Importance: ${(value * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 0.4,
                    ticks: {
                        color: 'white',
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    ticks: {
                        color: 'white'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
}

/**
 * Update charts with property data
 */
function updateCharts() {
    if (!propertyData.length) return;
    
    // Update Value Distribution Chart
    updateValueDistributionChart();
    
    // Update Neighborhood Comparison Chart
    updateNeighborhoodComparisonChart();
}

/**
 * Update the value distribution chart
 */
function updateValueDistributionChart() {
    // Group properties by price range
    const ranges = [
        { min: 0, max: 200000, label: '< $200K' },
        { min: 200000, max: 300000, label: '$200K - $300K' },
        { min: 300000, max: 400000, label: '$300K - $400K' },
        { min: 400000, max: 500000, label: '$400K - $500K' },
        { min: 500000, max: 750000, label: '$500K - $750K' },
        { min: 750000, max: 1000000, label: '$750K - $1M' },
        { min: 1000000, max: Number.POSITIVE_INFINITY, label: '> $1M' }
    ];
    
    const distribution = ranges.map(range => ({
        label: range.label,
        count: propertyData.filter(p => p.estimated_value >= range.min && p.estimated_value < range.max).length
    }));
    
    // Update chart data
    charts.valueDistribution.data.labels = distribution.map(d => d.label);
    charts.valueDistribution.data.datasets[0].data = distribution.map(d => d.count);
    charts.valueDistribution.update();
}

/**
 * Update the neighborhood comparison chart
 */
function updateNeighborhoodComparisonChart() {
    // Group properties by neighborhood and calculate average valuation
    const neighborhoodGroups = {};
    
    propertyData.forEach(property => {
        if (!property.neighborhood) return;
        
        if (!neighborhoodGroups[property.neighborhood]) {
            neighborhoodGroups[property.neighborhood] = {
                count: 0,
                totalValue: 0
            };
        }
        
        neighborhoodGroups[property.neighborhood].count++;
        neighborhoodGroups[property.neighborhood].totalValue += property.estimated_value;
    });
    
    const neighborhoods = Object.keys(neighborhoodGroups)
        .map(neighborhood => ({
            name: neighborhood,
            averageValue: neighborhoodGroups[neighborhood].totalValue / neighborhoodGroups[neighborhood].count
        }))
        .sort((a, b) => b.averageValue - a.averageValue)
        .slice(0, 10); // Show top 10 neighborhoods
    
    // Update chart data
    charts.neighborhoodComparison.data.labels = neighborhoods.map(n => n.name);
    charts.neighborhoodComparison.data.datasets[0].data = neighborhoods.map(n => n.averageValue);
    charts.neighborhoodComparison.update();
}

/**
 * Update the property table
 */
function updatePropertyTable() {
    if (!propertyData.length) {
        propertyTableBody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center py-4">
                    <i class="bi bi-house-slash fs-1 d-block mb-3 text-secondary"></i>
                    <p class="mb-0">No properties found matching your criteria</p>
                </td>
            </tr>
        `;
        return;
    }
    
    propertyTableBody.innerHTML = '';
    
    propertyData.forEach(property => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${property.property_id}</td>
            <td>${property.address}, ${property.city}, ${property.state} ${property.zip_code}</td>
            <td>${property.neighborhood || 'N/A'}</td>
            <td>${property.bedrooms || 'N/A'}</td>
            <td>${property.square_feet ? property.square_feet.toLocaleString() : 'N/A'}</td>
            <td class="format-currency">${formatCurrency(property.estimated_value)}</td>
            <td>${property.valuation_date ? new Date(property.valuation_date).toLocaleDateString() : 'N/A'}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary view-property-details" data-property-id="${property.id}">
                    <i class="bi bi-info-circle"></i>
                </button>
            </td>
        `;
        
        // Add event listener for viewing property details
        const viewDetailsButton = row.querySelector('.view-property-details');
        viewDetailsButton.addEventListener('click', function() {
            viewPropertyDetails(property.id);
        });
        
        propertyTableBody.appendChild(row);
    });
}

/**
 * Update the pagination controls
 */
function updatePagination() {
    // Calculate number of pages
    const totalPages = Math.ceil(totalProperties / itemsPerPage);
    
    // Update display counts
    document.getElementById('showing-start').textContent = totalProperties === 0 ? 0 : (currentPage - 1) * itemsPerPage + 1;
    document.getElementById('showing-end').textContent = Math.min(currentPage * itemsPerPage, totalProperties);
    document.getElementById('total-count').textContent = totalProperties;
    
    // Clear existing page links
    const pageItems = pagination.querySelectorAll('.page-item:not(:first-child):not(:last-child)');
    pageItems.forEach(item => item.remove());
    
    // Add page links
    const maxPages = 5; // Show at most 5 page links
    let startPage = Math.max(1, currentPage - Math.floor(maxPages / 2));
    let endPage = Math.min(totalPages, startPage + maxPages - 1);
    
    if (endPage - startPage + 1 < maxPages) {
        startPage = Math.max(1, endPage - maxPages + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
        const pageItem = document.createElement('li');
        pageItem.className = `page-item${i === currentPage ? ' active' : ''}`;
        
        const pageLink = document.createElement('a');
        pageLink.className = 'page-link';
        pageLink.href = '#';
        pageLink.textContent = i;
        pageLink.dataset.page = i;
        
        pageLink.addEventListener('click', function(e) {
            e.preventDefault();
            currentPage = parseInt(this.dataset.page);
            loadProperties();
        });
        
        pageItem.appendChild(pageLink);
        prevPageButton.parentElement.insertAdjacentElement('afterend', pageItem);
    }
    
    // Update previous and next buttons
    prevPageButton.parentElement.classList.toggle('disabled', currentPage === 1);
    nextPageButton.parentElement.classList.toggle('disabled', currentPage === totalPages);
}

/**
 * Update the summary metrics
 */
function updateSummaryMetrics() {
    // Update total properties
    totalPropertiesElement.textContent = totalProperties.toLocaleString();
    
    // Calculate average valuation
    if (propertyData.length > 0) {
        const totalValuation = propertyData.reduce((sum, property) => sum + property.estimated_value, 0);
        const avgValuation = totalValuation / propertyData.length;
        avgValuationElement.textContent = formatCurrency(avgValuation);
        
        // Calculate median valuation
        const valuations = [...propertyData].sort((a, b) => a.estimated_value - b.estimated_value);
        const medianIndex = Math.floor(valuations.length / 2);
        const medianValuation = valuations.length % 2 === 0
            ? (valuations[medianIndex - 1].estimated_value + valuations[medianIndex].estimated_value) / 2
            : valuations[medianIndex].estimated_value;
        medianValuationElement.textContent = formatCurrency(medianValuation);
    } else {
        avgValuationElement.textContent = 'N/A';
        medianValuationElement.textContent = 'N/A';
    }
    
    // Update filtered count
    filteredCountElement.textContent = propertyData.length.toLocaleString();
}

/**
 * View property details in a modal
 */
function viewPropertyDetails(propertyId) {
    // Show loading state
    propertyDetailContent.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Loading property details...</p>
        </div>
    `;
    
    // Set the view full details link
    viewFullDetailsButton.href = `/property/${propertyId}`;
    
    // Show the modal
    propertyDetailModal.show();
    
    // Fetch property details from API
    fetch(`/api/properties/${propertyId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch property details');
            }
            return response.json();
        })
        .then(property => {
            // Fetch valuation history
            return fetch(`/api/properties/${propertyId}/valuations`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to fetch valuation history');
                    }
                    return response.json();
                })
                .then(valuationData => {
                    // Update property details in modal
                    updatePropertyDetailsModal(property, valuationData.valuations || []);
                });
        })
        .catch(error => {
            console.error('Error loading property details:', error);
            propertyDetailContent.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    Failed to load property details. Please try again later.
                </div>
            `;
        });
}

/**
 * Update the property details modal
 */
function updatePropertyDetailsModal(property, valuations) {
    // Update modal title
    document.getElementById('propertyDetailModalLabel').textContent = `Property: ${property.address}`;
    
    // Format valuation date
    const valuationDate = property.valuation_date ? new Date(property.valuation_date).toLocaleDateString() : 'N/A';
    
    // Create property summary card
    let propertyHtml = `
        <div class="property-summary-card">
            <div class="property-image">
                <i class="bi bi-house"></i>
            </div>
            <div class="property-summary">
                <h6 class="property-address">${property.address}, ${property.city}, ${property.state} ${property.zip_code}</h6>
                <div class="property-features">
                    <div class="property-feature">
                        <i class="bi bi-door-open"></i> ${property.bedrooms || '?'} bd
                    </div>
                    <div class="property-feature">
                        <i class="bi bi-droplet"></i> ${property.bathrooms || '?'} ba
                    </div>
                    <div class="property-feature">
                        <i class="bi bi-square"></i> ${property.square_feet ? property.square_feet.toLocaleString() : '?'} sqft
                    </div>
                </div>
                <div class="property-valuation">${formatCurrency(property.estimated_value)}</div>
                <small class="text-secondary">Valuation as of ${valuationDate}</small>
            </div>
        </div>
    `;
    
    // Add property details
    propertyHtml += `
        <ul class="nav nav-tabs property-detail-tabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="details-tab" data-bs-toggle="tab" data-bs-target="#details" type="button" role="tab">Details</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="valuations-tab" data-bs-toggle="tab" data-bs-target="#valuations" type="button" role="tab">Valuation History</button>
            </li>
        </ul>
        
        <div class="tab-content">
            <div class="tab-pane fade show active" id="details" role="tabpanel" aria-labelledby="details-tab">
                <div class="detail-section">
                    <h6 class="detail-section-title">Property Information</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="detail-item">
                                <div class="detail-label">Property ID</div>
                                <div class="detail-value">${property.property_id}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Property Type</div>
                                <div class="detail-value">${property.property_type || 'N/A'}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Year Built</div>
                                <div class="detail-value">${property.year_built || 'N/A'}</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="detail-item">
                                <div class="detail-label">Neighborhood</div>
                                <div class="detail-value">${property.neighborhood || 'N/A'}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Lot Size</div>
                                <div class="detail-value">${property.lot_size ? property.lot_size.toLocaleString() + ' sqft' : 'N/A'}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Last Sale</div>
                                <div class="detail-value">${property.last_sale_date ? new Date(property.last_sale_date).toLocaleDateString() + ' - ' + formatCurrency(property.last_sale_price) : 'N/A'}</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="detail-section">
                    <h6 class="detail-section-title">Location</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="detail-item">
                                <div class="detail-label">Address</div>
                                <div class="detail-value">${property.address}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">City</div>
                                <div class="detail-value">${property.city}</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="detail-item">
                                <div class="detail-label">State</div>
                                <div class="detail-value">${property.state}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">ZIP Code</div>
                                <div class="detail-value">${property.zip_code}</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="detail-section">
                    <h6 class="detail-section-title">Valuation</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="detail-item">
                                <div class="detail-label">Estimated Value</div>
                                <div class="detail-value">${formatCurrency(property.estimated_value)}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Valuation Date</div>
                                <div class="detail-value">${valuationDate}</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="detail-item">
                                <div class="detail-label">Valuation Method</div>
                                <div class="detail-value">${property.latest_valuation?.valuation_method || 'N/A'}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Confidence Score</div>
                                <div class="detail-value">${property.latest_valuation?.confidence_score ? (property.latest_valuation.confidence_score * 100).toFixed(1) + '%' : 'N/A'}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="tab-pane fade" id="valuations" role="tabpanel" aria-labelledby="valuations-tab">
    `;
    
    // Add valuation history
    if (valuations.length === 0) {
        propertyHtml += `
            <div class="text-center py-4">
                <i class="bi bi-clock-history fs-1 d-block mb-3 text-secondary"></i>
                <p class="mb-0">No valuation history available</p>
            </div>
        `;
    } else {
        propertyHtml += `
            <div class="table-responsive mt-3">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Estimated Value</th>
                            <th>Method</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        valuations.forEach(valuation => {
            const date = new Date(valuation.valuation_date).toLocaleDateString();
            const confidence = valuation.confidence_score ? (valuation.confidence_score * 100).toFixed(1) + '%' : 'N/A';
            
            propertyHtml += `
                <tr>
                    <td>${date}</td>
                    <td>${formatCurrency(valuation.estimated_value)}</td>
                    <td>${valuation.valuation_method || 'N/A'}</td>
                    <td>${confidence}</td>
                </tr>
            `;
        });
        
        propertyHtml += `
                    </tbody>
                </table>
            </div>
        `;
    }
    
    propertyHtml += `
            </div>
        </div>
    `;
    
    // Update the modal content
    propertyDetailContent.innerHTML = propertyHtml;
}

/**
 * Collect filter values from form elements
 */
function collectFilters() {
    filters.search = searchInput.value.trim();
    filters.neighborhood = neighborhoodSelect.value;
    filters.propertyType = propertyTypeSelect.value;
    filters.minPrice = minPriceInput.value;
    filters.maxPrice = maxPriceInput.value;
    
    // Get selected bedroom value
    const selectedBedroom = document.querySelector('input[name="bedrooms"]:checked');
    filters.bedrooms = selectedBedroom ? selectedBedroom.value : '';
    
    filters.lastUpdated = lastUpdatedSelect.value;
}

/**
 * Reset all filters to default values
 */
function resetFilters() {
    // Reset filter form elements
    searchInput.value = '';
    neighborhoodSelect.selectedIndex = 0;
    propertyTypeSelect.selectedIndex = 0;
    minPriceInput.value = '';
    maxPriceInput.value = '';
    document.getElementById('any-bed').checked = true;
    lastUpdatedSelect.selectedIndex = 0;
    
    // Reset filter object
    filters = {
        search: '',
        neighborhood: '',
        propertyType: '',
        minPrice: '',
        maxPrice: '',
        bedrooms: '',
        lastUpdated: ''
    };
}

/**
 * Toggle auto-refresh on or off
 */
function toggleAutoRefresh() {
    autoRefreshEnabled = !autoRefreshEnabled;
    
    if (autoRefreshEnabled) {
        // Set up auto-refresh interval (every 30 seconds)
        autoRefreshInterval = setInterval(loadDashboardData, 30000);
        autoRefreshToggle.innerHTML = '<i class="bi bi-clock"></i> Auto Refresh: On';
        autoRefreshToggle.classList.remove('btn-outline-secondary');
        autoRefreshToggle.classList.add('btn-outline-success');
    } else {
        // Clear auto-refresh interval
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        autoRefreshToggle.innerHTML = '<i class="bi bi-clock"></i> Auto Refresh: Off';
        autoRefreshToggle.classList.remove('btn-outline-success');
        autoRefreshToggle.classList.add('btn-outline-secondary');
    }
}

/**
 * Export data in the specified format
 */
function exportData(format) {
    // This is a placeholder for data export functionality
    // In a real implementation, this would call an API endpoint to generate the export file
    
    showAlert(`Exporting data in ${format.toUpperCase()} format...`, 'info');
    
    // Build query parameters for export
    const params = new URLSearchParams();
    params.append('format', format);
    params.append('sort_by', sortColumn);
    params.append('sort_dir', sortDirection);
    
    // Add filters
    if (filters.search) params.append('search', filters.search);
    if (filters.neighborhood) params.append('neighborhood', filters.neighborhood);
    if (filters.propertyType) params.append('property_type', filters.propertyType);
    if (filters.minPrice) params.append('min_price', filters.minPrice);
    if (filters.maxPrice) params.append('max_price', filters.maxPrice);
    if (filters.bedrooms) params.append('bedrooms', filters.bedrooms);
    if (filters.lastUpdated) {
        const date = new Date();
        date.setDate(date.getDate() - parseInt(filters.lastUpdated));
        params.append('updated_since', date.toISOString().split('T')[0]);
    }
    
    // Redirect to export endpoint
    window.location.href = `/api/properties/export?${params.toString()}`;
}

/**
 * Show loading state in the property table
 */
function showLoadingState() {
    propertyTableBody.innerHTML = `
        <tr class="property-loading-placeholder">
            <td colspan="8" class="text-center py-5">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mb-0">Loading property data...</p>
            </td>
        </tr>
    `;
}

/**
 * Hide loading state in the property table
 */
function hideLoadingState() {
    const placeholder = propertyTableBody.querySelector('.property-loading-placeholder');
    if (placeholder) {
        placeholder.remove();
    }
}

/**
 * Show error state in the property table
 */
function showErrorState(message) {
    propertyTableBody.innerHTML = `
        <tr>
            <td colspan="8" class="text-center py-4">
                <div class="alert alert-danger mb-0" role="alert">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    ${message}
                </div>
            </td>
        </tr>
    `;
}

/**
 * Show an alert message
 */
function showAlert(message, type = 'info', duration = 5000) {
    // Create alert container if it doesn't exist
    let alertContainer = document.getElementById('alert-container');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alert-container';
        alertContainer.style.position = 'fixed';
        alertContainer.style.top = '1rem';
        alertContainer.style.right = '1rem';
        alertContainer.style.zIndex = '1050';
        document.body.appendChild(alertContainer);
    }
    
    // Create alert element
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add alert to container
    alertContainer.appendChild(alertElement);
    
    // Initialize Bootstrap alert
    const bsAlert = new bootstrap.Alert(alertElement);
    
    // Auto-dismiss alert after duration
    if (duration > 0) {
        setTimeout(() => {
            bsAlert.close();
        }, duration);
    }
    
    // Remove from DOM after hidden
    alertElement.addEventListener('hidden.bs.alert', function() {
        this.remove();
    });
}

/**
 * Format a number as currency
 */
function formatCurrency(value) {
    if (value === null || value === undefined) return 'N/A';
    
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

/**
 * Format a number as abbreviated currency
 */
function formatCurrencyShort(value) {
    if (value === null || value === undefined) return 'N/A';
    
    if (value >= 1000000) {
        return '$' + (value / 1000000).toFixed(1) + 'M';
    } else if (value >= 1000) {
        return '$' + (value / 1000).toFixed(0) + 'K';
    } else {
        return '$' + value.toFixed(0);
    }
}

/**
 * Get the CSS class for a status
 */
function getStatusClass(status) {
    switch (status.toLowerCase()) {
        case 'completed':
            return 'bg-success';
        case 'running':
            return 'bg-primary';
        case 'failed':
            return 'bg-danger';
        default:
            return 'bg-secondary';
    }
}

/**
 * Get the icon for a status
 */
function getStatusIcon(status) {
    switch (status.toLowerCase()) {
        case 'completed':
            return '<i class="bi bi-check-circle"></i>';
        case 'running':
            return '<i class="bi bi-play-fill"></i>';
        case 'failed':
            return '<i class="bi bi-exclamation-triangle"></i>';
        default:
            return '<i class="bi bi-question-circle"></i>';
    }
}

/**
 * Get the CSS class for an agent status
 */
function getAgentStatusClass(status) {
    switch (status.toLowerCase()) {
        case 'idle':
            return 'agent-status-idle';
        case 'running':
            return 'agent-status-running';
        case 'error':
            return 'agent-status-error';
        default:
            return 'agent-status-idle';
    }
}

/**
 * Get the icon for an agent status
 */
function getAgentStatusIcon(status) {
    switch (status.toLowerCase()) {
        case 'idle':
            return '<i class="bi bi-pause-fill"></i>';
        case 'running':
            return '<i class="bi bi-play-fill"></i>';
        case 'error':
            return '<i class="bi bi-exclamation-triangle"></i>';
        default:
            return '<i class="bi bi-question-circle"></i>';
    }
}

/**
 * Get the CSS class for a log level
 */
function getLevelClass(level) {
    switch (level.toLowerCase()) {
        case 'info':
            return 'log-level-info';
        case 'warning':
            return 'log-level-warning';
        case 'error':
            return 'log-level-error';
        case 'success':
            return 'log-level-success';
        default:
            return '';
    }
}

/**
 * Debounce function to limit how often a function can be called
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func.apply(this, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}