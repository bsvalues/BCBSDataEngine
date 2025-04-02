// Dashboard.js - Compiled version for use in the Flask app
// This file contains the compiled Dashboard React component

'use strict';

// Ensure React and ReactDOM are loaded
if (typeof React === 'undefined' || typeof ReactDOM === 'undefined' || typeof Chart === 'undefined') {
  console.error('Required libraries not loaded. Make sure React, ReactDOM, and Chart.js are included before this script.');
}

// Initialize Chart.js components
Chart.register(
  Chart.CategoryScale,
  Chart.LinearScale,
  Chart.PointElement,
  Chart.LineElement,
  Chart.BarElement,
  Chart.ArcElement,
  Chart.Title,
  Chart.Tooltip,
  Chart.Legend
);

// Dashboard Component
const Dashboard = () => {
  // State for storing property valuations data
  const [properties, setProperties] = React.useState([]);
  // State for tracking loading status
  const [loading, setLoading] = React.useState(true);
  // State for storing any error messages
  const [error, setError] = React.useState(null);
  // State for tracking filter values
  const [filters, setFilters] = React.useState({
    neighborhood: '',
    minValue: '',
    maxValue: '',
    propertyType: '',
    fromDate: '',
    toDate: ''
  });
  
  // State for chart data and configuration
  const [chartData, setChartData] = React.useState(null);
  
  // Refs for chart containers
  const valueDistributionChartRef = React.useRef(null);
  const neighborhoodChartRef = React.useRef(null);

  // Function to fetch property valuations from the API
  const fetchPropertyValuations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Build query parameters based on active filters
      const queryParams = new URLSearchParams();
      
      if (filters.minValue) queryParams.append('min_value', filters.minValue);
      if (filters.maxValue) queryParams.append('max_value', filters.maxValue);
      if (filters.propertyType) queryParams.append('property_type', filters.propertyType);
      
      // Make the API request
      const response = await fetch(`/api/valuations?${queryParams.toString()}`);
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      
      const data = await response.json();
      setProperties(data);
    } catch (err) {
      console.error('Error fetching property valuations:', err);
      setError('Failed to load property valuations. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Function to handle filter changes
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prevFilters => ({
      ...prevFilters,
      [name]: value
    }));
  };

  // Function to apply filters
  const applyFilters = (e) => {
    e.preventDefault();
    fetchPropertyValuations();
  };

  // Function to reset all filters
  const resetFilters = () => {
    setFilters({
      neighborhood: '',
      minValue: '',
      maxValue: '',
      propertyType: '',
      fromDate: '',
      toDate: ''
    });
    // Re-fetch data with cleared filters
    fetchPropertyValuations();
  };

  // Function to format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value);
  };

  // Function to extract neighborhood from address
  const extractNeighborhood = (address) => {
    // This is a simple extraction - you might need more complex logic
    // based on your actual address format
    const parts = address.split(',');
    if (parts.length >= 2) {
      return parts[1].trim();
    }
    return 'Unknown';
  };

  // Function to filter properties by neighborhood if that filter is active
  const filteredProperties = properties.filter(property => {
    // Apply neighborhood filter (client-side filtering since API doesn't support it)
    if (filters.neighborhood && !extractNeighborhood(property.address).toLowerCase().includes(filters.neighborhood.toLowerCase())) {
      return false;
    }
    
    // Apply date filters if they exist (assuming valuation_date is available)
    if (filters.fromDate && new Date(property.valuation_date) < new Date(filters.fromDate)) {
      return false;
    }
    
    if (filters.toDate && new Date(property.valuation_date) > new Date(filters.toDate)) {
      return false;
    }
    
    return true;
  });
  
  // Load property valuations when component mounts
  React.useEffect(() => {
    fetchPropertyValuations();
  }, []); // Empty dependency array means this runs once on mount
  
  // Prepare chart data when properties change
  React.useEffect(() => {
    if (filteredProperties.length > 0) {
      prepareChartData();
    }
  }, [filteredProperties]);
  
  // Function to prepare chart data based on filtered properties
  const prepareChartData = () => {
    // Group properties by value ranges for distribution chart
    const valueRanges = [
      '< $200K', 
      '$200K - $300K', 
      '$300K - $400K', 
      '$400K - $500K', 
      '$500K - $750K', 
      '$750K - $1M', 
      '> $1M'
    ];
    
    const valueCounts = [0, 0, 0, 0, 0, 0, 0];
    
    filteredProperties.forEach(property => {
      const value = property.estimated_value;
      if (value < 200000) {
        valueCounts[0]++;
      } else if (value < 300000) {
        valueCounts[1]++;
      } else if (value < 400000) {
        valueCounts[2]++;
      } else if (value < 500000) {
        valueCounts[3]++;
      } else if (value < 750000) {
        valueCounts[4]++;
      } else if (value < 1000000) {
        valueCounts[5]++;
      } else {
        valueCounts[6]++;
      }
    });
    
    // Group properties by neighborhood for neighborhood chart
    const neighborhoods = {};
    
    filteredProperties.forEach(property => {
      const neighborhood = extractNeighborhood(property.address);
      if (!neighborhoods[neighborhood]) {
        neighborhoods[neighborhood] = {
          count: 0,
          totalValue: 0
        };
      }
      neighborhoods[neighborhood].count++;
      neighborhoods[neighborhood].totalValue += property.estimated_value;
    });
    
    // Convert to arrays for chart data
    const neighborhoodNames = Object.keys(neighborhoods)
      .sort((a, b) => neighborhoods[b].count - neighborhoods[a].count)
      .slice(0, 8); // Top 8 neighborhoods by count
    
    const neighborhoodCounts = neighborhoodNames.map(name => neighborhoods[name].count);
    const neighborhoodAvgValues = neighborhoodNames.map(name => 
      Math.round(neighborhoods[name].totalValue / neighborhoods[name].count)
    );
    
    // Set chart data
    setChartData({
      valueDistribution: {
        labels: valueRanges,
        datasets: [
          {
            label: 'Property Count',
            data: valueCounts,
            backgroundColor: 'rgba(54, 162, 235, 0.6)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          }
        ]
      },
      neighborhoodComparison: {
        labels: neighborhoodNames,
        datasets: [
          {
            label: 'Property Count',
            data: neighborhoodCounts,
            backgroundColor: 'rgba(153, 102, 255, 0.6)',
            borderColor: 'rgba(153, 102, 255, 1)',
            borderWidth: 1,
            yAxisID: 'y'
          },
          {
            label: 'Average Value',
            data: neighborhoodAvgValues,
            type: 'line',
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderWidth: 2,
            fill: false,
            yAxisID: 'y1'
          }
        ]
      }
    });
  };

  return React.createElement(
    'div', 
    { className: 'container mx-auto px-4 py-8' },
    [
      React.createElement('h1', { className: 'text-3xl font-bold text-gray-800 mb-6', key: 'title' }, 'Property Valuations Dashboard'),
      
      // Filters Section
      React.createElement(
        'div', 
        { className: 'bg-white rounded-lg shadow-md p-6 mb-8', key: 'filters' },
        [
          React.createElement('h2', { className: 'text-xl font-semibold mb-4', key: 'filters-title' }, 'Filters'),
          React.createElement(
            'form', 
            { onSubmit: applyFilters, key: 'filters-form' },
            [
              React.createElement(
                'div', 
                { className: 'grid grid-cols-1 md:grid-cols-3 gap-4', key: 'filters-grid' },
                [
                  // Neighborhood Filter
                  React.createElement(
                    'div',
                    { key: 'neighborhood-filter' },
                    [
                      React.createElement('label', { className: 'block text-sm font-medium text-gray-700 mb-1', key: 'neighborhood-label' }, 'Neighborhood'),
                      React.createElement('input', {
                        type: 'text',
                        name: 'neighborhood',
                        value: filters.neighborhood,
                        onChange: handleFilterChange,
                        className: 'w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500',
                        placeholder: 'e.g. Richland',
                        key: 'neighborhood-input'
                      })
                    ]
                  ),
                  
                  // Min Value Filter
                  React.createElement(
                    'div',
                    { key: 'minValue-filter' },
                    [
                      React.createElement('label', { className: 'block text-sm font-medium text-gray-700 mb-1', key: 'minValue-label' }, 'Min Value ($)'),
                      React.createElement('input', {
                        type: 'number',
                        name: 'minValue',
                        value: filters.minValue,
                        onChange: handleFilterChange,
                        className: 'w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500',
                        placeholder: 'Min value',
                        key: 'minValue-input'
                      })
                    ]
                  ),
                  
                  // Max Value Filter
                  React.createElement(
                    'div',
                    { key: 'maxValue-filter' },
                    [
                      React.createElement('label', { className: 'block text-sm font-medium text-gray-700 mb-1', key: 'maxValue-label' }, 'Max Value ($)'),
                      React.createElement('input', {
                        type: 'number',
                        name: 'maxValue',
                        value: filters.maxValue,
                        onChange: handleFilterChange,
                        className: 'w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500',
                        placeholder: 'Max value',
                        key: 'maxValue-input'
                      })
                    ]
                  ),
                  
                  // Property Type Filter
                  React.createElement(
                    'div',
                    { key: 'propertyType-filter' },
                    [
                      React.createElement('label', { className: 'block text-sm font-medium text-gray-700 mb-1', key: 'propertyType-label' }, 'Property Type'),
                      React.createElement(
                        'select',
                        {
                          name: 'propertyType',
                          value: filters.propertyType,
                          onChange: handleFilterChange,
                          className: 'w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500',
                          key: 'propertyType-select'
                        },
                        [
                          React.createElement('option', { value: '', key: 'type-all' }, 'All Types'),
                          React.createElement('option', { value: 'residential', key: 'type-residential' }, 'Residential'),
                          React.createElement('option', { value: 'commercial', key: 'type-commercial' }, 'Commercial'),
                          React.createElement('option', { value: 'land', key: 'type-land' }, 'Land'),
                          React.createElement('option', { value: 'multifamily', key: 'type-multifamily' }, 'Multi-Family')
                        ]
                      )
                    ]
                  ),
                  
                  // From Date Filter
                  React.createElement(
                    'div',
                    { key: 'fromDate-filter' },
                    [
                      React.createElement('label', { className: 'block text-sm font-medium text-gray-700 mb-1', key: 'fromDate-label' }, 'From Date'),
                      React.createElement('input', {
                        type: 'date',
                        name: 'fromDate',
                        value: filters.fromDate,
                        onChange: handleFilterChange,
                        className: 'w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500',
                        key: 'fromDate-input'
                      })
                    ]
                  ),
                  
                  // To Date Filter
                  React.createElement(
                    'div',
                    { key: 'toDate-filter' },
                    [
                      React.createElement('label', { className: 'block text-sm font-medium text-gray-700 mb-1', key: 'toDate-label' }, 'To Date'),
                      React.createElement('input', {
                        type: 'date',
                        name: 'toDate',
                        value: filters.toDate,
                        onChange: handleFilterChange,
                        className: 'w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500',
                        key: 'toDate-input'
                      })
                    ]
                  )
                ]
              ),
              
              // Filter Action Buttons
              React.createElement(
                'div', 
                { className: 'mt-4 flex justify-end space-x-2', key: 'filter-actions' },
                [
                  React.createElement(
                    'button',
                    {
                      type: 'button',
                      onClick: resetFilters,
                      className: 'px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500',
                      key: 'reset-button'
                    },
                    'Reset Filters'
                  ),
                  React.createElement(
                    'button',
                    {
                      type: 'submit',
                      className: 'px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500',
                      key: 'apply-button'
                    },
                    'Apply Filters'
                  )
                ]
              )
            ]
          )
        ]
      ),
      
      // Results Section
      React.createElement(
        'div', 
        { className: 'bg-white rounded-lg shadow-md overflow-hidden', key: 'results' },
        [
          React.createElement(
            'h2', 
            { className: 'text-xl font-semibold p-6 border-b', key: 'results-title' },
            `Properties (${filteredProperties.length})`
          ),
          
          // Loading State
          loading && React.createElement(
            'div', 
            { className: 'flex justify-center items-center p-12', key: 'loading' },
            [
              React.createElement('div', { className: 'animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500', key: 'loading-spinner' }),
              React.createElement('span', { className: 'ml-3 text-gray-600', key: 'loading-text' }, 'Loading property data...')
            ]
          ),
          
          // Error State
          error && React.createElement(
            'div', 
            { className: 'bg-red-50 p-4 border-l-4 border-red-500', key: 'error' },
            React.createElement(
              'div', 
              { className: 'flex' },
              [
                React.createElement(
                  'div', 
                  { className: 'flex-shrink-0', key: 'error-icon-container' },
                  React.createElement(
                    'svg', 
                    { className: 'h-5 w-5 text-red-400', xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 20 20', fill: 'currentColor', key: 'error-icon' },
                    React.createElement('path', { 
                      fillRule: 'evenodd', 
                      d: 'M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z', 
                      clipRule: 'evenodd',
                      key: 'error-path'
                    })
                  )
                ),
                React.createElement(
                  'div', 
                  { className: 'ml-3', key: 'error-text-container' },
                  React.createElement('p', { className: 'text-sm text-red-700', key: 'error-text' }, error)
                )
              ]
            )
          ),
          
          // Results Table (Only shown when not loading and no errors)
          !loading && !error && (
            filteredProperties.length > 0 ? (
              React.createElement(
                'div', 
                { className: 'overflow-x-auto', key: 'table-container' },
                React.createElement(
                  'table', 
                  { className: 'min-w-full divide-y divide-gray-200', key: 'results-table' },
                  [
                    React.createElement(
                      'thead', 
                      { className: 'bg-gray-50', key: 'table-head' },
                      React.createElement(
                        'tr',
                        { key: 'header-row' },
                        [
                          React.createElement('th', { scope: 'col', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider', key: 'header-id' }, 'Property ID'),
                          React.createElement('th', { scope: 'col', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider', key: 'header-address' }, 'Address'),
                          React.createElement('th', { scope: 'col', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider', key: 'header-neighborhood' }, 'Neighborhood'),
                          React.createElement('th', { scope: 'col', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider', key: 'header-value' }, 'Estimated Value'),
                          React.createElement('th', { scope: 'col', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider', key: 'header-confidence' }, 'Confidence'),
                          React.createElement('th', { scope: 'col', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider', key: 'header-model' }, 'Model'),
                          React.createElement('th', { scope: 'col', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider', key: 'header-date' }, 'Valuation Date'),
                          React.createElement('th', { scope: 'col', className: 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider', key: 'header-actions' }, 'Actions')
                        ]
                      )
                    ),
                    React.createElement(
                      'tbody', 
                      { className: 'bg-white divide-y divide-gray-200', key: 'table-body' },
                      filteredProperties.map((property) => (
                        React.createElement(
                          'tr', 
                          { key: property.property_id, className: 'hover:bg-gray-50' },
                          [
                            React.createElement('td', { className: 'px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900', key: `${property.property_id}-id` }, property.property_id),
                            React.createElement('td', { className: 'px-6 py-4 whitespace-nowrap text-sm text-gray-500', key: `${property.property_id}-address` }, property.address),
                            React.createElement('td', { className: 'px-6 py-4 whitespace-nowrap text-sm text-gray-500', key: `${property.property_id}-neighborhood` }, extractNeighborhood(property.address)),
                            React.createElement('td', { className: 'px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium', key: `${property.property_id}-value` }, formatCurrency(property.estimated_value)),
                            React.createElement(
                              'td', 
                              { className: 'px-6 py-4 whitespace-nowrap', key: `${property.property_id}-confidence` },
                              React.createElement(
                                'div', 
                                { className: 'flex items-center' },
                                [
                                  React.createElement(
                                    'div', 
                                    { className: 'relative w-full h-2 bg-gray-200 rounded', key: `${property.property_id}-confidence-bar-container` },
                                    React.createElement('div', {
                                      className: 'absolute top-0 left-0 h-2 bg-green-500 rounded',
                                      style: { width: `${property.confidence_score * 100}%` },
                                      key: `${property.property_id}-confidence-bar`
                                    })
                                  ),
                                  React.createElement(
                                    'span', 
                                    { className: 'ml-2 text-sm text-gray-500', key: `${property.property_id}-confidence-text` },
                                    `${Math.round(property.confidence_score * 100)}%`
                                  )
                                ]
                              )
                            ),
                            React.createElement('td', { className: 'px-6 py-4 whitespace-nowrap text-sm text-gray-500', key: `${property.property_id}-model` }, property.model_used),
                            React.createElement('td', { className: 'px-6 py-4 whitespace-nowrap text-sm text-gray-500', key: `${property.property_id}-date` }, new Date(property.valuation_date).toLocaleDateString()),
                            React.createElement(
                              'td', 
                              { className: 'px-6 py-4 whitespace-nowrap text-sm font-medium', key: `${property.property_id}-actions` },
                              React.createElement(
                                'a', 
                                { href: `/property/${property.property_id}`, className: 'text-blue-600 hover:text-blue-900', key: `${property.property_id}-view-link` },
                                'View Details'
                              )
                            )
                          ]
                        )
                      ))
                    )
                  ]
                )
              )
            ) : (
              React.createElement(
                'div', 
                { className: 'p-6 text-center text-gray-500', key: 'no-results' },
                'No properties found matching your filters. Try adjusting your search criteria.'
              )
            )
          )
        ]
      ),
      
      // Summary Stats (Only when data is loaded)
      !loading && !error && filteredProperties.length > 0 && React.createElement(
        'div', 
        { className: 'mt-8 grid grid-cols-1 md:grid-cols-4 gap-4', key: 'summary-stats' },
        [
          // Average Value Card
          React.createElement(
            'div', 
            { className: 'bg-white rounded-lg shadow-md p-6', key: 'avg-value-card' },
            [
              React.createElement('h3', { className: 'text-sm text-gray-500 uppercase tracking-wider mb-1', key: 'avg-value-title' }, 'Average Value'),
              React.createElement(
                'p', 
                { className: 'text-2xl font-bold text-gray-900', key: 'avg-value' },
                formatCurrency(filteredProperties.reduce((acc, p) => acc + p.estimated_value, 0) / filteredProperties.length)
              )
            ]
          ),
          
          // Highest Value Card
          React.createElement(
            'div', 
            { className: 'bg-white rounded-lg shadow-md p-6', key: 'high-value-card' },
            [
              React.createElement('h3', { className: 'text-sm text-gray-500 uppercase tracking-wider mb-1', key: 'high-value-title' }, 'Highest Value'),
              React.createElement(
                'p', 
                { className: 'text-2xl font-bold text-gray-900', key: 'high-value' },
                formatCurrency(Math.max(...filteredProperties.map(p => p.estimated_value)))
              )
            ]
          ),
          
          // Lowest Value Card
          React.createElement(
            'div', 
            { className: 'bg-white rounded-lg shadow-md p-6', key: 'low-value-card' },
            [
              React.createElement('h3', { className: 'text-sm text-gray-500 uppercase tracking-wider mb-1', key: 'low-value-title' }, 'Lowest Value'),
              React.createElement(
                'p', 
                { className: 'text-2xl font-bold text-gray-900', key: 'low-value' },
                formatCurrency(Math.min(...filteredProperties.map(p => p.estimated_value)))
              )
            ]
          ),
          
          // Average Confidence Card
          React.createElement(
            'div', 
            { className: 'bg-white rounded-lg shadow-md p-6', key: 'avg-confidence-card' },
            [
              React.createElement('h3', { className: 'text-sm text-gray-500 uppercase tracking-wider mb-1', key: 'avg-confidence-title' }, 'Avg. Confidence'),
              React.createElement(
                'div', 
                { className: 'flex items-center', key: 'avg-confidence-container' },
                [
                  React.createElement(
                    'div', 
                    { className: 'relative w-full h-3 bg-gray-200 rounded', key: 'avg-confidence-bar-container' },
                    React.createElement(
                      'div', 
                      { 
                        className: 'absolute top-0 left-0 h-3 bg-green-500 rounded',
                        style: { width: `${(filteredProperties.reduce((acc, p) => acc + p.confidence_score, 0) / filteredProperties.length) * 100}%` },
                        key: 'avg-confidence-bar'
                      }
                    )
                  ),
                  React.createElement(
                    'span', 
                    { className: 'ml-2 text-lg font-bold text-gray-900', key: 'avg-confidence-text' },
                    `${Math.round((filteredProperties.reduce((acc, p) => acc + p.confidence_score, 0) / filteredProperties.length) * 100)}%`
                  )
                ]
              )
            ]
          )
        ]
      ),
      
      // Data Visualizations
      !loading && !error && filteredProperties.length > 0 && chartData && React.createElement(
        'div', 
        { className: 'mt-8', key: 'visualizations' },
        [
          React.createElement('h2', { className: 'text-xl font-semibold mb-4', key: 'visualizations-title' }, 'Valuation Analytics'),
          
          React.createElement(
            'div', 
            { className: 'grid grid-cols-1 lg:grid-cols-2 gap-8', key: 'charts-grid' },
            [
              // Property Value Distribution Chart
              React.createElement(
                'div', 
                { className: 'bg-white rounded-lg shadow-md p-6', key: 'value-chart-card' },
                [
                  React.createElement('h3', { className: 'text-lg font-semibold mb-4', key: 'value-chart-title' }, 'Value Distribution'),
                  React.createElement(
                    'div', 
                    { className: 'h-80', key: 'value-chart-container' },
                    React.createElement(BarChart, { 
                      data: chartData.valueDistribution, 
                      options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: 'top',
                          },
                          title: {
                            display: true,
                            text: 'Property Value Distribution'
                          },
                          tooltip: {
                            callbacks: {
                              label: function(context) {
                                return `${context.dataset.label}: ${context.raw} properties`;
                              }
                            }
                          }
                        },
                        scales: {
                          y: {
                            beginAtZero: true,
                            title: {
                              display: true,
                              text: 'Number of Properties'
                            },
                            ticks: {
                              precision: 0
                            }
                          }
                        }
                      }
                    })
                  )
                ]
              ),
              
              // Neighborhood Comparison Chart
              React.createElement(
                'div', 
                { className: 'bg-white rounded-lg shadow-md p-6', key: 'neighborhood-chart-card' },
                [
                  React.createElement('h3', { className: 'text-lg font-semibold mb-4', key: 'neighborhood-chart-title' }, 'Neighborhood Analysis'),
                  React.createElement(
                    'div', 
                    { className: 'h-80', key: 'neighborhood-chart-container' },
                    React.createElement(BarChart, { 
                      data: chartData.neighborhoodComparison, 
                      options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: 'top',
                          },
                          title: {
                            display: true,
                            text: 'Property Count and Values by Neighborhood'
                          },
                          tooltip: {
                            callbacks: {
                              label: function(context) {
                                const label = context.dataset.label;
                                const value = context.raw;
                                return label === 'Average Value' 
                                  ? `${label}: ${formatCurrency(value)}` 
                                  : `${label}: ${value}`;
                              }
                            }
                          }
                        },
                        scales: {
                          y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                              display: true,
                              text: 'Number of Properties'
                            },
                            beginAtZero: true,
                            ticks: {
                              precision: 0
                            }
                          },
                          y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                              display: true,
                              text: 'Average Value ($)'
                            },
                            beginAtZero: true,
                            grid: {
                              drawOnChartArea: false,
                            },
                            ticks: {
                              callback: function(value) {
                                return '$' + value.toLocaleString();
                              }
                            }
                          }
                        }
                      }
                    })
                  )
                ]
              )
            ]
          )
        ]
      )
    ]
  );
};

// Bar Chart Component - Wrapper for Chart.js
const BarChart = ({ data, options }) => {
  const chartRef = React.useRef(null);
  const [chart, setChart] = React.useState(null);
  
  React.useEffect(() => {
    // Clear previous chart instance if exists
    if (chart) {
      chart.destroy();
    }
    
    // Create new chart
    if (chartRef.current) {
      const newChart = new Chart(chartRef.current, {
        type: 'bar',
        data: data,
        options: options
      });
      
      setChart(newChart);
    }
    
    // Cleanup function
    return () => {
      if (chart) {
        chart.destroy();
      }
    };
  }, [data, options]);
  
  return React.createElement('canvas', { ref: chartRef });
};

// Initialize Dashboard when the page loads
document.addEventListener('DOMContentLoaded', function() {
  const dashboardRoot = document.getElementById('dashboardRoot');
  if (dashboardRoot) {
    ReactDOM.render(React.createElement(Dashboard), dashboardRoot);
  }
});